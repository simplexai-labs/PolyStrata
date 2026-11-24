// attend_batch(): a prompt batch's attention on the matrix units, after metal/kernels/qsa_prompt.metal's qsa_flash.
// A threadgroup takes QB of the call's tokens for one query head and walks the keys they attend KC at a time with
// an online softmax.  SIMD group g owns head values 64 g .. 64 g + 63: its share of S = Q K^T, summed over the
// groups through threadgroup memory, and those 64 values of the output, P V.  The softmax takes a row a lane.
// Q is rounded to fp16 in threadgroup memory; K and V are fp16 tensors in device memory that matmul2d reads where they
// lie: the cache itself, or (a q8 cache, or a ring) poly_flash_kv's fp16 copy of the call's keys, p[9] and p[4], key 0
// the call's first.  P is fp16; the sums are fp32.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct Args {   // attention.metal's
    device void* p[10]; long pos;
    int dim,heads,kv,rows,window,ring,whole,stride,rot,sections,kind;
    float scale,eps;
};
constant constexpr int QB=16,KC=64;   // tokens a threadgroup, keys a step
inline half cache_half(device void* raw,device void* codes,device void* scales,ulong at) {
    return codes?half(float(static_cast<device char*>(codes)[at])*float(static_cast<device half*>(scales)[at/32]))
                :static_cast<device half*>(raw)[at];
}
// The call's first key: the window's for its first token, else the context's
inline long call_first(constant Args& a) { return a.window>0?max(0L,a.pos-a.window+1):0; }
// K and V of the keys the call reads, from call_first() to its last, in fp16 as [key][kv head][value] (p[9], p[4])
kernel void poly_flash_kv(constant Args& a [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    const ulong row=ulong(a.kv)*a.dim,keys=ulong(a.pos+a.rows-call_first(a));
    if(i>=keys*row) return;
    const long at=call_first(a)+long(i/row);
    const ulong src=ulong(a.ring?at%a.ring:at)*row+i%row;
    static_cast<device half*>(a.p[9])[i]=cache_half(a.p[1],a.p[5],a.p[6],src);
    static_cast<device half*>(a.p[4])[i]=cache_half(a.p[2],a.p[7],a.p[8],src);
}
template<int D>
kernel void poly_flash(constant Args& a [[buffer(0)]],uint2 tg [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]],ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NSG=D/64,NTH=NSG*32;
    threadgroup half qs[QB*D];          // the tokens' queries: [token][value]
    threadgroup float sp[NSG][QB*KC];   // each group's share of S: [token][key]
    threadgroup half p[QB*KC];          // the weights
    threadgroup float fac[QB],lsum[QB],mrun[QB];
    const int h=int(tg.x),t0=int(tg.y)*QB,kh=h/(a.heads/a.kv),n=min(QB,a.rows-t0);
    const long start=a.pos,last=a.whole?start+a.rows-1:start+t0+n-1;
    const long first=a.window>0?max(0L,start+t0-a.window+1):0;
    device const float* q=static_cast<device const float*>(a.p[0]);
    for(int i=tid;i<QB*D;i+=NTH) {
        const int r=i/D;
        qs[i]=r<n?half(q[(ulong(t0+r)*a.heads+h)*D+i%D]):half(0.0h);
    }
    if(tid<QB) { mrun[tid]=-INFINITY;lsum[tid]=0.0f; }
    constexpr auto dS=matmul2d_descriptor(QB,KC,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dO=matmul2d_descriptor(QB,64,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS,execution_simdgroup> opS;
    matmul2d<dO,execution_simdgroup> opO;
    tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> tQ(qs,dextents<int32_t,2>(D,QB));
    tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> tP(p,dextents<int32_t,2>(KC,QB));
    // K and V of this head: [value][key] with the keys a row of kv heads apart, key 0 the call's first
    const long kfirst=call_first(a);
    const bool copied=a.p[9]!=nullptr;
    device half* kb=copied?static_cast<device half*>(a.p[9]):static_cast<device half*>(a.p[1])+ulong(kfirst)*a.kv*D;
    device half* vb=copied?static_cast<device half*>(a.p[4]):static_cast<device half*>(a.p[2])+ulong(kfirst)*a.kv*D;
    const int nkeys=int(start+a.rows-kfirst);
    tensor<device half,dextents<int32_t,2>,tensor_inline> tK(kb+kh*D,dextents<int32_t,2>(D,nkeys),array<int32_t,2>({1,a.kv*D}));
    tensor<device half,dextents<int32_t,2>,tensor_inline> tV(vb+kh*D,dextents<int32_t,2>(D,nkeys),array<int32_t,2>({1,a.kv*D}));
    auto tPh=tP.template slice<32,QB>(0,0);
    auto rv=tV.slice(0,0);
    using PType=__tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>;
    using VType=__tensor_ops_detail::__remove_addrspace_t<decltype(rv)>;
    // tot: the running output; acc: one step's, fresh each step (qsa_flash: a run does not accumulate into elements
    // rescaled in place on the M4)
    auto acc=opO.template get_destination_cooperative_tensor<PType,VType,float>();
    auto tot=opO.template get_destination_cooperative_tensor<PType,VType,float>();
    for(uint16_t i=0;i<tot.get_capacity();++i) if(tot.is_valid_element(i)) tot[i]=0.0f;
    const int d0=sg*64;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(long c0=first;c0<=last;c0+=KC) {
        const int nc=int(min(long(KC),last-c0+1)),rel=int(c0-kfirst);
        {
            auto rk=tK.slice(d0,rel);
            auto tA=tQ.template slice<32,QB>(d0,0);
            using AType=__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>;
            using KType=__tensor_ops_detail::__remove_addrspace_t<decltype(rk)>;
            auto s=opS.template get_destination_cooperative_tensor<AType,KType,float>();
            for(uint16_t i=0;i<s.get_capacity();++i) if(s.is_valid_element(i)) s[i]=0.0f;
            for(int k0=d0;k0<d0+64;k0+=32) {
                tA=tQ.template slice<32,QB>(k0,0);
                rk=tK.slice(k0,rel);
                opS.run(tA,rk,s);
            }
            tensor<threadgroup float,dextents<int32_t,2>,tensor_inline> tS(sp[sg],dextents<int32_t,2>(KC,QB));
            s.store(tS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // the softmax, TPR lanes of one SIMD group a token, KPT keys a lane: only its own keys (to itself, or the call's
        // last with whole; within the window) weigh.  The maximum is taken across the lanes; the sum of the weights
        // stays one lane's, in key order, so the result is the one-lane softmax's to the bit.
        {
            constexpr int TPR=NTH/QB,KPT=KC/TPR;
            const int r=tid/TPR,j=tid%TPR,k1=j*KPT;
            int c1=0,c2=0;
            float m0=0.0f,m=0.0f;
            if(r<n) {
                const long own=start+t0+r,lo=a.window>0?max(0L,own-a.window+1):0,hi=a.whole?start+a.rows-1:own;
                c1=int(clamp(lo-c0,0L,long(nc)));c2=int(clamp(hi-c0+1,0L,long(nc)));
                m0=mrun[r];
                m=m0;
                for(int c=max(c1,k1);c<min(c2,k1+KPT);++c) {
                    float v=0.0f;
                    for(int g=0;g<NSG;++g) v+=sp[g][r*KC+c];
                    sp[0][r*KC+c]=v*a.scale;
                    m=max(m,v*a.scale);
                }
                for(int o=1;o<TPR;o<<=1) m=max(m,simd_shuffle_xor(m,ushort(o)));
                for(int c=k1;c<k1+KPT;++c) {
                    const float e=c>=c1 && c<c2?exp(sp[0][r*KC+c]-m):0.0f;
                    sp[0][r*KC+c]=e;
                    p[r*KC+c]=half(e);
                }
            } else {
                for(int c=k1;c<k1+KPT;++c) p[r*KC+c]=half(0.0h);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);   // a token's lanes are one SIMD group's
            if(j==0) {
                if(r<n) {
                    const float f=m==-INFINITY?1.0f:exp(m0-m);
                    float l=lsum[r]*f;
                    for(int c=c1;c<c2;++c) l+=sp[0][r*KC+c];
                    mrun[r]=m;lsum[r]=l;fac[r]=f;
                } else fac[r]=0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint16_t i=0;i<acc.get_capacity();++i) if(acc.is_valid_element(i)) acc[i]=0.0f;
        for(int k0=0;k0<KC;k0+=32) {
            tPh=tP.template slice<32,QB>(k0,0);
            rv=tV.slice(d0,rel+k0);
            opO.run(tPh,rv,acc);
        }
        for(uint16_t i=0;i<acc.get_capacity();++i)
            if(acc.is_valid_element(i)) tot[i]=fma(tot[i],fac[acc.get_multidimensional_index(i)[1]],acc[i]);
    }
    device float* out=static_cast<device float*>(a.p[3]);
    for(uint16_t i=0;i<tot.get_capacity();++i)
        if(tot.is_valid_element(i)) {
            const auto ix=tot.get_multidimensional_index(i);   // (value, token)
            const int r=ix[1];
            if(r<n) out[(ulong(t0+r)*a.heads+h)*D+d0+ix[0]]=tot[i]/lsum[r];
        }
}
typedef decltype(poly_flash<64>) poly_flash_t;
template [[host_name("poly_flash_64")]] kernel poly_flash_t poly_flash<64>;
template [[host_name("poly_flash_128")]] kernel poly_flash_t poly_flash<128>;
template [[host_name("poly_flash_256")]] kernel poly_flash_t poly_flash<256>;
