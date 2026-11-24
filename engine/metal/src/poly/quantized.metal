// dequant.metal (the block decoders, after the quant header) is prepended by quantized.cpp.
struct QArgs {
    device void* p[24]; ulong stride,xstride,ystride,offset;
    int ni,no,rows,count,every,kind;
};
kernel void poly_quantize(constant QArgs& a [[buffer(0)]],uint2 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]],uint2 shape [[threads_per_threadgroup]]) {
    int row=a.p[2]?static_cast<device int*>(a.p[2])[group.y]:int(group.y);
    float v=static_cast<device float*>(a.p[0])[ulong(row)*a.xstride+group.x*32+lane];
    if(a.kind==1) {float u=static_cast<device float*>(a.p[3])[ulong(row)*a.xstride+group.x*32+lane];v=v/(1+exp(-v))*u;}
    float d=simd_max(abs(v))/127.0f,sum=v;
    for(uint k=16;k;k>>=1) sum+=simd_shuffle_xor(sum,k);
    device uchar* dst=static_cast<device uchar*>(a.p[1])+(ulong(group.y)*(a.ni/32)+group.x)*36;
    reinterpret_cast<device char*>(dst+4)[lane]=char(d?round(v/d):0);
    if(!lane) { reinterpret_cast<device half*>(dst)[0]=half(d);reinterpret_cast<device half*>(dst)[1]=half(sum); }
}
// Mmq::quantize() on the matrix-unit path (mm.metal): the rows in fp16, row i from row ids[i] of x; four values a
// thread (a row is whole 64-value steps), a threadgroup row of them to each row
kernel void poly_tohalf(constant QArgs& a [[buffer(0)]],uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],uint2 shape [[threads_per_threadgroup]]) {
    const uint k=(tg.x*shape.x+tid)*4,row=tg.y;
    if(k>=uint(a.ni)) return;
    const int from=a.p[2]?static_cast<device int*>(a.p[2])[row]:int(row);
    device const float* x=static_cast<device float*>(a.p[0])+ulong(from)*a.xstride+k;
    const float4 v=a.xstride%4?float4(x[0],x[1],x[2],x[3]):*reinterpret_cast<device const float4*>(x);
    *reinterpret_cast<device half4*>(static_cast<device half*>(a.p[1])+ulong(row)*a.ni+k)=half4(v);
}
// Sum over the 16-value piece il of a block of its weights times xv (the row's values there; xs their sum, where the
// dot asks for it with kSum): from the block decoder, or with a dot of the type's own (dot_<type>)
template<typename Block,void Deq(device const Block*,short,thread float4x4&)>
struct DeqDot {
    enum { kSum=0 };
    static float run(device const Block* b,short il,thread const float* xv,float) {
        float4x4 d; Deq(b,il,d);
        float s=0;
        for(int j=0;j<16;++j) s+=d[j/4][j%4]*xv[j];
        return s;
    }
};
// The types' own dots: the decoders' values with their scales (and minimums, against xs) out of the sum, and wider
// loads.  The same values as dequant.metal's decoders, summed in another order.
struct dot_q8_0 {
    enum { kSum=0 };
    static float run(device const block_q8_0* b,short il,thread const float* xv,float) {
        device const ushort* q=reinterpret_cast<device const ushort*>(b->qs+16*il);   // 2-byte aligned (34-byte blocks)
        float s=0;
        for(int j=0;j<8;++j) {
            const ushort v=q[j];
            s+=float(as_type<char>(uchar(v&255)))*xv[2*j]+float(as_type<char>(uchar(v>>8)))*xv[2*j+1];
        }
        return float(b->d)*s;
    }
};
struct dot_q2_0 {   // 2-bit codes, the lowest first, minus 1
    enum { kSum=1 };
    static float run(device const block_q2_0* b,short il,thread const float* xv,float xs) {
        device const ushort* q=reinterpret_cast<device const ushort*>(b->qs+4*il);
        const uint v=uint(q[0])|(uint(q[1])<<16);
        float s=0;
        for(int j=0;j<16;++j) s+=float((v>>(2*j))&3)*xv[j];
        return float(b->d)*(s-xs);
    }
};
struct dot_nvfp4 {
    enum { kSum=0 };
    static float run(device const block_nvfp4* b,short il,thread const float* xv,float) {
        device const uint* q=reinterpret_cast<device const uint*>(b->qs+8*il);
        const uint lo=q[0],hi=q[1];
        float s=0;
        for(int j=0;j<4;++j)
            s+=kvalues_nvfp4_f[(lo>>(8*j))&15]*xv[j]+kvalues_nvfp4_f[(hi>>(8*j))&15]*xv[4+j]+
               kvalues_nvfp4_f[(lo>>(8*j+4))&15]*xv[8+j]+kvalues_nvfp4_f[(hi>>(8*j+4))&15]*xv[12+j];
        return nvfp4_scale(b->d[il])*s;
    }
};
struct dot_q4_K {
    enum { kSum=1 };
    static float run(device const block_q4_K* b,short il,thread const float* xv,float xs) {
        const uchar2 sc=get_scale_min_k4_just2((il/4)*2,(il&3)/2,b->scales);
        device const uint* q=reinterpret_cast<device const uint*>(b->qs+(il/4)*32+16*(il&1));   // 144-byte blocks
        const uint sh=(il&2)?4:0;
        float s=0;
        for(int i=0;i<4;++i) {
            const uchar4 c=as_type<uchar4>((q[i]>>sh)&0x0F0F0F0Fu);
            s+=float(c[0])*xv[4*i]+float(c[1])*xv[4*i+1]+float(c[2])*xv[4*i+2]+float(c[3])*xv[4*i+3];
        }
        return float(b->d)*float(sc[0])*s-float(b->dmin)*float(sc[1])*xs;
    }
};
struct dot_q6_K {   // dequantize_q6_K's 6-bit codes, four a word, minus 32
    enum { kSum=1 };
    static float run(device const block_q6_K* b,short il,thread const float* xv,float xs) {
        device const ushort* ql=reinterpret_cast<device const ushort*>(b->ql)+32*(il/8)+16*((il/2)&1)+8*(il&1);
        device const ushort* qh=reinterpret_cast<device const ushort*>(b->qh)+16*(il/8)+8*(il&1);
        const float sc=reinterpret_cast<device const int8_t*>(b->scales)[(il%2)+2*(il/2)];
        const short k=(il/2)&3;
        const uint mask1=k>1?(k>2?0xC0C0C0C0u:0x30303030u):(k>0?0x0C0C0C0Cu:0x03030303u),mask2=k>1?0xF0F0F0F0u:0x0F0F0F0Fu;
        const uint shr_h=k>2?2:0,shl_h=k>1?0:(k>0?2:4),shr_l=k>1?4:0;
        float s=0;
        for(int i=0;i<4;++i) {
            const uint low=(uint(ql[2*i])|(uint(ql[2*i+1])<<16))&mask2,high=(uint(qh[2*i])|(uint(qh[2*i+1])<<16))&mask1;
            const uchar4 c=as_type<uchar4>(((high<<shl_h)>>shr_h)|(low>>shr_l));
            s+=float(c[0])*xv[4*i]+float(c[1])*xv[4*i+1]+float(c[2])*xv[4*i+2]+float(c[3])*xv[4*i+3];
        }
        return float(b->d)*sc*(s-32*xs);
    }
};
// eight grid bytes (two words) times xv with the signs' bits flipping xv's
static inline float signed_grid8(uint2 g,uint signs,thread const float* xv) {
    float s=0;
    for(int j=0;j<4;++j)
        s+=float((g.x>>(8*j))&255)*((signs>>j)&1?-xv[j]:xv[j])+float((g.y>>(8*j))&255)*((signs>>(j+4))&1?-xv[4+j]:xv[4+j]);
    return s;
}
struct dot_iq2_s {
    enum { kSum=0 };
    static float run(device const block_iq2_s* b,short il,thread const float* xv,float) {
        const int ib32=il/2,h=il%2;
        device const uint8_t* qs=b->qs+4*ib32+2*h;
        device const uint8_t* signs=qs+QK_K/8;
        const uint8_t qh=b->qh[ib32]>>4*h;
        const float dl=float(b->d)*(0.5f+((b->scales[ib32]>>4*h)&0xf))*0.25f;
        const uint2 g1=as_type<uint2>(iq2s_grid[qs[0]|((qh<<8)&0x300)]),g2=as_type<uint2>(iq2s_grid[qs[1]|((qh<<6)&0x300)]);
        return dl*(signed_grid8(g1,signs[0],xv)+signed_grid8(g2,signs[1],xv+8));
    }
};
struct dot_iq2_xxs {
    enum { kSum=0 };
    static float run(device const block_iq2_xxs* b,short il,thread const float* xv,float) {
        const int ib32=il/2,h=il%2;
        device const uint16_t* q2=b->qs+4*ib32;
        const uint aux_g=uint(q2[0])|(uint(q2[1])<<16),aux_s=uint(q2[2])|(uint(q2[3])<<16);
        const float dl=float(b->d)*(0.5f+(aux_s>>28))*0.25f;
        const uint2 g1=as_type<uint2>(iq2xxs_grid[(aux_g>>(16*h))&255]),g2=as_type<uint2>(iq2xxs_grid[(aux_g>>(16*h+8))&255]);
        return dl*(signed_grid8(g1,ksigns_iq2xs[(aux_s>>(14*h))&127],xv)+signed_grid8(g2,ksigns_iq2xs[(aux_s>>(14*h+7))&127],xv+8));
    }
};
struct dot_iq1_s {   // dl (grid + 1) + ml: dl (the grid's sum) + (dl + ml) xs
    enum { kSum=1 };
    static float run(device const block_iq1_s* b,short il,thread const float* xv,float xs) {
        const int ib32=il/2,h=il%2;
        device const uint8_t* qs=b->qs+4*ib32+2*h;
        const uint16_t qh=b->qh[ib32];
        const float dl=float(b->d)*(2*((qh>>12)&7)+1);
        const float ml=dl*(qh&0x8000?-1-IQ1S_DELTA:-1+IQ1S_DELTA);
        const uint16_t hh=qh>>6*h;
        const uint2 g1=as_type<uint2>(iq1s_grid[qs[0]|((hh<<8)&0x700)]),g2=as_type<uint2>(iq1s_grid[qs[1]|((hh<<5)&0x700)]);
        const char4 c0=as_type<char4>(g1.x),c1=as_type<char4>(g1.y),c2=as_type<char4>(g2.x),c3=as_type<char4>(g2.y);
        float s=0;
        for(int j=0;j<4;++j) s+=float(c0[j])*xv[j]+float(c1[j])*xv[4+j]+float(c2[j])*xv[8+j]+float(c3[j])*xv[12+j];
        return dl*s+(dl+ml)*xs;
    }
};
// A threadgroup is a SIMD group of NR outputs, the rows' values read once for them: 4 where a launch has 8,192 outputs
// or more (qrun() in quantized.cpp), else 1 (poly_mul_<type>_1).  Its 32 lanes take 16 values at a time, 512 apart.
template<typename Block,int QK,void Deq(device const Block*,short,thread float4x4&),typename Dot,int NR>
kernel void poly_mul(constant QArgs& a [[buffer(0)]],uint3 group [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]]) {
    uint row=group.y,gx=group.x,mj=0; ulong yrow=row;
    int no=a.no;
    device const uchar* w=static_cast<device uchar*>(a.p[0]);
    device float* out_y=static_cast<device float*>(a.p[2]);
    if(a.kind==9) {   // native_mmvq_many: matrix j's outputs, NR a group, after the groups of the ones before it
        constant uint* outputs=reinterpret_cast<constant uint*>(&a.stride);   // n_out[j], 32 bits each (quantized.cpp)
        while(mj+1<uint(a.count) && gx>=(outputs[mj]+NR-1)/NR) gx-=(outputs[mj++]+NR-1)/NR;
        w=static_cast<device const uchar*>(a.p[3*mj]);
        out_y=static_cast<device float*>(a.p[3*mj+2]);
        no=int(outputs[mj]);
        yrow=0;
    }
    else if(a.kind==1) { w=static_cast<device const uchar* device const*>(a.p[0])[static_cast<device int*>(a.p[3])[row]]+a.offset; }
    else if(a.kind==2 || a.kind==6) {
        if(a.kind==6 && a.p[4]) row+=static_cast<device int*>(a.p[4])[group.z];
        if(row>=uint(a.rows)) return;
        yrow=row;
        if(a.p[4]) { device const int* bounds=static_cast<device int*>(a.p[4]); if(row<uint(bounds[group.z]) || row>=uint(bounds[group.z+1])) return; }
        w+=ulong(group.z)*a.stride;
        if(a.p[3]) yrow=static_cast<device int*>(a.p[3])[row];
    }
    else if(a.kind==7) {
        device const int* bounds=static_cast<device int*>(a.p[4]);
        if(row<uint(bounds[0]) || row>=uint(bounds[a.count])) return;
        uint lo=0,hi=uint(a.count);
        while(lo+1<hi) {
            uint mid=(lo+hi)>>1;
            if(row<uint(bounds[mid])) hi=mid;
            else lo=mid;
        }
        w+=ulong(lo)*a.stride;
        if(a.p[3]) yrow=static_cast<device int*>(a.p[3])[row];
    }
    else if(a.kind==8) {
        w=static_cast<device const uchar*>(a.p[3*row]);
        out_y=static_cast<device float*>(a.p[3*row+2]);
        yrow=0;
    }
    ulong xrow=row;
    if(a.kind==4 || a.kind==5){
        if(group.z>=uint(*static_cast<device int*>(a.p[5])))return;
        device int* bounds=static_cast<device int*>(a.p[4]);
        if(row<uint(bounds[group.z]) || row>=uint(bounds[group.z+1]))return;
        w=static_cast<device const uchar* device const*>(a.p[0])[group.z]+a.offset;
        if(a.kind==4)xrow=static_cast<device int*>(a.p[3])[row];
        else yrow=static_cast<device int*>(a.p[3])[row];
    }
    device const uchar* x=a.kind==8 || a.kind==9
        ? static_cast<device const uchar*>(a.p[3*(a.kind==8?row:mj)+1])
        : static_cast<device uchar*>(a.p[1])+ulong(xrow/a.every)*a.xstride;
    // fewer in the last group; one a constant, so its loops are straight code (counted at run time, dense matrices
    // were 3-15% slower than before the 4)
    const int o0=int(gx)*NR,nr=NR==1?1:min(NR,no-o0);
    const ulong per=a.ni/QK;
    device const Block* weights=reinterpret_cast<device const Block*>(w)+ulong(o0)*per;
    float acc[NR]={};
    for(int start=lane*16;start<a.ni;start+=32*16) {
        float xv[16];
        if(a.kind==3) for(int j=0;j<16;++j) xv[j]=float(reinterpret_cast<device const half*>(x)[start+j]);
        else {   // a q8_1 block: its scale, then 32 codes (4-byte aligned: 36-byte blocks)
            device const uchar* block=x+(start/32)*36;
            const float d=float(*reinterpret_cast<device const half*>(block));
            device const char4* q=reinterpret_cast<device const char4*>(block+4+start%32);
            for(int j=0;j<4;++j) { const float4 v=d*float4(q[j]); xv[4*j]=v[0];xv[4*j+1]=v[1];xv[4*j+2]=v[2];xv[4*j+3]=v[3]; }
        }
        float xs=0;
        if(Dot::kSum) for(int j=0;j<16;++j) xs+=xv[j];
        for(int r=0;r<nr;++r) {
            device const Block* b=weights+r*per+start/QK;
            const short il=short((start%QK)/16);
            if(a.kind==3) { float4x4 deq; Deq(b,il,deq); for(int j=0;j<16;++j) acc[r]+=float(half(deq[j/4][j%4]))*xv[j]; }
            else acc[r]+=Dot::run(b,il,xv,xs);
        }
    }
    for(int r=0;r<nr;++r) {
        const float sum=simd_sum(acc[r]);
        if(!lane) out_y[yrow*a.ystride+o0+r]=sum;
    }
}
#define MUL_(name,QK,DOT,NR,SUFFIX) template [[host_name("poly_mul_" #name SUFFIX)]] kernel decltype(poly_mul<block_##name,QK,dequantize_##name<float4x4>,DOT,NR>) poly_mul<block_##name,QK,dequantize_##name<float4x4>,DOT,NR>;
#define MUL(name,QK) MUL_(name,QK,dot_##name,4,"") MUL_(name,QK,dot_##name,1,"_1")
#define DEQDOT(name,QK) typedef DeqDot<block_##name,dequantize_##name<float4x4>> dot_##name;
#define MV_TYPES(M) M(q4_0,32) M(q4_1,32) M(q5_0,32) M(q5_1,32) M(q8_0,32) M(q2_0,64) M(mxfp4,32) M(nvfp4,64) \
    M(q2_K,256) M(q3_K,256) M(q4_K,256) M(q5_K,256) M(q6_K,256) M(iq2_xxs,256) M(iq2_xs,256) M(iq3_xxs,256) M(iq1_s,256) \
    M(iq4_nl,32) M(iq3_s,256) M(iq2_s,256) M(iq4_xs,256) M(iq1_m,256)
#define MV_TYPES_DEQ(M) M(q4_0,32) M(q4_1,32) M(q5_0,32) M(q5_1,32) M(mxfp4,32) M(q2_K,256) M(q3_K,256) M(q5_K,256) \
    M(iq2_xs,256) M(iq3_xxs,256) M(iq4_nl,32) M(iq3_s,256) M(iq4_xs,256) M(iq1_m,256)
MV_TYPES_DEQ(DEQDOT)
MV_TYPES(MUL)
#ifdef MV_DEQ   // poly_mv_bench's: poly_mul_<type>_deq4 and _deq1, every type from its block decoder
template<typename Block,void Deq(device const Block*,short,thread float4x4&)> struct BenchDot {
    enum { kSum=0 };
    static float run(device const Block* b,short il,thread const float* xv,float xs) { return DeqDot<Block,Deq>::run(b,il,xv,xs); }
};
#define MULD(name,QK) typedef BenchDot<block_##name,dequantize_##name<float4x4>> benchdot_##name; \
    MUL_(name,QK,benchdot_##name,4,"_deq4") MUL_(name,QK,benchdot_##name,1,"_deq1")
MV_TYPES(MULD)
#endif

template<typename Block,int QK,void Deq(device const Block*,short,thread float4x4&)>
kernel void q36_embed(constant QArgs& a [[buffer(0)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    uint col=(group.x*32+lane)*16;if(col>=uint(a.ni))return;
    int id=static_cast<device int*>(a.p[3])[group.y];
    device const Block* w=reinterpret_cast<device const Block*>(static_cast<device uchar*>(a.p[0])+ulong(id)*a.stride);
    float4x4 d;Deq(w+col/QK,short(col%QK/16),d);
    for(int j=0;j<16;++j)static_cast<device float*>(a.p[2])[ulong(group.y)*a.ni+col+j]=d[j/4][j%4];
}
#define EMB(name,QK) template [[host_name("q36_embed_" #name)]] kernel decltype(q36_embed<block_##name,QK,dequantize_##name<float4x4>>) q36_embed<block_##name,QK,dequantize_##name<float4x4>>;
EMB(q4_0,32) EMB(q4_1,32) EMB(q5_0,32) EMB(q5_1,32) EMB(q8_0,32) EMB(q2_0,64) EMB(iq4_nl,32) EMB(iq4_xs,256)
EMB(iq2_xxs,256) EMB(iq2_xs,256) EMB(iq3_xxs,256) EMB(iq1_s,256) EMB(iq3_s,256) EMB(iq2_s,256) EMB(iq1_m,256) EMB(mxfp4,32) EMB(nvfp4,64)
EMB(q2_K,256) EMB(q3_K,256) EMB(q4_K,256) EMB(q5_K,256) EMB(q6_K,256)
template [[host_name("q36_embed_f32")]] kernel decltype(q36_embed<float4x4,16,dequantize_f32<float4x4>>) q36_embed<float4x4,16,dequantize_f32<float4x4>>;
template [[host_name("q36_embed_f16")]] kernel decltype(q36_embed<half4x4,16,dequantize_f16<float4x4>>) q36_embed<half4x4,16,dequantize_f16<float4x4>>;
// bfloat vectors require Metal 3.1; decode BF16 without that device feature.
kernel void q36_embed_bf16(constant QArgs& a [[buffer(0)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    uint col=group.x*32+lane;if(col>=uint(a.ni))return;int id=static_cast<device int*>(a.p[3])[group.y];
    device const ushort* w=reinterpret_cast<device const ushort*>(static_cast<device uchar*>(a.p[0])+ulong(id)*a.stride);
    static_cast<device float*>(a.p[2])[ulong(group.y)*a.ni+col]=as_type<float>(uint(w[col])<<16);
}
