// PolyStrata projections on the GGUF block decoders of dequant.metal. All dispatch
// and graph ownership stays in PolyStrata's Metal runtime.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/quant/types.hpp"
#include "launch.hpp"
#include "quantized.hpp"
#include "mmq.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
namespace strata::poly::metal_detail {
namespace {
// The block structs and tables (the quant header in its Metal mode) and the block decoders, ahead of a kernel file
std::string block_source() {
    return "#include <metal_stdlib>\nusing namespace metal;\n#define GGML_COMMON_DECL_METAL\n#define GGML_COMMON_IMPL_METAL\n"+
           read_source(STRATA_POLY_QUANT_COMMON)+read_source(STRATA_POLY_DEQUANT_SOURCE);
}
int64_t block_values(int t) { return quant::type_info(t)->blck; }
std::string quant_source() { return block_source()+read_source(STRATA_POLY_QUANT_SOURCE); }
const std::string& mm_source() {
    static const std::string source=block_source()+read_source(STRATA_POLY_MM_SOURCE);
    return source;
}
// The matrix-unit library compiles here (Metal 4's matmul2d); asked once
bool mm_ready() {
    static const bool ready=[] {
        std::string error;
        return gpu::metal::pipeline("poly_mm",mm_source().c_str(),"poly_mm_q4_K_64_64_64_4",error)!=nullptr;
    }();
    return ready;
}
struct MmTile { int tm=64,tn=64,tk=64,sg=4; };   // mm.metal's tile: rows, outputs, values a step, SIMD groups
struct MArgs {   // mm.metal's MmArgs
    const void* p[5]{};
    uint64_t stride=0,row_bytes=0,ld=0;
    int ni=0,no=0,total=0;
};
static_assert(sizeof(MArgs)==80);
struct GArgs {   // mm.metal's GemmArgs
    const void* p[3]{};
    int m=0,n=0,k=0,pad=0;
};
static_assert(sizeof(GArgs)==40);
// The fp16 copy of a dense matrix (poly_dq) that poly_gemm reads: grown to the largest one, kept.  Larger matrices
// than kDqMax bytes of halves (an output head) and calls of fewer than kDqRows rows stay with poly_mm.
void* dq_buf=nullptr;
size_t dq_bytes=0;
constexpr size_t kDqMax=size_t(64)<<20;
constexpr int64_t kDqRows=128;
}
namespace {
bool mm_allowed=!(std::getenv("STRATA_METAL_MM") && std::string(std::getenv("STRATA_METAL_MM"))=="0");
MmTile mm_tile;
}
void allow_mm_rows(bool on) { mm_allowed=on; }
void set_mm_tile(const char* tile) {
    mm_tile=MmTile{};
    if(tile) std::sscanf(tile,"%d,%d,%d,%d",&mm_tile.tm,&mm_tile.tn,&mm_tile.tk,&mm_tile.sg);
}
bool mm_rows(int64_t cols) { return mm_allowed && cols>0 && cols%mm_tile.tk==0 && gpu::metal::optimize() && mm_ready(); }
bool mm_gemm(const void* a,const void* b,float* y,int m,int n,int k,bool f32,void* s) {
    if(!mm_allowed || m<=0 || n<=0 || k<=0 || !gpu::metal::optimize() || !mm_ready()) return false;
    GArgs g;g.p[0]=a;g.p[1]=b;g.p[2]=y;g.m=m;g.n=n;g.k=k;
    const int tn=n<=32?32:64;
    const std::string name=std::string("poly_gemm_")+(f32?"float":"half")+"_64_"+std::to_string(tn)+"_4";
    dispatch("poly_mm",mm_source(),name.c_str(),g,s,{uint32_t((n+tn-1)/tn),uint32_t((m+63)/64),1},128);
    return true;
}
void qrun(const std::string& name,const QArgs& a,void* s,gpu::metal::Grid groups) {
    static const auto source=quant_source();
    std::string kernel=name;
    const bool mul=name.starts_with("poly_mul_");
    // groups.x: the outputs.  4 a threadgroup where the launch has 8,192 outputs or more; fewer leave the GPU too few
    // threadgroups (a dense 2,304 x 4,096 matrix: 13% slower on the M5 Ultra), so there 1 (poly_mv_bench).  Q5_1
    // always 1: its 4 were 8% slower on Qwen3.6's experts and no faster on Inkling 2's.
    const int nr=mul && name!="poly_mul_q5_1" && uint64_t(groups.x)*groups.y*groups.z>=8192?4:1;
    if(mul && nr==1) kernel+="_1";
    if(mul && a.kind==9) {   // native_mmvq_many: each matrix's outputs in groups of nr
        const uint64_t packed[4]={a.stride,a.xstride,a.ystride,a.offset};
        groups.x=0;
        for(int j=0;j<a.count;++j) groups.x+=(uint32_t(packed[j/2]>>(32*(j%2)))+nr-1)/nr;
    }
    else if(mul) groups.x=(groups.x+nr-1)/nr;
    const bool indirect = mul && (a.kind==1 || a.kind==4 || a.kind==5);
    dispatch("poly_quantized",source,kernel.c_str(),a,s,groups,32,
             {a.p,a.kind==8 || a.kind==9?size_t(3*a.count):6,true,indirect?a.p[0]:nullptr});
}
const char* quant_name(int t) {
    switch(t) {
#define Q(id,name) case id:return #name;
    Q(2,q4_0) Q(3,q4_1) Q(6,q5_0) Q(7,q5_1) Q(8,q8_0) Q(10,q2_K) Q(11,q3_K) Q(12,q4_K) Q(13,q5_K)
    Q(14,q6_K) Q(16,iq2_xxs) Q(17,iq2_xs) Q(18,iq3_xxs) Q(19,iq1_s) Q(20,iq4_nl) Q(21,iq3_s)
    Q(22,iq2_s) Q(23,iq4_xs) Q(29,iq1_m) Q(39,mxfp4) Q(40,nvfp4) Q(42,q2_0)
#undef Q
    default:return nullptr;
    }
}
}
namespace strata::kernels {
using namespace poly::metal_detail;
bool native_mmvq_supported(int t) noexcept { return quant_name(t); }
bool native_mmvq_ids_supported(int t) { return native_mmvq_supported(t); }
size_t native_q8_1_bytes(int n,int rows) {
    if(n<=0 || n%32 || rows<=0) throw std::invalid_argument("Q8_1 shape");
    return size_t(n/32)*36*rows;
}
size_t native_mmvq_weight_bytes(int t,int ni,int no) {
    if(!native_mmvq_supported(t) || ni<=0 || no<=0 || ni%block_values(t)) throw std::invalid_argument("Metal quantized matrix shape");
    return quant::row_bytes(t,ni)*no;
}
void native_quantize_q8_1(const float* x,void* q,int n,int rows,void* s) {
    native_q8_1_bytes(n,rows);
    QArgs a;a.p[0]=x;a.p[1]=q;a.ni=n;a.rows=rows;a.xstride=n;
    qrun("poly_quantize",a,s,{uint32_t(n/32),uint32_t(rows),1});
}
void native_add_rms_norm_q8_1(float* x,const float* add,const float* w,float* y,void* q,int n,int rows,float eps,int,void* s) {
    if(add) poly::axpy(add,1,x,n*rows,true,s);
    poly::rms_norm(x,w,y,n,rows,eps,s);
    native_quantize_q8_1(y,q,n,rows,s);
}
void native_mmvq(int type,const void* w,const void* x,float* y,int ni,int no,int rows,void* s) {
    native_mmvq_weight_bytes(type,ni,no);
    QArgs a;a.p[0]=w;a.p[1]=x;a.p[2]=y;a.ni=ni;a.no=no;a.rows=rows;a.xstride=native_q8_1_bytes(ni);a.ystride=no;
    qrun(std::string("poly_mul_")+quant_name(type),a,s,{uint32_t(no),uint32_t(rows),1});
}
bool native_mmvq_slots(int t,const MmvqSlots& a,int count,int ni,int no,void* s) {
    if(!native_mmvq_supported(t) || count<0 || count>8) return false;
    if(!count) return true;
    if(!gpu::metal::optimize()) {
        for(int i=0;i<count;++i) native_mmvq(t,a.w[i],a.x[i],a.y[i],ni,no,1,s);
        return true;
    }
    native_mmvq_weight_bytes(t,ni,no);
    QArgs q; q.ni=ni; q.no=no; q.rows=count; q.count=count; q.ystride=1; q.kind=8;
    for(int i=0;i<count;++i) { q.p[3*i]=a.w[i]; q.p[3*i+1]=a.x[i]; q.p[3*i+2]=a.y[i]; }
    qrun(std::string("poly_mul_")+quant_name(t),q,s,{uint32_t(no),uint32_t(count),1});
    return true;
}
bool native_mmvq_ids(int t,const MmvqIds& a,int count,int ni,int no,void* s) {
    if(!native_mmvq_supported(t)) return false;
    native_mmvq_weight_bytes(t,ni,no);
    QArgs q;q.p[0]=a.table;q.p[1]=a.x;q.p[2]=a.y;q.p[3]=a.ids;q.ni=ni;q.no=no;q.rows=count;
    q.xstride=a.x_stride;q.ystride=a.y_stride;q.offset=a.offset;q.every=a.x_every;q.kind=1;
    qrun(std::string("poly_mul_")+quant_name(t),q,s,{uint32_t(no),uint32_t(count),1});return true;
}
// Any type poly_mul has: the matrices' outputs one after another in one launch (kind 9), each output the sum
// native_mmvq() gives it
bool native_mmvq_many_fits(int t,const NativeMmvqMany& m,int ni) {
    if(!native_mmvq_supported(t) || m.count<1 || m.count>8 || ni<=0 || ni%block_values(t)) return false;
    for(int j=0;j<m.count;++j) if(!m.w[j] || !m.x_q8_1[j] || !m.y[j] || m.n_out[j]<=0) return false;
    return true;
}
void native_mmvq_many(int t,const NativeMmvqMany& m,int ni,void* s) {
    if(!native_mmvq_many_fits(t,m,ni)) throw std::invalid_argument("native_mmvq_many: not these matrices");
    QArgs q;q.ni=ni;q.rows=1;q.count=m.count;q.kind=9;
    uint64_t packed[4]={};   // n_out[j], 32 bits each, in the four 64-bit fields kind 9 does not otherwise read
    uint32_t total=0;
    for(int j=0;j<m.count;++j) {
        q.p[3*j]=m.w[j];q.p[3*j+1]=m.x_q8_1[j];q.p[3*j+2]=m.y[j];
        packed[j/2]|=uint64_t(uint32_t(m.n_out[j]))<<(32*(j%2));
        total+=uint32_t(m.n_out[j]);
    }
    q.stride=packed[0];q.xstride=packed[1];q.ystride=packed[2];q.offset=packed[3];
    qrun(std::string("poly_mul_")+quant_name(t),q,s,{total,1,1});
}
}
namespace strata::poly {
using namespace metal_detail;
// The products' pipelines (every type, both tiles) made at load: about 3 ms from Metal's cache, 0.3 s the first time
// after an update.  Made on the way in the first request, poly_mul's two kernels a type kept LFM2.5 UD-IQ2_M's first
// prompt at 0.20-0.22 s in 8 of 8 starts, where the engine before them read it in 0.16-0.17 s in about half.
Mmq::Mmq() {
    static const auto source=quant_source();
    std::string error;
    for(int t=0;t<64;++t) if(quant_name(t))
        for(const char* tile:{"","_1"})
            gpu::metal::pipeline("poly_quantized",source.c_str(),(std::string("poly_mul_")+quant_name(t)+tile).c_str(),error);
}
Mmq::~Mmq() = default;
bool Mmq::fits(int t,int64_t rows,int64_t cols) {
    return kernels::native_mmvq_supported(t) && rows>0 && rows<=INT32_MAX && cols>0 && cols<=INT32_MAX && cols%block_values(t)==0;
}
// room for either form of the rows: q8_1, or fp16 on the matrix-unit path (mm_rows())
size_t Mmq::q8_bytes(int64_t rows,int64_t cols) {
    return std::max(kernels::native_q8_1_bytes(int(cols),int(rows)),size_t(rows)*size_t(cols)*2);
}
void Mmq::quantize(const float* x,const int32_t* ids,void* q,int,int64_t cols,int64_t ld,int64_t rows,void* s) {
    QArgs a;a.p[0]=x;a.p[1]=q;a.p[2]=ids;a.ni=int(cols);a.rows=int(rows);a.xstride=ld;
    if(mm_rows(cols)) {
        static const auto source=quant_source();
        const uint32_t threads=uint32_t(std::min<int64_t>(256,cols/4));
        dispatch("poly_quantized",source,"poly_tohalf",a,s,{uint32_t((cols/4+threads-1)/threads),uint32_t(rows),1},threads,
                 {a.p,3,true});
        return;
    }
    qrun("poly_quantize",a,s,{uint32_t(cols/32),uint32_t(rows),1});
}
bool Mmq::reads_by_index(int,int64_t) { return false; }
bool Mmq::int8(int,int64_t) { return false; }   // rows in q8_1 or fp16 here, never gemm_iq_i8_quantize's
void Mmq::mul(const void* w,int type,int64_t nr,int64_t nc,size_t stride,int n,const void* x,
              const int32_t* bounds,const int32_t* ids,int64_t total,int64_t max_rows,float* y,int64_t ld,void* s,
              const int32_t* from) {
    if(from) throw std::logic_error("Metal MMQ reads no rows by index");
    if(n<=0 || total<=0 || max_rows<=0) return;
    if(!fits(type,nr,nc)) throw std::invalid_argument("Metal MMQ shape");
    const size_t half_bytes=size_t(nr)*size_t(nc)*2;
    if(mm_rows(nc) && !bounds && !ids && ld==nr && total>=kDqRows && half_bytes<=kDqMax) {
        if(half_bytes>dq_bytes) {
            if(dq_buf) { gpu::stream_synchronize(static_cast<gpu::Stream>(s)); gpu::free_device(dq_buf); }
            dq_buf=nullptr;dq_bytes=0;
            if(gpu::malloc_device_raw(&dq_buf,half_bytes)==gpu::kSuccess) dq_bytes=half_bytes;
            else dq_buf=nullptr;
        }
        if(dq_buf) {
            MArgs m;m.p[0]=w;m.p[1]=dq_buf;m.row_bytes=quant::row_bytes(type,nc);m.ni=int(nc);m.no=int(nr);
            const uint64_t threads=uint64_t(nr)*uint64_t(nc/16);
            dispatch("poly_mm",mm_source(),(std::string("poly_dq_")+quant_name(type)).c_str(),m,s,
                     {uint32_t((threads+255)/256),1,1},256);
            mm_gemm(x,dq_buf,y,int(total),int(nr),int(nc),false,s);
            return;
        }
    }
    if(mm_rows(nc)) {   // TM rows of one matrix and TN outputs a threadgroup, the rows past a matrix's bounds idle
        MArgs m;m.p[0]=w;m.p[1]=x;m.p[2]=y;m.p[3]=ids;m.p[4]=bounds;
        m.stride=stride;m.row_bytes=quant::row_bytes(type,nc);m.ld=uint64_t(ld);m.ni=int(nc);m.no=int(nr);m.total=int(total);
        const int64_t rows=bounds?max_rows:total;
        // matrices of 32 rows or fewer (most of Qwen3.6's 256 experts in a batch of 1,024 tokens) leave half of a
        // 64-row tile empty: the 32-row one.  The products are the same to the bit whatever the tile.
        MmTile t=mm_tile;
        if(rows<=32 && t.tm==64 && t.tn==64 && t.tk==64 && t.sg==4) t.tm=32;
        const std::string name=std::string("poly_mm_")+quant_name(type)+"_"+std::to_string(t.tm)+"_"+std::to_string(t.tn)+"_"+
                               std::to_string(t.tk)+"_"+std::to_string(t.sg);
        dispatch("poly_mm",mm_source(),name.c_str(),m,s,
                 {uint32_t((nr+t.tn-1)/t.tn),uint32_t((rows+t.tm-1)/t.tm),uint32_t(bounds?n:1)},uint32_t(32*t.sg));
        return;
    }
    const bool opt=gpu::metal::optimize(), flat=opt && bounds && total<max_rows*n;
    QArgs a;a.p[0]=w;a.p[1]=x;a.p[2]=y;a.p[3]=ids;a.p[4]=bounds;a.ni=int(nc);a.no=int(nr);a.rows=int(total);a.count=n;
    a.xstride=kernels::native_q8_1_bytes(int(nc));a.ystride=ld;a.stride=stride;a.kind=flat?7:(opt?6:2);
    qrun(std::string("poly_mul_")+quant_name(type),a,s,{uint32_t(nr),uint32_t(flat?total:(opt?max_rows:total)),uint32_t(flat?1:n)});
}
}
