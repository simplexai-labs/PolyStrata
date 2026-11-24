// NVFP4 on strata-poly's Metal quant layer against the CPU reference (src/quant's dequant_row), on random blocks: the
// decoders to the bit, every scale code among them, and the products to float rounding; and the experts' global
// scales (ExpertScales) as CUDA's activations and expert_weights() put them on.
#include "test_utils.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/poly/kernels.hpp"
#include "mmq.hpp"
#include "../src/poly/quantized.hpp"
#include "strata/gpu/metal.hpp"
#include <algorithm>
#include <cstring>
#include <random>
namespace k=strata::kernels;
namespace p=strata::poly;
namespace q=strata::quant;
constexpr int kNvfp4=40;
std::mt19937 rng(40);
// random 36-byte blocks: scale codes from those given (all 256 when null), random nibbles
std::vector<unsigned char> blocks(int rows,int ni,const std::vector<int>* codes=nullptr){
    std::vector<unsigned char> b(size_t(rows)*ni/64*36);
    for(size_t i=0;i<b.size();++i){const size_t at=i%36;b[i]=uint8_t(at<4&&codes?(*codes)[rng()%codes->size()]:rng());}
    if(!codes)for(size_t i=0;i<b.size()/36&&i<64;++i)for(int s=0;s<4;++s)b[i*36+s]=uint8_t(i*4+s);   // each code once
    return b;
}
std::vector<float> dequant(const std::vector<unsigned char>& b,int ni){
    std::vector<float> f(b.size()/36*64);
    for(size_t r=0;r<f.size()/ni;++r)if(!q::dequant_row(q::Type::NVFP4,b.data()+r*(ni/64)*36,f.data()+r*ni,ni))throw std::runtime_error("dequant_row");
    return f;
}
// zero_sign: whether -0 must stay -0 (a matrix product's sum of a -0 is +0)
int same_bits(const std::vector<float>& got,const std::vector<float>& ref,const char* label,bool zero_sign=true){
    int bad=0;for(size_t i=0;i<ref.size();++i)bad+=zero_sign?std::memcmp(&got[i],&ref[i],4)!=0:!(got[i]==ref[i]);
    std::printf("%s: %d of %zu values differ in their bits\n",label,bad,ref.size());return bad;
}
int main(){
    int bad=0;g::Stream stream;ck(g::stream_create(&stream));
    // q36_embed_nvfp4: the decoder to floats
    {
        const int ni=1024,rows=8;auto w=blocks(rows,ni);auto ref=dequant(w,ni);
        std::vector<int32_t> ids(rows);for(int r=0;r<rows;++r)ids[r]=rows-1-r;
        std::vector<float> want(ref.size());for(int r=0;r<rows;++r)std::memcpy(&want[r*ni],&ref[ids[r]*ni],ni*4);
        Buffer dw(w.size()),di(rows*4),out(ref.size()*4);dw.put(w.data(),w.size());di.put(ids.data(),rows*4);
        p::metal_detail::QArgs a;a.p[0]=dw.ptr;a.p[2]=out.ptr;a.p[3]=di.ptr;a.stride=ni/64*36;a.ni=ni;a.rows=rows;
        p::metal_detail::qrun("q36_embed_nvfp4",a,stream,{uint32_t(ni/512),uint32_t(rows),1});ck(g::stream_synchronize(stream));
        bad+=same_bits(out.read(ref.size()),want,"decoded floats");
    }
    // the fp16 decoders of the matrix units (poly_dq, poly_mm's tiles): rows that are each a unit vector give the
    // weights' halves back, to the bit
    g::metal::set_optimize(true);p::metal_detail::allow_mm_rows(true);
    if(p::metal_detail::mm_rows(1024)){
        const int ni=1024,no=72;auto w=blocks(2*no,ni);auto ref=dequant(w,ni);
        std::vector<float> eye(size_t(2*ni)*ni,0.f);for(int r=0;r<2*ni;++r)eye[size_t(r)*ni+r%ni]=1;
        Buffer dw(w.size()),dx(eye),xh(p::Mmq::q8_bytes(2*ni,ni)),out(size_t(2*ni)*no*4),db(12);dw.put(w.data(),w.size());
        p::Mmq::quantize(dx.as(),nullptr,xh.ptr,kNvfp4,ni,ni,2*ni,stream);
        std::vector<float> want(size_t(2*ni)*no);
        for(int r=0;r<2*ni;++r)for(int o=0;o<no;++o)want[size_t(r)*no+o]=float(_Float16(ref[size_t((r/ni)*no+o)*ni+r%ni]));
        p::Mmq mmq;mmq.mul(dw.ptr,kNvfp4,no,ni,0,1,xh.ptr,nullptr,nullptr,ni,ni,out.as(),no,stream);ck(g::stream_synchronize(stream));
        bad+=same_bits(out.read(size_t(ni)*no),std::vector<float>(want.begin(),want.begin()+size_t(ni)*no),"halves, poly_dq",false);
        int32_t bounds[]={0,ni,2*ni};db.put(bounds,12);
        mmq.mul(dw.ptr,kNvfp4,no,ni,w.size()/2,2,xh.ptr,db.as<int32_t>(),nullptr,2*ni,ni,out.as(),no,stream);ck(g::stream_synchronize(stream));
        bad+=same_bits(out.read(want.size()),want,"halves, poly_mm tiles of two matrices",false);
    } else std::printf("no matrix units here: the fp16 decoders are not run\n");
    // the products: Inkling 2's expert shapes (2,304 in, and 1,152, which leaves lanes a piece short), the decode
    // launches and the batch's, against double sums of the CPU's weights and the rows as the kernels read them
    std::vector<int> codes;for(int c=0x20;c<0x40;++c)codes.push_back(c);   // scales 1/8 to 15/16
    for(int ni:{2304,1152}){
        const int no=6,rows=4,matrices=2;auto w=blocks(matrices*no,ni,&codes);auto wf=dequant(w,ni);
        const size_t size=k::native_mmvq_weight_bytes(kNvfp4,ni,no);
        auto x=values(rows*ni,.8f);Buffer dw(w.size()),dx(x),qx(p::Mmq::q8_bytes(rows,ni)),y(rows*no*4);dw.put(w.data(),w.size());
        auto read_rows=[&](const std::vector<int>& from){   // the rows as the kernels read them: q8_1, or fp16 on the matrix units
            std::vector<float> v(size_t(rows)*ni);
            if(p::metal_detail::mm_rows(ni)){for(int r=0;r<rows;++r)for(int i=0;i<ni;++i)v[r*ni+i]=float(_Float16(x[from[r]*ni+i]));return v;}
            auto b=qx.read<unsigned char>(k::native_q8_1_bytes(ni,rows));
            for(size_t i=0;i<v.size();++i){_Float16 d;std::memcpy(&d,b.data()+(i/32)*36,2);v[i]=float(d)*reinterpret_cast<int8_t*>(b.data())[(i/32)*36+4+i%32];}
            return v;
        };
        auto product=[&](const std::vector<float>& in,const std::vector<int>& experts,const std::vector<int>& outputs){
            std::vector<float> ref(rows*no);
            for(int r=0;r<rows;++r)for(int o=0;o<no;++o){double s=0;for(int i=0;i<ni;++i)s+=double(wf[size_t(experts[r]*no+o)*ni+i])*in[r*ni+i];ref[outputs[r]*no+o]=float(s);}
            return ref;
        };
        char label[96];
        for(auto [optimized,units]:{std::pair{false,false},{true,false},{true,true}}){
            g::metal::set_optimize(optimized);p::metal_detail::allow_mm_rows(units);
            const char* how=!optimized?"plain":units&&p::metal_detail::mm_rows(ni)?"matrix units":"optimized";
            k::native_quantize_q8_1(dx.as(),qx.ptr,ni,rows,stream);ck(g::stream_synchronize(stream));
            p::metal_detail::allow_mm_rows(false);auto xq=read_rows({0,1,2,3});
            p::metal_detail::allow_mm_rows(units);
            k::native_mmvq(kNvfp4,dw.ptr,qx.ptr,y.as(),ni,no,rows,stream);ck(g::stream_synchronize(stream));
            std::snprintf(label,sizeof label,"%d in, %s: one matrix",ni,how);bad+=compare(y.read(rows*no),product(xq,{0,0,0,0},{0,1,2,3}),label,1e-4f);
            std::vector<const void*> table={dw.ptr,static_cast<unsigned char*>(dw.ptr)+size};std::vector<int32_t> ids={1,0,1,0};
            Buffer tab(table.size()*sizeof(void*)),di(16);tab.put(table.data(),table.size()*sizeof(void*));di.put(ids.data(),16);
            k::MmvqIds selected{tab.as<const void*>(),di.as<int32_t>(),0,qx.ptr,k::native_q8_1_bytes(ni),1,y.as(),size_t(no)};
            k::native_mmvq_ids(kNvfp4,selected,rows,ni,no,stream);ck(g::stream_synchronize(stream));
            std::snprintf(label,sizeof label,"%d in, %s: expert ids",ni,how);bad+=compare(y.read(rows*no),product(xq,{1,0,1,0},{0,1,2,3}),label,1e-4f);
            k::MmvqSlots slots{};
            for(int r=0;r<rows;++r){slots.w[r]=table[ids[r]];slots.x[r]=static_cast<unsigned char*>(qx.ptr)+r*k::native_q8_1_bytes(ni);slots.y[r]=y.as()+r*no;}
            k::native_mmvq_slots(kNvfp4,slots,rows,ni,no,stream);ck(g::stream_synchronize(stream));
            std::snprintf(label,sizeof label,"%d in, %s: expert slots",ni,how);bad+=compare(y.read(rows*no),product(xq,{1,0,1,0},{0,1,2,3}),label,1e-4f);
            // a batch: rows gathered, two matrices by bounds, scattered; optimized, {0,1,4} is the compact launch
            // (fewer rows than 2 matrices of the most rows) and {0,2,4} the launch by matrix
            for(auto bounds:{std::vector<int32_t>{0,1,4},std::vector<int32_t>{0,2,4}}){
                std::vector<int32_t> gather={2,0,3,1},scatter={1,3,0,2};
                Buffer dg(16),db(12),ds(16);dg.put(gather.data(),16);db.put(bounds.data(),12);ds.put(scatter.data(),16);
                p::Mmq::quantize(dx.as(),dg.as<int32_t>(),qx.ptr,kNvfp4,ni,ni,rows,stream);
                p::Mmq mmq;mmq.mul(dw.ptr,kNvfp4,no,ni,size,matrices,qx.ptr,db.as<int32_t>(),ds.as<int32_t>(),rows,std::max(bounds[1]-bounds[0],bounds[2]-bounds[1]),y.as(),no,stream);
                ck(g::stream_synchronize(stream));
                std::vector<int> experts(rows);for(int r=0;r<rows;++r)experts[r]=r<bounds[1]?0:1;
                std::snprintf(label,sizeof label,"%d in, %s: batch, bounds %d,%d,%d",ni,how,bounds[0],bounds[1],bounds[2]);
                bad+=compare(y.read(rows*no),product(read_rows({2,0,3,1}),experts,{1,3,0,2}),label,1e-4f);
            }
        }
    }
    // the global scales: scale_expert_rows (out of place and in place, by ids and by e[]) to the bit, the activations
    // to float rounding (CUDA's kernels.cu: gate and up times their scales first), expert_weights to the bit
    {
        const int n=73,rows=5,experts=6;const float limit=1.2f;
        auto gu=values(rows*2*n,3);std::vector<float> gs(experts),us(experts);
        for(int e=0;e<experts;++e){gs[e]=.5f+.1f*e;us[e]=1.5f-.15f*e;}
        std::vector<int32_t> ids={3,0,5,1,3};
        Buffer dgu(gu),dgs(gs),dus(us),di(rows*4),out(rows*2*n*4),act(rows*n*4);di.put(ids.data(),rows*4);
        for(int mode=0;mode<3;++mode){   // gate and up by ids, by e[], gate alone by ids
            p::ExpertScales sc;sc.gate=dgs.as();sc.up=mode<2?dus.as():nullptr;
            if(mode==1)std::copy(ids.begin(),ids.end(),sc.e);else sc.ids=di.as<int32_t>();
            std::vector<float> scaled(gu.size());
            for(int r=0;r<rows;++r)for(int i=0;i<2*n;++i)scaled[r*2*n+i]=gu[r*2*n+i]*(i<n?gs[ids[r]]:sc.up?us[ids[r]]:1.f);
            const char* how[]={"by ids","by e[]","gate only"};char label[96];
            p::scale_expert_rows(dgu.as(),out.as(),n,rows,&sc,stream);ck(g::stream_synchronize(stream));
            std::snprintf(label,sizeof label,"scaled rows, %s",how[mode]);bad+=same_bits(out.read(gu.size()),scaled,label);
            std::vector<float> ref(rows*n);
            for(int r=0;r<rows;++r)for(int i=0;i<n;++i){float g=std::min(scaled[r*2*n+i],limit),u=std::clamp(scaled[r*2*n+n+i],-limit,limit);ref[r*n+i]=g/(1+std::exp(-g))*u;}
            p::swiglu_clamp_rows(dgu.as(),act.as(),n,rows,limit,stream,&sc);ck(g::stream_synchronize(stream));
            std::snprintf(label,sizeof label,"swiglu_clamp_rows, %s",how[mode]);bad+=compare(act.read(rows*n),ref,label);
            for(int r=0;r<rows;++r)for(int i=0;i<n;++i){float g=scaled[r*2*n+i],u=scaled[r*2*n+n+i];ref[r*n+i]=.5f*g*(1+std::tanh(.7978845608028654f*g*(1+.044715f*g*g)))*u;}
            p::geglu_rows(dgu.as(),act.as(),n,rows,stream,&sc);ck(g::stream_synchronize(stream));
            std::snprintf(label,sizeof label,"geglu_rows, %s",how[mode]);bad+=compare(act.read(rows*n),ref,label);
            out.put(gu.data(),gu.size()*4);p::scale_expert_rows(out.as(),out.as(),n,rows,&sc,stream);ck(g::stream_synchronize(stream));
            std::snprintf(label,sizeof label,"scaled rows in place, %s",how[mode]);bad+=same_bits(out.read(gu.size()),scaled,label);
        }
        p::scale_expert_rows(dgu.as(),out.as(),n,rows,nullptr,stream);ck(g::stream_synchronize(stream));
        bad+=same_bits(out.read(gu.size()),gu,"no scales: a copy");
        auto w=values(rows,.1f);Buffer dw(w);std::vector<float> wref(rows);for(int r=0;r<rows;++r)wref[r]=w[r]*gs[ids[r]];
        p::expert_weights(dw.as(),dgs.as(),di.as<int32_t>(),rows,stream);ck(g::stream_synchronize(stream));
        bad+=same_bits(dw.read(rows),wref,"expert_weights");
    }
    p::metal_detail::allow_mm_rows(true);
    ck(g::stream_destroy(stream));return bad?1:0;
}
