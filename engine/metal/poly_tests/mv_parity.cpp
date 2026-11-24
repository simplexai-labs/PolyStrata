// One token's quantized products (poly_mul, every type) on the shapes of a decode step, against double sums of the
// CPU's weights (ggml's decoders; NVFP4's random blocks through src/quant's) and the rows as the kernels read them:
// a dense matrix, expert ids and slots, and a batch's launches, with output counts that leave a tile part empty.
#include "test_utils.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/quant/dequant.hpp"
#include "mmq.hpp"
#include "../src/poly/quantized.hpp"
#include "ggml.h"
#include "strata/gpu/metal.hpp"
#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
namespace k=strata::kernels;
namespace p=strata::poly;
int main(int argc,char** argv){
    int bad=0;g::Stream stream;ck(g::stream_create(&stream));
    std::vector<int> types={2,3,6,7,8,10,11,12,13,14,16,17,18,19,20,21,22,23,29,39,40,42};
    if(argc>1)types={std::atoi(argv[1])};
    struct Shape{int ni,no;};
    for(int type:types)for(Shape sh:{Shape{2304,37},Shape{1152,21},Shape{512,70},Shape{4096,9}}){
        const int ni=sh.ni,no=sh.no,rows=4,matrices=2;
        if(ni%int(ggml_blck_size(ggml_type(type))))continue;
        const size_t size=k::native_mmvq_weight_bytes(type,ni,no);
        std::vector<unsigned char> w(size*matrices);std::vector<float> wf(size_t(matrices)*no*ni);
        if(type==40){
            std::mt19937 rng(40);for(size_t i=0;i<w.size();++i)w[i]=uint8_t(i%36<4?0x20+rng()%32:rng());
            for(int r=0;r<matrices*no;++r)strata::quant::dequant_row(strata::quant::Type::NVFP4,w.data()+r*(ni/64)*36,wf.data()+size_t(r)*ni,ni);
        }else{
            auto t=ggml_type(type);ggml_quantize_init(t);auto src=values(wf.size(),.5f);std::vector<float> imp(ni,1);
            ggml_quantize_chunk(t,src.data(),w.data(),0,matrices*no,ni,imp.data());ggml_get_type_traits(t)->to_float(w.data(),wf.data(),wf.size());
        }
        auto x=values(rows*ni,.8f);Buffer dw(w.size()),dx(x),qx(p::Mmq::q8_bytes(rows,ni)),y(rows*no*4);dw.put(w.data(),w.size());
        auto q8_rows=[&]{   // the rows in q8_1 as the kernels read them
            std::vector<float> v(size_t(rows)*ni);auto b=qx.read<unsigned char>(k::native_q8_1_bytes(ni,rows));
            for(size_t i=0;i<v.size();++i){_Float16 d;std::memcpy(&d,b.data()+(i/32)*36,2);v[i]=float(d)*reinterpret_cast<int8_t*>(b.data())[(i/32)*36+4+i%32];}
            return v;
        };
        auto product=[&](const std::vector<float>& in,const std::vector<int>& experts,const std::vector<int>& outputs){
            std::vector<float> ref(rows*no);
            for(int r=0;r<rows;++r)for(int o=0;o<no;++o){double s=0;for(int i=0;i<ni;++i)s+=double(wf[size_t(experts[r]*no+o)*ni+i])*in[r*ni+i];ref[outputs[r]*no+o]=float(s);}
            return ref;
        };
        char label[96];const char* name=ggml_type_name(ggml_type(type));
        p::metal_detail::allow_mm_rows(false);
        for(bool optimized:{false,true}){
            g::metal::set_optimize(optimized);const char* how=optimized?"optimized":"plain";
            k::native_quantize_q8_1(dx.as(),qx.ptr,ni,rows,stream);ck(g::stream_synchronize(stream));auto xq=q8_rows();
            k::native_mmvq(type,dw.ptr,qx.ptr,y.as(),ni,no,rows,stream);ck(g::stream_synchronize(stream));
            std::snprintf(label,sizeof label,"%s %d x %d, %s: one matrix",name,ni,no,how);bad+=compare(y.read(rows*no),product(xq,{0,0,0,0},{0,1,2,3}),label,1e-4f);
            std::vector<const void*> table={dw.ptr,static_cast<unsigned char*>(dw.ptr)+size};std::vector<int32_t> ids={1,0,1,0};
            Buffer tab(table.size()*sizeof(void*)),di(16);tab.put(table.data(),table.size()*sizeof(void*));di.put(ids.data(),16);
            k::MmvqIds selected{tab.as<const void*>(),di.as<int32_t>(),0,qx.ptr,k::native_q8_1_bytes(ni),1,y.as(),size_t(no)};
            k::native_mmvq_ids(type,selected,rows,ni,no,stream);ck(g::stream_synchronize(stream));
            std::snprintf(label,sizeof label,"%s %d x %d, %s: expert ids",name,ni,no,how);bad+=compare(y.read(rows*no),product(xq,{1,0,1,0},{0,1,2,3}),label,1e-4f);
            k::MmvqSlots slots{};
            for(int r=0;r<rows;++r){slots.w[r]=table[ids[r]];slots.x[r]=static_cast<unsigned char*>(qx.ptr)+r*k::native_q8_1_bytes(ni);slots.y[r]=y.as()+r*no;}
            k::native_mmvq_slots(type,slots,rows,ni,no,stream);ck(g::stream_synchronize(stream));
            std::snprintf(label,sizeof label,"%s %d x %d, %s: expert slots",name,ni,no,how);bad+=compare(y.read(rows*no),product(xq,{1,0,1,0},{0,1,2,3}),label,1e-4f);
            for(auto bounds:{std::vector<int32_t>{0,1,4},std::vector<int32_t>{0,2,4}}){   // the compact launch and the one by matrix
                std::vector<int32_t> gather={2,0,3,1},scatter={1,3,0,2};
                Buffer dg(16),db(12),ds(16);dg.put(gather.data(),16);db.put(bounds.data(),12);ds.put(scatter.data(),16);
                p::Mmq::quantize(dx.as(),dg.as<int32_t>(),qx.ptr,type,ni,ni,rows,stream);
                p::Mmq mmq;mmq.mul(dw.ptr,type,no,ni,size,matrices,qx.ptr,db.as<int32_t>(),ds.as<int32_t>(),rows,std::max(bounds[1]-bounds[0],bounds[2]-bounds[1]),y.as(),no,stream);
                ck(g::stream_synchronize(stream));
                std::vector<int> experts(rows);for(int r=0;r<rows;++r)experts[r]=r<bounds[1]?0:1;
                std::snprintf(label,sizeof label,"%s %d x %d, %s: batch, bounds %d,%d,%d",name,ni,no,how,bounds[0],bounds[1],bounds[2]);
                bad+=compare(y.read(rows*no),product(q8_rows(),experts,{1,3,0,2}),label,1e-4f);
            }
        }
    }
    // native_mmvq_many: five matrices of a type (Inkling 2's q, k, v, relative positions' and gate's outputs, and a
    // launch of fewer outputs), three inputs among them, in one launch, against the five native_mmvq() calls: the same
    // bits.  Q8_0 also at Inkling 2's width.
    for(int type:types)for(int ni:{256,2304})for(bool big:{true,false}){
        if(ni%int(ggml_blck_size(ggml_type(type))) || (ni==2304 && type!=8))continue;
        const int outs[5]={big?3072:96,big?1024:32,big?1024:32,big?384:12,big?3072:96},from[5]={0,1,2,0,0};
        k::NativeMmvqMany m{};std::vector<std::unique_ptr<Buffer>> keep;
        auto xs=values(3*ni,.8f);Buffer dx(xs),qx(3*k::native_q8_1_bytes(ni));
        k::native_quantize_q8_1(dx.as(),qx.ptr,ni,3,stream);
        std::vector<std::unique_ptr<Buffer>> ys,refs;
        for(int j=0;j<5;++j){
            const size_t size=k::native_mmvq_weight_bytes(type,ni,outs[j]);std::vector<unsigned char> w(size);
            if(type==40){std::mt19937 rng(j);for(size_t i=0;i<size;++i)w[i]=uint8_t(i%36<4?0x20+rng()%32:rng());}
            else{auto t=ggml_type(type);ggml_quantize_init(t);auto src=values(size_t(outs[j])*ni,.5f+.1f*j);std::vector<float> imp(ni,1);
                 ggml_quantize_chunk(t,src.data(),w.data(),0,outs[j],ni,imp.data());}
            keep.push_back(std::make_unique<Buffer>(size));keep.back()->put(w.data(),size);
            ys.push_back(std::make_unique<Buffer>(outs[j]*4));refs.push_back(std::make_unique<Buffer>(outs[j]*4));
            m.w[j]=keep.back()->ptr;m.x_q8_1[j]=static_cast<unsigned char*>(qx.ptr)+from[j]*k::native_q8_1_bytes(ni);m.y[j]=ys.back()->as();m.n_out[j]=outs[j];
            k::native_mmvq(type,m.w[j],m.x_q8_1[j],refs.back()->as(),ni,outs[j],1,stream);
        }
        m.count=5;
        if(!k::native_mmvq_many_fits(type,m,ni)){std::printf("%s: native_mmvq_many_fits false\n",ggml_type_name(ggml_type(type)));++bad;continue;}
        k::native_mmvq_many(type,m,ni,stream);ck(g::stream_synchronize(stream));
        int differ=0;size_t n=0;
        for(int j=0;j<5;++j){auto got=ys[j]->read(outs[j]),want=refs[j]->read(outs[j]);n+=got.size();
                             for(int i=0;i<outs[j];++i)differ+=std::memcmp(&got[i],&want[i],4)!=0;}
        std::printf("%s many, %d wide, %d outputs: %d of %zu outputs differ in their bits from native_mmvq's\n",
                    ggml_type_name(ggml_type(type)),ni,big?8576:268,differ,n);
        bad+=differ;
    }
    p::metal_detail::allow_mm_rows(true);
    ck(g::stream_destroy(stream));ggml_quantize_free();return bad?1:0;
}
