#include "test_utils.hpp"
#include "q36_launch.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"
#include <algorithm>
#include <cstdint>
int main(){
 int bad=0;g::Stream s;ck(g::stream_create(&s));
 int n=2048,rows=3;auto x=values(n*rows),w=values(n,.1f);for(auto&v:w)v+=1;
 Buffer dx(x),dw(w),dy(n*rows*4);std::vector<float> ref(x.size());
 for(int r=0;r<rows;++r){double sum=0;for(int i=0;i<n;++i)sum+=double(x[r*n+i])*x[r*n+i];float inv=1/std::sqrt(sum/n+1e-6);for(int i=0;i<n;++i)ref[r*n+i]=x[r*n+i]*w[i]*inv;}
 launch_k_rms(rows,256,s,dx.as(),dw.as(),dy.as(),n,1e-6);ck(g::stream_synchronize(s));
 bad+=compare(dy.read(x.size()),ref,"q36 RMS");

 // State update and gated normalization against a double-precision recurrence.
 const int C=512,HV=2,T=3;auto h=values(C*T,.04f),ga=values(HV*T,.1f),be=values(HV*T,.1f),z=values(HV*128*T,.3f);
 for(auto& v:ga)v=-.2f+v;for(auto& v:be)v+=.5f;
 auto state=values(128*HV*128,.01f);std::vector<float> gamma(128,1),expected(T*HV*128);std::vector<double> st(state.begin(),state.end());
 for(int t=0;t<T;++t)for(int head=0;head<HV;++head){double o[128],ss=0;double decay=std::exp(ga[t*HV+head]);
  for(int col=0;col<128;++col){double kv=0;for(int r=0;r<128;++r)kv+=st[(r*HV+head)*128+col]*h[t*C+128+r];
   double delta=(h[t*C+256+head*128+col]-decay*kv)*be[t*HV+head];o[col]=0;
   for(int r=0;r<128;++r){auto& v=st[(r*HV+head)*128+col];v=decay*v+h[t*C+128+r]*delta;o[col]+=v*h[t*C+r];}
   o[col]/=std::sqrt(128.);ss+=o[col]*o[col];}
  for(int col=0;col<128;++col){int i=(t*HV+head)*128+col;expected[i]=o[col]/std::sqrt(ss/128+1e-6)*z[i]/(1+std::exp(-z[i]));}}
 Buffer dh(h),dg(ga),db(be),dz(z),ds(state),dn(gamma),out(expected.size()*4);
 launch_k_gdn_seq(HV,dim3(128,2),s,ds.as(),dh.as(),C,dg.as(),db.as(),dz.as(),dn.as(),1e-6,out.as(),1,HV,T,true);
 ck(g::stream_synchronize(s));bad+=compare(out.read(expected.size()),expected,"q36 GDN output");
 bad+=compare(ds.read(state.size()),std::vector<float>(st.begin(),st.end()),"q36 GDN state");
 // The prompt column-split path must implement the same recurrence.
 ds.put(state.data(),state.size()*4);Buffer raw(expected.size()*4);
 launch_k_gdn_seq_cols(dim3(HV,4),dim3(32,16),s,ds.as(),dh.as(),C,dg.as(),db.as(),raw.as(),1,HV,T);
 launch_k_gdn_norm(dim3(HV,T),128,s,raw.as(),dz.as(),dn.as(),1e-6,out.as(),HV);
 ck(g::stream_synchronize(s));bad+=compare(out.read(expected.size()),expected,"q36 prompt GDN output");
 bad+=compare(ds.read(state.size()),std::vector<float>(st.begin(),st.end()),"q36 prompt GDN state");
 // Split-K attention across a 256-cell boundary and pointer-table sessions.
 const int ctx=512,P=270;auto q=values(256,.2f),kc=values(ctx*256,.15f),vc=values(ctx*256,.3f),gate=values(256,.4f);
 Buffer dq(q),dk(kc),dv(vc),dgate(gate),dp(4),pa(2*256*4),pm(8),pl(8),ao(256*4);
 int position=P-1;dp.put(&position,4);std::vector<float> ar(256);double den=0;
 for(int t=0;t<P;++t){double dot=0;for(int j=0;j<256;++j)dot+=q[j]*kc[t*256+j];double e=std::exp(dot/16);den+=e;for(int j=0;j<256;++j)ar[j]+=e*vc[t*256+j];}
 for(int j=0;j<256;++j)ar[j]/=den*(1+std::exp(-gate[j]));
 Buffer kt(8),vt(8),slot(4);auto kp=dk.as(),vp=dv.as();int zero=0;kt.put(&kp,8);vt.put(&vp,8);slot.put(&zero,4);
 RowSel rs;rs.slot=slot.as<int32_t>();rs.kc=kt.as<float*>();rs.vc=vt.as<float*>();
 launch_k_attn_part(dim3(2,1),256,s,dq.as(),nullptr,nullptr,dp.as<int32_t>(),0,1,1,ctx,1.f/16,2,pa.as(),pm.as(),pl.as(),rs);
 launch_k_attn_merge(1,256,s,pa.as(),pm.as(),pl.as(),dgate.as(),dp.as<int32_t>(),0,1,2,ao.as(),rs);
 ck(g::stream_synchronize(s));bad+=compare(ao.read(256),ar,"q36 split-K session attention");
 // The routed experts' SwiGLU is unclamped, silu(g) * u: the prompt chunks' (mmq::swiglu) and the grouped experts' of a decode step.
 {const int F=512,R=2;auto gu=values(2*F*R,3.f);std::vector<float> ref(F*R);
  for(int r=0;r<R;++r)for(int i=0;i<F;++i){float a=gu[r*2*F+i];ref[r*F+i]=a/(1+std::exp(-a))*gu[r*2*F+F+i];}
  Buffer dgu(gu),dhh(ref.size()*4);strata::prefill::mmq::swiglu(dgu.as(),dhh.as(),R,F,false,s);
  ck(g::stream_synchronize(s));bad+=compare(dhh.read(ref.size()),ref,"q36 prompt expert SwiGLU");}
 {namespace K=strata::kernels;const int N=256,F=128,E=2;
  auto L=K::native_expert_layout(GGML_TYPE_Q8_0,GGML_TYPE_Q8_0,N,F);
  std::vector<uint8_t> blob(L.bytes*E);std::vector<float> wg(F*N*E),wu(F*N*E),wd(N*F*E),x(N);
  auto fill=[](std::vector<float>& v,float ph,float sc){for(size_t i=0;i<v.size();++i)v[i]=sc*std::sin(float(i)*.37f+ph);};
  fill(wg,.1f,.05f);fill(wu,.7f,.05f);fill(wd,1.3f,.08f);fill(x,2.f,.5f);
  const auto* q8=ggml_get_type_traits(GGML_TYPE_Q8_0);
  for(int e=0;e<E;++e){uint8_t* b=blob.data()+e*L.bytes;   // the GGUF's bytes, and the values the reference reads back from them
   ggml_quantize_chunk(GGML_TYPE_Q8_0,&wg[e*F*N],b,0,F,N,nullptr);q8->to_float(b,&wg[e*F*N],F*N);
   ggml_quantize_chunk(GGML_TYPE_Q8_0,&wu[e*F*N],b+L.up_off,0,F,N,nullptr);q8->to_float(b+L.up_off,&wu[e*F*N],F*N);
   ggml_quantize_chunk(GGML_TYPE_Q8_0,&wd[e*N*F],b+L.down_off,0,N,F,nullptr);q8->to_float(b+L.down_off,&wd[e*N*F],N*F);}
  // x and h are read in Q8_1 (32 values to an fp16 scale): the reference rounds them the same way
  auto q8_1=[](std::vector<double>& v){for(size_t b=0;b<v.size();b+=32){double m=0;for(int j=0;j<32;++j)m=std::max(m,std::abs(v[b+j]));
   const double d=ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(m/127)));for(int j=0;j<32;++j)v[b+j]=d?std::nearbyint(v[b+j]/d)*d:0;}};
  std::vector<double> xr(x.begin(),x.end());q8_1(xr);std::vector<float> ref(N*E);
  for(int e=0;e<E;++e){std::vector<double> hh(F);
   for(int i=0;i<F;++i){double gs=0,us=0;for(int k=0;k<N;++k){gs+=wg[(e*F+i)*N+k]*xr[k];us+=wu[(e*F+i)*N+k]*xr[k];}hh[i]=gs/(1+std::exp(-gs))*us;}
   q8_1(hh);
   for(int j=0;j<N;++j){double y=0;for(int i=0;i<F;++i)y+=wd[(e*N+j)*F+i]*hh[i];ref[e*N+j]=float(y);}}
  Buffer dblob(blob.size()),dx(x),xq(K::native_q8_1_bytes(N,1)),scr(K::native_expert_scratch_bytes(E,F)),dout(ref.size()*4),
         tab(8*E),bounds(4*(E+1)),count(4),dst(4*E),tok(4*E);
  dblob.put(blob.data(),blob.size());
  unsigned long long ptr[E];for(int e=0;e<E;++e)ptr[e]=(unsigned long long)(uintptr_t)(dblob.as<uint8_t>()+e*L.bytes);
  int32_t bnd[E+1]={0,1,2},cnt=E,dsts[E]={0,1},toks[E]={0,0};
  tab.put(ptr,sizeof ptr);bounds.put(bnd,sizeof bnd);count.put(&cnt,4);dst.put(dsts,sizeof dsts);tok.put(toks,sizeof toks);
  K::native_quantize_q8_1(dx.as(),xq.as(),N,1,s);
  K::native_expert_grouped(L,tab.as<unsigned long long>(),bounds.as<int32_t>(),count.as<int32_t>(),dst.as<int32_t>(),
                           tok.as<int32_t>(),E,E,xq.as(),scr.as(),dout.as(),s);
  ck(g::stream_synchronize(s));auto got=dout.read(ref.size());float most=0,big=0;
  for(size_t i=0;i<ref.size();++i){most=std::max(most,std::abs(got[i]-ref[i]));big=std::max(big,std::abs(ref[i]));}
  // gate and up agree to ~5e-4; an h value that rounds to the next Q8_1 step on one side moves its outputs by about 0.4% of
  // the largest (measured on an M5 Ultra: 0.6% at most).  An expert that adds nothing is off by 100%.
  const bool ok=most<=.02f*big;bad+=!ok;
  std::printf("q36 grouped Q8_0 experts: max error %g of %g %s\n",most,big,ok?"ok":"FAILED");}
 ck(g::stream_destroy(s));return bad?1:0;
}
