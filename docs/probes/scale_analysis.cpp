#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
static float h2f(uint16_t h){ uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023; float v; if(e==0) v=ldexpf((float)m,-24); else if(e==31) v=INFINITY; else v=ldexpf((float)(m+1024),(int)e-25); return s?-v:v; }
int main(int argc,char**argv){
  const char*fn=argv[1]; int K=atoi(argv[2]); int N=atoi(argv[3]);
  FILE*f=fopen(fn,"rb"); std::vector<uint8_t> d(28ull*(K/128)*N); fread(d.data(),1,d.size(),f); fclose(f);
  int G=K/128; std::vector<float> S((size_t)N*G); size_t zeros=0,tot=0; size_t cnt[3]={0,0,0};
  static const uint8_t p3[6]={1,3,9,27,81,243};
  for(int n=0;n<N;n++) for(int g=0;g<G;g++){
    const uint8_t*b=&d[((size_t)n*G+g)*28]; uint16_t h; memcpy(&h,b+26,2); S[(size_t)n*G+g]=h2f(h);
    for(int k=0;k<5;k++)for(int m=0;m<24;m++){ uint8_t q=b[m]*p3[k]; int xi=((uint16_t)q*3)>>8; cnt[xi]++; }
    for(int k=0;k<4;k++)for(int m=0;m<2;m++){ uint8_t q=b[24+m]*p3[k]; int xi=((uint16_t)q*3)>>8; cnt[xi]++; }
  }
  size_t T=cnt[0]+cnt[1]+cnt[2]; printf("trit dist: -1:%.3f  0:%.3f  +1:%.3f\n",cnt[0]/(double)T,cnt[1]/(double)T,cnt[2]/(double)T);
  std::vector<float> v(S); std::sort(v.begin(),v.end()); printf("scale min %.5g p1 %.5g med %.5g p99 %.5g max %.5g\n",v[0],v[v.size()/100],v[v.size()/2],v[v.size()*99/100],v.back());
  std::set<float> u(S.begin(),S.end()); printf("distinct fp16 scale values: %zu of %zu\n",u.size(),S.size());
  size_t nonpos=0; for(float x:S) if(!(x>0)) nonpos++; printf("non-positive scales: %zu\n",nonpos);
  // within-row spread
  double sumr=0,maxr=0; for(int n=0;n<N;n++){ float mn=1e30,mx=0; for(int g=0;g<G;g++){float x=S[(size_t)n*G+g]; mn=std::min(mn,x); mx=std::max(mx,x);} double r=mx/mn; sumr+=r; maxr=std::max(maxr,r);} printf("within-row max/min scale ratio: mean %.3f max %.3f\n",sumr/N,maxr);
  // rank-1 log fit
  std::vector<double> a(N,0),b(G,0); for(int n=0;n<N;n++){double s=0;for(int g=0;g<G;g++)s+=log(S[(size_t)n*G+g]); a[n]=s/G;}
  for(int g=0;g<G;g++){double s=0;for(int n=0;n<N;n++)s+=log(S[(size_t)n*G+g])-a[n]; b[g]=s/N;}
  double se=0,mxe=0,mean_rel=0; for(int n=0;n<N;n++)for(int g=0;g<G;g++){double r=log(S[(size_t)n*G+g])-a[n]-b[g]; se+=r*r; mxe=std::max(mxe,fabs(r)); mean_rel+=fabs(exp(r)-1);} 
  printf("rank-1 (row x group) log-fit: rms %.4f  max |log err| %.4f  mean rel err %.4f\n",sqrt(se/((double)N*G)),mxe,mean_rel/((double)N*G));
  // per-row-only fit (one scale per row)
  double se2=0,mx2=0; for(int n=0;n<N;n++)for(int g=0;g<G;g++){double r=log(S[(size_t)n*G+g])-a[n]; se2+=r*r; mx2=std::max(mx2,fabs(r));} printf("row-only log-fit: rms %.4f max %.4f\n",sqrt(se2/((double)N*G)),mx2);
  // group-only
  double se3=0; for(int n=0;n<N;n++)for(int g=0;g<G;g++){double r=log(S[(size_t)n*G+g])-b[g]; se3+=r*r;} 
  // power-of-two closeness: frac of log2(s) near integer
  size_t near=0; for(float x:S){ double l=log2(x); if(fabs(l-round(l))<0.02) near++; } printf("scales within 1.4%% of a power of two: %.4f\n",near/(double)S.size());
}
