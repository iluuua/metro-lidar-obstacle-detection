#include "path_fit.hpp"
#include "geometry.hpp"
#include <cassert>
#include <iostream>
using namespace metro;
int main(){
  Vec w(331,50.),curve(331),noisy(331);
  for(int i=0;i<331;++i)curve[i]=.1+.003*i+i*i/(2.*300.);
  Vec result=continuous_path_fit(curve,w,2e6,100.);
  for(int i=0;i<331;++i)assert(std::abs(result[i]-curve[i])<1e-5);
  PathModel transported(curve,Vec(331,-1.12),0.,300.);
  for(int step=0;step<20;++step)transported=transported.shifted(1.5,true);
  for(int i=0;i<250;++i)assert(std::abs(transported.lat[i]-curve[i])<1e-8);
  noisy=curve;noisy[150]+=2.;
  result=continuous_path_fit(noisy,w,2e6,100.);
  assert(std::abs(result[150]-curve[150])<.25);
  for(int i=0;i<331;++i)noisy[i]=4.*std::exp(-std::pow((i-160)/6.,2));
  result=continuous_path_fit(noisy,w,2e6,300.);
  double max_c=0.;for(int i=0;i<329;++i)max_c=std::max(max_c,std::abs(result[i]-2*result[i+1]+result[i+2]));
  assert(max_c<=1./300.+1e-5);
  assert(result[160]>.3); // a constraint must not replace all observations with a flat line
  for(int i=0;i<331;++i){noisy[i]=i*i/500.;w[i]=i<=3?1e8:50.;}
  result=continuous_path_fit(noisy,w,2e5,300.);
  for(int i=0;i<329;++i)assert(std::abs(result[i]-2*result[i+1]+result[i+2])<=1./300.+1e-5);
  // A spatial side-step in observations must become one joined curve, not
  // two straight pieces with a jump or a short, physically sharp zigzag.
  for(int i=0;i<331;++i){noisy[i]=i<150?0.:2.;w[i]=50.;}
  result=continuous_path_fit(noisy,w,2e6,100.);
  assert(std::abs(result.front())<.05&&std::abs(result.back()-2.)<.05);
  for(int i=0;i<329;++i){
    assert(std::abs(result[i]-2*result[i+1]+result[i+2])<=.01+1e-5);
    const double a=std::atan(result[i+1]-result[i]),b=std::atan(result[i+2]-result[i+1]);
    assert(std::abs(b-a)<.01001);
  }
  // Beyond the measured support a bend may not reverse: far observations that pull the curve the
  // other way (registration noise, a changed tunnel) must not turn it into an S or a hook.
  auto sign_flips=[](const Vec& f){int flips=0;double last=0.;for(int i=0;i+2<int(f.size());++i){const double c=f[i]-2*f[i+1]+f[i+2];if(std::abs(c)<1e-6)continue;if(last!=0.&&c*last<0.)++flips;last=c;}return flips;};
  const int support=150;
  Vec far(331),far_w(331);
  for(int i=0;i<331;++i){
    const bool seen=i<=support;
    far[i]=seen?i*i/(2.*600.):support*support/(2.*600.)+(i-support)*support/600.-(i-support)*(i-support)/(2.*120.);  // the far data bend back
    far_w[i]=seen?50.:.02;
  }
  const Vec free_fit=continuous_path_fit(far,far_w,2e6,100.);
  assert(sign_flips(free_fit)>=1);  // the failure this guards against: curvature crosses zero beyond the data
  const Vec held_fit=continuous_path_fit(far,far_w,2e6,100.,support,30.);
  assert(sign_flips(held_fit)==0);
  for(int i=0;i<=support;++i)assert(std::abs(held_fit[i]-i*i/1200.)<2e-3);  // the measured part keeps its shape
  for(int i=support;i+2<331;++i){const double c=held_fit[i]-2*held_fit[i+1]+held_fit[i+2];assert(c>=-1e-6&&c<=1./600.+1e-6);}
  // Without data the safer guess is a straight track: the tail curvature fades with distance.
  const Vec fading=continuous_path_fit(far,far_w,2e6,100.,support,30.,false,100.);
  assert(sign_flips(fading)==0);
  for(int i=support;i+2<331;++i){const double c=fading[i]-2*fading[i+1]+fading[i+2];assert(c>=-1e-6&&c<=std::exp(-(i+1.-support)/100.)/600.+1e-6);}
  assert(fading[330]<held_fit[330]-5.);  // the arc no longer runs away
  // A path declared straight continues exactly straight, however wild the unmeasured far data are.
  for(int i=0;i<331;++i){far[i]=i<=support?.5*i/100.:30.*std::sin((i-support)/25.);far_w[i]=i<=support?50.:.02;}
  const Vec straight=continuous_path_fit(far,far_w,2e6,100.,support,30.,true);
  assert(sign_flips(straight)==0);
  for(int i=0;i<=support;++i)assert(std::abs(straight[i]-.5*i/100.)<2e-3);
  for(int i=support+3;i+2<331;++i)assert(std::abs(straight[i]-2*straight[i+1]+straight[i+2])<1e-6);
  // Too little measured axis to judge a bend: the rule stays out of the way.
  const Vec unjudged=continuous_path_fit(far,far_w,2e6,100.,5.,30.),legacy=continuous_path_fit(far,far_w,2e6,100.);
  for(int i=0;i<331;++i)assert(std::abs(unjudged[i]-legacy[i])<1e-9);
  std::cout<<"Path fit: quadratic arc, repeated transport, outlier suppression, curvature constraints, no reversal beyond support passed; max D2="<<max_c<<"\n";
}
