// Joint path fit on the 1 m grid. Curvature changes are penalized, so a line
// and a constant-curvature arc are not pulled towards a straight track.
#pragma once
#include "numerics.hpp"

namespace metro {

class PathBandSolver {
  std::vector<std::array<double,4>> lower;
 public:
  explicit PathBandSolver(std::vector<std::array<double,4>> a) : lower(std::move(a)) {
    for (int i=0;i<int(lower.size());++i) for (int j=std::max(0,i-3);j<=i;++j) {
      double v=lower[i][i-j];
      for(int k=std::max({0,i-3,j-3});k<j;++k)v-=lower[i][i-k]*lower[j][j-k];
      lower[i][i-j]=i==j?std::sqrt(std::max(v,1e-12)):v/lower[j][0];
    }
  }
  Vec solve(Vec b) const {
    const int n=int(b.size());
    for(int i=0;i<n;++i){for(int d=1;d<=std::min(i,3);++d)b[i]-=lower[i][d]*b[i-d];b[i]/=lower[i][0];}
    for(int i=n-1;i>=0;--i){for(int d=1;d<=std::min(n-1-i,3);++d)b[i]-=lower[i+d][d]*b[i+d];b[i]/=lower[i][0];}
    return b;
  }
};

// Per-node interval of the discrete curvature D2 y = y[i]-2y[i+1]+y[i+2].
// Where the data end (`support` metres, 1 m grid) nothing measured says the track bends the other
// way: lidar looks along the track and the inner wall hides what lies behind a bend. Beyond the
// support the curvature may therefore only keep the sign it had on the last `window` metres of
// measured axis and may not grow past that mean value, which fades with distance beyond the support
// (e-folding length `decay`, 0 = no fading): without data the safer guess is a straight track. A path
// the detector declared straight (`straight`) continues exactly straight.
// support < 0 disables this and leaves only the radius bound.
inline void curvature_limits(const Vec& fit,double bound,double support,double window,bool straight,double decay,Vec& lo,Vec& hi) {
  const int nodes=int(fit.size())-2;
  lo.assign(nodes,-bound);hi.assign(nodes,bound);
  if(support<10.||nodes<1)return;
  const int end=std::min(nodes-1,int(std::floor(support))-2);  // last node fully inside the measured axis
  const int begin=std::max(0,end-int(window)+1);
  double mean=0.;
  if(end>=begin){
    for(int i=begin;i<=end;++i)mean+=fit[i]-2*fit[i+1]+fit[i+2];
    mean/=end-begin+1;
  }
  mean=straight?0.:std::clamp(mean,-bound,bound);
  for(int i=std::max(0,end+1);i<nodes;++i){
    const double limit=decay>0.?mean*std::exp(-(i+1-support)/decay):mean;
    lo[i]=std::min(0.,limit);hi[i]=std::max(0.,limit);
  }
}

inline Vec continuous_path_fit(const Vec& measurement,const Vec& weights,double lambda,double radius,double support=-1.,double window=30.,bool straight=false,double decay=0.,double far_weight=0.) {
  const int n=int(measurement.size());
  if(n<4)return measurement;
  std::vector<std::array<double,4>> a(n, {0.,0.,0.,0.});Vec rhs(n);
  // Past the measured support the "observations" are only a continuation of the near geometry. They
  // must not pull the measured part sideways through the stiff curve, so the extension is left to the
  // penalty (a constant-curvature arc) and to the one-sided curvature limits below.
  const bool judged=support>=10.;
  for(int i=0;i<n;++i){a[i][0]=judged&&i>std::ceil(support)?far_weight*weights[i]:std::max(weights[i],1e-6);rhs[i]=a[i][0]*measurement[i];}
  auto add_derivative=[&](const Vec& taps,double strength){
    for(int i=0;i+int(taps.size())<=n;++i)for(int j=0;j<int(taps.size());++j)for(int k=0;k<=j;++k)
      a[i+j][j-k]+=strength*taps[j]*taps[k];
  };
  add_derivative({-1.,3.,-3.,1.},lambda);
  Vec fit=PathBandSolver(a).solve(rhs);
  const double bound=1./std::max(radius,1.);
  auto curvature=[&](const Vec& f,int i){return f[i]-2*f[i+1]+f[i+2];};
  Vec lo,hi;curvature_limits(fit,bound,support,window,straight,decay,lo,hi);
  double violation=0.;for(int i=0;i<n-2;++i)violation=std::max({violation,curvature(fit,i)-hi[i],lo[i]-curvature(fit,i)});
  if(violation<=1e-7)return fit;
  // ADMM projection onto lo <= D2 y <= hi (|D2 y| <= 1/R, one-sided beyond the support). This is
  // conservative for true planar curvature y''/(1+y'^2)^(3/2). One band factorization serves all iterations.
  double rho=1e6;
  add_derivative({1.,-2.,1.},rho);PathBandSolver solver(a);
  Vec z(n-2),u(n-2,0.);
  for(int i=0;i<n-2;++i)z[i]=std::clamp(curvature(fit,i),lo[i],hi[i]);
  for(int iteration=0;iteration<400;++iteration){
    Vec b=rhs;
    for(int i=0;i<n-2;++i){double v=rho*(z[i]-u[i]);b[i]+=v;b[i+1]-=2*v;b[i+2]+=v;}
    fit=solver.solve(std::move(b));double residual=0.;
    for(int i=0;i<n-2;++i){double c=curvature(fit,i),next=std::clamp(c+u[i],lo[i],hi[i]);u[i]+=c-next;z[i]=next;residual=std::max({residual,c-hi[i],lo[i]-c});}
    if(residual<1e-7&&iteration>=15)break;
    if(iteration>=159&&iteration%40==39&&residual>1e-7&&rho<1e10){
      // Tight radii can conflict with a strongly anchored rail datum. Do not
      // silently return a visibly infeasible curve when the initial penalty
      // converges too slowly: strengthen it, retaining the unscaled dual.
      add_derivative({1.,-2.,1.},9*rho);rho*=10.;
      for(double& dual:u)dual/=10.;
      solver=PathBandSolver(a);
    }
  }
  return fit;
}
} // namespace metro
