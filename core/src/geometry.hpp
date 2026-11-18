// Track geometry: centre-line model, near rails, normal cross-section template, wall odometry.
#pragma once

#include <array>
#include <optional>
#include <vector>

#include "metro/detector.hpp"
#include "numerics.hpp"

namespace metro {

struct P3 {
  float x, y, z;
};
using Cloud = std::vector<P3>;

struct PathInfo {
  double support_m = 0., rail_support_m = 0.;
  bool extension_ok = false;
  int good_slabs = 0;
  double phi = 0.;
  int coasting_frames = 0;
  bool has_coasting = false;
  double analysis_limit_m = 0.;
  bool has_analysis_limit = false;
  double decision_limit_m = 0.;  // where this frame itself verifies the envelope (no held length)
  double break_m = 1e9;       // nearest slab the normal tunnel cannot explain (changed cross-section)
  bool support_held = false;
  double support_raw_m = 0.;  // this frame's own support, before the hold
  bool extension_raw = false;
};

// Centre line y = c(x), rail-head height z = r(x) + phi * u, sampled on the 1 m grid 0..330 (0..430 in
// libmetro_core_long for the 400 m reach of the Seyond Falcon). One library per grid: the longer grid
// changes the far geometry of a Pandar drive, so the Pandar keeps the validated one.
#ifndef METRO_GRID_N
#define METRO_GRID_N 331
#endif
struct PathModel {
  static constexpr int N = METRO_GRID_N;
  Vec lat, vert, dlat, arc, sigma_u, sigma_h;
  double phi = 0., support = 0.;
  PathInfo info;

  PathModel() = default;
  PathModel(Vec lat_, Vec vert_, double phi_, double support_, Vec su = {}, Vec sh = {}, PathInfo info_ = {});

  static PathModel flat(double z0);
  void idx(double x, long& i, double& f) const {
    double xf = std::min(std::max(x, 0.), N - 1 - 1e-9);
    i = static_cast<long>(xf);
    f = xf - i;
  }
  double at(const Vec& a, long i, double f) const { return a[i] * (1 - f) + a[i + 1] * f; }
  double c(double x) const {
    long i;
    double f;
    idx(x, i, f);
    return at(lat, i, f);
  }
  double dc(double x) const {
    long i;
    double f;
    idx(x, i, f);
    return at(dlat, i, f);
  }
  double r(double x) const {
    long i;
    double f;
    idx(x, i, f);
    return at(vert, i, f);
  }
  void to_track(const P3& p, double& s, double& u, double& h) const {
    long i;
    double f;
    idx(p.x, i, f);
    const double cc = at(lat, i, f), d = at(dlat, i, f);
    const double cs = 1. / std::sqrt(1. + d * d);
    const double dy = p.y - cc;
    u = dy * cs;
    h = p.z - at(vert, i, f) - phi * u;
    s = at(arc, i, f) + dy * d * cs;
  }
  PathModel shifted(double ds, bool second_order_transport = false) const;
  double arc_at_x(double x) const { return interp_grid(x, arc); }
};

struct NearFit {
  std::array<double, 3> dev{0, 0, 0};   // centre deviation from the prior: d0 + d1 x + d2 x^2
  std::array<double, 3> rail{0, 0, 0};  // rail plane relative to the prior: a + b x + phi u
  double support = 0.;
  int n = 0;
  double resid = 0.;
};

// Rail-pair ridges in slabs, de-skewed by the previous centre line.
class NearTrack {
 public:
  explicit NearTrack(const Config& cfg);
  std::optional<NearFit> fit(const Cloud& p, const PathModel& prior) const;

 private:
  const Config& cfg_;
  static constexpr double DY = .02, Y0 = -1.9;
  static constexpr int NY = 190;
  Vec edges_;
};

// Occupancy of the normal cross-section in (u, h), exponentially updated.
class Template {
 public:
  static constexpr double U0 = -10., U1 = 10., H0 = -1.5, H1 = 6.5, D = .05;
  explicit Template(const Config& cfg);
  void reset();
  void update(const Vec& u, const Vec& h, bool envelope_clear = true);
  bool ready() const { return ready_; }
  float dist(double u, double h) const { return dist_[cell(u, h)]; }
  // exp(-0.5 (d / 0.09)^2): the registration score of a point, precomputed per cell
  float score(double u, double h) const { return score_[cell(u, h)]; }

 private:
  size_t cell(double u, double h) const {
    const double tu = (u - U0) / D, th = (h - H0) / D;
    const int iu = tu <= 0 ? 0 : (tu >= nu_ - 1 ? nu_ - 1 : static_cast<int>(tu));
    const int ih = th <= 0 ? 0 : (th >= nh_ - 1 ? nh_ - 1 : static_cast<int>(th));
    return static_cast<size_t>(iu) * nh_ + ih;
  }
  const Config& cfg_;
  int nu_, nh_;
  std::vector<float> count_, dist_, score_;
  bool ready_ = false;
};

// Forward displacement from wall depth profiles of two consecutive frames.
class WallOdometry {
 public:
  static constexpr double X0 = 6., X1 = 52., DX = .1, H0 = -.3, H1 = 3.9, DH = .5;
  static constexpr double MAX_DT = 3., MAX_SPEED = 25.;
  void reset() {
    prev_.clear();
    has_v = false;
  }
  void prime(const Cloud& p, const PathModel& path);
  // returns false when no displacement could be measured
  bool update(const Cloud& p, const PathModel& path, bool has_dt, double dt, double& ds, double& quality);
  bool has_v = false;
  double v = 0.;

 private:
  std::vector<double> image(const Cloud& p, const PathModel& path, bool& ok) const;
  std::vector<double> prev_;  // 2 x nx x nh, NaN where empty
  int nx_ = static_cast<int>(std::ceil((X1 - X0) / DX)) + 1;
  int nh_ = static_cast<int>(std::ceil((H1 - H0) / DH)) + 1;
};

}  // namespace metro
