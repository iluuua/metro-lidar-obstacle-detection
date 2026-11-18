#include "geometry.hpp"

#include <algorithm>
#include <cmath>

namespace metro {

// ---------------------------------------------------------------- PathModel
PathModel::PathModel(Vec lat_, Vec vert_, double phi_, double support_, Vec su, Vec sh, PathInfo info_)
    : lat(std::move(lat_)), vert(std::move(vert_)), phi(phi_), support(support_), info(info_) {
  dlat = gradient(lat);
  arc.assign(N, 0.);
  for (int i = 1; i < N; ++i) {
    const double m = (dlat[i] + dlat[i - 1]) / 2;
    arc[i] = arc[i - 1] + std::sqrt(1 + m * m);
  }
  sigma_u = su.empty() ? Vec(N, .05) : std::move(su);
  sigma_h = sh.empty() ? Vec(N, .05) : std::move(sh);
}

PathModel PathModel::flat(double z0) { return PathModel(Vec(N, 0.), Vec(N, z0), 0., 0.); }

PathModel PathModel::shifted(double ds, bool second_order_transport) const {
  // The sensor keeps its mounting offset and yaw/pitch relative to the track, so
  // offset and heading at x = 0 are preserved and only the shape ahead is transported.
  if (ds == 0.) return PathModel(lat, vert, phi, support, sigma_u, sigma_h, info);
  Vec g(N);
  for (int i = 0; i < N; ++i) g[i] = i;
  Vec dvert = gradient(vert), tangent = dlat;
  if (second_order_transport) {
    // First-order edge differences bias the heading by half a grid cell on a curve, so repeated
    // transport injects a fictitious yaw every frame (found by Igor, 19.10).
    auto edge = [](const Vec& v, Vec& d) {
      const int n = static_cast<int>(v.size());
      d[0] = (-3 * v[0] + 4 * v[1] - v[2]) / 2.;
      d[n - 1] = (3 * v[n - 1] - 4 * v[n - 2] + v[n - 3]) / 2.;
    };
    edge(lat, tangent), edge(vert, dvert);
  }
  const double l0 = interp_grid(ds, lat) - lat[0], d0 = interp_grid(ds, tangent) - tangent[0];
  const double v0 = interp_grid(ds, vert) - vert[0], e0 = interp_grid(ds, dvert) - dvert[0];
  Vec nl(N), nv(N);
  for (int i = 0; i < N; ++i) {
    const double xo = std::min(i + ds, double(N - 1));
    nl[i] = interp_grid(xo, lat) - l0 - d0 * i;
    nv[i] = interp_grid(xo, vert) - v0 - e0 * i;
  }
  return PathModel(nl, nv, phi, std::max(0., support - ds), sigma_u, sigma_h, info);
}

// ---------------------------------------------------------------- NearTrack
NearTrack::NearTrack(const Config& cfg) : cfg_(cfg) {
  edges_ = arange(3., 20., .75);
  for (double v : arange(20., 34., 1.5)) edges_.push_back(v);
  for (double v : arange(34., 46.1, 2.5)) edges_.push_back(v);
}

std::optional<NearFit> NearTrack::fit(const Cloud& p, const PathModel& prior) const {
  const Vec& E = edges_;
  const int ns = static_cast<int>(E.size()) - 1;
  const double NEG = -kInf;
  std::vector<float> M(static_cast<size_t>(ns) * NY, -std::numeric_limits<float>::infinity());
  struct Cell {
    int key;
    double z;
  };
  std::vector<Cell> cells;
  double lo = kInf;
  for (const P3& q : p) {
    const double x = q.x;
    if (!(x >= E.front() && x < E.back() && std::abs(q.y) < 4.)) continue;
    const double y = q.y - prior.c(x), z = q.z - prior.r(x);
    if (!(std::abs(y) < 1.85 && z > -.45 && z < .35)) continue;
    const int ix = static_cast<int>(std::upper_bound(E.begin(), E.end(), x) - E.begin()) - 1;
    const int iy = static_cast<int>((y - Y0) / DY);
    if (ix < 0 || ix >= ns || iy < 0 || iy >= NY) continue;
    cells.push_back({ix * NY + iy, z});
    lo = std::min(lo, z);
  }
  // per-cell maximum, quantized to 1 mm above the lowest height (as the packed-integer max in Python)
  for (const Cell& c : cells) {
    const double q = std::floor((c.z - lo) / .001);
    const float v = static_cast<float>(q * .001 + lo);
    M[c.key] = std::max(M[c.key], v);
  }
  auto Mat = [&](int j, int i) -> double {
    if (i < 0 || i >= NY) return NEG;
    return M[static_cast<size_t>(j) * NY + i];
  };
  // R: ridge height above the highest neighbour 10-20 cm to either side
  std::vector<double> R(static_cast<size_t>(ns) * NY, 0.), R2(static_cast<size_t>(ns) * NY, 0.);
  for (int j = 0; j < ns; ++j) {
    for (int i = 0; i < NY; ++i) {
      double nb = NEG;
      for (int k = 5; k <= 10; ++k) nb = std::max({nb, Mat(j, i - k), Mat(j, i + k)});
      const double r = Mat(j, i) - nb;
      R[static_cast<size_t>(j) * NY + i] = std::isfinite(r) ? r : 0.;
    }
  }
  for (int j = 0; j < ns; ++j)
    for (int i = 0; i < NY; ++i) {
      double v = R[static_cast<size_t>(j) * NY + i];
      for (int d : {1, -1, 2, -2}) {
        const int k = i - d;
        if (k >= 0 && k < NY) v = std::max(v, R[static_cast<size_t>(j) * NY + k]);
        else v = std::max(v, 0.);
      }
      R2[static_cast<size_t>(j) * NY + i] = v;
    }
  const int half = static_cast<int>(std::lround(cfg_.gauge / 2 / DY));
  auto Rv = [&](int j, int i) { return (i < 0 || i >= NY) ? 0. : R2[static_cast<size_t>(j) * NY + i]; };
  std::vector<double> P(static_cast<size_t>(ns) * NY, 0.);
  for (int j = 0; j < ns; ++j)
    for (int i = 0; i < NY; ++i) {
      const double hl = Mat(j, i - half), hr = Mat(j, i + half);
      const double dz = std::abs(hl - hr);  // NaN when both are empty
      double pv = std::min(Rv(j, i - half), Rv(j, i + half));
      P[static_cast<size_t>(j) * NY + i] = (dz < .2 && pv > .02) ? std::min(pv, .25) : 0.;
    }
  // Viterbi over slabs; deviation from the prior is nearly constant.
  Vec centres(NY);
  for (int i = 0; i < NY; ++i) centres[i] = Y0 + DY * (i + .5);
  auto score = [&](int j, int i) { return std::log1p(40 * P[static_cast<size_t>(j) * NY + i]); };
  Vec D(NY), best(NY), cand(NY);
  std::vector<int> arg(NY), back(static_cast<size_t>(ns) * NY, 0);
  for (int i = 0; i < NY; ++i) D[i] = score(0, i) - .5 * (centres[i] / .35) * (centres[i] / .35);
  for (int j = 1; j < ns; ++j) {
    best = D;
    for (int i = 0; i < NY; ++i) arg[i] = i;
    for (int d : {-2, -1, 1, 2}) {
      for (int i = 0; i < NY; ++i) {
        const int src = i - d;
        const double cv = (src >= 0 && src < NY) ? D[src] - .15 * d * d : -1e9;
        if (cv > best[i]) {
          best[i] = cv;
          arg[i] = src;
        }
      }
    }
    for (int i = 0; i < NY; ++i) {
      D[i] = best[i] + score(j, i) - .03;
      back[static_cast<size_t>(j) * NY + i] = arg[i];
    }
  }
  std::vector<int> path(ns);
  path[ns - 1] = static_cast<int>(std::max_element(D.begin(), D.end()) - D.begin());
  for (int j = ns - 1; j > 0; --j) path[j - 1] = back[static_cast<size_t>(j) * NY + path[j]];
  Vec xs_good, dev_good, sup_good, hl_good, hr_good;
  for (int j = 0; j < ns; ++j) {
    const double sup = P[static_cast<size_t>(j) * NY + path[j]];
    if (sup > .03) {
      xs_good.push_back((E[j] + E[j + 1]) / 2);
      dev_good.push_back(centres[path[j]]);
      sup_good.push_back(sup);
      hl_good.push_back(Mat(j, path[j] - half));
      hr_good.push_back(Mat(j, path[j] + half));
    }
  }
  if (xs_good.size() < 5 || *std::min_element(xs_good.begin(), xs_good.end()) > 15) return std::nullopt;
  NearFit out;
  std::vector<std::array<double, 3>> A;
  for (double x : xs_good) A.push_back({1., x, x * x});
  out.dev = robust_lstsq(A, 3, dev_good, sup_good);
  out.dev[2] = std::clamp(out.dev[2], -1 / 500., 1 / 500.);
  Vec ar;
  for (size_t k = 0; k < xs_good.size(); ++k)
    ar.push_back(std::abs(dev_good[k] - (out.dev[0] + out.dev[1] * xs_good[k] + out.dev[2] * xs_good[k] * xs_good[k])));
  out.resid = ar.empty() ? 0. : median(ar);
  // rail-head plane (relative to prior r): z = a + b x + phi u
  std::vector<std::array<double, 3>> B;
  Vec zz;
  for (size_t k = 0; k < xs_good.size(); ++k)
    if (std::isfinite(hl_good[k])) B.push_back({1., xs_good[k], -cfg_.gauge / 2}), zz.push_back(hl_good[k]);
  for (size_t k = 0; k < xs_good.size(); ++k)
    if (std::isfinite(hr_good[k])) B.push_back({1., xs_good[k], cfg_.gauge / 2}), zz.push_back(hr_good[k]);
  out.rail = robust_lstsq(B, 3, zz, Vec(zz.size(), 1.), .008);
  out.support = *std::max_element(xs_good.begin(), xs_good.end());
  out.n = static_cast<int>(xs_good.size());
  return out;
}

// ---------------------------------------------------------------- Template
Template::Template(const Config& cfg) : cfg_(cfg) {
  nu_ = static_cast<int>(std::lround((U1 - U0) / D));
  nh_ = static_cast<int>(std::lround((H1 - H0) / D));
  count_.assign(static_cast<size_t>(nu_) * nh_, 0.f);
}

void Template::reset() {
  std::fill(count_.begin(), count_.end(), 0.f);
  ready_ = false;
}

void Template::update(const Vec& u, const Vec& h, bool envelope_clear) {
  std::vector<uint8_t> now(count_.size(), 0);
  for (size_t k = 0; k < u.size(); ++k) {
    if (envelope_clear && std::abs(u[k]) < cfg_.half_width + .05 && h[k] > .04 && h[k] < cfg_.height + .1) continue;
    const double tu = (u[k] - U0) / D, th = (h[k] - H0) / D;
    if (!(tu > -1e9 && tu < 1e9 && th > -1e9 && th < 1e9)) continue;
    const int iu = static_cast<int>(tu), ih = static_cast<int>(th);  // truncation, like astype(int32)
    if (iu < 0 || iu >= nu_ || ih < 0 || ih >= nh_) continue;
    now[static_cast<size_t>(iu) * nh_ + ih] = 1;
  }
  std::vector<uint8_t> occ(count_.size());
  size_t n_occ = 0;
  for (size_t i = 0; i < count_.size(); ++i) {
    count_[i] = count_[i] * static_cast<float>(cfg_.template_decay) + now[i];
    occ[i] = count_[i] > .5f;
    n_occ += occ[i];
  }
  if (n_occ < 50) {
    ready_ = false;
    return;
  }
  dist_ = distance_to_occupied(occ, nu_, nh_);
  score_.resize(dist_.size());
  for (size_t i = 0; i < dist_.size(); ++i) {
    dist_[i] = static_cast<float>(dist_[i] * D);
    const double t = dist_[i] / .09;
    score_[i] = static_cast<float>(std::exp(-.5 * t * t));
  }
  ready_ = true;
}

// ---------------------------------------------------------------- WallOdometry
std::vector<double> WallOdometry::image(const Cloud& p, const PathModel& path, bool& ok) const {
  // tall (0.5 m) cells, short along the track (0.1 m); a 2 m high-pass removes the smooth lining
  std::vector<double> tot(static_cast<size_t>(2) * nx_ * nh_, 0.), cnt(tot.size(), 0.);
  size_t selected = 0;
  for (const P3& q : p) {
    const double x = q.x;
    if (!(x > X0 && x < X1)) continue;
    const double u = q.y - path.c(x), h = q.z - path.r(x);
    if (!(std::abs(u) > 1.3 && std::abs(u) < 4.5 && h > H0 && h < H1)) continue;
    ++selected;
    const int side = u > 0;
    const long ix = static_cast<long>((x - X0) / DX), ih = static_cast<long>((h - H0) / DH);
    if (ix < 0 || ix >= nx_ || ih < 0 || ih >= nh_) continue;
    const size_t key = (static_cast<size_t>(side) * nx_ + ix) * nh_ + ih;
    tot[key] += std::abs(u);
    cnt[key] += 1;
  }
  ok = selected >= 200;
  std::vector<double> img(tot.size(), kNaN);
  if (!ok) return img;
  for (size_t i = 0; i < tot.size(); ++i)
    if (cnt[i] > 0) img[i] = tot[i] / cnt[i];
  const int w = static_cast<int>(2. / DX);
  std::vector<double> v(nx_), n(nx_), ms(nx_), mn(nx_), out(img.size(), kNaN);
  for (int side = 0; side < 2; ++side)
    for (int ih = 0; ih < nh_; ++ih) {
      for (int ix = 0; ix < nx_; ++ix) {
        const double a = img[(static_cast<size_t>(side) * nx_ + ix) * nh_ + ih];
        v[ix] = std::isfinite(a) ? a : 0.;
        n[ix] = std::isfinite(a) ? 1. : 0.;
      }
      uniform_filter_reflect(v.data(), ms.data(), nx_, w);
      uniform_filter_reflect(n.data(), mn.data(), nx_, w);
      for (int ix = 0; ix < nx_; ++ix) {
        const size_t key = (static_cast<size_t>(side) * nx_ + ix) * nh_ + ih;
        if (std::isfinite(img[key])) out[key] = img[key] - ms[ix] / std::max(mn[ix], 1e-6);
      }
    }
  return out;
}

void WallOdometry::prime(const Cloud& p, const PathModel& path) {
  bool ok;
  std::vector<double> b = image(p, path, ok);
  prev_ = ok ? std::move(b) : std::vector<double>();
}

bool WallOdometry::update(const Cloud& p, const PathModel& path, bool has_dt, double dt, double& ds, double& quality) {
  bool ok;
  std::vector<double> b = image(p, path, ok);
  std::vector<double> a = std::move(prev_);
  prev_ = ok ? b : std::vector<double>();
  ds = 0.;
  quality = 0.;
  if (a.empty() || !ok || !has_dt || dt <= 0 || dt > MAX_DT) return false;
  Vec cands = has_v ? arange(std::max(0., v * dt - .6), v * dt + .61, .05)
                    : arange(0., std::max(3.2, std::min(20., MAX_SPEED * dt)) + .01, .05);
  Vec costs(cands.size(), kInf);
  for (size_t ci = 0; ci < cands.size(); ++ci) {
    const double k = cands[ci] / DX;
    const long k0 = static_cast<long>(std::floor(k));
    const double f = k - k0;
    if (k0 + 1 >= nx_) continue;
    const long len = nx_ - 1 - k0;
    double sum = 0.;
    long count = 0;
    for (int side = 0; side < 2; ++side)
      for (long ix = 0; ix < len; ++ix)
        for (int ih = 0; ih < nh_; ++ih) {
          const double a0 = a[(static_cast<size_t>(side) * nx_ + ix + k0) * nh_ + ih];
          const double a1 = a[(static_cast<size_t>(side) * nx_ + ix + k0 + 1) * nh_ + ih];
          const double sh = (1 - f) * a0 + f * a1;
          const double c = b[(static_cast<size_t>(side) * nx_ + ix) * nh_ + ih];
          if (std::isfinite(sh) && std::isfinite(c)) {
            sum += std::min(std::abs(sh - c), .1);
            ++count;
          }
        }
    if (count < 60) continue;
    costs[ci] = sum / count;
  }
  long best = -1;
  for (size_t i = 0; i < costs.size(); ++i)
    if (std::isfinite(costs[i]) && (best < 0 || costs[i] < costs[best])) best = static_cast<long>(i);
  if (best < 0) return false;
  ds = cands[best];
  if (best > 0 && best + 1 < static_cast<long>(costs.size()) && std::isfinite(costs[best - 1]) &&
      std::isfinite(costs[best + 1])) {
    const double den = costs[best - 1] - 2 * costs[best] + costs[best + 1];
    if (den > 1e-9) ds += std::clamp(.5 * (costs[best - 1] - costs[best + 1]) / den, -.5, .5) * .05;
  }
  Vec finite;
  for (double c : costs)
    if (std::isfinite(c)) finite.push_back(c);
  const double med = median(finite);
  quality = (med - costs[best]) / std::max(med, 1e-6);
  const double vv = std::max(0., ds) / dt;
  v = has_v ? .5 * v + .5 * vv : vv;
  has_v = true;
  ds = std::max(0., ds);
  return true;
}

}  // namespace metro
