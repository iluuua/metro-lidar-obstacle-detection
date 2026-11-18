// Numerical helpers with the semantics of the numpy/scipy calls they replace
// (np.interp, np.gradient, np.quantile, scipy.ndimage, solveh_banded).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace metro {

using Vec = std::vector<double>;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// np.arange(start, stop, step) for float steps: ceil((stop - start) / step) values.
inline Vec arange(double start, double stop, double step) {
  Vec out;
  const double n = std::ceil((stop - start) / step);
  for (long k = 0; k < static_cast<long>(n); ++k) out.push_back(start + k * step);
  return out;
}

inline Vec linspace(double a, double b, int n) {
  Vec out(n);
  for (int k = 0; k < n; ++k) out[k] = n == 1 ? a : a + (b - a) * k / (n - 1);
  if (n > 1) out[n - 1] = b;
  return out;
}

// np.interp on an increasing grid xp: clamps outside the range.
inline double interp(double x, const Vec& xp, const Vec& fp) {
  const size_t n = xp.size();
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  const size_t j = std::upper_bound(xp.begin(), xp.end(), x) - xp.begin();  // xp[j-1] <= x < xp[j]
  const double t = (x - xp[j - 1]) / (xp[j] - xp[j - 1]);
  return fp[j - 1] + t * (fp[j] - fp[j - 1]);
}

// Uniform 1 m grid 0..n-1 (PathModel.GRID): faster interp with the same result.
inline double interp_grid(double x, const Vec& fp) {
  const long n = static_cast<long>(fp.size());
  if (x <= 0.) return fp[0];
  if (x >= n - 1) return fp[n - 1];
  const long i = static_cast<long>(x);
  const double t = x - i;
  return fp[i] + t * (fp[i + 1] - fp[i]);
}

// np.gradient(f, unit spacing), edge_order=1.
inline Vec gradient(const Vec& f) {
  const size_t n = f.size();
  Vec g(n, 0.);
  if (n < 2) return g;
  g[0] = f[1] - f[0];
  g[n - 1] = f[n - 1] - f[n - 2];
  for (size_t i = 1; i + 1 < n; ++i) g[i] = (f[i + 1] - f[i - 1]) / 2.;
  return g;
}

// np.quantile(v, q), linear method. v is copied.
inline double quantile(Vec v, double q) {
  if (v.empty()) return kNaN;
  std::sort(v.begin(), v.end());
  const double pos = q * (v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(lo + 1, v.size() - 1);
  const double t = pos - lo;
  return v[lo] + t * (v[hi] - v[lo]);
}

inline double median(Vec v) { return quantile(std::move(v), .5); }

// Smooth f on the uniform grid g (0, 1, ..., n-1):
// min sum w (f(x) - y)^2 + lam * sum (second difference)^2.
// The normal matrix is symmetric pentadiagonal; solved by banded Cholesky.
inline Vec penalized_fit(int n, const Vec& x, const Vec& y, const Vec& w, double lam) {
  Vec d0(n, 0.), d1(n - 1, 0.), d2(n - 2, 0.), b(n, 0.);
  for (size_t k = 0; k < x.size(); ++k) {
    if (!(w[k] != 0.)) continue;
    // idx = clip(searchsorted(g, x) - 1, 0, n - 2) on the grid g = 0..n-1
    long idx = static_cast<long>(std::ceil(x[k])) - 1;  // searchsorted left: first g >= x
    if (x[k] <= 0.) idx = -1;
    idx = std::clamp(idx, 0L, static_cast<long>(n - 2));
    const double t = x[k] - idx;
    d0[idx] += w[k] * (1 - t) * (1 - t);
    d0[idx + 1] += w[k] * t * t;
    d1[idx] += w[k] * t * (1 - t);
    b[idx] += w[k] * (1 - t) * y[k];
    b[idx + 1] += w[k] * t * y[k];
  }
  // lam * D^T D for second differences: bands [1, -4, 6, -4, 1] with edge corrections
  for (int i = 0; i < n; ++i) {
    double c = 6.;
    if (i == 0 || i == n - 1) c = 1.;
    else if (i == 1 || i == n - 2) c = 5.;
    d0[i] += lam * c + 1e-6;
  }
  for (int i = 0; i < n - 1; ++i) d1[i] += lam * ((i == 0 || i == n - 2) ? -2. : -4.);
  for (int i = 0; i < n - 2; ++i) d2[i] += lam;
  // A = L L^T with L lower-triangular of bandwidth 2
  Vec l0(n), l1(n, 0.), l2(n, 0.);
  for (int i = 0; i < n; ++i) {
    if (i >= 2) l2[i] = d2[i - 2] / l0[i - 2];
    if (i >= 1) l1[i] = (d1[i - 1] - (i >= 2 ? l2[i] * l1[i - 1] : 0.)) / l0[i - 1];
    double s = d0[i] - l1[i] * l1[i] - l2[i] * l2[i];
    l0[i] = std::sqrt(std::max(s, 1e-300));
  }
  Vec z(n);
  for (int i = 0; i < n; ++i) {
    double s = b[i];
    if (i >= 1) s -= l1[i] * z[i - 1];
    if (i >= 2) s -= l2[i] * z[i - 2];
    z[i] = s / l0[i];
  }
  Vec f(n);
  for (int i = n - 1; i >= 0; --i) {
    double s = z[i];
    if (i + 1 < n) s -= l1[i + 1] * f[i + 1];
    if (i + 2 < n) s -= l2[i + 2] * f[i + 2];
    f[i] = s / l0[i];
  }
  return f;
}

// Huber weights of observations against a data-only smooth fit (one iteration).
inline Vec robust_weights(int n, const Vec& x, const Vec& y, const Vec& w, double lam, double tau) {
  bool any = false;
  for (double v : w) any |= v > 0;
  if (x.empty() || !any) return w;
  const Vec f = penalized_fit(n, x, y, w, lam);
  Vec ww(w.size());
  for (size_t k = 0; k < x.size(); ++k) {
    const double r = std::abs(y[k] - interp_grid(x[k], f));
    ww[k] = w[k] * std::min(1., tau / std::max(r, 1e-6));
  }
  return ww;
}

// Weighted RMS of residuals r(x) around each grid node within +-half metres;
// beyond the last observation the last value is kept.
inline Vec local_rms(int n, const Vec& x, const Vec& r, const Vec& w, double half) {
  Vec out(n, 0.);
  if (x.empty()) return out;
  std::vector<char> near(n, 0);
  for (int g = 0; g < n; ++g) {
    double num = 0., den = 0.;
    for (size_t k = 0; k < x.size(); ++k) {
      if (std::abs(g - x[k]) <= half) {
        num += w[k] * r[k] * r[k];
        den += w[k];
      }
    }
    if (den > 0) {
      near[g] = 1;
      out[g] = std::sqrt(num / den);
    }
  }
  int first = -1;
  for (int g = 0; g < n; ++g)
    if (near[g]) { first = g; break; }
  if (first < 0) return out;
  int last = first;
  for (int g = 0; g < n; ++g) {
    if (near[g]) last = g;
    else out[g] = out[g < first ? first : last];
  }
  return out;
}

// Exact squared Euclidean distance transform along one line (Felzenszwalb & Huttenlocher).
// Empty cells carry a large finite value, so they never win against an occupied one.
constexpr double kFar = 1e12;

inline void edt_1d(const double* f, double* d, int n, int* v, double* z) {
  int k = 0;
  v[0] = 0;
  z[0] = -kInf;
  z[1] = kInf;
  auto meet = [&](int q, int p) { return ((f[q] + double(q) * q) - (f[p] + double(p) * p)) / (2. * q - 2. * p); };
  for (int q = 1; q < n; ++q) {
    double s = meet(q, v[k]);
    while (s <= z[k]) {
      --k;
      s = meet(q, v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = kInf;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    d[q] = double(q - v[k]) * (q - v[k]) + f[v[k]];
  }
}

// Distance (in cells) from every cell to the nearest cell with occ != 0; 0 on occupied cells.
// Grid is row-major [nu][nh] like the numpy array.
inline std::vector<float> distance_to_occupied(const std::vector<uint8_t>& occ, int nu, int nh) {
  std::vector<double> g(static_cast<size_t>(nu) * nh);
  const int n = std::max(nu, nh);
  std::vector<double> f(n), d(n), z(n + 1);
  std::vector<int> v(n);
  // along h (rows of the array)
  for (int i = 0; i < nu; ++i) {
    for (int j = 0; j < nh; ++j) f[j] = occ[static_cast<size_t>(i) * nh + j] ? 0. : kFar;
    edt_1d(f.data(), d.data(), nh, v.data(), z.data());
    for (int j = 0; j < nh; ++j) g[static_cast<size_t>(i) * nh + j] = d[j];
  }
  std::vector<float> out(static_cast<size_t>(nu) * nh);
  for (int j = 0; j < nh; ++j) {
    for (int i = 0; i < nu; ++i) f[i] = g[static_cast<size_t>(i) * nh + j];
    edt_1d(f.data(), d.data(), nu, v.data(), z.data());
    for (int i = 0; i < nu; ++i) out[static_cast<size_t>(i) * nh + j] = static_cast<float>(std::sqrt(d[i]));
  }
  return out;
}

// scipy.ndimage.uniform_filter1d(values, size, mode='reflect') along a line of n samples:
// output[i] = mean(input[i - size/2 .. i + size - size/2 - 1]) with half-sample reflection.
inline void uniform_filter_reflect(const double* in, double* out, int n, int size) {
  const int left = size / 2;
  const int right = size - left - 1;
  auto at = [&](int i) {
    if (n == 1) return in[0];
    while (i < 0 || i >= n) {
      if (i < 0) i = -i - 1;
      if (i >= n) i = 2 * n - i - 1;
    }
    return in[i];
  };
  double s = 0.;
  for (int j = -left; j <= right; ++j) s += at(j);
  out[0] = s / size;
  for (int i = 1; i < n; ++i) {
    s += at(i + right) - at(i - left - 1);
    out[i] = s / size;
  }
}

// Weighted least squares with rows scaled by w (np.linalg.lstsq(A * w, b * w)), m <= 3 columns.
inline bool weighted_lstsq(const std::vector<std::array<double, 3>>& A, int m, const Vec& b, const Vec& w,
                           std::array<double, 3>& coef) {
  double M[3][3] = {{0}}, r[3] = {0};
  for (size_t k = 0; k < b.size(); ++k) {
    const double ww = w[k] * w[k];
    for (int i = 0; i < m; ++i) {
      r[i] += ww * A[k][i] * b[k];
      for (int j = 0; j < m; ++j) M[i][j] += ww * A[k][i] * A[k][j];
    }
  }
  // Gaussian elimination with partial pivoting; a tiny ridge keeps rank-deficient fits finite
  double scale = 0.;
  for (int i = 0; i < m; ++i) scale = std::max(scale, std::abs(M[i][i]));
  for (int i = 0; i < m; ++i) M[i][i] += 1e-12 * (scale > 0 ? scale : 1.);
  int piv[3] = {0, 1, 2};
  for (int c = 0; c < m; ++c) {
    int best = c;
    for (int i = c + 1; i < m; ++i)
      if (std::abs(M[i][c]) > std::abs(M[best][c])) best = i;
    if (std::abs(M[best][c]) < 1e-300) return false;
    if (best != c) {
      for (int j = 0; j < m; ++j) std::swap(M[c][j], M[best][j]);
      std::swap(r[c], r[best]);
      std::swap(piv[c], piv[best]);
    }
    for (int i = c + 1; i < m; ++i) {
      const double f = M[i][c] / M[c][c];
      for (int j = c; j < m; ++j) M[i][j] -= f * M[c][j];
      r[i] -= f * r[c];
    }
  }
  for (int i = m - 1; i >= 0; --i) {
    double s = r[i];
    for (int j = i + 1; j < m; ++j) s -= M[i][j] * coef[j];
    coef[i] = s / M[i][i];
  }
  for (int i = m; i < 3; ++i) coef[i] = 0.;
  return true;
}

// Iteratively reweighted fit with weights shrinking for residuals beyond 2 robust sigmas.
inline std::array<double, 3> robust_lstsq(const std::vector<std::array<double, 3>>& A, int m, const Vec& b, Vec w,
                                          double floor = .02, int iters = 3) {
  std::array<double, 3> coef{0., 0., 0.};
  if (b.empty()) return coef;
  weighted_lstsq(A, m, b, w, coef);
  Vec r(b.size()), ww(b.size());
  for (int it = 0; it < iters; ++it) {
    Vec ar(b.size());
    for (size_t k = 0; k < b.size(); ++k) {
      double v = 0.;
      for (int i = 0; i < m; ++i) v += A[k][i] * coef[i];
      r[k] = b[k] - v;
      ar[k] = std::abs(r[k]);
    }
    const double s = std::max(floor, 1.4826 * median(ar));
    for (size_t k = 0; k < b.size(); ++k) ww[k] = w[k] * std::min(1., 2. * s / std::max(std::abs(r[k]), 1e-9));
    weighted_lstsq(A, m, b, ww, coef);
  }
  return coef;
}

}  // namespace metro
