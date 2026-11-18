// C interface for ctypes (metro/native.py) and other non-C++ callers.
#include <cstring>
#include <string>

#include "metro/detector.hpp"

namespace {
struct Handle {
  metro::Detector* detector = nullptr;
  std::string result, error;
};
thread_local std::string g_error;
}  // namespace

extern "C" {

// config_json: flat JSON object with Config field overrides (may be empty or NULL).
void* metro_create(const char* config_json) {
  try {
    metro::Config cfg;
    if (config_json && std::strlen(config_json)) cfg.update_from_json(config_json);
    auto* h = new Handle;
    h->detector = new metro::Detector(cfg);
    return h;
  } catch (const std::exception& e) {
    g_error = e.what();
    return nullptr;
  }
}

const char* metro_create_error() { return g_error.c_str(); }

void metro_destroy(void* handle) {
  auto* h = static_cast<Handle*>(handle);
  if (!h) return;
  delete h->detector;
  delete h;
}

void metro_reset(void* handle) { static_cast<Handle*>(handle)->detector->reset(); }

// xyz: n float32 triples in the analysis frame. Returns JSON valid until the next call, or NULL on error.
const char* metro_process(void* handle, const float* xyz, size_t n, double stamp, int has_stamp) {
  auto* h = static_cast<Handle*>(handle);
  try {
    h->result = h->detector->process(xyz, n, stamp, has_stamp != 0);
    return h->result.c_str();
  } catch (const std::exception& e) {
    h->error = e.what();
    h->detector->reset();
    return nullptr;
  }
}

const char* metro_error(void* handle) { return static_cast<Handle*>(handle)->error.c_str(); }
}

// ---- test hooks: numerical helpers compared against numpy/scipy in eval/parity_units.py
#include "numerics.hpp"
extern "C" {
void metro_test_edt(const unsigned char* occ, int nu, int nh, float* out) {
  std::vector<uint8_t> o(occ, occ + static_cast<size_t>(nu) * nh);
  auto d = metro::distance_to_occupied(o, nu, nh);
  std::copy(d.begin(), d.end(), out);
}
void metro_test_fit(int n, const double* x, const double* y, const double* w, size_t m, double lam, double* out) {
  metro::Vec xv(x, x + m), yv(y, y + m), wv(w, w + m);
  auto f = metro::penalized_fit(n, xv, yv, wv, lam);
  std::copy(f.begin(), f.end(), out);
}
void metro_test_uniform(const double* in, int n, int size, double* out) { metro::uniform_filter_reflect(in, out, n, size); }
}
