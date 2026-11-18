// Garbage in, no crash out: random and hostile clouds through the detector and random payloads
// through the PointCloud2 parser. Build with sanitizers (see core/CMakeLists.txt, METRO_FUZZ):
//   cmake -S core -B core/build-fuzz -DMETRO_FUZZ=ON && cmake --build core/build-fuzz && core/build-fuzz/metro_fuzz 20000
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "metro/detector.hpp"

extern "C" long metro_decode(const uint8_t* data, size_t nbytes, uint32_t height, uint32_t width, uint32_t point_step,
                             uint32_t row_step, int bigendian, const int* offsets, const int* types,
                             double min_distance, const double* transform, float* out_xyz, uint16_t* out_ring);

namespace {

std::mt19937_64 rng(12345);
double uni(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }
int pick(int n) { return static_cast<int>(rng() % static_cast<uint64_t>(n)); }

// a plausible tunnel frame (bed, rails, wall) to which damage is applied
std::vector<float> tunnel(size_t n) {
  std::vector<float> p;
  p.reserve(3 * n);
  for (size_t k = 0; k < n; ++k) {
    const double x = 3 + 150 * std::pow(uni(0, 1), 2);
    const int part = pick(4);
    double y, z;
    if (part == 0) y = uni(-1.4, 1.4), z = -1.4;
    else if (part == 1) y = (pick(2) ? .795 : -.795) + uni(-.03, .03), z = -1.12;
    else { const double a = uni(-1.9, 1.9); y = 2.6 * std::sin(a), z = .5 + 2.6 * std::cos(a); }
    p.insert(p.end(), {float(x), float(y), float(z)});
  }
  return p;
}

std::vector<float> hostile(int kind) {
  const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
  switch (kind) {
    case 0: return {};                                                      // empty
    case 1: return std::vector<float>(3 * 50000, 0.f);                      // all at the sensor
    case 2: { std::vector<float> p(3 * 20000); for (auto& v : p) v = nan; return p; }
    case 3: { std::vector<float> p(3 * 20000); for (auto& v : p) v = pick(2) ? inf : -inf; return p; }
    case 4: { std::vector<float> p(3 * 20000); for (auto& v : p) v = float(uni(-1e30, 1e30)); return p; }
    case 5: { std::vector<float> p; for (int k = 0; k < 40000; ++k) p.insert(p.end(), {10.f, 0.f, -0.5f}); return p; }
    case 6: { auto p = tunnel(60000); for (size_t k = 0; k < p.size(); k += 7) p[k] = nan; return p; }
    case 7: { auto p = tunnel(60000); for (size_t k = 0; k < p.size(); k += 11) p[k] = float(uni(-1e6, 1e6)); return p; }
    case 8: { std::vector<float> p(3 * 30000); for (auto& v : p) v = float(uni(-400, 400)); return p; }
    case 9: { auto p = tunnel(400000); return p; }                          // a heavy frame
    default: return tunnel(80000);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const long frames = argc > 1 ? std::atol(argv[1]) : 2000;
  metro::Detector det;
  double t = 0.;
  long json_bytes = 0;
  for (long f = 0; f < frames; ++f) {
    const int kind = pick(14);
    std::vector<float> p = hostile(kind);
    // time: mostly 10 Hz, sometimes backwards, a long gap, or no stamp at all
    const int clock = pick(20);
    if (clock == 0) t -= uni(0, 5);
    else if (clock == 1) t += uni(3, 100);
    else t += .1;
    const bool has_t = clock != 2;
    try {
      const std::string out = det.process(p.data(), p.size() / 3, t, has_t);
      json_bytes += static_cast<long>(out.size());
    } catch (const std::exception& e) {
      std::printf("frame %ld kind %d: exception %s\n", f, kind, e.what());
      det.reset();
    }
    // the parser on a random layout and random bytes
    const uint32_t step = 4 + pick(60), width = static_cast<uint32_t>(pick(2000)), height = 1 + pick(3);
    const uint32_t row = width * step + static_cast<uint32_t>(pick(16));
    std::vector<uint8_t> data(static_cast<size_t>(row) * height + static_cast<size_t>(pick(2) ? 0 : pick(64)));
    for (auto& b : data) b = static_cast<uint8_t>(rng());
    if (pick(5) == 0 && !data.empty()) data.resize(data.size() / 2);      // truncated
    int offsets[4] = {pick(64) - 4, pick(64) - 4, pick(64) - 4, pick(3) ? -1 : pick(64)};
    int types[4] = {pick(10), pick(10), pick(10), pick(10)};
    std::vector<float> xyz(3 * static_cast<size_t>(width) * height);
    std::vector<uint16_t> ring(static_cast<size_t>(width) * height);
    const double tr[16] = {0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    metro_decode(data.data(), data.size(), height, width, step, row, pick(2), offsets, types, 1.,
                 pick(2) ? tr : nullptr, xyz.empty() ? nullptr : xyz.data(), ring.empty() ? nullptr : ring.data());
    if ((f + 1) % 1000 == 0) std::printf("%ld frames, %ld bytes of answers\n", f + 1, json_bytes);
  }
  std::printf("done: %ld frames without a crash\n", frames);
  return 0;
}
