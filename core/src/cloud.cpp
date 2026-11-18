// PointCloud2 payload -> analysis-frame XYZ, the same points in the same order as metro/io.py decode.
// The buffer comes straight from the sensor driver or a bag, so every size and offset is checked
// before a byte is read; a malformed message is refused, never read past its end.
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

enum Type { I8 = 1, U8 = 2, I16 = 3, U16 = 4, I32 = 5, U32 = 6, F32 = 7, F64 = 8 };

int type_size(int t) {
  switch (t) {
    case I8: case U8: return 1;
    case I16: case U16: return 2;
    case I32: case U32: case F32: return 4;
    case F64: return 8;
    default: return 0;
  }
}

// one scalar at p, converted to double; big-endian payloads are swapped first
double read(const uint8_t* p, int t, bool big) {
  uint8_t b[8];
  const int n = type_size(t);
  if (big) for (int k = 0; k < n; ++k) b[k] = p[n - 1 - k];
  else std::memcpy(b, p, n);
  switch (t) {
    case I8: { int8_t v; std::memcpy(&v, b, 1); return v; }
    case U8: return b[0];
    case I16: { int16_t v; std::memcpy(&v, b, 2); return v; }
    case U16: { uint16_t v; std::memcpy(&v, b, 2); return v; }
    case I32: { int32_t v; std::memcpy(&v, b, 4); return v; }
    case U32: { uint32_t v; std::memcpy(&v, b, 4); return v; }
    case F32: { float v; std::memcpy(&v, b, 4); return v; }
    case F64: { double v; std::memcpy(&v, b, 8); return v; }
  }
  return NAN;
}

}  // namespace

extern "C" {

// offsets/types: x, y, z and (optionally, offset < 0 when absent) ring. transform: row-major 4x4
// source -> analysis, or NULL for the default sensor convention (forward -y, left +x).
// out_xyz must hold 3*width*height floats, out_ring width*height values (may be NULL).
// Returns the number of kept points, or -1 for an invalid layout.
long metro_decode(const uint8_t* data, size_t nbytes, uint32_t height, uint32_t width, uint32_t point_step,
                  uint32_t row_step, int bigendian, const int* offsets, const int* types, double min_distance,
                  const double* transform, float* out_xyz, uint16_t* out_ring) {
  if (!data || !offsets || !types || !out_xyz || point_step == 0) return -1;
  if (static_cast<uint64_t>(width) * point_step > row_step) return -1;
  if (static_cast<uint64_t>(height) * row_step > nbytes) return -1;
  for (int k = 0; k < 3; ++k) {
    const int s = type_size(types[k]);
    if (s == 0 || offsets[k] < 0 || static_cast<uint64_t>(offsets[k]) + s > point_step) return -1;
  }
  const bool has_ring = offsets[3] >= 0 && out_ring;
  if (has_ring) {
    const int s = type_size(types[3]);
    if (s == 0 || static_cast<uint64_t>(offsets[3]) + s > point_step) return -1;
  }
  const bool big = bigendian != 0;
  const bool f32 = !big && types[0] == F32 && types[1] == F32 && types[2] == F32;
  const float md2 = static_cast<float>(min_distance * min_distance);
  long n = 0;
  for (uint32_t r = 0; r < height; ++r) {
    const uint8_t* row = data + static_cast<size_t>(r) * row_step;
    for (uint32_t c = 0; c < width; ++c) {
      const uint8_t* p = row + static_cast<size_t>(c) * point_step;
      float x, y, z;
      if (f32) {
        std::memcpy(&x, p + offsets[0], 4), std::memcpy(&y, p + offsets[1], 4), std::memcpy(&z, p + offsets[2], 4);
      } else {
        x = static_cast<float>(read(p + offsets[0], types[0], big));
        y = static_cast<float>(read(p + offsets[1], types[1], big));
        z = static_cast<float>(read(p + offsets[2], types[2], big));
      }
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
      if (!(x * x + y * y + z * z > md2)) continue;
      float* o = out_xyz + 3 * n;
      if (!transform) {
        o[0] = -y, o[1] = x, o[2] = z;
      } else {
        for (int k = 0; k < 3; ++k)
          o[k] = static_cast<float>(transform[4 * k] * x + transform[4 * k + 1] * y + transform[4 * k + 2] * z +
                                    transform[4 * k + 3]);
      }
      if (has_ring) {
        // a ring number outside 0..65535 (a float field, garbage) would be undefined to convert
        const double rv = read(p + offsets[3], types[3], big);
        out_ring[n] = std::isfinite(rv) && rv >= 0. && rv <= 65535. ? static_cast<uint16_t>(rv) : 0;
      }
      ++n;
    }
  }
  return n;
}
}
