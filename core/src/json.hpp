// Minimal JSON: a writer for detector results and a reader for flat config objects
// ({"key": number | true | false | [numbers]}).
#pragma once

#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace metro {

class JsonWriter {
 public:
  std::string out;

  void raw(const char* s) { out += s; }
  void sep() {
    if (need_comma_) out += ',';
    need_comma_ = true;
  }
  void key(const char* k) {
    sep();
    out += '"';
    out += k;
    out += "\":";
    need_comma_ = false;
  }
  void open_obj() {
    sep();
    out += '{';
    need_comma_ = false;
  }
  void close_obj() {
    out += '}';
    need_comma_ = true;
  }
  void open_arr() {
    sep();
    out += '[';
    need_comma_ = false;
  }
  void close_arr() {
    out += ']';
    need_comma_ = true;
  }
  void num(double v) {
    sep();
    if (!std::isfinite(v)) {
      out += "null";
      return;
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.9g", v);
    out += buf;
  }
  void integer(long long v) {
    sep();
    out += std::to_string(v);
  }
  void boolean(bool v) {
    sep();
    out += v ? "true" : "false";
  }
  void null() {
    sep();
    out += "null";
  }
  void str(const std::string& s) {
    sep();
    out += '"';
    for (unsigned char c : s) {
      if (c == '"' || c == '\\') {
        out += '\\';
        out += static_cast<char>(c);
      } else if (c < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof buf, "\\u%04x", c);
        out += buf;
      } else {
        out += static_cast<char>(c);
      }
    }
    out += '"';
  }
  // key/value shortcuts
  void kv(const char* k, double v) { key(k), num(v); }
  void kv_int(const char* k, long long v) { key(k), integer(v); }
  void kv_bool(const char* k, bool v) { key(k), boolean(v); }
  void kv_str(const char* k, const std::string& v) { key(k), str(v); }
  void kv_null(const char* k) { key(k), null(); }
  void kv_opt(const char* k, bool has, double v) {
    key(k);
    if (has) num(v);
    else null();
  }
  template <class It>
  void kv_arr(const char* k, It begin, It end) {
    key(k);
    open_arr();
    for (It it = begin; it != end; ++it) num(*it);
    close_arr();
  }

 private:
  bool need_comma_ = false;
};

// Values of a flat JSON object.
struct JsonValue {
  bool is_array = false;
  double number = 0.;
  std::vector<double> array;
};

inline std::map<std::string, JsonValue> parse_flat_json(const std::string& text) {
  std::map<std::string, JsonValue> out;
  size_t i = 0;
  auto ws = [&] {
    while (i < text.size() && (text[i] == ' ' || text[i] == '\n' || text[i] == '\t' || text[i] == '\r')) ++i;
  };
  auto expect = [&](char c) {
    ws();
    if (i >= text.size() || text[i] != c) throw std::runtime_error(std::string("config JSON: expected ") + c);
    ++i;
  };
  auto number = [&]() -> double {
    ws();
    if (text.compare(i, 4, "true") == 0) {
      i += 4;
      return 1.;
    }
    if (text.compare(i, 5, "false") == 0) {
      i += 5;
      return 0.;
    }
    size_t used = 0;
    const double v = std::stod(text.substr(i, 64), &used);
    i += used;
    return v;
  };
  expect('{');
  ws();
  if (i < text.size() && text[i] == '}') return out;
  while (true) {
    expect('"');
    const size_t end = text.find('"', i);
    if (end == std::string::npos) throw std::runtime_error("config JSON: unterminated key");
    const std::string key = text.substr(i, end - i);
    i = end + 1;
    expect(':');
    ws();
    JsonValue v;
    if (i < text.size() && text[i] == '[') {
      ++i;
      v.is_array = true;
      ws();
      if (text[i] != ']') {
        while (true) {
          v.array.push_back(number());
          ws();
          if (text[i] == ',') {
            ++i;
            continue;
          }
          break;
        }
      }
      expect(']');
    } else {
      v.number = number();
    }
    out[key] = v;
    ws();
    if (i < text.size() && text[i] == ',') {
      ++i;
      continue;
    }
    expect('}');
    break;
  }
  return out;
}

}  // namespace metro
