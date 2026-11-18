#include "metro/detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "geometry.hpp"
#include "json.hpp"
#include "path_fit.hpp"

namespace metro {

// ---------------------------------------------------------------- config
Config::Config() {
  for (double v : arange(20., 100., 5.)) slab_edges.push_back(v);
  for (double v : arange(100., 310., 10.)) slab_edges.push_back(v);
}

void Config::update_from_json(const std::string& json) {
  const auto values = parse_flat_json(json);
  std::map<std::string, double*> d = {
      {"half_width", &half_width}, {"height", &height}, {"min_range", &min_range}, {"max_range", &max_range}, {"long_range", &long_range},
      {"candidate_margin_m", &candidate_margin_m}, {"candidate_margin_sigma", &candidate_margin_sigma}, {"candidate_sigma_floor", &candidate_sigma_floor},
      {"path_min_radius", &path_min_radius}, {"path_vertical_radius", &path_vertical_radius}, {"path_tau", &path_tau}, {"path_spatial_lambda", &path_spatial_lambda}, {"path_tail_m", &path_tail_m}, {"path_far_weight", &path_far_weight}, {"envelope_max_drop", &envelope_max_drop}, {"confirm_range_tol_m", &confirm_range_tol_m}, {"envelope_max_rise", &envelope_max_rise}, {"path_release_k", &path_release_k}, {"hold_expect_points", &hold_expect_points}, {"merge_gap_m", &merge_gap_m}, {"hold_near_m", &hold_near_m}, {"path_extension_decay_m", &path_extension_decay_m},
      {"gauge", &gauge}, {"rail_z_prior", &rail_z_prior}, {"rail_over_bed", &rail_over_bed},
      {"rail_max_x", &rail_max_x}, {"template_decay", &template_decay}, {"min_slab_inliers", &min_slab_inliers},
      {"max_ambiguity", &max_ambiguity}, {"min_explained", &min_explained}, {"slab_edge_frac", &slab_edge_frac}, {"slab_edge_margin_m", &slab_edge_margin_m}, {"slab_edge_memory_m", &slab_edge_memory_m}, {"break_explained", &break_explained}, {"confirm_explained", &confirm_explained}, {"confirm_du", &confirm_du}, {"confirm_free_points", &confirm_free_points}, {"confirm_du_per_m", &confirm_du_per_m}, {"slab_weight_power", &slab_weight_power}, {"slab_amb_sigma_k", &slab_amb_sigma_k}, {"slab_amb_free", &slab_amb_free}, {"slab_weight_floor", &slab_weight_floor}, {"break_bridge_m", &break_bridge_m}, {"extension_break_margin_m", &extension_break_margin_m},
      {"sigma_lat0", &sigma_lat0}, {"sigma_lat_per_m", &sigma_lat_per_m}, {"sigma_lat_per_m_far", &sigma_lat_per_m_far},
      {"sigma_vert0", &sigma_vert0}, {"sigma_vert_per_m", &sigma_vert_per_m}, {"sigma_extrap_per_m", &sigma_extrap_per_m},
      {"margin_k", &margin_k}, {"max_sigma_support", &max_sigma_support}, {"support_buffer", &support_buffer},
      {"extension_min_support", &extension_min_support}, {"extension_max_curvature", &extension_max_curvature},
      {"extension_sigma_per_m", &extension_sigma_per_m}, {"extension_corridor", &extension_corridor},
      {"min_height", &min_height}, {"rail_exclusion", &rail_exclusion}, {"voxel_s", &voxel_s}, {"voxel_uh", &voxel_uh},
      {"max_hold_s", &max_hold_s}, {"hold_min_range", &hold_min_range}, {"hold_min_travel", &hold_min_travel},
      {"still_ds", &still_ds}, {"change_voxel", &change_voxel}, {"change_ref_s", &change_ref_s},
      {"change_min_age_s", &change_min_age_s}, {"change_occupancy", &change_occupancy},
      {"change_new_frac", &change_new_frac}, {"move_evidence_k", &move_evidence_k}, {"move_resid", &move_resid},
      {"move_min_speed", &move_min_speed}, {"move_max_speed", &move_max_speed}, {"move_rel_min", &move_rel_min},
      {"move_hold_s", &move_hold_s}, {"gate_s", &gate_s}, {"static_tol", &static_tol}, {"wall_extent_s", &wall_extent_s},
      {"wall_extent_uh", &wall_extent_uh}, {"wall_clear_center", &wall_clear_center},
      {"wall_end_margin", &wall_end_margin}, {"gate_h", &gate_h}, {"evidence_near", &evidence_near},
      {"evidence_per_m", &evidence_per_m}, {"frame_period_s", &frame_period_s}, {"max_gap_s", &max_gap_s},
      {"coast_sigma_h", &coast_sigma_h}, {"coast_sigma_u", &coast_sigma_u}, {"temporal_weight", &temporal_weight},
      {"fuse_q_abs", &fuse_q_abs}, {"fuse_q_rel", &fuse_q_rel}, {"fuse_q_rel_vert", &fuse_q_rel_vert}, {"fuse_gate", &fuse_gate}, {"fuse_damp_min", &fuse_damp_min}, {"fuse_damp_lo", &fuse_damp_lo}, {"fuse_damp_hi", &fuse_damp_hi}, {"fuse_floor0", &fuse_floor0},
      {"fuse_floor_per_m", &fuse_floor_per_m}, {"fuse_start", &fuse_start}, {"fuse_ramp", &fuse_ramp},
      {"hold_support_tol", &hold_support_tol}, {"structure_memory_s", &structure_memory_s},
      {"sigma_lat_per_m_rail", &sigma_lat_per_m_rail}, {"sigma_vert_per_m_rail", &sigma_vert_per_m_rail},
      {"boundary_max_len", &boundary_max_len}, {"boundary_min_top", &boundary_min_top},
      {"boundary_max_range", &boundary_max_range}, {"speck_range", &speck_range}, {"speck_size", &speck_size},
      {"far_min_range", &far_min_range}, {"far_max_sigma_u", &far_max_sigma_u},
      {"geometry_min_range", &geometry_min_range}, {"geometry_max_step", &geometry_max_step},
      {"fuse_extrap_r", &fuse_extrap_r}, {"fuse_extrap_floor_k", &fuse_extrap_floor_k}, {"tail_window_m", &tail_window_m}, {"tail_curvature_floor", &tail_curvature_floor}, {"tail_curvature_max", &tail_curvature_max}, {"display_tau_s", &display_tau_s}, {"soft_tau_s", &soft_tau_s}, {"soft_reference", &soft_reference},
      {"soft_threshold", &soft_threshold}, {"display_release", &display_release},
      {"display_snap", &display_snap}, {"display_max_gap_s", &display_max_gap_s}, {"display_join_sigma", &display_join_sigma}, {"display_far_tau_s", &display_far_tau_s}, {"display_far_from", &display_far_from}, {"display_far_gain_half_m", &display_far_gain_half_m}, {"display_far_support_tau_s", &display_far_support_tau_s}, {"display_far_ramp", &display_far_ramp}, {"display_far_tau_max", &display_far_tau_max}, {"display_far_release_k", &display_far_release_k}, {"display_far_pass_k", &display_far_pass_k}, {"display_far_pass_m", &display_far_pass_m}, {"display_join_lambda", &display_join_lambda}, {"prior_tol_lat", &prior_tol_lat}, {"prior_tol_vert", &prior_tol_vert}};
  std::map<std::string, int*> i = {
      {"slab_points", &slab_points}, {"strong_points", &strong_points}, {"strong_hits", &strong_hits},
      {"weak_hits", &weak_hits}, {"window", &window}, {"hold_frames", &hold_frames},
      {"change_persist", &change_persist}, {"change_window", &change_window}, {"move_min_obs", &move_min_obs},
      {"move_strong", &move_strong}, {"max_coast_frames", &max_coast_frames}, {"fuse_sigma", &fuse_sigma},
      {"hold_support_frames", &hold_support_frames}, {"attention_strong_frames", &attention_strong_frames}, {"extension_stops_at_break", &extension_stops_at_break}, {"attention_gap_frames", &attention_gap_frames}, {"path_reversal_parts", &path_reversal_parts}, {"hold_frames_strong", &hold_frames_strong}, {"envelope_hold_frames", &envelope_hold_frames}, {"envelope_rise_frames", &envelope_rise_frames}, {"envelope_end_frames", &envelope_end_frames}, {"confirm_range_frames", &confirm_range_frames}, {"hold_empty_frames", &hold_empty_frames}, {"hold_slab_points", &hold_slab_points}, {"soft_min_hits", &soft_min_hits}, {"boundary_points", &boundary_points}, {"slab_scatter_weighted", &slab_scatter_weighted},
      {"boundary_hits", &boundary_hits}, {"speck_points", &speck_points},
      {"far_window", &far_window}, {"far_attention_hits", &far_attention_hits}, {"far_confirm_hits", &far_confirm_hits},
      {"far_min_points", &far_min_points}, {"geometry_frames", &geometry_frames}, {"geometry_points", &geometry_points},
      {"fuse_still_frames", &fuse_still_frames}};
  std::map<std::string, std::vector<double>*> a = {
      {"template_near", &template_near}, {"slab_edges", &slab_edges}, {"chunk_edges", &chunk_edges}};
  for (const auto& [key, value] : values) {
    if (auto it = d.find(key); it != d.end()) *it->second = value.number;
    else if (auto it = i.find(key); it != i.end()) *it->second = static_cast<int>(std::lround(value.number));
    else if (auto it = a.find(key); it != a.end()) *it->second = value.array;
    else if (key == "extension") extension = value.number != 0.;
    else if (key == "fuse") fuse = value.number != 0.;
    else if (key == "fuse_inflate") fuse_inflate = value.number != 0.;
    else if (key == "fuse_extrap_floor") fuse_extrap_floor = value.number != 0.;
    else if (key == "explain") explain = value.number != 0.;
    else if (key == "display_smoothing") display_smoothing = value.number != 0.;
    else if (key == "display_far_smoothing") display_far_smoothing = value.number != 0.;
    else if (key == "display_join") display_join = value.number != 0.;
    else if (key == "soft_confirmation") soft_confirmation = value.number != 0.;
    else if (key == "farfield") farfield = value.number != 0.;
    else if (key == "edge_confirmation") edge_confirmation = value.number != 0.;
    else if (key == "hold_observed_geometry") hold_observed_geometry = value.number != 0.;
    else if (key == "continuous_path") continuous_path = value.number != 0.;
    else if (key == "spatial_path") spatial_path = value.number != 0.;
    else if (key == "path_no_reversal") path_no_reversal = value.number != 0.;
    else if (key == "hold_check") hold_check = value.number != 0.;
    else if (key == "merge_boxes") merge_boxes = value.number != 0.;
    else if (key == "envelope_hold") envelope_hold = value.number != 0.;
    else throw std::runtime_error("Unknown config key: " + key);
  }
}

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Candidate {
  double distance_m = 0., s = 0., s_max = 0., s_med = 0., path_lateral_m = 0.;
  double u_lo = 0., u_hi = 0., h_lo = 0., h_hi = 0.;
  std::array<double, 3> center{}, lo{}, hi{};
  int point_count = 0, confident_points = 0, unique_point_count = 0, unique_conf = 0;
  int nominal_points = 0;
  double soft_z = 0.;
  double appeared_frac = 0., margin_m = 0., sigma_u = 0., sigma_h = 0., evidence_need = 0.;
  int strong_recent = 0;
  int far_points = 0, far_hits = 0, far_total = 0;
  double far_hlo = kInf, far_hhi = -kInf, far_u = 0., far_ulo = kInf, far_uhi = -kInf;
  bool far_attention = false, far_confirmed = false;
  std::string decision_rule;
  bool wall_like = false;
  bool boundary = false;  // a compact object only in the uncertain band at the envelope edge
  int attention_run = 0;  // frames in a row (gaps of a few frames allowed) this object has been under ATTENTION
  bool attention_strong = false;  // ... long enough for the stronger level
  bool edge_confirmed = false;
  bool soft_confirmed = false;  // by the continuous evidence of intrusion over frames
  bool unverified = false;  // beyond this frame's own support (in the held part of the envelope)
  // tracking
  double S = 0., evidence_obs = 0., evidence = 0., S_drift = 0.;
  int track_id = -1, hits = 0;
  bool tracked = false, confirmed = false, appeared = false, moving = false, held = false;
  double speed_mps = 0.;
  int held_frames = 0;
  double h_mean() const { return (h_lo + h_hi) / 2; }
};

struct Track {
  struct FarHit { int frame, points; double S, u, hlo, hhi; };
  std::deque<FarHit> far;
  std::deque<std::pair<int, int>> raw;
  int id = 0;
  double S = 0., u = 0., h = 0.;
  int first = 0, last = 0, hits = 1, strong = 0;
  std::vector<int> frames;
  std::deque<double> Ss, ts;
  std::deque<std::pair<int, double>> conf;
  bool has_confirmed = false;
  int confirmed_at = 0;
  double confirmed_t = 0.;
  double offset = 0.;
  std::optional<Candidate> last_c;
  double S_ego_c = 0.;
  std::vector<double> obs_ego;
  bool appeared = false, moving = false;
  double speed = 0.;
  int n_moving = 0;
  // continuous signed evidence of intrusion (CUSUM-like, with forgetting)
  double soft = 0., soft_t = -1.;
  int soft_hits = 0;
  int empty_hold = 0;  // consecutive held frames with no point where the box should be
  int att_run = 0, att_frame = -1000;  // ATTENTION persistence of this object
};

struct Slabs {
  Vec x, y, z, du, dh, inl, alt, n, wv, rel, env;  // rel: the coarse lateral shift as a share of the search half-width; env: points inside the predicted envelope
  void append(const Slabs& o) {
    Vec* dst[] = {&x, &y, &z, &du, &dh, &inl, &alt, &n, &wv, &rel, &env};
    const Vec* src[] = {&o.x, &o.y, &o.z, &o.du, &o.dh, &o.inl, &o.alt, &o.n, &o.wv, &o.rel, &o.env};
    for (int k = 0; k < 11; ++k) dst[k]->insert(dst[k]->end(), src[k]->begin(), src[k]->end());
  }
};

struct Corridor {
  Cloud p;
  Vec s, u, h;
};

struct Vox {
  int first;
  double first_t;
  int count;
};

}  // namespace

struct Detector::Impl {
  Config cfg;
  NearTrack near;
  Template tmpl;
  WallOdometry odometry;
  bool has_surface = false;
  double surf_floor = 0.;
  std::optional<double> surf_ceil;
  int miss = 0;
  std::optional<PathModel> path;  // the envelope axis shown and used for decisions (held, fused)
  std::optional<PathModel> est;   // this frame's own estimate: the prior of the next frame
  std::vector<Track> tracks;
  int next_id = 0, frame = -1;
  double now = 0., S_ego = 0.;
  std::optional<double> last_time;
  std::optional<double> cur_ds;
  // standstill change detection
  bool still = false;
  int still_f0 = 0;
  double still_t0 = 0.;
  std::unordered_map<int64_t, Vox> vox;
  // fused far axis (Kalman state per grid node)
  bool fused = false;
  Vec f_lat, f_vert, f_Plat, f_Pvert;
  int standing = 0;  // consecutive frames without travel
  int support_drops = 0, extension_drops = 0;
  // the envelope as drawn: small frame-to-frame jitter damped, real changes let through
  Vec show_lat, show_vert;
  double show_phi = 0., show_envelope = 0., show_t = 0.;
  bool show_valid = false;
  double env_drawn = 0., env_t = 0., env_break = 1e9, env_cum = 0.;
  struct EnvMark {
    double end;    // raw envelope end along the path (travel included), so a frame is comparable with the next ones
    bool visible;  // a changed cross-section was seen at the same place twice in a row
  };
  std::deque<EnvMark> env_hist;
  std::deque<double> limit_hist;  // along the track: where the verified range ended in the last frames
  bool env_valid = false;
  struct Mark {
    double S, t;  // world position along the track, time seen
  };
  std::vector<Mark> structures;  // starts of structure in the straight corridor
  double extension_observed = 0., extension_remembered = 0.;
  double frame_measured_support = 0.;
  std::optional<Slabs> debug_slabs;
  double support_recent = 0.;  // farthest range the path was verified to in the last frames (forgotten with travel)
  Vec dbg_raw, dbg_fused, dbg_spatial;  // lateral axis after registration / fusion / joint fit (explain output)
  const Cloud* cloud_now = nullptr;  // the frame's points while its candidates are tracked
  struct GeometryFrame { Cloud p; PathModel near; double S; };
  std::deque<GeometryFrame> geometry_history;
  double geometry_s = 0.;
  int geometry_used = 1;
  struct GeometryCertificate { PathModel path; double S, residual; int frame; };
  std::deque<GeometryCertificate> certificates;
  std::optional<GeometryCertificate> current_certificate;

  explicit Impl(const Config& c) : cfg(c), near(cfg), tmpl(cfg) { reset(); }

  void reset() {
    has_surface = false;
    surf_ceil.reset();
    miss = 0;
    path.reset();
    est.reset();
    tmpl.reset();
    odometry.reset();
    tracks.clear();
    next_id = 0;
    frame = -1;
    now = 0.;
    S_ego = 0.;
    last_time.reset();
    reset_change();
    fused = false;
    standing = 0;
    support_drops = extension_drops = 0;
    structures.clear();
    show_valid = false;
    env_valid = false;
    env_hist.clear();
    limit_hist.clear();
    support_recent = 0.;
    debug_slabs.reset();
    geometry_history.clear();
    geometry_s = 0.;
    certificates.clear();
    current_certificate.reset();
  }
  void reset_change() {
    still = false;
    vox.clear();
  }

  // ------------------------------------------------------------------ priors
  double rail_height_guess(const Cloud& p) const {
    // the bed is the largest horizontal surface 4-15 m ahead within the track width
    Vec z;
    for (const P3& q : p)
      if (q.x > 4 && q.x < 15 && std::abs(q.y) < 1.5 && q.z < -.3 && q.z > -4.) z.push_back(q.z);
    if (z.size() < 200) return cfg.rail_z_prior;
    const Vec edges = arange(-4., -.29, .05);
    std::vector<long> hist(edges.size() - 1, 0);
    for (double v : z) {
      if (v < edges.front() || v > edges.back()) continue;
      long k = static_cast<long>(std::upper_bound(edges.begin(), edges.end(), v) - edges.begin()) - 1;
      if (k == static_cast<long>(hist.size())) k -= 1;  // the last bin includes its right edge
      ++hist[k];
    }
    const long k = std::max_element(hist.begin(), hist.end()) - hist.begin();
    return (edges[k] + edges[k + 1]) / 2 + cfg.rail_over_bed;
  }

  PathModel prior_for(std::optional<double> ds, const Cloud& p) const {
    if (!est) return PathModel::flat(rail_height_guess(p));
    return est->shifted(ds.value_or(0.),cfg.continuous_path);
  }

  // ------------------------------------------------------------------ registration
  void slab_points(const Cloud& p, Cloud& q, std::vector<int>& slab) const {
    const Vec& E = cfg.slab_edges;
    const int nsl = static_cast<int>(E.size()) - 1;
    std::vector<size_t> idx;
    std::vector<int> sl;
    std::vector<long> counts(nsl, 0);
    for (size_t k = 0; k < p.size(); ++k) {
      const double x = p[k].x;
      if (x >= E.front() && x < E.back()) {
        const int j = static_cast<int>(std::upper_bound(E.begin(), E.end(), x) - E.begin()) - 1;
        idx.push_back(k);
        sl.push_back(j);
        ++counts[j];
      }
    }
    q.clear();
    slab.clear();
    for (size_t t = 0; t < idx.size(); ++t) {
      const double keep = std::min(1., 2. * cfg.slab_points / std::max(counts[sl[t]], 1L));
      const uint64_t h = (static_cast<uint64_t>(idx[t]) * 2654435761ULL) % 1000ULL;
      if (static_cast<double>(h) < keep * 1000.) {
        q.push_back(p[idx[t]]);
        slab.push_back(sl[t]);
      }
    }
  }

  void vertical_from_surfaces(const std::vector<int>& slab, const Vec& u, const Vec& h, int nsl, Vec& dh,
                              Vec& w) const {
    // A real grade change moves both the floor and the crown; a low object only the floor,
    // a hanging one only the crown. Disagreement lowers the weight.
    dh.assign(nsl, 0.);
    w.assign(nsl, 0.);
    if (!has_surface) return;
    std::vector<Vec> fl(nsl), ce(nsl);
    std::vector<char> present(nsl, 0);
    for (size_t k = 0; k < slab.size(); ++k) {
      present[slab[k]] = 1;
      // A few distant obstacle returns must not become the estimated floor or crown.
      const double section_x = .5 * (cfg.slab_edges[slab[k]] + cfg.slab_edges[slab[k] + 1]);
      if (section_x >= cfg.geometry_min_range && std::abs(u[k]) < cfg.half_width + .1 &&
          h[k] > cfg.min_height && h[k] < cfg.height + .1) continue;
      if (std::abs(u[k]) < 1.3 && h[k] > -1.3 && h[k] < 1.6) fl[slab[k]].push_back(h[k]);
      if (surf_ceil && std::abs(u[k]) < 1.0 && h[k] > *surf_ceil - 1.4 && h[k] < *surf_ceil + 1.6)
        ce[slab[k]].push_back(h[k]);
    }
    for (int j = 0; j < nsl; ++j) {
      if (!present[j]) continue;
      std::vector<std::pair<double, double>> ests;
      if (fl[j].size() >= 6) ests.push_back({quantile(fl[j], .3) - surf_floor, std::min<double>(fl[j].size(), 40) / 40.});
      if (ce[j].size() >= 4 && surf_ceil)
        ests.push_back({quantile(ce[j], .9) - *surf_ceil, std::min<double>(ce[j].size(), 30) / 30.});
      if (ests.empty()) continue;
      if (ests.size() == 2) {
        const double d0 = ests[0].first, d1 = ests[1].first;
        if (std::abs(d0 - d1) < .3) {
          dh[j] = (d0 * ests[0].second + d1 * ests[1].second) / (ests[0].second + ests[1].second);
          w[j] = 1.;
        } else {
          dh[j] = std::abs(d0) < std::abs(d1) ? d0 : d1;
          w[j] = .2;
        }
      } else {
        dh[j] = ests[0].first;
        w[j] = .5 * ests[0].second;
      }
    }
  }

  std::optional<Slabs> register_slabs(const PathModel& base, double width_scale, double xlo, double xhi,
                                      const Cloud& pre_q, const std::vector<int>& pre_slab) const {
    const Vec& E = cfg.slab_edges;
    if (!tmpl.ready()) return std::nullopt;
    const int nsl = static_cast<int>(E.size()) - 1;
    Vec xc(nsl);
    for (int j = 0; j < nsl; ++j) xc[j] = (E[j] + E[j + 1]) / 2;
    std::vector<int> slab_all;
    Vec u_all, h_all;
    for (size_t k = 0; k < pre_q.size(); ++k) {
      const double c = xc[pre_slab[k]];
      if (!(c >= xlo && c < xhi)) continue;
      double s, u, h;
      base.to_track(pre_q[k], s, u, h);
      slab_all.push_back(pre_slab[k]);
      u_all.push_back(u);
      h_all.push_back(h);
    }
    if (slab_all.empty()) return std::nullopt;
    // points in the predicted envelope are left out of the lateral fit: potential obstacles
    std::vector<int> slab;
    Vec u, h;
    Vec corridor(nsl, 0.);
    for (size_t k = 0; k < slab_all.size(); ++k) {
      const double uu = u_all[k], hh = h_all[k];
      if (std::abs(uu) < cfg.half_width + .35 && hh > .05 && hh < cfg.height + .3) {
        corridor[slab_all[k]] += 1.;
        continue;
      }
      if (!(std::abs(uu) < 9.8 && hh > -1.45 && hh < 6.45)) continue;
      slab.push_back(slab_all[k]);
      u.push_back(uu);
      h.push_back(hh);
    }
    if (slab.empty()) return std::nullopt;
    Vec W(nsl), Wh(nsl);
    for (int j = 0; j < nsl; ++j) {
      W[j] = std::clamp((.5 + .012 * xc[j]) * width_scale, .4, 3.5);
      Wh[j] = std::clamp(.15 + .004 * xc[j], .15, 1.2);
    }
    const Vec grid = linspace(-1, 1, 41), hgrid = linspace(-1, 1, 21), fine = linspace(-.12, .12, 13);
    const int K = static_cast<int>(grid.size()), KH = static_cast<int>(hgrid.size()), KF = static_cast<int>(fine.size());
    std::vector<double> tot(static_cast<size_t>(nsl) * K, 0.);
    for (size_t k = 0; k < slab.size(); ++k) {
      const int j = slab[k];
      for (int g = 0; g < K; ++g) tot[static_cast<size_t>(j) * K + g] += tmpl.score(u[k] - grid[g] * W[j], h[k]);
    }
    Vec du(nsl), dh_t(nsl), rel(nsl);
    for (int j = 0; j < nsl; ++j) {
      int best = 0;
      for (int g = 1; g < K; ++g)
        if (tot[static_cast<size_t>(j) * K + g] > tot[static_cast<size_t>(j) * K + best]) best = g;
      du[j] = grid[best] * W[j];
      rel[j] = std::abs(grid[best]);
    }
    // vertical search with the lateral shift applied
    std::vector<double> toth(static_cast<size_t>(nsl) * KH, 0.);
    for (size_t k = 0; k < slab.size(); ++k) {
      const int j = slab[k];
      const double uu = u[k] - du[j];
      for (int g = 0; g < KH; ++g) toth[static_cast<size_t>(j) * KH + g] += tmpl.score(uu, h[k] - hgrid[g] * Wh[j]);
    }
    for (int j = 0; j < nsl; ++j) {
      int best = 0;
      for (int g = 1; g < KH; ++g)
        if (toth[static_cast<size_t>(j) * KH + g] > toth[static_cast<size_t>(j) * KH + best]) best = g;
      dh_t[j] = hgrid[best] * Wh[j];
    }
    // refine the lateral shift on a fine grid
    std::vector<double> tf(static_cast<size_t>(nsl) * KF, 0.);
    for (size_t k = 0; k < slab.size(); ++k) {
      const int j = slab[k];
      const double hh = h[k] - dh_t[j];
      for (int g = 0; g < KF; ++g) tf[static_cast<size_t>(j) * KF + g] += tmpl.score(u[k] - (du[j] + fine[g]), hh);
    }
    Vec peak(nsl);
    for (int j = 0; j < nsl; ++j) {
      int best = 0;
      for (int g = 1; g < KF; ++g)
        if (tf[static_cast<size_t>(j) * KF + g] > tf[static_cast<size_t>(j) * KF + best]) best = g;
      du[j] += fine[best];
      peak[j] = tf[static_cast<size_t>(j) * KF + best];
    }
    // ambiguity: best lateral score away from the peak
    Vec alt(nsl, 0.), n(nsl, 0.);
    for (int j = 0; j < nsl; ++j)
      for (int g = 0; g < K; ++g)
        if (std::abs(grid[g] * W[j] - du[j]) > .25) alt[j] = std::max(alt[j], tot[static_cast<size_t>(j) * K + g]);
    for (int j : slab) n[j] += 1;
    Vec ush(slab_all.size());
    for (size_t k = 0; k < slab_all.size(); ++k) ush[k] = u_all[k] - du[slab_all[k]];
    Vec dhv, wv;
    vertical_from_surfaces(slab_all, ush, h_all, nsl, dhv, wv);
    Slabs out;
    for (int j = 0; j < nsl; ++j) {
      if (!(xc[j] >= xlo && xc[j] < xhi)) continue;
      const bool use_t = wv[j] <= 0;
      const double dhj = use_t ? dh_t[j] : dhv[j];
      const double d = base.dc(xc[j]);
      const double cs = 1. / std::sqrt(1. + d * d);
      out.x.push_back(xc[j]);
      out.y.push_back(base.c(xc[j]) + du[j] / cs);
      out.z.push_back(base.r(xc[j]) + dhj);
      out.du.push_back(du[j]);
      out.dh.push_back(dhj);
      out.inl.push_back(peak[j]);
      out.alt.push_back(alt[j]);
      out.n.push_back(n[j]);
      out.wv.push_back(use_t ? .3 : wv[j]);
      out.rel.push_back(rel[j]);
      out.env.push_back(corridor[j]);
    }
    return out;
  }

  PathModel smooth_path(const PathModel& prior, const std::optional<NearFit>& near_fit, const Slabs* slabs,
                        bool first, bool quick = false) const {
    const int N = PathModel::N;
    double phi = prior.phi, support = 0., rail_support = 0.;
    Vec ox, oy, ow, vx, vy, vw;
    if (near_fit) {
      const auto& d = near_fit->dev;
      const auto& r = near_fit->rail;
      for (double x : arange(3., std::min(near_fit->support, 40.) + .1, 1.)) {
        ox.push_back(x), oy.push_back(prior.c(x) + d[0] + d[1] * x + d[2] * x * x), ow.push_back(400.);
        vx.push_back(x), vy.push_back(prior.r(x) + r[0] + r[1] * x), vw.push_back(400.);
      }
      phi = r[2];
      support = rail_support = near_fit->support;
    }
    std::vector<char> good;
    Vec amb, wfit;
    double break_m = 1e9;
    if (slabs) {
      const size_t m = slabs->x.size();
      good.assign(m, 0);
      std::vector<char> confirmed(m, 0);
      Vec w(m);
      amb.assign(m, 0.);
      // a slab counts only if the normal-tunnel template explains most of its points
      for (size_t k = 0; k < m; ++k) {
        const double q = slabs->inl[k];
        const double ratio = slabs->alt[k] / std::max(q, 1e-6);
        amb[k] = ratio;
        const double frac = q / std::max(slabs->n[k], 1.);
        good[k] = q >= cfg.min_slab_inliers && ratio < cfg.max_ambiguity && frac >= cfg.min_explained;
        w[k] = good[k] ? std::min(q, 60.) * std::pow(std::max(1 - ratio, cfg.slab_weight_floor), cfg.slab_weight_power) * frac : 0.;
        // Only partly explained, but on the predicted axis and with nothing in the envelope: the same tunnel seen with a platform
        // or a niche beside it. It supports the path where it is, without steering it (its shift from the axis is about zero).
        confirmed[k] = !first && cfg.confirm_explained > 0. && !good[k] && q >= cfg.min_slab_inliers && ratio < cfg.max_ambiguity &&
                       frac >= cfg.confirm_explained && slabs->rel[k] < cfg.slab_edge_frac &&
                       std::abs(slabs->du[k]) <= cfg.confirm_du + cfg.confirm_du_per_m * slabs->x[k] && slabs->env[k] <= cfg.confirm_free_points;
        if (confirmed[k]) good[k] = 1, w[k] = std::min(q, 60.) * std::pow(std::max(1 - ratio, cfg.slab_weight_floor), cfg.slab_weight_power) * frac;
      }
      // A best match on the edge of the search window is no match: the tunnel lies farther out, or nowhere. A sharp bend is found
      // that way step by step, but only beyond the range the path has recently been verified to: inside it the path already
      // follows the tunnel, and a slab that suddenly sits on the edge there is something else (a platform wall, a structure by
      // the track) and does not steer the axis.
      if (!first && cfg.slab_edge_frac <= 1.) {
        for (size_t k = 0; k < m; ++k)
          if (good[k] && slabs->rel[k] >= cfg.slab_edge_frac && slabs->x[k] <= support_recent + cfg.slab_edge_margin_m) good[k] = 0, w[k] = 0.;
      }
      // a slab the template clearly cannot explain marks a changed tunnel: nothing beyond it counts
      // unless the axis is back under the next slabs: a platform entrance or a niche, not another tunnel
      const auto on_axis = [&](size_t k) {
        return (good[k] || confirmed[k]) && slabs->rel[k] < cfg.slab_edge_frac &&
               std::abs(slabs->du[k]) <= cfg.confirm_du + cfg.confirm_du_per_m * slabs->x[k] && slabs->env[k] <= cfg.confirm_free_points;
      };
      double cut = kInf;
      for (size_t k = 0; k < m && !std::isfinite(cut); ++k) {
        if (!(slabs->n[k] >= 20 && !confirmed[k] && slabs->inl[k] / std::max(slabs->n[k], 1.) < cfg.break_explained)) continue;
        bool bridged = false;
        for (size_t k2 = k + 1; k2 < m && slabs->x[k2] - slabs->x[k] <= cfg.break_bridge_m && !bridged; ++k2) bridged = on_axis(k2);
        if (!bridged) cut = slabs->x[k];
      }
      if (std::isfinite(cut))
        for (size_t k = 0; k < m; ++k)
          if (!(slabs->x[k] < cut)) good[k] = 0, w[k] = 0.;
      break_m = cut;
      for (size_t k = 0; k < m; ++k) {
        ox.push_back(slabs->x[k]), oy.push_back(slabs->y[k]), ow.push_back(w[k]);
        vx.push_back(slabs->x[k]), vy.push_back(slabs->z[k]);
        vw.push_back((good[k] || slabs->wv[k] > .4) ? 20. * slabs->wv[k] : 0.);
      }
      wfit = w;
      double last = support;
      for (size_t k = 0; k < m; ++k) {
        if (!good[k]) continue;
        if (slabs->x[k] - last > 25.) break;
        last = slabs->x[k];
      }
      support = std::max(support, last);
    }
    // robust weights from a data-only fit: a wrong previous path must not turn
    // consistent new observations into outliers
    if (!quick) {
      ow = robust_weights(N, ox, oy, ow, 4e4, .25);
      vw = robust_weights(N, vx, vy, vw, 4e6, .2);
    }
    // temporal prior: the previous path moved by odometry, trusted only inside the
    // range it was itself supported by data. In a transition curve the track ahead looks different
    // after every metre run, so where this frame's own data disagree with the moved path, the
    // path gives way; otherwise it pulled the fit off the slabs and cut the support every other frame.
    Vec lat_d, vert_d;
    const bool compare = !first && slabs && cfg.prior_tol_lat > 0;
    if (compare) lat_d = penalized_fit(N, ox, oy, ow, 4e4), vert_d = penalized_fit(N, vx, vy, vw, 4e6);
    for (int g = 0; g < N; ++g) {
      const double tw = first ? 0. : cfg.temporal_weight * std::clamp(1. - (g - prior.support) / 20., 0., 1.);
      double twl = tw, twv = tw;
      if (compare && g <= support) {
        const double dl = (prior.lat[g] - lat_d[g]) / cfg.prior_tol_lat;
        const double dv = (prior.vert[g] - vert_d[g]) / cfg.prior_tol_vert;
        twl /= 1. + dl * dl;
        twv /= 1. + dv * dv;
      }
      ox.push_back(g), oy.push_back(prior.lat[g]), ow.push_back(twl);
      vx.push_back(g), vy.push_back(prior.vert[g]), vw.push_back(twv);
    }
    Vec lat = penalized_fit(N, ox, oy, ow, 4e4);
    {
      const Vec curv = gradient(gradient(lat));
      double mx = 0.;
      for (int g = 0; g <= std::min(N - 1, static_cast<int>(support)); ++g) mx = std::max(mx, std::abs(curv[g]));
      if (mx > 1 / 250.) lat = penalized_fit(N, ox, oy, ow, 4e5);
    }
    Vec vert = penalized_fit(N, vx, vy, vw, 4e6);
    Vec su(N), sh(N);
    // where the rails are seen the centre line and the rail plane are measured, beyond they are
    // carried by the walls
    const double R = std::min(rail_support, 100.);
    for (int g = 0; g < N; ++g) {
      const double near = std::min<double>(g, R), mid = std::clamp(g - R, 0., 100. - R);
      su[g] = cfg.sigma_lat0 + cfg.sigma_lat_per_m_rail * near + cfg.sigma_lat_per_m * mid +
              cfg.sigma_lat_per_m_far * std::max(0., g - 100.);
      sh[g] = cfg.sigma_vert0 + cfg.sigma_vert_per_m_rail * near + cfg.sigma_vert_per_m * std::max(0., g - R);
    }
    int n_good = 0;
    for (char gd : good) n_good += gd;
    if (slabs && n_good > 0) {
      // empirical consistency of the registered slabs around the smooth path
      Vec xs, ru, rz, wq;
      for (size_t k = 0; k < good.size(); ++k) {
        if (!good[k]) continue;
        xs.push_back(slabs->x[k]);
        ru.push_back(slabs->y[k] - interp_grid(slabs->x[k], lat));
        rz.push_back(slabs->z[k] - interp_grid(slabs->x[k], vert));
        wq.push_back(cfg.slab_scatter_weighted ? std::max(wfit[k], 1e-3) : std::min(slabs->inl[k], 60.));
      }
      const Vec eu = local_rms(N, xs, ru, wq, 12.), ez = local_rms(N, xs, rz, wq, 12.);
      for (int g = 0; g < N; ++g) {
        su[g] = std::sqrt(su[g] * su[g] + eu[g] * eu[g]);
        sh[g] = std::sqrt(sh[g] * sh[g] + ez[g] * ez[g]);
      }
      // where the profile fits about as well at other shifts (a platform, a niche) the shifts of the slabs scatter and lean to one
      // side: the axis there is less certain than the scatter around the fit says
      if (cfg.slab_amb_sigma_k > 0.) {
        Vec ex;
        for (size_t k = 0; k < good.size(); ++k)
          if (good[k]) ex.push_back(std::max(0., amb[k] - cfg.slab_amb_free));
        const Vec ea = local_rms(N, xs, ex, wq, 12.);
        for (int g = 0; g < N; ++g) {
          su[g] = std::hypot(su[g], cfg.slab_amb_sigma_k * ea[g]);
          sh[g] = std::hypot(sh[g], .25 * cfg.slab_amb_sigma_k * ea[g]);
        }
      }
      // support ends where the lateral geometry becomes inconsistent
      for (int g = 0; g < N; ++g)
        if (g > rail_support && eu[g] > cfg.max_sigma_support) {
          support = std::min(support, double(g));
          break;
        }
    }
    // straight continuation only when the last 60 m of supported path are straight
    const int k0 = static_cast<int>(std::max(0., support - 60.));
    const int k1 = static_cast<int>(std::min(double(N - 1), support));
    bool straight = false;
    if (cfg.extension && support >= cfg.extension_min_support && k1 - k0 >= 30) {
      const Vec part(lat.begin() + k0, lat.begin() + k1 + 1);
      const Vec curv = gradient(gradient(part));
      double mx = 0.;
      for (double c : curv) mx = std::max(mx, std::abs(c));
      straight = mx < cfg.extension_max_curvature;
    }
    for (int g = 0; g < N; ++g) {
      const double beyond = std::max(0., g - support);
      if (straight) {
        su[g] += cfg.extension_sigma_per_m * beyond;
        sh[g] += cfg.extension_sigma_per_m * beyond;
      } else {
        su[g] += cfg.sigma_extrap_per_m * beyond + beyond * beyond / (2 * 800.);
        sh[g] += .5 * cfg.sigma_extrap_per_m * beyond;
      }
    }
    PathInfo info;
    info.support_m = support;
    info.rail_support_m = rail_support;
    info.extension_ok = straight;
    info.good_slabs = n_good;
    info.phi = phi;
    info.break_m = break_m;
    return PathModel(lat, vert, phi, support, su, sh, info);
  }

  void update_surface_ref(const Vec& s, const Vec& u, const Vec& h) {
    Vec fl, ce;
    for (size_t k = 0; k < s.size(); ++k) {
      if (s[k] > 10 && s[k] < 32 && std::abs(u[k]) < 1.3 && h[k] > -1.3 && h[k] < 1.6) fl.push_back(h[k]);
      if (s[k] > 12 && s[k] < 40 && std::abs(u[k]) < 1.0 && h[k] > 2.3 && h[k] < 7.) ce.push_back(h[k]);
    }
    if (fl.size() < 50) return;
    const double floor = quantile(fl, .3);
    std::optional<double> ceil;
    if (ce.size() >= 20) ceil = quantile(ce, .9);
    if (!has_surface) {
      has_surface = true;
      surf_floor = floor;
      surf_ceil = ceil;
    } else {
      surf_floor = .8 * surf_floor + .2 * floor;
      if (ceil) surf_ceil = surf_ceil ? .8 * *surf_ceil + .2 * *ceil : *ceil;
    }
  }

  // ------------------------------------------------------------------ envelope
  int64_t voxel_key(const P3& q) const {
    const double v = cfg.change_voxel;
    const int64_t ix = static_cast<int64_t>(std::floor(q.x / v));
    const int64_t iy = static_cast<int64_t>(std::floor(q.y / v)) + 512;
    const int64_t iz = static_cast<int64_t>(std::floor(q.z / v)) + 512;
    return (ix << 22) | (iy << 11) | iz;
  }

  void update_change(const Cloud& p) {
    // occupancy history of the scene in sensor voxels while the train stands
    if (!cur_ds || std::abs(*cur_ds) >= cfg.still_ds) {
      if (still) reset_change();
      return;
    }
    if (!still) {
      still = true;
      still_f0 = frame;
      still_t0 = now;
    }
    std::unordered_map<int64_t, char> seen;
    seen.reserve(p.size() / 4 + 16);
    for (const P3& q : p) {
      if (!(q.x < 150. && std::abs(q.y) < 12. && q.z > -8. - .12 * q.x && q.z < 8. + .12 * q.x)) continue;
      seen.emplace(voxel_key(q), 1);
    }
    for (const auto& kv : seen) {
      auto it = vox.find(kv.first);
      if (it != vox.end()) ++it->second.count;
      else vox.emplace(kv.first, Vox{frame, now, 1});
    }
  }

  double appeared_frac(const Cloud& q) const {
    // share of points in voxels that were empty early in the stop and are now steadily occupied
    const double eps = .5 * cfg.frame_period_s;
    if (!still || vox.empty() || now - still_t0 < cfg.change_ref_s + cfg.change_min_age_s - eps) return 0.;
    if (q.empty()) return 0.;
    long n_new = 0;
    for (const P3& p : q) {
      auto it = vox.find(voxel_key(p));
      if (it == vox.end()) continue;
      const Vox& v = it->second;
      const int age = frame - v.first + 1;
      if (v.first_t >= still_t0 + cfg.change_ref_s - eps && now - v.first_t >= cfg.change_min_age_s - eps && age >= 2 &&
          v.count >= cfg.change_occupancy * age)
        ++n_new;
    }
    return static_cast<double>(n_new) / q.size();
  }

  double extension_limit(const Cloud& p, const PathModel& pm) const {
    // how far the straight continuation is observably free of tunnel structure
    const double s0 = pm.arc_at_x(pm.support);
    struct Hit { double s, u, h; };
    std::vector<Hit> hits;
    for (const P3& q : p) {
      if (!(q.x > pm.support - 2.)) continue;
      double s, u, h;
      pm.to_track(q, s, u, h);
      if (std::abs(u) < cfg.extension_corridor && h > -.3 && h < cfg.height + .8 && s > s0) hits.push_back({s, u, h});
    }
    if (hits.empty()) return cfg.max_range;
    std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) { return a.s < b.s; });
    Vec S, U, H;
    for (const auto& q : hits) S.push_back(q.s), U.push_back(q.u), H.push_back(q.h);
    // Opposite sparse wall returns and a compact central object are not one structure.
    // Require the same longitudinal, lateral and height connectivity as physical candidates.
    // A continuous transverse wall still has a wide connected component.
    const auto labels = cluster_labels(S, U, H);
    struct Extent { double s0 = kInf, s1 = -kInf, u0 = kInf, u1 = -kInf; };
    std::vector<Extent> components(*std::max_element(labels.begin(), labels.end()) + 1);
    for (size_t k = 0; k < hits.size(); ++k) {
      auto& e = components[labels[k]];
      e.s0 = std::min(e.s0, S[k]), e.s1 = std::max(e.s1, S[k]);
      e.u0 = std::min(e.u0, U[k]), e.u1 = std::max(e.u1, U[k]);
    }
    double limit = cfg.max_range;
    for (const auto& e : components)
      if (e.s1 - e.s0 > 6. || e.u1 - e.u0 > 2.6) limit = std::min(limit, std::max(s0, e.s0 - 2.));
    return limit;
  }

  Corridor corridor(const Cloud& p, PathModel& pm) {
    Cloud q;
    for (const P3& v : p) {
      const long xi = std::clamp(static_cast<long>(v.x), 0L, static_cast<long>(PathModel::N - 1));
      if (std::abs(v.y - pm.lat[xi]) < cfg.half_width + 1.2 + std::abs(pm.dlat[xi]) && v.x > cfg.min_range - .5)
        q.push_back(v);
    }
    // standstill occupancy history over the whole corridor, independent of the verified range
    update_change(q);
    double limit = std::min(cfg.max_range, interp_grid(std::max(0., pm.support - cfg.support_buffer), pm.arc));
    const double measured_support = frame_measured_support;
    const double measured_limit = std::min(cfg.max_range, pm.arc_at_x(std::max(0., measured_support - cfg.support_buffer)));
    extension_observed = extension_remembered = cfg.max_range;
    if (pm.info.extension_ok) {
      // Structure in the straight corridor is seen intermittently at long range (a few points
      // more or less decide it). Where it was seen during the last moments it is kept, moved with
      // the train, so the envelope end does not jump between it and the full range.
      PathModel observed = pm;
      observed.support = measured_support;
      double ext = extension_limit(p, observed);
      extension_observed = ext;
      if (ext < cfg.max_range) structures.push_back({S_ego + ext, now});
      structures.erase(std::remove_if(structures.begin(), structures.end(),
                                      [&](const Mark& m) { return now - m.t > cfg.structure_memory_s || m.S < S_ego; }),
                       structures.end());
      for (const Mark& m : structures) ext = std::min(ext, std::max(pm.arc_at_x(measured_support), m.S - S_ego));
      extension_remembered = ext;
      // A tunnel the template cannot explain ahead ends the straight continuation too: nothing beyond it is known.
      if (cfg.extension_stops_at_break && pm.info.break_m < 1e8) ext = std::min(ext, std::max(pm.arc_at_x(measured_support), pm.arc_at_x(std::max(0., pm.info.break_m - cfg.extension_break_margin_m))));
      // A current straight continuation is verified against current structures. If only the
      // previous length survives, retain its drawn end but do not turn it into verified space.
      limit = std::max(pm.info.extension_raw ? measured_limit : limit, ext);
    }
    pm.info.analysis_limit_m = limit;
    pm.info.has_analysis_limit = true;
    // A held length only keeps the drawn envelope steady. Beyond this frame's own support nothing
    // is verified: the tunnel may change right there (a wall across the straight continuation).
    double decide = limit;
    if (pm.info.support_held && !pm.info.extension_raw)
      decide = std::min(limit, interp_grid(std::max(0., pm.info.support_raw_m - cfg.support_buffer), pm.arc));
    else if (pm.info.extension_ok && !pm.info.extension_raw)
      decide = std::min(limit, interp_grid(std::max(0., pm.support - cfg.support_buffer), pm.arc));
    pm.info.decision_limit_m = decide;
    Corridor c;
    for (const P3& v : q) {
      double s, u, h;
      pm.to_track(v, s, u, h);
      if (std::abs(u) < cfg.half_width + .4 && h > -.05 && h < cfg.height + .4 && s > cfg.min_range && s < limit) {
        c.p.push_back(v);
        c.s.push_back(s);
        c.u.push_back(u);
        c.h.push_back(h);
      }
    }
    return c;
  }

  static std::vector<long> cluster_labels(const Vec& S, const Vec& U, const Vec& H) {
    // gap-based grouping: along-track gaps first, then lateral and vertical; gaps grow with range
    const size_t n = S.size();
    std::vector<long> labels(n, 0);
    long next = 0;
    size_t start = 0;
    for (size_t k = 1; k <= n; ++k) {
      if (k < n && !(S[k] - S[k - 1] > std::max(1.0, .012 * S[k]))) continue;
      std::vector<size_t> grp;
      double mean = 0.;
      for (size_t t = start; t < k; ++t) grp.push_back(t), mean += S[t];
      mean /= grp.size();
      const double gu = std::max(.45, .006 * mean);
      std::stable_sort(grp.begin(), grp.end(), [&](size_t a, size_t b) { return U[a] < U[b]; });
      size_t s1 = 0;
      for (size_t t = 1; t <= grp.size(); ++t) {
        if (t < grp.size() && !(U[grp[t]] - U[grp[t - 1]] > gu)) continue;
        std::vector<size_t> sub(grp.begin() + s1, grp.begin() + t);
        std::stable_sort(sub.begin(), sub.end(), [&](size_t a, size_t b) { return H[a] < H[b]; });
        size_t s2 = 0;
        for (size_t r = 1; r <= sub.size(); ++r) {
          if (r < sub.size() && !(H[sub[r]] - H[sub[r - 1]] > gu)) continue;
          for (size_t z = s2; z < r; ++z) labels[sub[z]] = next;
          ++next;
          s2 = r;
        }
        s1 = t;
      }
      start = k;
    }
    return labels;
  }

  std::vector<Candidate> clusters(const Corridor& cor, const PathModel& pm) const {
    std::vector<Candidate> out;
    const size_t n = cor.s.size();
    if (n == 0) return out;
    const double k = cfg.margin_k;
    std::vector<size_t> ins;
    std::vector<char> conf;
    for (size_t i = 0; i < n; ++i) {
      long gi = std::lower_bound(pm.arc.begin(), pm.arc.end(), cor.s[i]) - pm.arc.begin();
      gi = std::clamp(gi, 0L, static_cast<long>(pm.arc.size()) - 1);
      const double su = pm.sigma_u[gi], sh = pm.sigma_h[gi];
      const double u = cor.u[i], h = cor.h[i];
      const double d_lat = cfg.half_width - std::abs(u), d_top = cfg.height - h, d_bot = h;
      const bool rail = std::abs(std::abs(u) - cfg.gauge / 2) < cfg.rail_exclusion + std::min(.03, su) && h < .12 + sh;
      const bool inside = d_lat > 0 && d_top > 0 && d_bot > cfg.min_height && !rail;
      const double outer_u = std::min(cfg.candidate_margin_m, cfg.candidate_margin_sigma * std::max(su, cfg.candidate_sigma_floor));
      const double outer_h = std::min(cfg.candidate_margin_m, cfg.candidate_margin_sigma * std::max(sh, cfg.candidate_sigma_floor));
      const bool observable = !rail && d_lat > -outer_u && d_top > -outer_h && d_bot > cfg.min_height;
      if (!inside && !(cfg.candidate_margin_m > 0. && observable)) continue;
      ins.push_back(i);
      conf.push_back(cor.s[i] <= pm.info.decision_limit_m && d_lat > k * su && d_top > k * sh &&
                     d_bot > std::max(cfg.min_height, k * sh) && !rail);
    }
    if (ins.empty()) return out;
    std::vector<size_t> order(ins.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return cor.s[ins[a]] < cor.s[ins[b]]; });
    Vec S(order.size()), U(order.size()), H(order.size());
    std::vector<char> C(order.size());
    Cloud Q(order.size());
    for (size_t t = 0; t < order.size(); ++t) {
      const size_t i = ins[order[t]];
      S[t] = cor.s[i], U[t] = cor.u[i], H[t] = cor.h[i], C[t] = conf[order[t]], Q[t] = cor.p[i];
    }
    const std::vector<long> labels = cluster_labels(S, U, H);
    const double support_arc = interp_grid(pm.info.support_held ? pm.info.support_raw_m : pm.support, pm.arc);
    const long nlab = labels.empty() ? 0 : *std::max_element(labels.begin(), labels.end()) + 1;
    std::vector<std::vector<size_t>> members(nlab);
    for (size_t t = 0; t < labels.size(); ++t) members[labels[t]].push_back(t);
    for (const auto& idx : members) {
      if (idx.empty()) continue;
      Candidate c;
      Vec ss, uu, hh, xs, ys, zs;
      double smin_conf = kInf, smin = kInf, smax = -kInf;
      int nc = 0;
      c.u_lo = c.h_lo = kInf;
      c.u_hi = c.h_hi = -kInf;
      c.lo = {kInf, kInf, kInf};
      c.hi = {-kInf, -kInf, -kInf};
      Cloud members_q;
      double margin = -kInf, umin_abs = kInf;
      std::vector<std::array<long, 3>> keys_all, keys_conf, keys_far;
      std::map<std::array<long, 3>, double> soft_spots;
      double far_sum_u = 0.;
      int far_returns = 0;
      auto key = [](const P3& q) {
        return std::array<long, 3>{std::lround(q.x * 100.f), std::lround(q.y * 100.f), std::lround(q.z * 100.f)};
      };
      for (size_t t : idx) {
        keys_all.push_back(key(Q[t]));
        if (cfg.candidate_margin_m > 0.) {
          long gi = std::lower_bound(pm.arc.begin(), pm.arc.end(), S[t]) - pm.arc.begin();
          gi = std::clamp(gi, 0L, static_cast<long>(pm.arc.size()) - 1);
          const double zu = (cfg.half_width - std::abs(U[t])) / std::max(pm.sigma_u[gi], cfg.candidate_sigma_floor);
          const double zt = (cfg.height - H[t]) / std::max(pm.sigma_h[gi], cfg.candidate_sigma_floor);
          const double zb = (H[t] - cfg.min_height) / std::max(pm.sigma_h[gi], cfg.candidate_sigma_floor);
          soft_spots[key(Q[t])] = std::min({zu, zt, zb});
        }
        if (std::abs(U[t]) < cfg.half_width && H[t] > cfg.min_height && H[t] < cfg.height) ++c.nominal_points;
        if (C[t]) keys_conf.push_back(key(Q[t]));
        if (S[t] >= cfg.far_min_range && S[t] <= pm.info.decision_limit_m &&
            std::abs(U[t]) < cfg.half_width - .05 && H[t] > .3 && H[t] < cfg.height - .4) {
          keys_far.push_back(key(Q[t]));
          far_sum_u += U[t];
          ++far_returns;
          c.far_hlo = std::min(c.far_hlo, H[t]), c.far_hhi = std::max(c.far_hhi, H[t]);
          c.far_ulo = std::min(c.far_ulo, U[t]), c.far_uhi = std::max(c.far_uhi, U[t]);
        }
        ss.push_back(S[t]), uu.push_back(U[t]), hh.push_back(H[t]);
        xs.push_back(Q[t].x), ys.push_back(Q[t].y), zs.push_back(Q[t].z);
        members_q.push_back(Q[t]);
        smin = std::min(smin, S[t]), smax = std::max(smax, S[t]);
        if (C[t]) smin_conf = std::min(smin_conf, S[t]);
        c.u_lo = std::min(c.u_lo, U[t]), c.u_hi = std::max(c.u_hi, U[t]);
        c.h_lo = std::min(c.h_lo, H[t]), c.h_hi = std::max(c.h_hi, H[t]);
        const double pt[3] = {Q[t].x, Q[t].y, Q[t].z};
        for (int a = 0; a < 3; ++a) c.lo[a] = std::min(c.lo[a], pt[a]), c.hi[a] = std::max(c.hi[a], pt[a]);
        margin = std::max(margin, std::min(std::min(cfg.half_width - std::abs(U[t]), cfg.height - H[t]), H[t] - cfg.min_height));
        umin_abs = std::min(umin_abs, std::abs(U[t]));
      }
      auto distinct = [](std::vector<std::array<long, 3>>& k) {
        std::sort(k.begin(), k.end());
        return static_cast<int>(std::unique(k.begin(), k.end()) - k.begin());
      };
      // the Pandar128 recordings repeat almost every return (two echoes of one target), so the
      // thresholds count returns; distinct spots are reported and used against specks of dust
      nc = static_cast<int>(keys_conf.size());
      c.unique_conf = distinct(keys_conf);
      c.unique_point_count = distinct(keys_all);
      if (!soft_spots.empty()) {
        std::vector<double> depths;
        for (const auto& spot : soft_spots) depths.push_back(spot.second);
        std::sort(depths.begin(), depths.end(), std::greater<double>());
        c.soft_z = depths[std::min(size_t(1), depths.size()-1)];  // require two distinct supporting spots
      }
      c.far_points = distinct(keys_far);
      c.far_u = far_returns ? far_sum_u / far_returns : 0.;
      // distance to the nearest point that is confidently inside; low, uncertain
      // points only contribute to the extent
      const double sm = nc ? smin_conf : smin;
      c.distance_m = c.s = sm;
      c.s_max = smax;
      c.s_med = median(ss);
      c.path_lateral_m = median(uu);
      c.center = {median(xs), median(ys), median(zs)};
      c.point_count = static_cast<int>(idx.size());
      c.confident_points = nc;
      c.appeared_frac = appeared_frac(members_q);
      c.margin_m = margin;
      long gi = std::lower_bound(pm.arc.begin(), pm.arc.end(), sm) - pm.arc.begin();
      c.sigma_u = pm.sigma_u[std::min(static_cast<long>(pm.arc.size()) - 1, gi)];
      c.sigma_h = pm.sigma_h[std::min(static_cast<long>(pm.arc.size()) - 1, gi)];
      // A wall crossing the envelope is seen obliquely: long along the track AND wide AND tall.
      const double ext_s = smax - smin, ext_u = c.u_hi - c.u_lo, ext_h = c.h_hi - c.h_lo;
      const bool off_centre = umin_abs > cfg.wall_clear_center;
      const bool at_axis_end = smax > support_arc - cfg.wall_end_margin;
      if (ext_s > cfg.wall_extent_s && ext_u > cfg.wall_extent_uh && ext_h > cfg.wall_extent_uh &&
          (off_centre || at_axis_end)) {
        c.wall_like = true;
        c.confident_points = 0;
      }
      if (c.distance_m > pm.info.decision_limit_m) c.confident_points = 0, c.unverified = true;
      out.push_back(c);
    }
    std::stable_sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) { return a.distance_m < b.distance_m; });
    return out;
  }

  // ------------------------------------------------------------------ continuous evidence
  // An object near the envelope edge may never reach the margin for the axis uncertainty in one
  // frame. Its depth inside the edge, in units of that uncertainty, is turned into a smooth weight
  // Phi(z) and accumulated per track with forgetting (the team's CUSUM-like experiment, Igor's
  // conservative variant): A = max(0, e^(-dt/tau) A + tau (1 - e^(-dt/tau)) q (Phi(z) - b)).
  // Points outside lower A; a constant intrusion of half a sigma passes the threshold in about 3 s.
  // One vote per track and frame, compact bodies with several distinct returns only.
  std::map<int, bool> soft_evidence(const std::vector<Candidate>& cands) {
    std::map<int, bool> out;
    if (!cfg.soft_confirmation) return out;
    std::map<int, std::pair<double, double>> best;  // track -> (z, q) of its best fragment
    for (const Candidate& c : cands) {
      if (!c.tracked || c.unique_point_count < 2 || c.wall_like || c.unverified) continue;
      if (c.s_max - c.s >= 3. || c.u_hi - c.u_lo >= 1.5 || c.h_hi - c.h_lo >= 2.5) continue;
      long gi = std::lower_bound(path->arc.begin(), path->arc.end(), c.s) - path->arc.begin();
      gi = std::clamp(gi, 0L, static_cast<long>(path->arc.size()) - 1);
      const double sig = std::max({c.sigma_u, path->sigma_h[gi], .02});
      const double z = c.margin_m / sig;
      const double q = std::min(c.unique_point_count / 4., 1.) * std::min(1., .15 / sig);
      auto it = best.find(c.track_id);
      if (it == best.end() || z * q > it->second.first * it->second.second) best[c.track_id] = {z, q};
    }
    for (const auto& [id, zq] : best) {
      Track* t = find_track(id);
      if (!t) continue;
      double dt = t->soft_t < 0 ? cfg.frame_period_s : now - t->soft_t;
      if (t->soft_t >= 0 && (dt <= 0 || dt > cfg.display_max_gap_s)) t->soft = 0., t->soft_hits = 0, dt = cfg.frame_period_s;
      t->soft_t = now;
      const double w = .5 * (1. + std::erf(zq.first / std::sqrt(2.)));
      const double decay = std::exp(-dt / cfg.soft_tau_s);
      t->soft = std::max(0., t->soft * decay + cfg.soft_tau_s * (1. - decay) * zq.second * (w - cfg.soft_reference));
      t->soft_hits += zq.second > 0;
      out[id] = t->soft_hits >= cfg.soft_min_hits && t->soft >= cfg.soft_threshold;
    }
    return out;
  }

  // ------------------------------------------------------------------ tracking
  int observation_window() const { return cfg.farfield ? std::max(cfg.window, cfg.far_window) : cfg.window; }
  bool holdable(const Track& t) const {
    const double limit = t.moving ? cfg.move_hold_s : cfg.max_hold_s;
    return t.has_confirmed && now - t.confirmed_t <= limit;
  }
  double expected_S(const Track& t) const {
    // a moving object has travelled on at its speed
    if (t.moving && !t.ts.empty()) return t.Ss.back() + t.speed * (now - t.ts.back());
    return t.S;
  }
  double evidence(const Candidate& c) const {
    // confident points, discounted when a close cluster has fewer returns than any real object of >= 10 cm
    // Identical duplicated returns still describe one ray. Cap only that
    // degenerate case; multi-ray Pandar clusters retain their established weight.
    const int n = c.unique_point_count <= 1 ? std::min(c.confident_points, 1) : c.confident_points;
    const double expected = std::min(8., std::max(1., 40. / std::max(c.s, 1.)));
    // a speck: a few rays into one spot near the train (dust); a real object there spans neighbouring rays
    double extent = 0.;
    for (int a = 0; a < 3; ++a) extent = std::max(extent, c.hi[a] - c.lo[a]);
    if (c.s < cfg.speck_range && c.unique_conf <= cfg.speck_points && extent < cfg.speck_size) return .1 * std::min(n, 12);
    return std::min(n, 12) * (n >= expected ? 1. : .3);
  }
  Track* find_track(int id) {
    for (Track& t : tracks)
      if (t.id == id) return &t;
    return nullptr;
  }

  std::optional<double> moving_speed(const Track& t, const Candidate& c, double ev, double need, int recent) const {
    // world speed of a track that moves steadily along the path with strong evidence
    const int f = frame;
    if (static_cast<int>(t.Ss.size()) < cfg.move_min_obs || recent < cfg.move_min_obs || t.ts.size() != t.Ss.size())
      return std::nullopt;
    const double tmin = *std::min_element(t.ts.begin(), t.ts.end()), tmax = *std::max_element(t.ts.begin(), t.ts.end());
    if (tmax - tmin < .2) return std::nullopt;
    const size_t m = t.ts.size();
    double tm = 0.;
    for (double v : t.ts) tm += v;
    tm /= m;
    double sxx = 0., sxy = 0., sy = 0.;
    for (size_t k = 0; k < m; ++k) sy += t.Ss[k];
    const double ym = sy / m;
    for (size_t k = 0; k < m; ++k) sxx += (t.ts[k] - tm) * (t.ts[k] - tm), sxy += (t.ts[k] - tm) * (t.Ss[k] - ym);
    const double speed = sxx > 0 ? sxy / sxx : 0.;
    double rss = 0.;
    for (size_t k = 0; k < m; ++k) {
      const double r = t.Ss[k] - (speed * (t.ts[k] - tm) + ym);
      rss += r * r;
    }
    if (std::sqrt(rss / m) > cfg.move_resid + .004 * c.s) return std::nullopt;
    if (!(cfg.move_min_speed <= std::abs(speed) && std::abs(speed) <= cfg.move_max_speed)) return std::nullopt;
    if (std::abs(speed - (odometry.has_v ? odometry.v : 0.)) < cfg.move_rel_min) {
      const bool wide = c.u_hi - c.u_lo > 1.2 && c.h_hi - c.h_lo > 1.2;
      if (!(wide && std::abs(c.path_lateral_m) < .6)) return std::nullopt;
    }
    if (ev < cfg.move_evidence_k * need) return std::nullopt;
    int strong = 0;
    for (const auto& [fr, nn] : t.conf)
      if (f - fr < cfg.window && nn >= cfg.strong_points) ++strong;
    if (strong < cfg.move_strong) return std::nullopt;
    long gi = std::lower_bound(path->arc.begin(), path->arc.end(), c.s) - path->arc.begin();
    gi = std::min(static_cast<long>(path->arc.size()) - 1, gi);
    if (std::abs(c.path_lateral_m) >= cfg.half_width - path->sigma_u[gi]) return std::nullopt;
    return speed;
  }

  void track(std::vector<Candidate>& cands, double analysis) {
    const int f = frame;
    limit_hist.push_back(S_ego + analysis);
    while (static_cast<int>(limit_hist.size()) > std::max(cfg.confirm_range_frames, 1)) limit_hist.pop_front();
    for (Candidate& c : cands) {
      c.S = c.s_med + S_ego;
      Track* best = nullptr;
      double best_score = 0.;
      for (Track& t : tracks) {
        if (f - t.last > observation_window() && !holdable(t)) continue;
        const double gate = cfg.gate_s + .01 * c.s;
        const double dS = std::abs(c.S - expected_S(t));
        if (dS < gate && std::abs(c.path_lateral_m - t.u) < .9 && std::abs(c.h_mean() - t.h) < cfg.gate_h) {
          // an already confirmed track wins ties, so one object keeps one track
          const double score = dS + (t.has_confirmed ? 0. : .5);
          if (!best || score < best_score) best = &t, best_score = score;
        }
      }
      const bool strong = c.confident_points >= cfg.strong_points;
      c.evidence_obs = evidence(c);
      Track* t;
      if (!best) {
        Track nt;
        nt.id = next_id++;
        nt.S = c.S, nt.u = c.path_lateral_m, nt.h = c.h_mean();
        nt.first = nt.last = f;
        nt.strong = strong;
        nt.frames = {f};
        nt.Ss = {c.S};
        nt.conf = {{f, c.evidence_obs}};
        nt.ts = {now};
        tracks.push_back(nt);
        t = &tracks.back();
      } else {
        t = best;
        if (t->last != f) {
          ++t->hits;
          t->frames.push_back(f);
        }
        t->last = f;
        const double a = t->moving ? 1. : .5;  // a moving track follows its object without lag
        t->S = (1 - a) * t->S + a * c.S;
        t->u = (1 - a) * t->u + a * c.path_lateral_m;
        t->h = (1 - a) * t->h + a * c.h_mean();
        t->strong += strong;
        t->Ss.push_back(c.S);
        while (static_cast<int>(t->Ss.size()) > cfg.window) t->Ss.pop_front();
        t->ts.push_back(now);
        while (static_cast<int>(t->ts.size()) > cfg.window) t->ts.pop_front();
        // one object may fall apart into clusters: a frame counts once, by its best cluster
        const double e = evidence(c);
        if (!t->conf.empty() && t->conf.back().first == f) t->conf.back().second = std::max(t->conf.back().second, e);
        else t->conf.push_back({f, e});
        while (static_cast<int>(t->conf.size()) > cfg.window) t->conf.pop_front();
      }
      c.track_id = t->id;
      c.tracked = true;
      while (!t->raw.empty() && f - t->raw.front().first >= cfg.window) t->raw.pop_front();
      const int raw = std::min(c.unique_point_count, 12);
      if (!t->raw.empty() && t->raw.back().first == f) t->raw.back().second = std::max(t->raw.back().second, raw);
      else t->raw.push_back({f, raw});
      while (!t->far.empty() && f - t->far.front().frame >= cfg.far_window) t->far.pop_front();
      const bool far_ok = cfg.farfield && c.far_points > 0 && !c.wall_like && !c.unverified &&
                          c.s >= cfg.far_min_range && c.sigma_u <= cfg.far_max_sigma_u &&
                          std::abs(c.far_u) + c.sigma_u <= cfg.half_width &&
                          c.s_max - c.s < 3. && c.u_hi - c.u_lo < 1.5 && c.far_uhi - c.far_ulo >= .05;
      if (far_ok) {
        Track::FarHit hit{f, c.far_points, c.S, c.far_u, c.far_hlo, c.far_hhi};
        if (!t->far.empty() && t->far.back().frame == f) {
          if (hit.points > t->far.back().points) t->far.back() = hit;
        } else t->far.push_back(hit);
      }
      if (c.confident_points > 0 && (t->obs_ego.empty() || t->obs_ego.back() != S_ego)) {
        t->obs_ego.push_back(S_ego);
        if (t->obs_ego.size() > 100) t->obs_ego.erase(t->obs_ego.begin());
      }
    }
    // prune
    std::vector<Track> kept;
    for (Track& t : tracks)
      if (f - t.last <= std::max(observation_window(), cfg.hold_frames) + 2 || holdable(t)) kept.push_back(std::move(t));
    tracks = std::move(kept);
    const std::map<int, bool> soft = soft_evidence(cands);
    for (Candidate& c : cands) {
      Track* t = find_track(c.track_id);
      int recent = 0;
      for (int x : t->frames) recent += f - x < cfg.window;
      const bool conf_pts = c.confident_points > 0;
      double drift = 0.;
      if (t->Ss.size() > 1)
        drift = *std::max_element(t->Ss.begin(), t->Ss.end()) - *std::min_element(t->Ss.begin(), t->Ss.end());
      const bool stat = drift < cfg.static_tol + .006 * c.s;
      double ev = 0.;
      for (const auto& [fr, nn] : t->conf)
        if (f - fr < cfg.window) ev += std::min(nn, 12.);
      const double need = cfg.evidence_near + cfg.evidence_per_m * std::max(0., c.s - 40.);
      c.evidence_need = need;
      for (const auto& [fr, nn] : t->conf)
        if (f - fr < cfg.window && nn >= cfg.strong_points) ++c.strong_recent;
      bool ok = conf_pts && stat && ev >= need &&
                ((t->strong >= cfg.strong_hits && recent >= cfg.strong_hits) || recent >= cfg.weak_hits);
      // A compact edge object can supply several distinct confident returns in
      // three frames without ever reaching the per-frame 'strong' threshold.
      // Keep the geometric margin and the total evidence requirement unchanged.
      int confident_frames = 0;
      for (const auto& [fr, nn] : t->conf)
        if (f - fr < cfg.window && nn > 0.) ++confident_frames;
      const bool near_edge = std::max(std::abs(c.u_lo), std::abs(c.u_hi)) > cfg.half_width - .35 ||
                             c.h_hi > cfg.height - .35;
      c.edge_confirmed = cfg.edge_confirmation && !ok && conf_pts && c.unique_conf >= 2 &&
                         stat && !c.wall_like && !c.unverified && near_edge &&
                         c.s < cfg.boundary_max_range && c.s_max - c.s < cfg.boundary_max_len &&
                         c.u_hi - c.u_lo < 1.5 && c.h_hi - c.h_lo < 2.5 && c.h_hi > cfg.boundary_min_top &&
                         confident_frames >= cfg.boundary_hits && recent >= cfg.boundary_hits && ev >= need;
      ok |= c.edge_confirmed;
      if (!ok && stat) {
        auto it = soft.find(c.track_id);
        c.soft_confirmed = it != soft.end() && it->second;
        ok |= c.soft_confirmed;
      }
      if (cfg.farfield && c.far_points > 0 && !c.wall_like && !c.unverified &&
          c.s >= cfg.far_min_range && c.sigma_u <= cfg.far_max_sigma_u &&
          std::abs(c.far_u) + c.sigma_u <= cfg.half_width) {
        double hlo = kInf, hhi = -kInf, umean = 0.;
        const long cell = static_cast<long>(std::floor(c.S));
        for (const auto& hit : t->far) {
          if (f - hit.frame >= cfg.far_window || std::abs(static_cast<long>(std::floor(hit.S)) - cell) > 1) continue;
          ++c.far_hits;
          c.far_total += hit.points;
          umean += hit.u;
          hlo = std::min(hlo, hit.hlo), hhi = std::max(hhi, hit.hhi);
        }
        // Repeated rays on a grounded compact body, rather than a single persistent dust spot.
        const bool body = c.far_hits && std::abs(umean / c.far_hits) < cfg.half_width - .15 &&
                          hlo < 1.1 && hhi - hlo > .15;
        c.far_attention = body && c.far_total >= cfg.far_min_points && c.far_hits >= cfg.far_attention_hits;
        c.far_confirmed = c.far_attention && c.far_hits >= cfg.far_confirm_hits;
        ok |= c.far_confirmed;
      }
      if (!ok && stat && still && !c.unverified && (t->appeared || c.appeared_frac >= cfg.change_new_frac)) {
        // the train stands still and the object appeared where the sensor saw nothing before
        int seen = 0;
        for (int x : t->frames) seen += f - x < cfg.change_window;
        long gi = std::lower_bound(path->arc.begin(), path->arc.end(), c.s) - path->arc.begin();
        gi = std::min(static_cast<long>(path->arc.size()) - 1, gi);
        const bool lateral_ok = std::abs(c.path_lateral_m) < cfg.half_width - path->sigma_u[gi];
        if (lateral_ok && (seen >= cfg.change_persist || (t->appeared && seen >= 3))) {
          ok = true;
          t->appeared = true;
          c.appeared = true;
        }
      }
      if (!ok && !stat && conf_pts) {
        if (auto speed = moving_speed(*t, c, ev, need, recent)) {
          ok = true;
          if (!(t->has_confirmed && t->confirmed_at == f)) ++t->n_moving;
          t->moving = true;
          t->speed = *speed;
          c.moving = true;
          c.speed_mps = *speed;
        }
      }
      c.evidence = ev;
      c.S_drift = drift;
      c.hits = recent;
      if (cfg.candidate_margin_m > 0. && c.nominal_points == 0) ok = false;
      // a new confirmation beyond the range that held in the last frames is only attention
      if (ok && !t->has_confirmed && cfg.confirm_range_frames > 1 && static_cast<int>(limit_hist.size()) >= std::min(cfg.confirm_range_frames, 3) &&
          S_ego + c.s > *std::min_element(limit_hist.begin(), limit_hist.end()) - cfg.confirm_range_tol_m)
        ok = false;
      c.confirmed = ok;
      // Inside the envelope but only within the margin for the axis uncertainty: not proof of an
      // obstacle, yet worth the driver's attention when it is compact and keeps its place.
      int raw_points = 0, raw_frames = 0;
      for (const auto& [fr, n] : t->raw) if (f - fr < cfg.window) raw_points += n, ++raw_frames;
      const bool boundary_evidence =
          (!cfg.edge_confirmation || c.unique_point_count >= 2) &&
          ((c.point_count >= cfg.boundary_points && recent >= cfg.boundary_hits) ||
           (raw_points >= cfg.boundary_points &&
            raw_frames >= (cfg.edge_confirmation ? cfg.boundary_hits : cfg.weak_hits)));
      c.boundary = !ok && !conf_pts && !c.wall_like && !c.unverified && stat && boundary_evidence &&
                   c.s_max - c.s < cfg.boundary_max_len &&
                   c.h_hi > cfg.boundary_min_top && c.s < cfg.boundary_max_range;
      // The same object staying under ATTENTION frame after frame is worth more than a passing flicker: stronger ATTENTION
      // (never an alarm; confirmation is decided above).
      const bool attention = !ok && (c.confident_points > 0 || c.appeared_frac > 0. || c.boundary || c.far_attention);
      if (attention) {
        if (f - t->att_frame > cfg.attention_gap_frames + 1) t->att_run = 0;
        if (t->att_frame != f) ++t->att_run;
        t->att_frame = f;
      } else if (f - t->att_frame > cfg.attention_gap_frames + 1) {
        t->att_run = 0;
      }
      c.attention_run = attention ? t->att_run : 0;
      c.attention_strong = attention && stat && cfg.attention_strong_frames > 0 && t->att_run >= cfg.attention_strong_frames;
      if (cfg.explain)
        c.decision_rule = ok ? (c.edge_confirmed ? "confirmed_edge" : c.soft_confirmed ? "confirmed_soft" : c.far_confirmed ? "confirmed_farfield" : c.appeared ? "confirmed_appeared" : (c.moving ? "confirmed_moving" : "confirmed_static"))
                            : c.unverified ? "unverified" : c.wall_like ? "wall_like" : c.boundary ? "boundary_attention"
                            : c.far_attention ? "far_attention"
                            : !conf_pts ? "axis_margin" : !stat ? "not_static" : ev < need ? "insufficient_evidence" : "insufficient_frames";
      if (ok) {
        t->has_confirmed = true;
        t->confirmed_at = f;
        t->confirmed_t = now;
        t->offset = c.distance_m - c.s_med;
        t->last_c = c;
        t->S_ego_c = S_ego;
      }
    }
    held(cands, analysis);
  }

  // True when a held box has lost its object: its own cluster would give several returns at this range,
  // yet almost none are in the box now. Far objects stay held: a few returns are all they ever gave.
  bool hold_empty(Track& t, const Candidate& c) {
    if (!cfg.hold_check || !cloud_now || !t.last_c) return false;
    const Candidate& last = *t.last_c;
    double expected = last.point_count * std::pow(std::max(last.distance_m, 1.) / std::max(c.distance_m, 1.), 2.);
    if (c.distance_m < cfg.hold_near_m) expected = std::max(expected, cfg.hold_expect_points);
    if (expected < cfg.hold_expect_points) return t.empty_hold = 0, false;
    const double mid_u = .5 * (c.lo[1] + c.hi[1]), half_u = .5 * (c.hi[1] - c.lo[1]) + .4;
    int n = 0, slab = 0;
    for (const P3& q : *cloud_now) {
      if (q.x < c.lo[0] - 5. || q.x > c.hi[0] + 5. || std::abs(q.y) > 4.) continue;
      ++slab;
      if (q.x >= c.lo[0] - 1. && q.x <= c.hi[0] + 1. && std::abs(q.y - mid_u) <= half_u && q.z >= c.lo[2] - .3 && q.z <= c.hi[2] + .3) ++n;
    }
    // The sensor returned (almost) nothing at this range at all: a dropout, not a gone object. Cannot judge.
    if (slab < cfg.hold_slab_points) return t.empty_hold = 0, false;
    if (n >= std::max(2., .15 * expected)) return t.empty_hold = 0, false;
    return ++t.empty_hold >= cfg.hold_empty_frames;
  }

  // frames a confirmed track is carried when it is not confirmed in a frame
  int hold_limit(const Track& t) const {
    const bool close_call = t.last_c && (t.last_c->soft_confirmed || t.last_c->edge_confirmed);
    return close_call ? cfg.hold_frames : std::max(cfg.hold_frames, cfg.hold_frames_strong);
  }

  void held(std::vector<Candidate>& cands, double analysis) {
    // keep a confirmed obstacle that is not confirmed in this frame
    const int f = frame;
    std::map<int, size_t> by_track;
    for (size_t k = 0; k < cands.size(); ++k)
      if (cands[k].tracked) by_track[cands[k].track_id] = k;
    std::vector<Candidate> added;
    for (Track& t : tracks) {
      if (!t.has_confirmed || t.confirmed_at == f) continue;
      const int age = f - t.confirmed_at;
      auto seen_it = by_track.find(t.id);
      Candidate* seen = seen_it == by_track.end() ? nullptr : &cands[seen_it->second];
      if (t.moving) {
        if (t.n_moving < 2 || t.ts.empty()) continue;
        const double S_pred = expected_S(t);
        const double d = S_pred - S_ego + t.offset;
        if (d < std::max(cfg.min_range, cfg.hold_min_range)) continue;
        if (t.last_c && t.last_c->far_confirmed && d > analysis) continue;
        if (seen && (seen->confident_points > 0 || (cfg.hold_observed_geometry && seen->unique_point_count >= 2 && !seen->wall_like && !seen->unverified))) {
          if (age > hold_limit(t)) continue;
          seen->confirmed = seen->held = seen->moving = true;
          seen->decision_rule = "held";
          seen->speed_mps = t.speed;
          continue;
        }
        const bool beyond = d > analysis - 2.;
        if (!(age <= hold_limit(t) || (beyond && now - t.confirmed_t <= cfg.move_hold_s))) continue;
        Candidate c = *t.last_c;
        const double shift = d - c.distance_m;
        c.lo[0] += shift, c.hi[0] += shift, c.center[0] += shift;
        c.distance_m = c.s = d;
        c.s_med = S_pred - S_ego;
        c.s_max += shift;
        c.S = S_pred;
        c.point_count = c.confident_points = c.hits = 0;
        c.confirmed = c.held = c.moving = true;
        c.decision_rule = "held";
        c.held_frames = age;
        c.track_id = t.id;
        c.tracked = true;
        c.speed_mps = t.speed;
        if (hold_empty(t, c)) { t.has_confirmed = false; continue; }
        added.push_back(c);
        continue;
      }
      const bool shrt = age <= hold_limit(t);
      const double d = t.S - S_ego + t.offset;
      if (d < std::max(cfg.min_range, cfg.hold_min_range)) continue;
      if (t.last_c && t.last_c->far_confirmed && d > analysis) continue;
      const bool beyond = d > analysis - 2.;
      if (seen && (seen->confident_points > 0 || (cfg.hold_observed_geometry && seen->unique_point_count >= 2 && !seen->wall_like && !seen->unverified))) {
        // observed with confident points, just not confirmed this frame
        if (shrt) seen->confirmed = seen->held = true, seen->decision_rule = "held";
        continue;
      }
      // not observed, or only as unconfident points: keep the last confirmed state
      double parallax = 0.;
      if (!t.obs_ego.empty())
        parallax = *std::max_element(t.obs_ego.begin(), t.obs_ego.end()) -
                   *std::min_element(t.obs_ego.begin(), t.obs_ego.end());
      const bool long_hold = beyond && parallax >= cfg.hold_min_travel && now - t.confirmed_t <= cfg.max_hold_s;
      if (shrt || long_hold) {
        const double dx = S_ego - t.S_ego_c;
        Candidate c = *t.last_c;
        c.lo[0] -= dx, c.hi[0] -= dx, c.center[0] -= dx;
        c.distance_m = c.s = d;
        c.s_med = t.S - S_ego;
        c.s_max -= dx;
        c.S = t.S;
        c.point_count = c.confident_points = c.hits = 0;
        c.confirmed = c.held = true;
        c.decision_rule = "held";
        c.held_frames = age;
        c.track_id = t.id;
        c.tracked = true;
        if (hold_empty(t, c)) { t.has_confirmed = false; continue; }
        added.push_back(c);
      }
    }
    for (auto& c : added) cands.push_back(c);
    std::stable_sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) { return a.distance_m < b.distance_m; });
  }

  // ------------------------------------------------------------------ output
  void write_candidate(JsonWriter& w, const Candidate& c) const {
    w.open_obj();
    w.kv("distance_m", c.distance_m);
    w.kv("s", c.s);
    w.kv("s_max", c.s_max);
    w.kv("s_med", c.s_med);
    w.kv("path_lateral_m", c.path_lateral_m);
    w.key("u_range"), w.open_arr(), w.num(c.u_lo), w.num(c.u_hi), w.close_arr();
    w.key("h_range"), w.open_arr(), w.num(c.h_lo), w.num(c.h_hi), w.close_arr();
    w.kv_arr("center", c.center.begin(), c.center.end());
    w.key("bbox");
    w.open_arr();
    w.open_arr();
    for (double v : c.lo) w.num(v);
    w.close_arr();
    w.open_arr();
    for (double v : c.hi) w.num(v);
    w.close_arr();
    w.close_arr();
    w.kv_int("point_count", c.point_count);
    w.kv_int("confident_points", c.confident_points);
    w.kv("appeared_frac", c.appeared_frac);
    w.kv("margin_m", c.margin_m);
    w.kv("sigma_u", c.sigma_u);
    if (cfg.candidate_margin_m > 0.) {
      w.kv("sigma_h", c.sigma_h);
      w.kv("soft_z", c.soft_z);
      w.kv_int("nominal_points", c.nominal_points);
    }
    if (c.wall_like) w.kv_bool("wall_like", true);
    if (c.boundary) w.kv_bool("boundary", true);
    if (c.attention_run > 0) w.kv_int("attention_run", c.attention_run);
    if (c.attention_strong) w.kv_bool("attention_strong", true);
    if (c.edge_confirmed) w.kv_bool("edge_confirmed", true);
    if (c.soft_confirmed) w.kv_bool("soft_confirmed", true);
    if (c.far_attention) w.kv_bool("far_attention", true);
    if (c.unverified) w.kv_bool("unverified", true);
    w.kv_int("unique_point_count", c.unique_point_count);
    w.kv_int("unique_confident_points", c.unique_conf);
    if (c.tracked) {
      w.kv("S", c.S);
      w.kv("evidence_obs", c.evidence_obs);
      w.kv_int("track_id", c.track_id);
      w.kv("evidence", c.evidence);
      w.kv("S_drift", c.S_drift);
      w.kv_int("hits", c.hits);
    }
    w.kv_bool("confirmed", c.confirmed);
    if (c.appeared) w.kv_bool("appeared", true);
    if (c.moving) w.kv_bool("moving", true), w.kv("speed_mps", c.speed_mps);
    if (c.held) w.kv_bool("held", true), w.kv_int("held_frames", c.held_frames);
    if (cfg.explain) {
      w.key("explain"), w.open_obj();
      w.kv("sigma_h", c.sigma_h);
      w.kv("evidence_required", c.evidence_need);
      w.kv_int("strong_recent", c.strong_recent);
      w.kv_int("far_frames", c.far_hits);
      w.kv_int("far_points", c.far_total);
      w.kv("lateral_margin", cfg.half_width - std::abs(c.path_lateral_m));
      w.kv("bottom_margin", c.h_lo - cfg.min_height);
      w.kv("top_margin", cfg.height - c.h_hi);
      w.kv_str("rule", c.decision_rule);
      w.close_obj();
    }
    w.close_obj();
  }

  std::string empty(const std::string& reason, Clock::time_point tic) const {
    JsonWriter w;
    w.open_obj();
    w.kv_str("model_id", "tunnel");
    w.kv_str("model_revision", "tunnel-profile-v2-cpp");
    w.kv_bool("obstacle", false);
    w.kv_null("distance_m");
    w.kv_str("status", "unknown");
    w.key("candidates"), w.open_arr(), w.close_arr();
    w.kv_str("geometry_reason", reason);
    w.kv("latency_ms", ms_since(tic));
    w.close_obj();
    return w.out;
  }

  // One reported box per physical object: confirmed boxes closer than merge_gap_m are merged into their union.
  std::vector<Candidate> merged(const std::vector<Candidate>& cands) const {
    if (!cfg.merge_boxes) return cands;
    auto gap = [](const Candidate& a, const Candidate& b) {
      double s = 0.;
      for (int k = 0; k < 3; ++k) {
        const double d = std::max({0., a.lo[k] - b.hi[k], b.lo[k] - a.hi[k]});
        s += d * d;
      }
      return std::sqrt(s);
    };
    std::vector<int> group(cands.size(), -1);
    int groups = 0;
    for (size_t i = 0; i < cands.size(); ++i) {
      if (!cands[i].confirmed) continue;
      if (group[i] < 0) group[i] = groups++;
      for (size_t j = i + 1; j < cands.size(); ++j)
        if (cands[j].confirmed && gap(cands[i], cands[j]) <= cfg.merge_gap_m) {
          if (group[j] < 0) group[j] = group[i];
          else if (group[j] != group[i]) {
            const int from = group[j];
            for (int& g : group) if (g == from) g = group[i];
          }
        }
    }
    std::vector<Candidate> out;
    std::map<int, size_t> at;
    for (size_t i = 0; i < cands.size(); ++i) {
      if (group[i] < 0) { out.push_back(cands[i]); continue; }
      auto it = at.find(group[i]);
      if (it == at.end()) { at[group[i]] = out.size(); out.push_back(cands[i]); continue; }
      Candidate& m = out[it->second];
      const Candidate& c = cands[i];
      if (c.distance_m < m.distance_m) {  // the nearest box speaks for the object
        const int pts = m.point_count + c.point_count, conf = m.confident_points + c.confident_points;
        const int uniq = m.unique_point_count + c.unique_point_count, uconf = m.unique_conf + c.unique_conf;
        const bool all_held = m.held && c.held;
        const auto lo = m.lo, hi = m.hi;
        const double umin = std::min(m.u_lo, c.u_lo), umax = std::max(m.u_hi, c.u_hi), hmin = std::min(m.h_lo, c.h_lo), hmax = std::max(m.h_hi, c.h_hi), smax = std::max(m.s_max, c.s_max);
        m = c;
        m.point_count = pts, m.confident_points = conf, m.unique_point_count = uniq, m.unique_conf = uconf, m.held = all_held;
        for (int k = 0; k < 3; ++k) m.lo[k] = std::min(lo[k], c.lo[k]), m.hi[k] = std::max(hi[k], c.hi[k]);
        m.u_lo = umin, m.u_hi = umax, m.h_lo = hmin, m.h_hi = hmax, m.s_max = smax;
      } else {
        m.point_count += c.point_count, m.confident_points += c.confident_points;
        m.unique_point_count += c.unique_point_count, m.unique_conf += c.unique_conf;
        m.held = m.held && c.held;
        for (int k = 0; k < 3; ++k) m.lo[k] = std::min(m.lo[k], c.lo[k]), m.hi[k] = std::max(m.hi[k], c.hi[k]);
        m.u_lo = std::min(m.u_lo, c.u_lo), m.u_hi = std::max(m.u_hi, c.u_hi);
        m.h_lo = std::min(m.h_lo, c.h_lo), m.h_hi = std::max(m.h_hi, c.h_hi), m.s_max = std::max(m.s_max, c.s_max);
      }
      for (int k = 0; k < 3; ++k) m.center[k] = .5 * (m.lo[k] + m.hi[k]);
    }
    return out;
  }

  std::string result(const PathModel& pm, const PathModel& shown, const std::vector<Candidate>& raw_cands, bool near_ok,
                     double analysis, double envelope, std::optional<double> ds, double oq, const std::vector<std::pair<const char*, double>>& timing,
                     Clock::time_point tic, size_t npoints, size_t nroi) const {
    const std::vector<Candidate> cands = merged(raw_cands);
    bool significant = false, obstacle = false;
    double distance = kInf;
    for (const Candidate& c : cands) {
      significant |= c.confident_points > 0 || c.appeared_frac > 0 || c.boundary || c.far_attention;
      if (c.confirmed) obstacle = true, distance = std::min(distance, c.distance_m);
    }
    const char* status = obstacle ? "obstacle" : (significant ? "candidate" : "clear");
    int attention_level = significant && !obstacle ? 1 : 0;
    for (const Candidate& c : cands) if (!obstacle && c.attention_strong && !c.confirmed) attention_level = 2;
    JsonWriter w;
    w.open_obj();
    w.kv_str("model_id", "tunnel");
    w.kv_str("model_revision", "tunnel-profile-v2-cpp");
    w.kv_str("status", status);
    w.kv_bool("obstacle", obstacle);
    w.kv_int("attention_level", attention_level);  // 0 none, 1 ATTENTION, 2 stronger ATTENTION (the same object for a long time)
    w.kv_opt("distance_m", obstacle, distance);
    w.key("candidates");
    w.open_arr();
    for (const Candidate& c : cands) write_candidate(w, c);
    w.close_arr();
    w.kv_int("points", static_cast<long long>(npoints));
    w.kv_int("roi_points", static_cast<long long>(nroi));
    w.kv("analysis_range_m", analysis);
    w.kv("envelope_range_m", envelope);
    w.key("path");
    w.open_arr();
    // the axis is exported up to the analysed region (max_range + the ROI margin), not the whole grid
    const int path_end = std::min(PathModel::N - 1, static_cast<int>(cfg.max_range) + 25);
    for (int g = 0; g <= path_end; g += 2) {
      w.open_arr();
      w.num(g), w.num(shown.lat[g]), w.num(shown.vert[g]), w.num(shown.arc[g]);
      w.close_arr();
    }
    w.close_arr();
    w.key("path_info");
    w.open_obj();
    w.kv("support_m", pm.info.support_m);
    w.kv("rail_support_m", pm.info.rail_support_m);
    w.kv_bool("extension_ok", pm.info.extension_ok);
    if (pm.info.support_held) w.kv("support_raw_m", pm.info.support_raw_m);
    if (pm.info.break_m < 1e8) w.kv("break_m", pm.info.break_m);
    w.kv_int("good_slabs", pm.info.good_slabs);
    w.kv("phi", pm.info.phi);
    if (pm.info.has_coasting) w.kv_int("coasting_frames", pm.info.coasting_frames);
    if (pm.info.has_analysis_limit) w.kv("analysis_limit_m", pm.info.analysis_limit_m);
    w.kv_opt("odometry_ds", ds.has_value(), ds.value_or(0.));
    w.kv("odometry_quality", oq);
    w.kv("ego_s", S_ego);
    if (cfg.explain) {
      w.key("explain"), w.open_obj();
      w.kv("extension_observed_m", extension_observed);
      w.kv("extension_remembered_m", extension_remembered);
      w.kv("decision_limit_m", pm.info.decision_limit_m);
      w.kv_int("geometry_frames", geometry_used);
      w.key("stages"), w.open_arr();
      for (int g = 0; g < PathModel::N && g < static_cast<int>(dbg_raw.size()); g += 5) {
        w.open_arr();
        w.num(g), w.num(dbg_raw[g]), w.num(dbg_fused[g]), w.num(dbg_spatial[g]), w.num(shown.lat[g]);
        w.close_arr();
      }
      w.close_arr();
      w.key("slabs"), w.open_arr();
      if (debug_slabs) for (size_t j = 0; j < debug_slabs->x.size(); ++j) {
        w.open_obj();
        w.kv("x", debug_slabs->x[j]);
        w.kv("points", debug_slabs->n[j]);
        w.kv("lateral", debug_slabs->y[j]);
        w.kv("height", debug_slabs->z[j]);
        w.kv("du", debug_slabs->du[j]);
        w.kv("inliers", debug_slabs->inl[j]);
        w.kv("corridor_points", debug_slabs->env[j]);
        w.kv("edge_share", debug_slabs->rel[j]);
        w.kv("ambiguity", debug_slabs->alt[j] / std::max(debug_slabs->inl[j], 1e-6));
        w.close_obj();
      }
      w.close_arr(), w.close_obj();
    }
    w.kv_str("method", cfg.spatial_path ? "rails + adaptive smoothing + joined spatial path" : cfg.continuous_path ? "rails + joint continuous tunnel path" : "rails + tunnel profile");
    if(cfg.continuous_path)w.kv_bool("continuous_path",true);
    if(cfg.spatial_path)w.kv_bool("spatial_path",true);
    if((cfg.continuous_path||cfg.spatial_path)&&cfg.path_no_reversal)w.kv_bool("no_reversal",true);
    w.close_obj();
    w.key("floor"), w.open_arr(), w.num(0.), w.num(0.), w.num(pm.vert[0]), w.close_arr();
    // [s, centre xyz, lateral xyz, up xyz] every 2 m for the viewer
    w.key("section_frames");
    w.open_arr();
    for (int g = 0; g < PathModel::N; g += 2) {
      if (shown.arc[g] > envelope + 2) continue;
      double t[3] = {1., shown.dlat[g], 0.};
      double nt = std::sqrt(t[0] * t[0] + t[1] * t[1]);
      t[0] /= nt, t[1] /= nt;
      double l[3] = {-t[1], t[0], shown.phi};
      const double nl = std::sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
      for (double& v : l) v /= nl;
      const double up[3] = {t[1] * l[2] - t[2] * l[1], t[2] * l[0] - t[0] * l[2], t[0] * l[1] - t[1] * l[0]};
      auto r4 = [](double v) { return std::round(v * 1e4) / 1e4; };
      w.open_arr();
      for (double v : {shown.arc[g], double(g), shown.lat[g], shown.vert[g], l[0], l[1], l[2], up[0], up[1], up[2]}) w.num(r4(v));
      w.close_arr();
    }
    w.close_arr();
    w.key("envelope");
    w.open_arr();
    for (double a : arange(0, analysis + 1, 2.)) {
      w.open_arr();
      for (double v : {a, -cfg.half_width, cfg.half_width, -cfg.half_width, cfg.half_width}) w.num(v);
      w.close_arr();
    }
    w.close_arr();
    w.key("vehicle");
    w.open_obj();
    w.kv_str("envelope_mode", "section");
    w.kv("car_width", 2 * cfg.half_width);
    w.kv("max_height", cfg.height);
    w.kv("body_bottom", 0.);
    w.kv("safety_margin", 0.);
    w.kv_str("projection_mode", "rail_section");
    w.close_obj();
    w.kv("half_width_m", cfg.half_width);
    w.kv("min_height_m", cfg.min_height);
    w.key("geometry_contract");
    w.open_obj();
    w.kv_str("height_datum", "rail_heads");
    w.kv("section_width_m", 2 * cfg.half_width);
    w.kv("section_height_m", cfg.height);
    w.kv("curve_margin_m", 0.);
    w.kv("lower_boundary_tolerance_m", cfg.min_height);
    w.kv_bool("datum_observed", near_ok);
    w.close_obj();
    w.key("timing");
    w.open_obj();
    for (const auto& [k, v] : timing) w.kv(k, v);
    w.close_obj();
    w.kv("latency_ms", ms_since(tic));
    w.close_obj();
    return w.out;
  }

  // ------------------------------------------------------------------ drawn envelope
  // Display only (the team's browser filter of 18.10 moved into the core): the previous drawn axis is
  // moved by the travel; a change smaller than display_release is damped with time constant
  // display_tau_s, from display_release to display_snap the new frame takes over smoothly, beyond it
  // the new axis is drawn at once. Decisions keep the computed axis.
  // The drawn envelope length. The raw length flips between a far structure seen or not seen; a
  // drawn box that blinks is read as unstable. A longer envelope is drawn at once, a shorter one only
  // once it persists and then gradually; decisions keep their own, unsmoothed range.
  double draw_envelope(double raw, std::optional<double> ds, double break_now) {
    const double dt = now - env_t;
    if (!cfg.envelope_hold || raw <= 0. || !env_valid || !ds || dt <= 0. || dt > cfg.display_max_gap_s) {
      env_valid = raw > 0., env_drawn = raw, env_cum = 0., env_t = now;
      env_hist.clear();
      if (env_valid) env_hist.push_back({raw, false});
      return raw;
    }
    // the raw end is kept along the path, so the train moving closer does not look like a change
    env_cum += *ds;
    const double moved = env_drawn - *ds;
    const bool visible_end = break_now < moved && std::abs(break_now - (env_break - *ds)) <= 15.;
    env_break = break_now;
    env_hist.push_back({env_cum + raw, visible_end});
    const size_t keep = static_cast<size_t>(std::max({cfg.envelope_hold_frames, cfg.envelope_rise_frames, cfg.envelope_end_frames, 1}));
    while (env_hist.size() > keep) env_hist.pop_front();
    auto over = [&](int n, auto pick, double start) {  // the extreme of the last n raw ends, or `start` when fewer frames exist
      if (static_cast<int>(env_hist.size()) < n) return start;
      double v = env_hist[env_hist.size() - 1].end;
      for (int i = 1; i <= n; ++i) v = pick(v, env_hist[env_hist.size() - i].end);
      return v;
    };
    const double here = env_drawn + env_cum - *ds;  // the drawn end along the path
    const double hi = over(std::max(cfg.envelope_hold_frames, 1), [](double a, double b) { return std::max(a, b); }, here);
    const double lo = over(std::max(cfg.envelope_rise_frames, 1), [](double a, double b) { return std::min(a, b); }, here);
    bool cut = false;
    if (cfg.envelope_end_frames > 0 && static_cast<int>(env_hist.size()) >= cfg.envelope_end_frames) {
      cut = true;
      for (int i = 1; i <= cfg.envelope_end_frames; ++i) cut = cut && env_hist[env_hist.size() - i].visible;
    }
    double end = here;
    if (cut && raw < moved) end = env_cum + raw;
    else if (lo > here) end = std::min(lo, here + cfg.envelope_max_rise);
    else if (hi < here) end = std::max(hi, here - cfg.envelope_max_drop);
    env_drawn = end - env_cum;
    env_t = now;
    return env_drawn;
  }

  PathModel display_path(const PathModel& pm, std::optional<double> ds, double envelope) {
    const int N = PathModel::N;
    // The joint spatial axis already carries this filter, as one curve. Smoothing it again node by node
    // (20.10, measured) halved the motion of the drawn end but brought back S-bends and hooks beyond the
    // data (61-518 frames with a reversal over 5 degrees instead of 0-15): the spatial axis is drawn as is.
    // Without it the drawing is smoothed and then joined into one curve, so no side-step comes back.
    const bool joint = cfg.spatial_path || cfg.continuous_path;
    if (!cfg.display_smoothing || (joint && !cfg.display_far_smoothing)) return pm;
    const double dt = now - show_t;
    Vec lat = pm.lat, vert = pm.vert;
    double phi = pm.phi;
    if (show_valid && ds && dt > 0 && dt <= cfg.display_max_gap_s) {
      const PathModel prev = PathModel(show_lat, show_vert, show_phi, 0.).shifted(*ds);
      const double base = 1. - std::exp(-dt / cfg.display_tau_s);
      auto gain = [&](double r) {
        const double t = std::clamp((r - cfg.display_release) / (cfg.display_snap - cfg.display_release), 0., 1.);
        return base + (1. - base) * t * t * (3. - 2. * t);
      };
      // beyond the measured support the drawn axis is only a continuation: it changes there slowly, without
      // the instant pass of large changes, since a jump of the support swings it by metres
      const double far = 1. - std::exp(-dt / cfg.display_far_tau_s);
      const double limit = std::min(envelope, show_envelope - *ds);
      // The offset and the heading of the train against the track change from frame to frame and move the whole axis: that is
      // no jitter and passes at once, or the far part would lag behind the near part and bend the curve (an S-bend).
      // Only the rest, the change of the shape, is damped far out.
      double ry0 = 0., ry1 = 0., rz0 = 0., rz1 = 0.;
      if (joint) {
        double n = 0., sg = 0., sgg = 0., sy = 0., sgy = 0., sz = 0., sgz = 0.;
        for (int g = 10; g <= std::min(60., pm.support) && g + *ds < N - 1; ++g) {
          const double dy = pm.lat[g] - prev.lat[g], dz = pm.vert[g] - prev.vert[g];
          n += 1., sg += g, sgg += double(g) * g, sy += dy, sgy += g * dy, sz += dz, sgz += g * dz;
        }
        const double det = n * sgg - sg * sg;
        if (n >= 16. && std::abs(det) > 1e-9) {
          ry1 = (n * sgy - sg * sy) / det, ry0 = (sy - ry1 * sg) / n;
          rz1 = (n * sgz - sg * sz) / det, rz0 = (sz - rz1 * sg) / n;
          // what the near nodes did is a rigid motion only if a straight line explains it; a slope that is a few metres at
          // the far end is noise of the near part, not a turn of the train
          double ey = 0., ez = 0.;
          for (int g = 10; g <= std::min(60., pm.support) && g + *ds < N - 1; ++g) {
            const double dy = pm.lat[g] - prev.lat[g] - ry0 - ry1 * g, dz = pm.vert[g] - prev.vert[g] - rz0 - rz1 * g;
            ey += dy * dy, ez += dz * dz;
          }
          if (std::sqrt(ey / n) > .08 || std::abs(ry1) > .004) ry0 = ry1 = 0.;
          if (std::sqrt(ez / n) > .08 || std::abs(rz1) > .004) rz0 = rz1 = 0.;
        }
      }
      // the gains of the far smoothing are themselves smoothed along the axis (31 m): a gain that changes from node to node
      // would bend the drawn curve
      Vec gy(N, 1.), gz(N, 1.);
      const double far_support = 1. - std::exp(-dt / std::max(cfg.display_far_support_tau_s, 1e-3));
      if (joint) {
        for (int g = 0; g < N; ++g) {  // every node, also beyond the drawn end: a gain that stops there would leave a step
          const double dy = pm.lat[g] - prev.lat[g] - (ry0 + ry1 * g), dz = pm.vert[g] - prev.vert[g] - (rz0 + rz1 * g);
          const double w = std::clamp((g - pm.support) / 20., 0., 1.);
          double ay = 1., az = 1.;
          const double wf = std::clamp((g - cfg.display_far_from) / std::max(cfg.display_far_ramp, 1.), 0., 1.);
          if (wf > 0. && g + *ds < N - 1) {
            const double base_far = 1. - std::exp(-dt / std::max(cfg.display_far_tau_max * wf, 1e-3));
            const auto adaptive = [&](double r, double sigma) {
              const double lo = cfg.display_far_release_k * sigma, hi = lo + std::max(cfg.display_far_pass_m, cfg.display_far_pass_k * sigma);
              const double t = std::clamp((r - lo) / (hi - lo), 0., 1.);
              return base_far + (1. - base_far) * t * t * (3. - 2. * t);
            };
            ay = adaptive(std::abs(dy), pm.sigma_u[g]);
            az = adaptive(std::abs(dz), pm.sigma_h[g]);
          }
          gy[g] = (1. - w) * ay + w * far_support;
          gz[g] = (1. - w) * az + w * far_support;
        }
        Vec sy(N), sz(N);
        for (int g = 0; g < N; ++g) {
          const int half = static_cast<int>(std::max(1., cfg.display_far_gain_half_m));
          const int a = std::max(0, g - half), b = std::min(N - 1, g + half);
          double cy = 0., cz = 0.;
          for (int k = a; k <= b; ++k) cy += gy[k], cz += gz[k];
          sy[g] = cy / (b - a + 1), sz[g] = cz / (b - a + 1);
        }
        gy = sy, gz = sz;
      }
      for (int g = 0; g < N && (joint || pm.arc[g] <= limit); ++g) {
        const double rigid_y = ry0 + ry1 * g, rigid_z = rz0 + rz1 * g;
        const double dy = pm.lat[g] - prev.lat[g] - rigid_y, dz = pm.vert[g] - prev.vert[g] - rigid_z;
        // the change of memory is spread over 20 m beyond the support, so it leaves no step there
        const double w = std::clamp((g - pm.support) / 20., 0., 1.);
        const double ay = joint ? gy[g] : (1. - w) * gain(std::hypot(dy, dz)) + w * far;
        const double az = joint ? gz[g] : ay;
        lat[g] = prev.lat[g] + rigid_y + ay * dy;
        vert[g] = prev.vert[g] + rigid_z + az * dz;
      }
      // the bank tilts the upper corners even when the centre line stands still
      const double dphi = pm.phi - show_phi;
      phi = show_phi + gain(3. * std::abs(dphi)) * dphi;
    }
    if (cfg.display_join && !joint) {
      Vec wu(N), wh(N);
      for (int g = 0; g < N; ++g) {
        wu[g] = 1. / std::pow(std::clamp(pm.sigma_u[g], .025, cfg.display_join_sigma), 2);
        wh[g] = 1. / std::pow(std::clamp(pm.sigma_h[g], .035, cfg.display_join_sigma), 2);
      }
      // the same one-sided curvature limits as the axis beyond its measured support (no S-bend or hook
      // where nothing is measured), with the drawn points kept there: dropping them let the end follow
      // every jump of the support
      lat = continuous_path_fit(lat, wu, cfg.display_join_lambda, cfg.path_min_radius, lateral_support(pm), cfg.path_tail_m,
                                (cfg.path_reversal_parts & 2) && pm.info.extension_ok, cfg.path_extension_decay_m, 1.);
      vert = continuous_path_fit(vert, wh, cfg.display_join_lambda, cfg.path_vertical_radius);
    }
    show_lat = lat, show_vert = vert, show_phi = phi, show_envelope = envelope, show_t = now, show_valid = true;
    return PathModel(lat, vert, phi, pm.support, pm.sigma_u, pm.sigma_h, pm.info);
  }

  // ------------------------------------------------------------------ envelope length
  // Slabs near the quality thresholds come and go between frames, and the support (with it the
  // envelope length) jumped by tens of metres. A drop is taken only when it persists, or at once
  // where the normal tunnel visibly ends (a changed cross-section before the old end).
  void hold_support(PathModel& pm, const PathModel& prev, double ds) {
    const double prev_end = prev.support - ds;
    const bool visible_end = pm.info.break_m < prev_end;
    if (pm.support < prev_end - cfg.hold_support_tol && !visible_end && support_drops + 1 < cfg.hold_support_frames) {
      ++support_drops;
      pm.support = pm.info.support_m = prev_end;
      pm.info.support_held = true;
    } else {
      support_drops = 0;
    }
    if (prev.info.extension_ok && !pm.info.extension_ok && !visible_end && extension_drops + 1 < cfg.hold_support_frames &&
        pm.support >= cfg.extension_min_support) {
      ++extension_drops;
      pm.info.extension_ok = true;  // the corridor beyond is still checked for walls
    } else if (pm.info.extension_ok || !prev.info.extension_ok) {
      extension_drops = 0;
    }
  }

  // ------------------------------------------------------------------ fusion
  // Per-node Kalman filter of the axis (lateral and vertical) along the 1 m grid. The previous
  // state is transported by the travel like the path itself (offset and heading at the sensor
  // stay, the shape ahead moves towards the train). Near the train the rails decide; far ahead the
  // fused estimate takes over, so the envelope end stops jumping between frames and its
  // uncertainty shrinks where the same geometry keeps being observed.
  void fuse_path(PathModel& pm, double ds, bool restart) {
    const int N = PathModel::N;
    auto init = [&] {
      f_lat = pm.lat, f_vert = pm.vert;
      f_Plat.resize(N), f_Pvert.resize(N);
      for (int g = 0; g < N; ++g) f_Plat[g] = pm.sigma_u[g] * pm.sigma_u[g], f_Pvert[g] = pm.sigma_h[g] * pm.sigma_h[g];
      fused = true;
    };
    if (!cfg.fuse) return;
    if (!fused || restart) return init();
    const PathModel prev = PathModel(f_lat, f_vert, pm.phi, 0.).shifted(ds);
    const double Q = cfg.fuse_q_abs * cfg.fuse_q_abs + (cfg.fuse_q_rel * ds) * (cfg.fuse_q_rel * ds);
    const double Qv = cfg.fuse_q_abs * cfg.fuse_q_abs + (cfg.fuse_q_rel_vert * ds) * (cfg.fuse_q_rel_vert * ds);
    const double start = pm.info.rail_support_m + cfg.fuse_start;
    Vec lat(N), vert(N), su(N), sh(N), Plat(N), Pvert(N);
    auto step = [&](double meas, double R, double prior, double Pp, double w, bool extrapolated, double& out, double& outP) {
      double est, P;
      const double nu = meas - prior;
      // beyond this frame's support the axis is only continued: it says little about the track there,
      // so the geometry gathered earlier, while it was supported, keeps the lead
      if (extrapolated) R *= cfg.fuse_extrap_r;
      if (!(Pp < 1e9) || nu * nu > cfg.fuse_gate * cfg.fuse_gate * (Pp + R)) {
        est = meas, P = R;  // new geometry (or nothing before): restart the node
      } else {
        double K = Pp / (Pp + R);
        if (cfg.fuse_damp_min < 1.) {
          // like the stabilizer of a phone: scatter within the filter's own uncertainty is damped, a change beyond it passes
          const double z = std::abs(nu) / std::sqrt(Pp + R);
          const double t = std::clamp((z - cfg.fuse_damp_lo) / std::max(cfg.fuse_damp_hi - cfg.fuse_damp_lo, 1e-6), 0., 1.);
          K *= cfg.fuse_damp_min + (1. - cfg.fuse_damp_min) * t * t * (3. - 2. * t);
        }
        est = prior + K * nu, P = (1 - K) * Pp;
        // beyond the supported range every frame extrapolates the same way: repeating it adds no
        // information, the node only averages the frame-to-frame scatter
        if (extrapolated && cfg.fuse_extrap_floor) P = std::max(P, cfg.fuse_extrap_floor_k * R);
      }
      out = (1 - w) * meas + w * est;
      outP = (1 - w) * R + w * P;
    };
    for (int g = 0; g < N; ++g) {
      const double xo = g + ds;
      const bool known = xo <= N - 1;
      const double Pl = known ? interp_grid(xo, f_Plat) + Q : 1e12;
      const double Pv = known ? interp_grid(xo, f_Pvert) + Qv : 1e12;
      const double w = std::clamp((g - start) / cfg.fuse_ramp, 0., 1.);
      const bool extrapolated = g > pm.info.support_raw_m;
      step(pm.lat[g], pm.sigma_u[g] * pm.sigma_u[g], prev.lat[g], Pl, w, extrapolated, lat[g], Plat[g]);
      step(pm.vert[g], pm.sigma_h[g] * pm.sigma_h[g], prev.vert[g], Pv, w, extrapolated, vert[g], Pvert[g]);
      const double floor = cfg.fuse_floor0 + cfg.fuse_floor_per_m * g;
      const bool shrink = cfg.fuse_sigma == 1 || (cfg.fuse_sigma == 2 && standing >= cfg.fuse_still_frames);
      su[g] = shrink ? std::min(pm.sigma_u[g], std::sqrt(Plat[g] + floor * floor)) : pm.sigma_u[g];
      sh[g] = shrink ? std::min(pm.sigma_h[g], std::sqrt(Pvert[g] + floor * floor)) : pm.sigma_h[g];
      // where the smoothed axis disagrees with this frame (it lags a new curve or grade), the margin
      // grows by the disagreement: a lagging envelope must not take a wall inside
      if (cfg.fuse_inflate) {
        const double du = lat[g] - pm.lat[g], dv = vert[g] - pm.vert[g];
        su[g] = std::sqrt(su[g] * su[g] + du * du);
        sh[g] = std::sqrt(sh[g] * sh[g] + dv * dv);
      }
    }
    f_lat = lat, f_vert = vert, f_Plat = Plat, f_Pvert = Pvert;
    pm = PathModel(lat, vert, pm.phi, pm.support, su, sh, pm.info);
  }

  // ------------------------------------------------------------------ frame
  // Measured end of the lateral axis for the no-reversal rule; negative disables it.
  double lateral_support(const PathModel& pm) const { return cfg.path_no_reversal ? pm.support : -1.; }

  void continuous_axis(PathModel& pm,double ds,double dt,bool restart) {
    const int N=PathModel::N;
    const auto previous=path&&!restart?std::optional<PathModel>(path->shifted(ds,true)):std::nullopt;
    Vec target_u=pm.lat,target_h=pm.vert,wu(N),wh(N);
    const double alpha=1-std::exp(-std::max(dt,.001)/std::max(cfg.path_tau,.001));
    const double memory=(1-alpha)/std::max(alpha,1e-6);
    const double start=std::max(15.,pm.info.rail_support_m);
    // Fit current observations and transported history as one curve. The
    // near datum is weighted by rail uncertainty, rather than frozen exactly.
    for(int g=0;g<N;++g){
      wu[g]=1./std::pow(std::max(pm.sigma_u[g],.025),2);
      wh[g]=1./std::pow(std::max(pm.sigma_h[g],.035),2);
      if(previous&&g+ds<N-1){
        const double temporal=memory*std::clamp((g-start)/30.,0.,1.);
        // A previously extrapolated curve is not as strong as newly visible
        // rails/walls. Precision-weight the transported prior, so good fresh
        // geometry releases an uncertain history rather than sticking to it.
        // Frame fits share many structural returns, so their uncertainty is
        // correlated and can be overconfident. Limit a one-frame release of
        // history; sustained good observations still take over quickly.
        const double pu=temporal*std::max(.5*wu[g],1./std::pow(std::max(interp_grid(g+ds,path->sigma_u),.025),2));
        const double ph=temporal*std::max(.5*wh[g],1./std::pow(std::max(interp_grid(g+ds,path->sigma_h),.035),2));
        target_u[g]=(wu[g]*target_u[g]+pu*previous->lat[g])/(wu[g]+pu);
        target_h[g]=(wh[g]*target_h[g]+ph*previous->vert[g])/(wh[g]+ph);
        wu[g]+=pu;wh[g]+=ph;
      }
    }
    Vec lat=continuous_path_fit(target_u,wu,cfg.path_spatial_lambda,cfg.path_min_radius,lateral_support(pm),cfg.path_tail_m,(cfg.path_reversal_parts&2)&&pm.info.extension_ok,cfg.path_extension_decay_m,(cfg.path_reversal_parts&1)?0.:cfg.path_far_weight);
    Vec vert=continuous_path_fit(target_h,wh,cfg.path_spatial_lambda,cfg.path_vertical_radius);
    Vec su=pm.sigma_u,sh=pm.sigma_h;
    for(int g=0;g<N;++g){
      // Penalize disagreement with fresh geometry as well as measurement
      // scatter. A smoother line must not gain confidence by ignoring data.
      su[g]=std::hypot(su[g],lat[g]-pm.lat[g]);
      sh[g]=std::hypot(sh[g],vert[g]-pm.vert[g]);
    }
    pm=PathModel(lat,vert,pm.phi,pm.support,su,sh,pm.info);
  }

  void spatial_axis(PathModel& pm,double ds,double dt,bool restart) {
    // Port the established adaptive display filter into the actual geometry.
    // Release large innovations, smooth small noise; then jointly constrain the
    // full curve so node-wise release cannot introduce a spatial side-step.
    Vec target_u=pm.lat,target_h=pm.vert,wu(PathModel::N),wh(PathModel::N);
    double phi=pm.phi;
    const double base=1.-std::exp(-std::max(dt,.001)/std::max(cfg.path_tau,.001));
    auto gain=[&](double residual,double sigma=0.){const double lo=std::max(.10,cfg.path_release_k*sigma),hi=std::max(.50,lo+.40);const double t=std::clamp((residual-lo)/(hi-lo),0.,1.);return base+(1.-base)*t*t*(3.-2*t);};
    if(path&&!restart){
      const auto previous=path->shifted(ds,true);
      for(int g=0;g<PathModel::N;++g)if(g+ds<PathModel::N-1){
        double a=gain(std::hypot(pm.lat[g]-previous.lat[g],pm.vert[g]-previous.vert[g]),pm.sigma_u[g]);
        target_u[g]=previous.lat[g]+a*(pm.lat[g]-previous.lat[g]);
        target_h[g]=previous.vert[g]+a*(pm.vert[g]-previous.vert[g]);
      }
      phi=path->phi+gain(cfg.height*std::abs(pm.phi-path->phi))*(pm.phi-path->phi);
    }
    for(int g=0;g<PathModel::N;++g){wu[g]=1./std::pow(std::max(pm.sigma_u[g],.025),2);wh[g]=1./std::pow(std::max(pm.sigma_h[g],.035),2);}
    Vec lat=continuous_path_fit(target_u,wu,cfg.path_spatial_lambda,cfg.path_min_radius,lateral_support(pm),cfg.path_tail_m,(cfg.path_reversal_parts&2)&&pm.info.extension_ok,cfg.path_extension_decay_m,(cfg.path_reversal_parts&1)?0.:cfg.path_far_weight);
    Vec vert=continuous_path_fit(target_h,wh,cfg.path_spatial_lambda,cfg.path_vertical_radius);
    Vec su=pm.sigma_u,sh=pm.sigma_h;
    for(int g=0;g<PathModel::N;++g){su[g]=std::hypot(su[g],lat[g]-pm.lat[g]);sh[g]=std::hypot(sh[g],std::hypot(vert[g]-pm.vert[g],cfg.half_width*(phi-pm.phi)));}
    pm=PathModel(lat,vert,phi,pm.support,su,sh,pm.info);
  }

  // Beyond the end of the supported axis nothing is measured, and the smoothing filters would let the axis relax towards
  // straight ahead whatever the track did before. Here it goes on from its last point along its last direction with the
  // curvature the last tail_window_m showed: the track entering a bend keeps bending, one on a straight stays straight.
  void continue_axis(PathModel& pm) const {
    const int N = PathModel::N;
    if (cfg.tail_window_m <= 0.) return;
    const int end = static_cast<int>(std::floor(pm.support));
    if (end < 30 || end > N - 8) return;
    const int begin = std::max(10, end - static_cast<int>(cfg.tail_window_m));
    if (end - begin < 12) return;
    // least-squares parabola y = y0 + b t + c t^2 / 2 on the tail, t counted from its end
    double s[5] = {0., 0., 0., 0., 0.}, r[3] = {0., 0., 0.};
    for (int g = begin; g <= end; ++g) {
      const double t = g - end, y = pm.lat[g];
      double p = 1.;
      for (int k = 0; k < 5; ++k) s[k] += p, p *= t;
      r[0] += y, r[1] += y * t, r[2] += y * t * t;
    }
    double a[3][4] = {{s[0], s[1], s[2], r[0]}, {s[1], s[2], s[3], r[1]}, {s[2], s[3], s[4], r[2]}};
    for (int i = 0; i < 3; ++i) {
      int piv = i;
      for (int k = i + 1; k < 3; ++k) if (std::abs(a[k][i]) > std::abs(a[piv][i])) piv = k;
      if (std::abs(a[piv][i]) < 1e-9) return;
      for (int j = 0; j < 4; ++j) std::swap(a[i][j], a[piv][j]);
      for (int k = 0; k < 3; ++k) {
        if (k == i) continue;
        const double f = a[k][i] / a[i][i];
        for (int j = i; j < 4; ++j) a[k][j] -= f * a[i][j];
      }
    }
    const double kappa = 2. * a[2][3] / a[2][2], slope_fit = a[1][3] / a[1][1];
    const double excess = std::abs(kappa) - cfg.tail_curvature_floor;
    const double c = excess > 0. ? std::copysign(std::min(excess, cfg.tail_curvature_max), kappa) : 0.;
    // Direction at the end point: that of the axis itself (a line through its last 14 m, carried to the end along the bend).
    const int near = std::max(begin, end - 14);
    double sn = 0., st = 0., sy = 0., syt = 0., stt = 0.;
    for (int g = near; g <= end; ++g) {
      const double t = g - end, y = pm.lat[g];
      sn += 1., st += t, sy += y, syt += y * t, stt += t * t;
    }
    const double den = stt - st * st / sn;
    const double s_end = (den > 1e-9 ? (syt - st * sy / sn) / den : 0.) + c * (end - near) / 2.;
    // The direction settles over `settle` metres to the one of the tail fit, when that bends the same way as the track: no kink
    // where the continuation starts, and it never turns against the bend.
    const double s_far = (c != 0. && (s_end - slope_fit) * c < 0.) ? slope_fit : s_end, settle = 15.;
    const double y0 = pm.lat[end];
    Vec lat = pm.lat;
    for (int g = end + 1; g < N; ++g) {
      const double t = g - end;
      lat[g] = y0 + s_far * t + .5 * c * t * t + (s_end - s_far) * settle * (1. - std::exp(-t / settle));
    }
    pm = PathModel(lat, pm.vert, pm.phi, pm.support, pm.sigma_u, pm.sigma_h, pm.info);
  }

  void certify_far_path(PathModel& pm, const Cloud& registration) {
    current_certificate.reset();
    if (geometry_used < 3 || !pm.info.extension_raw) { certificates.clear(); return; }
    Cloud q;
    std::vector<int> ids;
    slab_points(registration, q, ids);
    const auto measured = register_slabs(pm, .1, cfg.geometry_min_range, cfg.max_range + 5., q, ids);
    if (!measured) { certificates.clear(); return; }
    double last = frame_measured_support, sum_du2 = 0., total = 0.;
    int left = 0, right = 0, count = 0;
    for (size_t j = 0; j < measured->x.size(); ++j) {
      const double x = measured->x[j];
      if (x <= frame_measured_support) continue;
      if (x >= pm.info.break_m) break;
      const double n = measured->n[j], inl = measured->inl[j];
      // Test the already measured straight continuation. A different best alignment does not
      // earn support and cannot pull the axis towards an obstacle or a neighbouring tunnel.
      if (inl < cfg.min_slab_inliers || inl / std::max(n, 1.) < cfg.min_explained || std::abs(measured->du[j]) > .2) continue;
      if (x - last > 35.) break;
      int l = 0, r = 0;
      for (const auto& p : q) {
        if (std::abs(p.x - x) >= 5.) continue;
        double s, u, h;
        pm.to_track(p, s, u, h);
        if (tmpl.score(u - measured->du[j], h - measured->dh[j]) < .6) continue;
        l += u < -cfg.half_width - .35;
        r += u > cfg.half_width + .35;
      }
      left += l, right += r;
      last = x;
      sum_du2 += inl * measured->du[j] * measured->du[j];
      total += inl;
      ++count;
    }
    // Bilateral walls and several longitudinal sections are needed; a lone reflector is not
    // a measurement of the track frame. Height uncertainty keeps its conservative frame value.
    if (count >= 3 && left >= 6 && right >= 6 && last > frame_measured_support + 20.) {
      PathModel observed = pm;
      observed.support = last;
      certificates.push_back({std::move(observed), geometry_s, std::sqrt(sum_du2 / std::max(total, 1.)), frame});
    }
    while (!certificates.empty() && frame - certificates.front().frame >= cfg.geometry_frames) certificates.pop_front();
    for (const auto& c : certificates) {
      const double travel = geometry_s - c.S;
      const double end = c.path.support - travel;
      if (pm.info.break_m < end) continue;
      if (std::abs(travel) > cfg.geometry_max_step * cfg.geometry_frames) continue;
      const PathModel moved = c.path.shifted(travel,cfg.continuous_path);
      if (std::abs(pm.c(end) - moved.c(end)) > .35) continue;
      if (!current_certificate || end > current_certificate->path.support - (geometry_s - current_certificate->S)) current_certificate = c;
    }
    if (current_certificate) {
      pm.support = pm.info.support_m = std::max(pm.support, current_certificate->path.support - (geometry_s - current_certificate->S));
      pm.info.good_slabs += count;
    }
  }

  void certificate_uncertainty(PathModel& pm) const {
    if (!current_certificate) return;
    const auto& cert = *current_certificate;
    const double travel = geometry_s - cert.S;
    const PathModel moved = cert.path.shifted(travel,cfg.continuous_path);
    for (int g = static_cast<int>(cfg.geometry_min_range); g < PathModel::N; ++g) {
      // The certificate adds observations beyond the current measured support; it must not
      // erase the original local residual inside already measured sections.
      if (g <= frame_measured_support) continue;
      const double base = cfg.sigma_lat0 + cfg.sigma_lat_per_m * std::min(g, 100) +
                          cfg.sigma_lat_per_m_far * std::max(0, g - 100);
      const double disagreement = pm.lat[g] - moved.lat[g];
      pm.sigma_u[g] = std::sqrt(base * base + cert.residual * cert.residual + disagreement * disagreement) +
                      cfg.extension_sigma_per_m * std::max(0., g - moved.support);
    }
  }

  Cloud registration_cloud(const Cloud& p, const PathModel& near_path, std::optional<double> ds, bool restart) {
    geometry_used = 1;
    const bool slow = ds && std::abs(*ds) <= cfg.geometry_max_step;
    if (restart || !slow || cfg.geometry_frames <= 1) geometry_history.clear(), certificates.clear();
    geometry_s += ds.value_or(0.);
    Cloud combined = p;
    for (const auto& past : geometry_history) {
      const double travel = geometry_s - past.S;
      // Carry only the measured travel and curvature. A noisy rail fit is not a measurement
      // of sensor yaw/pitch: using its frame-to-frame slope would move far walls by metres.
      const double dy = past.near.c(0.) - past.near.c(travel);
      const double dz = past.near.r(0.) - past.near.r(travel);
      const double yaw = past.near.dc(0.) - past.near.dc(travel);
      const double pitch = past.near.r(1.) - past.near.r(0.) - (past.near.r(travel + 1.) - past.near.r(travel));
      for (const auto& q : past.p) {
        const double x = q.x - travel;
        const double y = q.y + dy + yaw * x;
        const double z = q.z + dz + pitch * x;
        combined.push_back({static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)});
      }
      ++geometry_used;
    }
    if (cfg.geometry_frames > 1 && cfg.geometry_points > 0 && (slow || restart)) {
      Cloud far;
      for (const auto& q : p) {
        if (q.x < cfg.geometry_min_range || q.x >= cfg.slab_edges.back()) continue;
        double s, u, h;
        near_path.to_track(q, s, u, h);
        if (std::abs(u) < cfg.half_width + .35 && h > .05 && h < cfg.height + .3) continue;
        if (std::abs(u) < 9.8 && h > -1.45 && h < 6.45) far.push_back(q);
      }
      if (far.size() > static_cast<size_t>(cfg.geometry_points)) {
        Cloud sampled;
        for (int k = 0; k < cfg.geometry_points; ++k) sampled.push_back(far[static_cast<size_t>(k) * far.size() / cfg.geometry_points]);
        far = std::move(sampled);
      }
      geometry_history.push_back({std::move(far), near_path, geometry_s});
      while (geometry_history.size() >= static_cast<size_t>(cfg.geometry_frames)) geometry_history.pop_front();
    }
    return combined;
  }

  std::string process(const float* xyz, size_t npts, double timestamp, bool has_ts) {
    const auto tic = Clock::now();
    ++frame;
    std::optional<double> dt;
    if (has_ts && last_time) {
      dt = timestamp - *last_time;
      if (*dt <= 0 || *dt > cfg.max_gap_s) {
        reset();
        frame = 0;
        dt.reset();
      }
    }
    if (has_ts) last_time = timestamp;
    else last_time.reset();
    // clock for time windows; frames may be skipped when the computer is busy
    now = has_ts ? timestamp : frame * cfg.frame_period_s;
    Cloud p;
    p.reserve(npts);
    for (size_t k = 0; k < npts; ++k) {
      const float X = xyz[3 * k], Y = xyz[3 * k + 1], Z = xyz[3 * k + 2];
      // the height window opens with range: a pitched sensor (Seyond Falcon, ~2.5 deg) sees the
      // track 10 m below its horizon at 200 m
      const float slope = .12f * X;
      if (X > .5 && X < cfg.max_range + 25 && std::abs(Y) < 45 && Z > -6 - slope && Z < 10 + slope) p.push_back({X, Y, Z});
    }
    std::vector<std::pair<const char*, double>> timing;
    // 1) odometry on the previous path (wall depth images)
    std::optional<double> ds;
    double oq = 0.;
    const bool had_path = path.has_value();
    if (had_path) {
      double d, q;
      if (odometry.update(p, *est, dt.has_value(), dt.value_or(0.), d, q)) ds = d, oq = q;
      if (miss) oq = 0.;
      if (!ds && odometry.has_v && dt && *dt != 0.) ds = odometry.v * *dt;
    }
    timing.push_back({"odometry_ms", ms_since(tic)});
    PathModel prior = prior_for(ds, p);
    // 2) rails: with the moved previous path, then with a neutral prior
    const auto t1 = Clock::now();
    std::optional<NearFit> near_fit = near.fit(p, prior);
    bool reacquired = false;
    if (!near_fit && path) {
      PathModel neutral = PathModel::flat(rail_height_guess(p));
      near_fit = near.fit(p, neutral);
      if (near_fit) {
        prior = neutral;
        reacquired = true;
        tmpl.reset();
      }
    }
    if (!near_fit) {
      ++miss;
      if (!path || miss > cfg.max_coast_frames) {
        reset();
        return empty("Не найдены головки рельсов вблизи поезда", tic);
      }
    } else {
      miss = 0;
    }
    timing.push_back({"rails_ms", ms_since(t1)});
    // provisional near path for the template (only from fresh rails)
    const auto t2 = Clock::now();
    const bool first = !path || reacquired;
    PathModel prov = smooth_path(prior, near_fit, nullptr, first);
    if (near_fit) {
      Vec sn, un, hn;
      size_t count = 0;
      for (const P3& q : p) {
        if (!(q.x > cfg.template_near[0] && q.x < cfg.template_near[1])) continue;
        if (count++ % 3) continue;
        double s, u, h;
        prov.to_track(q, s, u, h);
        sn.push_back(s), un.push_back(u), hn.push_back(h);
      }
      tmpl.update(un, hn);
      update_surface_ref(sn, un, hn);
    }
    timing.push_back({"template_ms", ms_since(t2)});
    // 3) far registration: each chunk is searched around the path fitted from all nearer chunks
    const auto t3 = Clock::now();
    PathModel pm = prov;
    std::optional<Slabs> slabs;
    Cloud pre_q;
    std::vector<int> pre_slab;
    const Cloud registration = registration_cloud(p, prov, ds, first || !near_fit);
    slab_points(p, pre_q, pre_slab);
    for (size_t c = 0; c + 1 < cfg.chunk_edges.size(); ++c) {
      auto sl = register_slabs(pm, first ? 1.4 : 1., cfg.chunk_edges[c], cfg.chunk_edges[c + 1], pre_q, pre_slab);
      if (!sl || sl->x.empty()) continue;
      if (!slabs) slabs = *sl;
      else slabs->append(*sl);
      pm = smooth_path(prior, near_fit, &*slabs, first, true);
    }
    if (slabs) pm = smooth_path(prior, near_fit, &*slabs, first);
    if (cfg.explain) debug_slabs = slabs;
    timing.push_back({"register_ms", ms_since(t3)});
    if (ds) S_ego += *ds;
    cur_ds = ds;
    standing = (ds && std::abs(*ds) < cfg.still_ds) ? standing + 1 : 0;
    if (!near_fit) {
      // coasting on odometry: the rail datum is not re-observed
      pm.info.has_coasting = true;
      pm.info.coasting_frames = miss;
      pm.support = std::min(pm.support, 40.);
      for (int g = 0; g < PathModel::N; ++g) {
        pm.sigma_h[g] += cfg.coast_sigma_h * miss;
        pm.sigma_u[g] += cfg.coast_sigma_u * miss;
      }
    }
    // the estimate feeds the next frame unchanged; holding and fusion only shape the output, so
    // they cannot pull the next registration off the data
    est = pm;
    support_recent = std::max(pm.support, support_recent - ds.value_or(0.) - cfg.slab_edge_memory_m);
    frame_measured_support = pm.support;
    pm.info.support_raw_m = pm.support;
    pm.info.extension_raw = pm.info.extension_ok;
    dbg_raw = pm.lat;
    if (had_path && !reacquired) hold_support(pm, *path, ds.value_or(0.));
    if(cfg.continuous_path){
      continuous_axis(pm,ds.value_or(0.),dt.value_or(cfg.frame_period_s),!had_path||reacquired);
      dbg_fused=dbg_spatial=pm.lat;
    }
    else {
      fuse_path(pm, ds.value_or(0.), !had_path || reacquired);
      dbg_fused = pm.lat;
      if(cfg.spatial_path)spatial_axis(pm,ds.value_or(0.),dt.value_or(cfg.frame_period_s),!had_path);
      dbg_spatial = pm.lat;
    }
    // Validate the actual output axis against structural returns; certifying a different,
    // noisy pre-fusion line would add its disagreement back as fictitious range uncertainty.
    continue_axis(pm);
    certify_far_path(pm, registration);
    certificate_uncertainty(pm);
    path = pm;
    if (!had_path) odometry.prime(p, *est);
    // 4) envelope membership
    const auto t4 = Clock::now();
    Corridor cor = corridor(p, *path);
    std::vector<Candidate> cands = clusters(cor, *path);
    timing.push_back({"clusters_ms", ms_since(t4)});
    // 5) tracking and decision
    // the envelope is drawn to `envelope` (a held length keeps it steady); decisions and the
    // verified range use only what this frame itself supports
    const double envelope = path->info.has_analysis_limit
                                ? path->info.analysis_limit_m
                                : std::min(cfg.max_range, interp_grid(std::max(0., path->support - cfg.support_buffer), path->arc));
    const double analysis = path->info.has_analysis_limit ? path->info.decision_limit_m : envelope;
    cloud_now = &p;
    track(cands, analysis);
    cloud_now = nullptr;
    const double drawn = draw_envelope(envelope, ds, path->info.break_m);
    const PathModel shown = display_path(*path, ds, drawn);
    return result(*path, shown, cands, near_fit.has_value(), analysis, drawn, ds, oq, timing, tic, npts, p.size());
  }
};

Detector::Detector(const Config& config) : impl_(std::make_unique<Impl>(config)) {}
Detector::~Detector() = default;
void Detector::reset() { impl_->reset(); }
std::string Detector::process(const float* xyz, size_t n, double timestamp, bool has_timestamp) {
  return impl_->process(xyz, n, timestamp, has_timestamp);
}

}  // namespace metro
