// Tunnel-profile obstacle detector (C++ core of metro/tunnel.py).
//
// A metro tunnel is an extruded cross-section around the track. Near the train
// the rail pair fixes the track frame; the near cross-section becomes a template
// of the normal tunnel; far slabs are registered to it and a smooth centre line
// through them carries the clearance envelope (2.1 x 3.0 m from the rail heads).
// Points inside the envelope that the normal tunnel cannot explain are evidence,
// accumulated per world position along the track.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace metro {

struct Config {
  // Envelope dimensions are relative to the rail heads and normal to the centre line.
  double half_width = 1.05;
  double height = 3.0;
  double min_range = 1.0;
  // The Python pipeline selects the long-range core for Falcon clouds.
  double max_range = 300.0;
  double long_range = 400.0;
  // Track / sensor priors.
  double gauge = 1.59;
  double rail_z_prior = -1.12;
  double rail_over_bed = .25;
  double rail_max_x = 45.0;
  // Far registration.
  double template_decay = .75;
  std::vector<double> template_near{4.0, 32.0};
  std::vector<double> slab_edges;  // filled in the constructor
  int slab_points = 260;
  std::vector<double> chunk_edges{0., 50., 80., 120., 180., 310.};
  double min_slab_inliers = 6.;
  double max_ambiguity = .98;
  double min_explained = .7;
  double slab_edge_frac = .9;  // a slab whose best lateral shift is this share of the search half-width or more sits on the edge of the search window (>1: rule off)
  double slab_edge_margin_m = 5.;   // such a slab is believed only farther than the recently verified range plus this
  double slab_edge_memory_m = 1.;   // that range is forgotten by the travel and this much more per frame
  double break_explained = .6;
  double confirm_explained = .45;       // a slab explained this much, on the predicted axis and with an empty envelope, is another cross-section of the same tunnel (a platform), not a changed one
  double confirm_du = .3;               // how far (m) such a slab may sit from the predicted axis
  double confirm_free_points = 15.;     // and how many points it may have inside the predicted envelope
  double break_bridge_m = 15.;          // an unexplained slab is no break when slabs back on the axis (confirmed, nothing in the envelope) follow within this many metres (0: off)
  double slab_amb_sigma_k = 0.;        // m of added axis uncertainty per unit of ambiguity above slab_amb_free (0 = off): slabs that fit about as well at other shifts scatter
  double slab_amb_free = .85;
  int slab_scatter_weighted = 0;       // 1: the scatter of the slabs around the axis (its uncertainty, where the support ends) is weighed like the fit, not by the points
  double slab_weight_power = 1.;       // a slab weighs (1 - ambiguity)^power: where the profile fits about as well at other shifts its shift says little
  double slab_weight_floor = .1;       // the lowest (1 - ambiguity) used
  double confirm_du_per_m = 0.;         // the allowed offset grows with range by this much per metre
  // Uncertainty (1 sigma) used for margin-aware decisions.
  double sigma_lat0 = .02;
  double sigma_lat_per_m = .003;
  double sigma_lat_per_m_far = .0015;
  double sigma_lat_per_m_rail = .003;   // the same within the range where the rails are seen
  double sigma_vert_per_m_rail = .002;
  double sigma_vert0 = .03;
  double sigma_vert_per_m = .002;
  double sigma_extrap_per_m = .012;
  double margin_k = 1.25;
  double candidate_margin_m = 0.;  // opt-in observation band, never expands the nominal clearance
  double candidate_margin_sigma = 2.;
  double candidate_sigma_floor = .06;
  bool hold_observed_geometry = false;  // opt-in: short holds use current matched returns
  double max_sigma_support = .35;
  double support_buffer = 5.;
  // Straight extension beyond the supported geometry into empty space.
  bool extension = true;
  double extension_min_support = 80.;
  double extension_max_curvature = 1. / 3000.;
  double extension_sigma_per_m = .004;
  double extension_corridor = 2.0;
  double min_height = .07;
  double rail_exclusion = .07;
  // Clusters and confirmation.
  double voxel_s = .5;
  double voxel_uh = .25;
  int strong_points = 6;
  int strong_hits = 2;
  int weak_hits = 4;
  int window = 8;
  int hold_frames = 3;
  double max_hold_s = 15.;
  double hold_min_range = 10.;
  double hold_min_travel = 10.;
  // Objects that appear while the train stands still.
  double still_ds = .01;
  double change_voxel = .2;
  double change_ref_s = 2.;
  double change_min_age_s = .4;
  double change_occupancy = .6;
  int change_persist = 8;
  int change_window = 12;
  double change_new_frac = .5;
  // Moving objects.
  int move_min_obs = 4;
  int move_strong = 3;
  double move_evidence_k = 2.;
  double move_resid = .3;
  double move_min_speed = .5;
  double move_max_speed = 30.;
  double move_rel_min = 1.;
  double move_hold_s = 2.;
  double gate_s = 1.6;
  double static_tol = 2.0;
  double wall_extent_s = 8.;
  double wall_extent_uh = 1.4;
  double wall_clear_center = .4;
  double wall_end_margin = 10.;
  double gate_h = .6;
  double evidence_near = 8.;
  double evidence_per_m = .16;
  double frame_period_s = .1;
  int max_coast_frames = 5;
  double max_gap_s = 3.;
  double coast_sigma_h = .06;
  double coast_sigma_u = .03;
  double temporal_weight = 1.;
  // the previous axis gives way where this frame's data disagree with it by more than this (0 = never)
  double prior_tol_lat = .25;
  double prior_tol_vert = .15;
  // Opt-in joint spatial/temporal fit used by the actual detector, not just display.
  bool continuous_path = false;
  bool spatial_path = true;  // preserve legacy measurement/fusion, then join the whole output curve into one smooth axis
  double path_min_radius = 100.;  // configurable conservative prior; does not classify track type
  double path_vertical_radius = 1000.;
  double path_tau = .35;  // memory of the axis filter (the joint continuous axis uses .65)
  double path_spatial_lambda = 2e6;  // penalty on change of curvature, not curvature itself
  // Beyond the measured support the lateral axis may not bend the other way than its measured tail
  // (nothing there can show a reverse bend), only relax to straight. Affects continuous/spatial path.
  // A held box (a confirmed obstacle not seen in this frame) must still have points where it should be, at a
  // range where its own cluster would give some; otherwise the object has left and the box is dropped.
  // Confirmed boxes of one object (fragments of a long object, a held box beside a fresh one) are reported as one.
  bool merge_boxes = true;
  double merge_gap_m = 1.;
  bool hold_check = true;
  // A track confirmed by strong evidence (not by the edge rules or the continuous evidence, which are
  // close calls) is carried over longer gaps; its box is still dropped once it is empty (hold_check).
  int hold_frames_strong = 12;
  int hold_empty_frames = 3;
  double hold_expect_points = 6.;
  int hold_slab_points = 12;  // fewer returns than this around the box at its range: a dropout, the box cannot be judged
  double hold_near_m = 30.;  // closer than this a real object always gives returns: an empty held box is gone
  // The drawn envelope length: a drop is shown only once it persists, then gradually (decisions keep their own range).
  bool envelope_hold = true;
  int envelope_hold_frames = 12;  // a shorter envelope is drawn only once the last N frames all agree
  int envelope_rise_frames = 3;    // a longer envelope is drawn once the last N frames all support it
  // A new confirmation needs the verified range to have held beyond the object for the last N frames: where the range
  // flickers (a tunnel section being registered in and out) a structure at its end is not told from an obstacle.
  int confirm_range_frames = 5;
  double confirm_range_tol_m = 0.;
  int envelope_end_frames = 3;     // a visible end of the normal tunnel seen in N frames in a row cuts the envelope at once (0: only by the hold)
  double envelope_max_drop = 8.;  // m per frame beyond the travel
  double envelope_max_rise = 30.;  // a longer envelope grows by at most this per frame
  // Far nodes of the axis are noisier, so the adaptive axis filter lets a change through only beyond this many sigmas.
  double path_release_k = 0.;
  // Parts of the no-reversal rule, as bits: 1 observations beyond the support are ignored, 2 a path declared
  // straight continues exactly straight, 4 one-sided curvature limits. All three: the strictest form.
  int path_reversal_parts = 4;
  double path_far_weight = 1.;  // weight of the observations beyond the measured support in the axis fit (1 = as measured)
  bool path_no_reversal = true;
  double path_tail_m = 30.;  // length of the measured tail whose mean curvature the extension may not exceed
  double path_extension_decay_m = 100.;  // that curvature fades with this e-folding length beyond the support (0 = constant)
  // Beyond the end of the supported axis the output axis goes on from the end point along the end tangent and the curvature
  // of the last tail_window_m of the supported axis (a circular arc), not straight ahead. Curvature below the noise floor counts as
  // none, and what exceeds it is capped (tail_window_m = 0: off).
  double tail_window_m = 80.;
  double tail_curvature_floor = 1. / 3000.;
  double tail_curvature_max = 1. / 150.;
  // Temporal fusion of the far axis: per-node Kalman filter over frames (stable envelope end,
  // smaller uncertainty where the geometry is observed again and again).
  bool fuse = true;
  double fuse_q_abs = .01;       // m per frame: how much the far axis may change by itself
  double fuse_q_rel = .1;        // ... plus this share of the travel since the last frame
  double fuse_q_rel_vert = .3;   // the same for the height profile: the floor must not lag behind a grade change
  double fuse_damp_min = 1.;     // innovation-adaptive gain: an innovation within fuse_damp_lo sigmas (jitter) gets this share of the gain, from fuse_damp_hi sigmas (a real change) all of it; 1 = off
  double fuse_damp_lo = .5;
  double fuse_damp_hi = 2.;
  double fuse_gate = 10.;        // innovation beyond this many sigmas restarts the node
  double fuse_floor0 = .03;      // m: bias floor of the fused uncertainty
  double fuse_floor_per_m = .0012;
  double fuse_start = 10.;       // m beyond the rail support where fusion starts
  double fuse_ramp = 20.;        // m over which it takes over from the per-frame estimate
  // When the fused uncertainty may replace the per-frame one: 0 never, 1 always, 2 only while the
  // train stands (the scene repeats exactly; moving, the per-frame errors are correlated).
  int fuse_sigma = 0;
  int fuse_still_frames = 10;
  bool fuse_extrap_floor = true;   // beyond the support the node uncertainty never drops below the frame's: a curve entered later is not held back by a straight guess repeated for many frames
  double fuse_extrap_r = 4.;        // the frame's continuation beyond its support weighs this much less
  double fuse_extrap_floor_k = .25;  // with fuse_extrap_floor: that floor is this share of the frame's own uncertainty
  bool fuse_inflate = true;      // the margin grows by the disagreement of the smoothed axis with the frame
  // The envelope length does not flicker: a shorter support than the previous one (moved by the
  // travel) is accepted after this many frames in a row, or at once when the tunnel visibly changes.
  int hold_support_frames = 5;
  double hold_support_tol = 5.;
  double structure_memory_s = 1.5;  // structure seen in the straight corridor is kept this long
  // ATTENTION for a compact object seen only within the margin at the envelope edge
  // ATTENTION that keeps its place: the same object under ATTENTION for this many frames (gaps up to attention_gap_frames)
  // raises the stronger level (attention_level 2). 0: off.
  double extension_break_margin_m = 10.;  // ...and ends this far before it: the slab that is no longer explained (10 m wide far out) begins before its centre, and what it holds is the change itself
  int extension_stops_at_break = 1;     // the straight continuation is verified no farther than a changed tunnel (break) ahead
  int attention_strong_frames = 15;
  int attention_gap_frames = 2;
  int boundary_points = 8;
  int boundary_hits = 3;
  bool edge_confirmation = true;  // earlier confirmation with independent confident returns
  double boundary_max_len = 3.;
  double boundary_min_top = .3;      // rises above the track bed (the bed itself sits in the lower margin)
  double boundary_max_range = 150.;  // attention only; sparse boundary evidence never confirms an alarm
  // Dust: within this range a cluster of at most so many distinct spots, all within this size,
  // is a speck and barely counts as evidence.
  double speck_range = 20.;
  int speck_points = 16;  // a tiny cluster (extent below speck_size) of up to this many distinct returns is dust
  double speck_size = .05;
  // Independent far observations accumulate before a per-frame cluster is strong enough.
  bool farfield = true;
  int far_window = 15;
  int far_attention_hits = 6;
  int far_confirm_hits = 7;
  int far_min_points = 8;
  double far_min_range = 120.;
  double far_max_sigma_u = .9;  // also require mean lateral clearance >= sigma_u
  // Only slow, geometrically aligned history contributes to the distant tunnel fit.
  int geometry_frames = 6;
  int geometry_points = 8000;
  double geometry_min_range = 120.;
  double geometry_max_step = .05;
  // Drawn envelope (JSON path and section_frames, RViz): small jitter is damped, real changes pass.
  bool display_smoothing = true;
  double display_tau_s = .35;
  double display_release = .10;  // m: below this the change is treated as jitter
  double display_snap = .50;     // m: from this the new axis is drawn at once
  double display_max_gap_s = .5;
  // The smoothed drawing is joined again into one curve with the curvature bounded by path_min_radius
  // (core/src/path_fit.hpp): node-wise blending would otherwise bring back side-steps. Drawing only;
  // with equal weights the curvature bound alone removes the kinks.
  bool display_join = true;
  double display_join_sigma = .025;
  // The drawn axis only (decisions use the axis before it): far from the train, where the estimate scatters from frame to frame,
  // small changes are damped by a memory that grows with range, a change beyond the scatter of the estimate passes (like the
  // stabilizer of a phone). The thresholds are multiples of the axis uncertainty, so a noisier lidar is damped more.
  bool display_far_smoothing = true;
  double display_far_from = 100.;     // m: nothing is added nearer
  double display_far_ramp = 100.;     // m: the memory reaches display_far_tau_max this far beyond display_far_from
  double display_far_support_tau_s = .05;  // s: memory beyond the measured support (the continuation is no measurement but a longer memory there bent it)
  double display_far_tau_max = 1.5;   // s
  double display_far_release_k = .5;  // a change below this many sigmas is jitter
  double display_far_pass_k = 1.;     // ... and one of this many sigmas more passes at once
  double display_far_gain_half_m = 30.;  // m: the gains are smoothed along the axis over this half-width (a gain that jumps from node to node bends the curve)
  double display_far_pass_m = .3;     // ... at least this many metres more
  double display_far_tau_s = 3.;  // s: memory of the drawn axis beyond the measured support (ramped over 20 m)
  double display_join_lambda = 1e4;  // weak: the curvature bound removes the kinks, the curve keeps to the drawing
  // Continuous evidence of intrusion for objects at the envelope edge (see soft_evidence).
  bool soft_confirmation = true;
  double soft_tau_s = 2.;
  double soft_reference = .55;
  double soft_threshold = .25;
  int soft_min_hits = 3;
  bool explain = false;  // optional decision diagnostics; does not change the normal JSON contract

  Config();
  // Overrides from a flat JSON object with the field names above; unknown keys throw.
  void update_from_json(const std::string& json);
};

class Detector {
 public:
  explicit Detector(const Config& config = Config());
  ~Detector();
  Detector(const Detector&) = delete;
  Detector& operator=(const Detector&) = delete;

  void reset();
  // xyz: n points (x forward, y left, z up), float32 triples. Returns the result as JSON.
  std::string process(const float* xyz, size_t n, double timestamp, bool has_timestamp = true);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace metro
