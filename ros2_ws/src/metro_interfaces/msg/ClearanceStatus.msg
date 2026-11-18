uint8 UNKNOWN=0
uint8 CLEAR=1
uint8 CANDIDATE=2
uint8 OBSTACLE=3
# Sensor header stamp. For unavailability events: zero stamp, has_source_stamp=false.
std_msgs/Header header
bool has_source_stamp
string source_frame_id
uint8 state
bool valid
string reason
bool obstacle
# 0 none, 1 ATTENTION (state CANDIDATE), 2 stronger ATTENTION: the same object has been under ATTENTION for a long time.
# It is still not an alarm: obstacle stays false.
uint8 attention_level
bool has_distance
float32 distance_along_path_m
# Algorithm analysis extent, NOT a guarantee that all space is visible/free.
float32 analysis_range_m
# How far the envelope is drawn: over short dropouts of the far geometry it keeps its previous
# length, while decisions stay within analysis_range_m.
float32 envelope_range_m
string model_revision
uint64 generation
uint64 sequence
float32 receive_to_publish_ms
float32 compute_ms
float32 max_result_age_s
uint64 received_frames
uint64 superseded_frames
uint64 stale_results
uint64 worker_restarts
metro_interfaces/ObstacleCandidate[] candidates
