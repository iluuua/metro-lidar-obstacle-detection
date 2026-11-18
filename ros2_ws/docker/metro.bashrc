source /opt/ros/humble/setup.bash
source /opt/metro_ws/install/setup.bash
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
PS1='metro:\w\$ '

# AUTOPLAY=/data/<bag or number> [PLAY_RATE=0.5] [PLAY_ARGS=--loop]: start playing once the detector is up (see run.sh)
if [ -n "${AUTOPLAY:-}" ] && [ -z "${METRO_AUTOPLAYED:-}" ]; then
  export METRO_AUTOPLAYED=1
  echo "Автозапуск записи $AUTOPLAY: жду детектор..."
  for _ in $(seq 40); do ros2 node list 2>/dev/null | grep -q metro_obstacle_detector && break; sleep 1; done
  play "$AUTOPLAY" "${PLAY_RATE:-1.0}" ${PLAY_ARGS:-}
fi
