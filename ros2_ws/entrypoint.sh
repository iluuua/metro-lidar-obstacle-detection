#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /opt/metro_ws/install/setup.bash
# NumPy work is element-wise; BLAS thread pools only compete with ROS for cores
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
exec "$@"
