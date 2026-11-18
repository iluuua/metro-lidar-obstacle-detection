FROM ros:humble-ros-base-jammy
# RViz runs inside the container: on the host's X server (-e DISPLAY) or on a virtual screen
# shown in the browser through noVNC (port 6080), see ros2_ws/docker/studio.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-numpy python3-scipy python3-colcon-common-extensions \
    ros-humble-ament-cmake-python ros-humble-rosidl-default-generators \
    ros-humble-sensor-msgs ros-humble-std-msgs ros-humble-std-srvs \
    ros-humble-geometry-msgs ros-humble-visualization-msgs ros-humble-diagnostic-msgs \
    ros-humble-tf2-ros ros-humble-launch-ros ros-humble-rosbag2-storage-default-plugins \
    ros-humble-rviz2 xvfb x11-utils x11vnc novnc websockify openbox \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY core /app/core
COPY metro /app/metro
COPY ros2_ws /app/ros2_ws
# A checkout on Windows turns line endings into CRLF and the shell scripts then fail (`set: -: invalid option`): strip them.
RUN find /app/ros2_ws /app/core /app/metro -type f -exec sed -i 's/\r$//' {} +
# metro_perception also builds the C++ core of the detector (libmetro_core)
RUN . /opt/ros/humble/setup.sh && colcon build --base-paths /app/ros2_ws/src \
    --build-base /opt/metro_ws/build --install-base /opt/metro_ws/install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && install -m 755 /app/ros2_ws/docker/studio /app/ros2_ws/docker/play /app/ros2_ws/docker/bags \
       /app/ros2_ws/docker/status /usr/local/bin/ \
    && install -m 644 /app/ros2_ws/docker/metro.bashrc /etc/metro.bashrc
ENV PYTHONUNBUFFERED=1
EXPOSE 6080
ENTRYPOINT ["/bin/bash", "/app/ros2_ws/entrypoint.sh"]
CMD ["ros2", "launch", "metro_bringup", "detector.launch.py"]
