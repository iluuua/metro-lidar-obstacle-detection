cmake_minimum_required(VERSION 3.8)
project(metro_perception)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
# One shared implementation for offline research and the ROS worker.
ament_python_install_package(metro PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../metro")
# The C++ core of the detector: libmetro_core in <prefix>/lib, loaded by metro.native.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../core" core)
install(PROGRAMS scripts/metro_obstacle_detector scripts/metro_status_echo DESTINATION lib/${PROJECT_NAME})
ament_package()
