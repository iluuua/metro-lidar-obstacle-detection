cmake_minimum_required(VERSION 3.16)
project(metro_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_library(metro_core SHARED src/detector.cpp src/geometry.cpp src/c_api.cpp src/cloud.cpp)
target_include_directories(metro_core PUBLIC include PRIVATE src)
target_compile_options(metro_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
set_target_properties(metro_core PROPERTIES CXX_VISIBILITY_PRESET default)

# The same core with the axis grid to 430 m, loaded by metro.native for a reach beyond 325 m (the Falcon).
add_library(metro_core_long SHARED src/detector.cpp src/geometry.cpp src/c_api.cpp src/cloud.cpp)
target_include_directories(metro_core_long PUBLIC include PRIVATE src)
target_compile_options(metro_core_long PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_compile_definitions(metro_core_long PRIVATE METRO_GRID_N=431)
set_target_properties(metro_core_long PROPERTIES CXX_VISIBILITY_PRESET default)

install(TARGETS metro_core metro_core_long LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

include(CTest)
if(BUILD_TESTING)
  add_executable(path_fit_test tests/path_fit_test.cpp src/geometry.cpp)
  target_include_directories(path_fit_test PRIVATE src include)
  add_test(NAME continuous_path_fit COMMAND path_fit_test)
endif()

# Garbage clouds and payloads under AddressSanitizer and UBSan: no crash, no memory error.
option(METRO_FUZZ "Build the sanitized fuzz driver core/tools/fuzz.cpp" OFF)
if(METRO_FUZZ)
  add_executable(metro_fuzz tools/fuzz.cpp src/detector.cpp src/geometry.cpp src/cloud.cpp)
  target_include_directories(metro_fuzz PRIVATE include src)
  target_compile_options(metro_fuzz PRIVATE -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined)
  target_link_options(metro_fuzz PRIVATE -fsanitize=address,undefined)
endif()
