cmake_minimum_required(VERSION 3.0.2)
project(tiny_vslam)
find_package( OpenCV REQUIRED )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES mono_node
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(mono_node src/mono_node.cc)

target_link_libraries(mono_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

include_directories(${PROJECT_SOURCE_DIR}/include)