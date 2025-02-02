cmake_minimum_required(VERSION 3.5)
project(ur5_planning)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(localization_interfaces REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(ur5_motion_interfaces REQUIRED)

find_package(std_srvs REQUIRED)

include_directories(include)

add_executable(planning1 src/planning_1.cpp)
ament_target_dependencies(planning1 rclcpp localization_interfaces tf2
tf2_ros
tf2_geometry_msgs
geometry_msgs
ur5_motion_interfaces
std_srvs
)

install(TARGETS
  planning1
  DESTINATION lib/${PROJECT_NAME})

ament_package()