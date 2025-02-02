cmake_minimum_required(VERSION 3.8)
project(ur5_motion_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ur5_motion_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(tf2_eigen REQUIRED)


add_library(action_client SHARED
  src/grabmove_client_test.cpp)
target_include_directories(action_client PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_definitions(action_client
  PRIVATE "CUSTOM_ACTION_CPP_BUILDING_DLL")
ament_target_dependencies(action_client
  "ur5_motion_interfaces"
  "rclcpp"
  "rclcpp_action"
  "rclcpp_components")
rclcpp_components_register_node(action_client PLUGIN "ur5_motion_cpp::GrabMoveActionClient" EXECUTABLE grabmove_client_test)
install(TARGETS
  action_client
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

add_library(action_server SHARED
  src/grabmove_server.cpp)
target_include_directories(action_server PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_definitions(action_server
  PRIVATE "CUSTOM_ACTION_CPP_BUILDING_DLL")
  ament_target_dependencies(action_server
  "ur5_motion_interfaces"
  "rclcpp"
  "rclcpp_action"
  "rclcpp_components"
  "geometry_msgs"
  "sensor_msgs"
  "tf2_ros"
  "tf2_geometry_msgs"
  "orocos_kdl"
  "control_msgs"
  "trajectory_msgs"
  "eigen3_cmake_module"
  "tf2_eigen")
rclcpp_components_register_node(action_server PLUGIN "ur5_motion_cpp::GrabMoveActionServer" EXECUTABLE grabmove_server)
install(TARGETS
  action_server
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
