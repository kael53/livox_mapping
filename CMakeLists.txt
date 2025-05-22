cmake_minimum_required(VERSION 3.8)
project(livox_mapping)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

SET(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -Wno-unused-local-typedefs")

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

# Livox Repub Node
add_executable(livox_repub src/livox_repub.cpp)
ament_target_dependencies(livox_repub
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  pcl_ros
  livox_ros_driver2
)
target_link_libraries(livox_repub
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

# Scan Registration Node
add_executable(loam_scanRegistration src/scanRegistration.cpp)
ament_target_dependencies(loam_scanRegistration
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  pcl_ros
  livox_ros_driver2
)
target_link_libraries(loam_scanRegistration
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

# Scan Registration Horizon Node
add_executable(loam_scanRegistration_horizon src/scanRegistration_horizon.cpp)
ament_target_dependencies(loam_scanRegistration_horizon
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  pcl_ros
  livox_ros_driver2
)
target_link_libraries(loam_scanRegistration_horizon
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

# Laser Mapping Node
add_executable(loam_laserMapping src/laserMapping.cpp)
ament_target_dependencies(loam_laserMapping
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  pcl_ros
  livox_ros_driver2
)
target_link_libraries(loam_laserMapping
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

# Install
install(TARGETS
  livox_repub
  loam_scanRegistration
  loam_scanRegistration_horizon
  loam_laserMapping
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  rviz_cfg
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
