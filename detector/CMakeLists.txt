cmake_minimum_required(VERSION 2.8.3)
project(pcl_monitor)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  genmsg
)

## Declare ROS messages and Services
## add_message_files(FILES Num.msg)
## add_service_files(FILES AddTwoInts.srv)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs)

## Declare a catkin package
catkin_package()

## Build Talker and Listener
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(testPub src/testPub.cpp)
target_link_libraries(testPub ${catkin_LIBRARIES})
add_dependencies(testPub pcl_monitor_generate_messages_cpp)

add_executable(testSub src/testSub.cpp)
target_link_libraries(testSub ${catkin_LIBRARIES})
add_dependencies(testSub pcl_monitor_generate_messages_cpp)

#catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES pcl_monitor
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
#)


#  ${catkin_INCLUDE_DIRS}
#)


