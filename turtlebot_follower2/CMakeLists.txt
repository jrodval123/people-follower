cmake_minimum_required(VERSION 2.8.3)
project(turtlebot_follower2)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  nodelet
  pcl_ros
  roscpp
  visualization_msgs
  turtlebot_msgs
  dynamic_reconfigure
  rospy
  std_msgs
)

find_package(Boost REQUIRED)
find_package(PCL)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

generate_dynamic_reconfigure_options(cfg/Follower2.cfg)

catkin_package(INCLUDE_DIRS LIBRARIES ${PROJECT_NAME} CATKIN_DEPENDS nodelet pcl_ros roscpp visualization_msgs turtlebot_msgs dynamic_reconfigure DEPENDS Boost PCL)


################################################
## Declare ROS messages, services and actions ##
################################################



###################################
## catkin specific configuration ##
###################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES turtlebot_follower2
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

## Declare a cpp library
#add_library(turtlebot_follower2
#   /src/turtlebot_follower2.cpp
# )

add_library(${PROJECT_NAME} src/follower2.cpp)
add_dependencies(${PROJECT_NAME} turtlebot_msgs_gencpp)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#############
## Install ##
#############


install(TARGETS ${PROJECT_NAME}
	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY param
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY plugins
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


#############
## Testing ##
#############


