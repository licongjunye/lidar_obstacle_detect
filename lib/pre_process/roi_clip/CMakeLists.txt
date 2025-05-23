cmake_minimum_required(VERSION 2.8.3)
project(roi_clip)

# 添加C++14支持
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

aux_source_directory(. DIR_SRCS)

#SHARED 动态库
#STATIC 静态库
add_library(roi_clip STATIC ${DIR_SRCS})

target_link_libraries(roi_clip
  ${PCL_LIBRARIES}
 )

 