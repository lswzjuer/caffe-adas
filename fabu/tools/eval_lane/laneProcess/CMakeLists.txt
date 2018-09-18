cmake_minimum_required(VERSION 2.8)

project(laneProcess)

add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)

#set(CMAKE_CXX_FLAGS "-g")

add_executable(test_laneProcess test_laneProcess.cc laneProcess.cc)
target_link_libraries(test_laneProcess ${OpenCV_LIBS})
