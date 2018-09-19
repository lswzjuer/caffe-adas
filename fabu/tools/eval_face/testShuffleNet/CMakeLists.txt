cmake_minimum_required(VERSION 2.8.7)

project(shufflenet)
find_package(OpenCV)

set (LIB_ROOT "/home/hujinming")

include_directories ("${LIB_ROOT}/ncnn/build/install/include")
link_directories("${LIB_ROOT}/ncnn/build/install/lib")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -fopenmp -O3")

add_executable (exec ${MTCNN_SRCS} testShuffleNet.cpp)
target_link_libraries(exec ncnn ${OpenCV_LIBS})
