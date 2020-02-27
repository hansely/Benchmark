cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -msse4.2 -mfma -mf16c")
project( benchmark )
find_package( OpenCV REQUIRED )
add_executable( benchmark main.cpp Benchmark.cpp FileQueue.cpp Utils.cpp)
target_link_libraries( benchmark ${OpenCV_LIBS} -lpthread)
