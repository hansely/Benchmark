cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)
project( benchmark )
find_package( OpenCV REQUIRED )
add_executable( benchmark benchmark.cpp)
target_link_libraries( benchmark ${OpenCV_LIBS} -lpthread)
