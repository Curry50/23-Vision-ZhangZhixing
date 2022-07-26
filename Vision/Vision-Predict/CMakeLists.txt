cmake_minimum_required(VERSION 3.22)
project(ImgTest)

set(CMAKE_CXX_STANDARD 14)

add_executable(ImgTest main.cpp Kalm.cpp Kalm.h)
find_package(OpenCV REQUIRED)
target_link_libraries(ImgTest ${OpenCV_LIBS})