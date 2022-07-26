# 在此处进行 CMake 设置（最小版本号，项目名）
cmake_minimum_required(VERSION 3.22)
project(Test)
set(CMAKE_CXX_STANDARD 14)
####### 以下禁止修改 #######
add_subdirectory(common)   #
add_subdirectory(modules)  #
####### 以上禁止修改 #######
find_package(OpenCV REQUIRED)
# 在此处生成可执行程序
add_executable(test main.cpp)

# 在此处链接库
target_include_directories(test PUBLIC ${kalman_INCLUDE_DIRS} ./modules/M1/include ./modules/M2/include ./common/math/include)
target_link_libraries(test ${kalman_LIBS} M1 M2 math)
