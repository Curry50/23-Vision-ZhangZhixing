add_subdirectory(src)
find_package(OpenCV REQUIRED)
target_include_directories(math PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(math ${OpenCV_LIBS})
