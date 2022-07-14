add_subdirectory(src)
target_include_directories(M1 PUBLIC ../A1/include)
target_link_libraries(M1 A1)
