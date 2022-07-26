include_directories(../include)
aux_source_directory(. math_DIRS)
add_library(math ${math_DIRS})
