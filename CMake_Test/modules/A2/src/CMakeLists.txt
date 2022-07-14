include_directories(../include)
aux_source_directory(. A2_DIRS)
add_library(A2 ${A2_DIRS})
