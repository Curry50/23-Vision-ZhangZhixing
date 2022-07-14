include_directories(../include)
aux_source_directory(. A1_DIRS)
add_library(A1 ${A1_DIRS})
