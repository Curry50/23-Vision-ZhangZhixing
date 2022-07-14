include_directories(../include)
aux_source_directory(. M1_DIRS)
add_library(M1 ${M1_DIRS})
