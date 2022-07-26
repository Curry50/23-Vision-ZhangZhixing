include_directories(../include)
aux_source_directory(. M2_DIRS)
add_library(M2 ${M2_DIRS})
