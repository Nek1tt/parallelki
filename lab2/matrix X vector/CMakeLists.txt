cmake_minimum_required(VERSION 3.10)

set(project_name MatrixVector)
project(${project_name})

find_package(OpenMP REQUIRED)

option(NUM_THREADS "Number of OpenMP threads" 4)

add_executable(${project_name} src/matrix.c)

target_link_libraries(${project_name} PRIVATE OpenMP::OpenMP_C)

target_compile_options(${project_name} PRIVATE -O2)

target_compile_definitions(${project_name} PRIVATE NUM_THREADS=${NUM_THREADS})