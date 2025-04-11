cmake_minimum_required(VERSION 3.10)

set(project_name matrixThreads)
project(${project_name})

find_package(Boost REQUIRED COMPONENTS program_options)
add_executable(${project_name} src/matrixThreads.cpp)
target_compile_options(${project_name} PRIVATE -O2)
target_link_libraries(${project_name} Boost::program_options)

