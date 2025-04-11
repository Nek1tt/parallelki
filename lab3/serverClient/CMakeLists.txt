cmake_minimum_required(VERSION 3.10)

set(project_name ServerClient)
project(${project_name})

add_executable(${project_name} src/main.cpp)
target_compile_options(${project_name} PRIVATE -O2)
target_include_directories(${project_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/headers)
