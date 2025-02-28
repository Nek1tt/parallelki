cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME SimpleIter)
project(${PROJECT_NAME})

find_package(OpenMP REQUIRED)

option(NUM_THREADS "Number of OpenMP threads" 1)

set(EXECUTABLE_ONE SimpleIterVarOne)
set(EXECUTABLE_TWO SimpleIterVarTwo)

add_executable(${EXECUTABLE_ONE} src/SimpleIterVarOne.cpp)
add_executable(${EXECUTABLE_TWO} src/SimpleIterVarTwo.cpp)

target_link_libraries(${EXECUTABLE_ONE} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${EXECUTABLE_TWO} PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(${EXECUTABLE_ONE} PRIVATE -O2)
target_compile_options(${EXECUTABLE_TWO} PRIVATE -O2)

target_compile_definitions(${EXECUTABLE_ONE} PRIVATE NUM_THREADS=${NUM_THREADS})
target_compile_definitions(${EXECUTABLE_TWO} PRIVATE NUM_THREADS=${NUM_THREADS})
