cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 17)

project(OpenVINO-Test)

find_package(Threads REQUIRED)

# set(PARALLEL_LIB_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/openvino/install-tbb/runtime/lib/intel64/libopenvino.so"
#     "${CMAKE_CURRENT_SOURCE_DIR}/openvino/install-tbb/runtime/3rdparty/tbb/lib/libtbb.so"
#     "${CMAKE_THREAD_LIBS_INIT}")

# set(PARALLEL_HEADER_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/openvino/install-tbb/runtime/include")

set(PARALLEL_LIB_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/openvino/install-omp/runtime/lib/intel64/libopenvino.so"
    "${CMAKE_THREAD_LIBS_INIT}")

set(PARALLEL_HEADER_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/openvino/install-omp/runtime/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(${PROJECT_NAME} main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "openvino-test"
    OUTPUT_NAME_DEBUG "openvino-test-debug")

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PARALLEL_HEADER_PATHS}")
target_link_libraries(${PROJECT_NAME} PRIVATE
    "${PARALLEL_LIB_PATHS}")
