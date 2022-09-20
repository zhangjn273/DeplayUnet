cmake_minimum_required(VERSION 3.0.0)
project(read_onnx VERSION 0.1.0)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_INCLUDE_DIRS})

include(CTest)
enable_testing()

add_executable(read_onnx main.cpp)
target_link_libraries(read_onnx ${OpenCV_LIBS})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
