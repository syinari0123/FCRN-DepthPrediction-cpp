cmake_minimum_required(VERSION 3.10)
project(depthnet_cpp)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
add_library(fcrn_cpp
    src/Model.cpp
    src/Tensor.cpp
    src/FCRN.cpp
)
target_include_directories(fcrn_cpp PRIVATE include)
target_link_libraries(fcrn_cpp -ltensorflow ${OpenCV_LIBS})

# FCRN Depth Prediction
add_executable(inference_fcrn inference_fcrn.cpp)
target_link_libraries(inference_fcrn fcrn_cpp)
