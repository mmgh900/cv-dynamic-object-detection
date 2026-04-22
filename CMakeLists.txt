cmake_minimum_required(VERSION 3.10)
project(DynamicObjectDetection CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(detect
    src/main.cpp
    src/feature_tracker.cpp
    src/motion_segmenter.cpp
    src/bbox_estimator.cpp
    src/silhouette_refiner.cpp
    src/evaluator.cpp
)
target_link_libraries(detect ${OpenCV_LIBS})
