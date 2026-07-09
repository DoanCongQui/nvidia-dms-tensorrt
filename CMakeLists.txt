cmake_minimum_required(VERSION 3.16)

project(FaceAiDms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

find_library(NVINFER_LIBRARY nvinfer)
if(NOT NVINFER_LIBRARY)
    message(FATAL_ERROR "TensorRT library 'nvinfer' was not found")
endif()

add_library(face_ai_lib
    lib/camera_capture.cpp
    lib/dms.cpp
    lib/letterbox.cpp
    lib/nms.cpp
    lib/shared_state.cpp
    lib/trt_logger.cpp
    lib/yolo_trt.cpp
    lib/yolo_worker.cpp
)

target_include_directories(face_ai_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(face_ai_lib
    PUBLIC
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${NVINFER_LIBRARY}
)

add_executable(face_ai src/main.cpp)
target_link_libraries(face_ai PRIVATE face_ai_lib)
