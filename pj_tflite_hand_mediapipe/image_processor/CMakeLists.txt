cmake_minimum_required(VERSION 3.0)

set(LibraryName "ImageProcessor")

# Create library
add_library (${LibraryName} 
    image_processor.cpp
    image_processor.h
    palm_detection_engine.cpp
    palm_detection_engine.h
    hand_landmark_engine.cpp
    hand_landmark_engine.h
    meidapipe/transpose_conv_bias.cc
    meidapipe/transpose_conv_bias.h
    meidapipe/ssd_anchors_calculator.cc
    meidapipe/ssd_anchors_calculator.h
    meidapipe/tflite_tensors_to_detections_calculator.cc
    meidapipe/tflite_tensors_to_detections_calculator.h
)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LibraryName} ${OpenCV_LIBS})

# Link Common Helper module
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../common_helper common_helper)
target_include_directories(${LibraryName} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../common_helper)
target_link_libraries(${LibraryName} CommonHelper)

# For InferenceHelper
set(INFERENCE_HELPER_DIR ${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper/)
set(INFERENCE_HELPER_ENABLE_TFLITE ON CACHE BOOL "TFLITE")
add_subdirectory(${INFERENCE_HELPER_DIR}/inference_helper inference_helper)
target_include_directories(${LibraryName} PUBLIC ${INFERENCE_HELPER_DIR}/inference_helper)
target_link_libraries(${LibraryName} InferenceHelper)


