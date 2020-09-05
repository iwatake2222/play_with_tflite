cmake_minimum_required(VERSION 3.0)

set(LibraryName "ImageProcessor")

# Create library
add_library (${LibraryName}
	ImageProcessor.cpp
	ImageProcessor.h
	PalmDetection.cpp
	PalmDetection.h
	HandLandmark.cpp
	HandLandmark.h
	Classify.cpp
	Classify.h
	AreaSelector.cpp
	AreaSelector.h
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

# Link InferenceHelper module
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper InferenceHelper)
target_include_directories(${LibraryName} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper)
target_link_libraries(${LibraryName} InferenceHelper)


if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
	add_definitions(-DTFLITE_DELEGATE_EDGETPU)
endif()
if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU)
	add_definitions(-DTFLITE_DELEGATE_GPU)
endif()
if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK)
	add_definitions(-DTFLITE_DELEGATE_XNNPACK)
endif()

