cmake_minimum_required(VERSION 3.0)

set(LibraryName "ImageProcessor")

# Create library
add_library (${LibraryName} 
	ImageProcessor.cpp
	ImageProcessor.h
	PalmDetectionEngine.cpp
	PalmDetectionEngine.h
	HandLandmarkEngine.cpp
	HandLandmarkEngine.h
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
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../CommonHelper CommonHelper)
target_include_directories(${LibraryName} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../CommonHelper)
target_link_libraries(${LibraryName} CommonHelper)

# Link InferenceHelper module
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper InferenceHelper)
target_include_directories(${LibraryName} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper)
target_link_libraries(${LibraryName} InferenceHelper)

