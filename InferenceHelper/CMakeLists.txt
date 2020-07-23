cmake_minimum_required(VERSION 3.0)

set(LibraryName "InferenceHelper")

set(TFLITE_DELEGATE_EDGETPU off CACHE BOOL "With Tflite Delegate EdgeTPU? [on/off]")
set(TFLITE_DELEGATE_GPU off CACHE BOOL "With Tflite Delegate GPU? [on/off]")
set(TFLITE_DELEGATE_XNNPACK off CACHE BOOL "With Tflite Delegate XNNPACK? [on/off]")

# Create library
add_library (${LibraryName} 
	InferenceHelper.h
	InferenceHelper.cpp
	InferenceHelperTensorflowLite.h
	InferenceHelperTensorflowLite.cpp
)

# For Tensorflow Lite
include(${CMAKE_CURRENT_LIST_DIR}/../third_party/cmakes/tflite.cmake)
target_include_directories(${LibraryName} PUBLIC ${TFLITE_INC})
target_link_libraries(${LibraryName} ${TFLITE_LIB})

# For Tensorflow Lite Delegate(Edge TPU)
if(TFLITE_DELEGATE_EDGETPU)
	include(${CMAKE_SOURCE_DIR}/../third_party/cmakes/tflite_edgetpu.cmake)
	target_include_directories(${LibraryName} PUBLIC ${TFLITE_EDGETPI_INC})
	target_link_libraries(${LibraryName} ${TFLITE_EDGETPI_LIB})
	add_definitions(-DTFLITE_DELEGATE_EDGETPU)
endif()

# For Tensorflow Lite Delegate(GPU)
if(TFLITE_DELEGATE_GPU)
	find_package(OpenCL)
	if(OpenCL_Found)
		target_include_directories(${LibraryName} PUBLIC ${OpenCL_INCLUDE_DIRS})
		target_link_libraries(${LibraryName} ${OpenCL_LIBRARIES})
	endif()
	include(${CMAKE_SOURCE_DIR}/../third_party/cmakes/tflite_gpu.cmake)
	target_include_directories(${LibraryName} PUBLIC ${TFLITE_GPU_INC})
	target_link_libraries(${LibraryName} ${TFLITE_GPU_LIB} EGL GLESv2)
	add_definitions(-DTFLITE_DELEGATE_GPU)
endif()

# For Tensorflow Lite Delegate(XNNPACK)
if(TFLITE_DELEGATE_XNNPACK)
	add_definitions(-DTFLITE_DELEGATE_XNNPACK)
endif()
