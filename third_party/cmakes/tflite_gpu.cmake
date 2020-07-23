set(TFLITE_GPU_INC
	${CMAKE_CURRENT_LIST_DIR}/../tensorflow
	${CMAKE_CURRENT_LIST_DIR}/../tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
	${CMAKE_CURRENT_LIST_DIR}/../tensorflow/tensorflow/lite/tools/make/downloads/absl
)

if(DEFINED  ANDROID_ABI)
	# set(TFLITE_GPU_LIB ${CMAKE_CURRENT_LIST_DIR}/../tensorflow_prebuilt/android/${ANDROID_ABI}/libtensorflowlite_gpu_delegate.so)
	add_library(TFLITE_GPU SHARED IMPORTED GLOBAL)
	set_target_properties(
		TFLITE_GPU
		PROPERTIES IMPORTED_LOCATION
		${CMAKE_CURRENT_LIST_DIR}/../tensorflow_prebuilt/android/${ANDROID_ABI}/libtensorflowlite_gpu_delegate.so
	)
	set(TFLITE_GPU_LIB TFLITE_GPU)
elseif(MSVC_VERSION)
	if((MSVC_VERSION GREATER_EQUAL 1910) AND (MSVC_VERSION LESS 1920))
		set(TFLITE_GPU_LIB ${CMAKE_CURRENT_LIST_DIR}/../tensorflow_prebuilt/x64_windows/libtensorflowlite_gpu_delegate.so.if.lib)
		file(COPY ${CMAKE_CURRENT_LIST_DIR}/../tensorflow_prebuilt/x64_windows/libtensorflowlite_gpu_delegate.so DESTINATION ${CMAKE_BINARY_DIR})
	else()
		message(FATAL_ERROR "[tflite_gpu] unsupported MSVC version")
	endif()
else()
	if(${BUILD_SYSTEM} STREQUAL "x64_linux")
		set(TFLITE_GPU_LIB ${CMAKE_CURRENT_LIST_DIR}/../tensorflow_prebuilt/x64_linux/libtensorflowlite_gpu_delegate.so)
	elseif(${BUILD_SYSTEM} STREQUAL "armv7")
		set(TFLITE_GPU_LIB ${CMAKE_CURRENT_LIST_DIR}/../tensorflow_prebuilt/armv7/libtensorflowlite_gpu_delegate.so)
	elseif(${BUILD_SYSTEM} STREQUAL "aarch64")
		set(TFLITE_GPU_LIB ${CMAKE_CURRENT_LIST_DIR}/../tensorflow_prebuilt/aarch64/libtensorflowlite_gpu_delegate.so)
	else()	
		message(FATAL_ERROR "[tflite_gpu] unsupported platform")
	endif()
	file(COPY ${TFLITE_GPU_LIB} DESTINATION ${CMAKE_BINARY_DIR})
endif()
