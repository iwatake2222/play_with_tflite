/*** Include ***/
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "InferenceHelper.h"
#ifdef INFERENCE_HELPER_ENABLE_TENSORRT
#include "InferenceHelperTensorRt.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE
#include "InferenceHelperTensorflowLite.h"
#endif

#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define _PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define _PRINT(...) printf(__VA_ARGS__)
#endif
#define PRINT(...) _PRINT("[InferenceHelper] " __VA_ARGS__)

InferenceHelper* InferenceHelper::create(const InferenceHelper::HELPER_TYPE type)
{
	InferenceHelper* p = NULL;
	switch (type) {
#ifdef INFERENCE_HELPER_ENABLE_TENSORRT
	case TENSOR_RT:
		PRINT("Use TensorRT \n");
		p = new InferenceHelperTensorRt();
		break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE
	case TENSORFLOW_LITE:
		PRINT("Use TensorflowLite\n");
		p = new InferenceHelperTensorflowLite();
		break;
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU
	case TENSORFLOW_LITE_EDGETPU:
		PRINT("Use TensorflowLite EdgeTPU Delegate\n");
		p = new InferenceHelperTensorflowLite();
		break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU
	case TENSORFLOW_LITE_GPU:
		PRINT("Use TensorflowLite GPU Delegate\n");
		p = new InferenceHelperTensorflowLite();
		break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK
	case TENSORFLOW_LITE_XNNPACK:
		PRINT("Use TensorflowLite XNNPACK Delegate\n");
		p = new InferenceHelperTensorflowLite();
		break;
#endif
#endif
	default:
		PRINT("not supported\n");
		exit(1);
		break;
	}
	p->m_helperType = type;
	return p;
}


TensorInfo::TensorInfo()
{
	index = -1;
	type = TENSOR_TYPE_NONE;
	data = NULL;;
	dims.clear();
	quant.scale = 0;
	quant.zeroPoint = 0;
	m_dataFp32 = NULL;
}

TensorInfo::~TensorInfo()
{
	if (m_dataFp32 != NULL) {
		delete[] m_dataFp32;
	}
}

float* TensorInfo::getDataAsFloat()
{
	if (type == TENSOR_TYPE_UINT8) {
		int dataNum = 1;
		for (int i = 0; i < (int)dims.size(); i++) dataNum *= dims[i];
		if (m_dataFp32 == NULL) {
			m_dataFp32 = new float[dataNum];
		}
		for (int i = 0; i < dataNum; i++) {
			const uint8_t* valUint8 = (uint8_t*)data;
			float valFloat = (valUint8[i] - quant.zeroPoint) * quant.scale;
			m_dataFp32[i] = valFloat;
		}
		return m_dataFp32;
	} else if (type == TENSOR_TYPE_FP32) {
		return (float*)data;
	} else {
		PRINT("invalid call");
		return NULL;
	}
}

