/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>

/* for My modules */
#include "CommonHelper.h"
#include "InferenceHelper.h"

#ifdef INFERENCE_HELPER_ENABLE_OPENCV
#include "InferenceHelperOpenCV.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_TENSORRT
#include "InferenceHelperTensorRt.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE
#include "InferenceHelperTensorflowLite.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_NCNN
#include "InferenceHelperNcnn.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_MNN
#include "InferenceHelperMnn.h"
#endif

/*** Macro ***/
#define TAG "InferenceHelper"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)


InferenceHelper* InferenceHelper::create(const InferenceHelper::HELPER_TYPE type)
{
	InferenceHelper* p = nullptr;
	switch (type) {
#ifdef INFERENCE_HELPER_ENABLE_OPENCV
	case OPEN_CV:
	case OPEN_CV_GPU:
		PRINT("Use OpenCV \n");
		p = new InferenceHelperOpenCV();
		break;
#endif
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
#ifdef INFERENCE_HELPER_ENABLE_NCNN
	case NCNN:
		PRINT("Use NCNN\n");
		p = new InferenceHelperNcnn();
		break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_MNN
	case MNN:
		PRINT("Use MNN\n");
		p = new InferenceHelperMnn();
		break;
#endif
	default:
		PRINT_E("Unsupported inference helper type (%d)\n", type);
		break;
	}
	if (p == nullptr) {
		PRINT_E("Failed to create inference helper\n");
	} else {
		p->m_helperType = type;
	}
	return p;
}

#ifdef INFERENCE_HELPER_ENABLE_PRE_PROCESS_BY_OPENCV
#include <opencv2/opencv.hpp>
void InferenceHelper::preProcessByOpenCV(const InputTensorInfo& inputTensorInfo, bool isNCHW, cv::Mat& imgBlob)
{
	/* Generate mat from original data */
	cv::Mat imgSrc = cv::Mat(cv::Size(inputTensorInfo.imageInfo.width, inputTensorInfo.imageInfo.height), (inputTensorInfo.imageInfo.channel == 3) ? CV_8UC3 : CV_8UC1, inputTensorInfo.data);

	/* Crop image */
	if (inputTensorInfo.imageInfo.width == inputTensorInfo.imageInfo.cropWidth && inputTensorInfo.imageInfo.height == inputTensorInfo.imageInfo.cropHeight) {
		/* do nothing */
	} else {
		imgSrc = imgSrc(cv::Rect(inputTensorInfo.imageInfo.cropX, inputTensorInfo.imageInfo.cropY, inputTensorInfo.imageInfo.cropWidth, inputTensorInfo.imageInfo.cropHeight));
	}

	/* Resize image */
	if (inputTensorInfo.imageInfo.cropWidth == inputTensorInfo.tensorDims.width && inputTensorInfo.imageInfo.cropHeight == inputTensorInfo.tensorDims.height) {
		/* do nothing */
	} else {
		cv::resize(imgSrc, imgSrc, cv::Size(inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height));
	}

	/* Convert color type */
	if (inputTensorInfo.imageInfo.channel == inputTensorInfo.tensorDims.channel) {
		if (inputTensorInfo.imageInfo.channel == 3 && inputTensorInfo.imageInfo.swapColor) {
			cv::cvtColor(imgSrc, imgSrc, cv::COLOR_BGR2RGB);
		}
	} else if (inputTensorInfo.imageInfo.channel == 3 && inputTensorInfo.tensorDims.channel == 1) {
		cv::cvtColor(imgSrc, imgSrc, (inputTensorInfo.imageInfo.isBGR) ? cv::COLOR_BGR2GRAY : cv::COLOR_RGB2GRAY);
	} else if (inputTensorInfo.imageInfo.channel == 1 && inputTensorInfo.tensorDims.channel == 3) {
		cv::cvtColor(imgSrc, imgSrc, cv::COLOR_GRAY2BGR);
	}

	if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_FP32) {
		/* Normalize image */
		if (inputTensorInfo.tensorDims.channel == 3) {
#if 1
			imgSrc.convertTo(imgSrc, CV_32FC3);
			cv::subtract(imgSrc, cv::Scalar(cv::Vec<float, 3>(inputTensorInfo.normalize.mean)), imgSrc);
			cv::multiply(imgSrc, cv::Scalar(cv::Vec<float, 3>(inputTensorInfo.normalize.norm)), imgSrc);
#else
			imgSrc.convertTo(imgSrc, CV_32FC3, 1.0 / 255);
			cv::subtract(imgSrc, cv::Scalar(cv::Vec<float, 3>(inputTensorInfo.normalize.mean)), imgSrc);
			cv::divide(imgSrc, cv::Scalar(cv::Vec<float, 3>(inputTensorInfo.normalize.norm)), imgSrc);
#endif
		} else {
#if 1
			imgSrc.convertTo(imgSrc, CV_32FC1);
			cv::subtract(imgSrc, cv::Scalar(cv::Vec<float, 1>(inputTensorInfo.normalize.mean)), imgSrc);
			cv::multiply(imgSrc, cv::Scalar(cv::Vec<float, 1>(inputTensorInfo.normalize.norm)), imgSrc);
#else
			imgSrc.convertTo(imgSrc, CV_32FC1, 1.0 / 255);
			cv::subtract(imgSrc, cv::Scalar(cv::Vec<float, 1>(inputTensorInfo.normalize.mean)), imgSrc);
			cv::divide(imgSrc, cv::Scalar(cv::Vec<float, 1>(inputTensorInfo.normalize.norm)), imgSrc);
#endif
		}
	} else {
		/* do nothing */
	}

	if (isNCHW) {
		/* Convert to 4-dimensional Mat in NCHW */
		imgSrc = cv::dnn::blobFromImage(imgSrc);
	}

	imgBlob = imgSrc;
	//memcpy(blobData, imgSrc.data, imgSrc.cols * imgSrc.rows * imgSrc.channels());

}

#else 
/* For the environment where OpenCV is not supported */
void InferenceHelper::preProcessByOpenCV(const InputTensorInfo& inputTensorInfo, bool isNCHW, cv::Mat& imgBlob)
{
}
#endif
