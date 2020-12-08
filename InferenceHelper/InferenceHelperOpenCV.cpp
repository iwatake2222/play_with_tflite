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

/* for OpenCV */
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/* for My modules */
#include "CommonHelper.h"
#include "InferenceHelperOpenCV.h"

/*** Macro ***/
#define TAG "InferenceHelperOpenCV"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Function ***/
InferenceHelperOpenCV::InferenceHelperOpenCV()
{
}

InferenceHelperOpenCV::~InferenceHelperOpenCV()
{
}

int32_t InferenceHelperOpenCV::setNumThread(const int32_t numThread)
{
	cv::setNumThreads(numThread);
	return RET_OK;
}

int32_t InferenceHelperOpenCV::setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps)
{
	PRINT("[WARNING] This method is not supported\n");
	return RET_OK;
}

int32_t InferenceHelperOpenCV::initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	/*** Create network ***/
	m_net = cv::dnn::readNetFromONNX(modelFilename);
	if (m_net.empty() == true) {
		PRINT_E("Failed to create inference engine (%s)\n", modelFilename.c_str());
		return RET_ERR;
	}
	
	if (m_helperType == OPEN_CV) {
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	} else if (m_helperType == OPEN_CV_GPU) {
		m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		m_net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
		// m_net.setPreferableTarget(cv::dnn::DNN_TARGET_VULKAN);
	} else {
		PRINT_E("Invalid helper type (%d)\n", m_helperType);
		return RET_ERR;
	}

	/*** Check tensor information  ***/
	/* Only one input tensor is supported. The input tensor name changes for some reasons. */
	if (inputTensorInfoList.size() == 1) {
		const auto& inputTensor = m_net.getLayer(0);
		inputTensorInfoList[0].name = inputTensor->name;
		inputTensorInfoList[0].id = m_net.getLayerId(inputTensor->name);
	} else {
		PRINT_E("Invalid input tensor num (%zu)\n", inputTensorInfoList.size());
	}

	/* Check output tensor name */
	for (auto& outputTensorInfo : outputTensorInfoList) {
		bool isFound = false;
		for (const auto& layerName : m_net.getLayerNames()) {
			if (outputTensorInfo.name == layerName) {
				isFound = true;
				outputTensorInfo.id = m_net.getLayerId(layerName);
				break;
			}
		}
		if (isFound == false) {
			PRINT_E("Output name (%s) not found\n", outputTensorInfo.name.c_str());
			return RET_ERR;
		}
	}

	/* Convert normalize parameter to speed up */
	for (auto& inputTensorInfo : inputTensorInfoList) {
		convertNormalizeParameters(inputTensorInfo);
	}

	return RET_OK;
};


int32_t InferenceHelperOpenCV::finalize(void)
{
	return RET_ERR;
}

int32_t InferenceHelperOpenCV::preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList)
{
	m_inMatList.clear();
	for (const auto& inputTensorInfo : inputTensorInfoList) {
		cv::Mat imgBlob;
		if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_IMAGE) {
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
			} else {
				PRINT_E("Unsupported color conversion (%d, %d)\n", inputTensorInfo.imageInfo.channel, inputTensorInfo.tensorDims.channel);
				return RET_ERR;
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
				} else if (inputTensorInfo.tensorDims.channel == 1) {
#if 1
					imgSrc.convertTo(imgSrc, CV_32FC1);
					cv::subtract(imgSrc, cv::Scalar(cv::Vec<float, 1>(inputTensorInfo.normalize.mean)), imgSrc);
					cv::multiply(imgSrc, cv::Scalar(cv::Vec<float, 1>(inputTensorInfo.normalize.norm)), imgSrc);
#else
					imgSrc.convertTo(imgSrc, CV_32FC1, 1.0 / 255);
					cv::subtract(imgSrc, cv::Scalar(cv::Vec<float, 1>(inputTensorInfo.normalize.mean)), imgSrc);
					cv::divide(imgSrc, cv::Scalar(cv::Vec<float, 1>(inputTensorInfo.normalize.norm)), imgSrc);
#endif
				} else {
					PRINT_E("Unsupported channel num (%d)\n", inputTensorInfo.tensorDims.channel);
					return RET_ERR;
				}
				/* Convert to 4-dimensional Mat in NCHW */
				imgBlob = cv::dnn::blobFromImage(imgSrc);
			} else if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_UINT8) {
				/* Convert to 4-dimensional Mat in NCHW */
				imgBlob = cv::dnn::blobFromImage(imgSrc);
			} else {
				PRINT_E("Unsupported tensorType (%d)\n", inputTensorInfo.tensorType);
				return RET_ERR;
			}

		} else if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NHWC) {
			cv::Mat imgSrc;
			if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_FP32) {
				imgSrc = cv::Mat(cv::Size(inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height), (inputTensorInfo.tensorDims.channel == 3) ? CV_32FC3 : CV_32FC1, inputTensorInfo.data);
			} else if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_UINT8) {
				imgSrc = cv::Mat(cv::Size(inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height), (inputTensorInfo.tensorDims.channel == 3) ? CV_8UC3 : CV_8UC1, inputTensorInfo.data);
			} else {
				PRINT_E("Unsupported tensorType (%d)\n", inputTensorInfo.tensorType);
				return RET_ERR;
			}
			imgBlob = cv::dnn::blobFromImage(imgSrc);
		} else if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NCHW) {
			PRINT_E("Unsupported dataType (%d)\n", inputTensorInfo.dataType);
			return RET_ERR;
		} else {
			PRINT_E("Unsupported data type (%d)\n", inputTensorInfo.dataType);
			return RET_ERR;
		}
		m_inMatList.push_back(imgBlob);
	}
	return RET_OK;
}

int32_t InferenceHelperOpenCV::invoke(std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	if (m_inMatList.size() != 1) {
		PRINT_E("Input tensor is not set\n");
		return RET_ERR;
	}
	m_net.setInput(m_inMatList[0]);

	/*** Run inference ***/
	std::vector<cv::String> outNameList;
	for (const auto& outputTensorInfo : outputTensorInfoList) {
		outNameList.push_back(outputTensorInfo.name);
	}
	m_outMatList.clear();
	m_net.forward(m_outMatList, outNameList);

	/*** Retrieve the results ***/
	if (m_outMatList.size() != outputTensorInfoList.size()) {
		PRINT_E("Unexpected output tensor num (%zu)\n", m_outMatList.size());
		return RET_ERR;
	}
	for (int32_t i = 0; i < static_cast<int32_t>(m_outMatList.size()); i++) {
		outputTensorInfoList[i].data = m_outMatList[i].data;
		outputTensorInfoList[i].tensorDims.batch = 1;
		outputTensorInfoList[i].tensorDims.width = m_outMatList[i].cols;
		outputTensorInfoList[i].tensorDims.height = m_outMatList[i].rows;
		outputTensorInfoList[i].tensorDims.channel = m_outMatList[i].channels();
	}

	return RET_OK;
}

void InferenceHelperOpenCV::convertNormalizeParameters(InputTensorInfo& inputTensorInfo)
{
	if (inputTensorInfo.dataType != InputTensorInfo::DATA_TYPE_IMAGE) return;

#if 0
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm  = src * 1 / (255 * norm) - (mean / norm) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] /= inputTensorInfo.normalize.norm[i];
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif
#if 1
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif
}
