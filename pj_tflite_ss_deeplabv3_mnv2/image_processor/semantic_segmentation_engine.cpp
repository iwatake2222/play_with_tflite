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
#include <fstream>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "inference_helper.h"
#include "semantic_segmentation_engine.h"

/*** Macro ***/
#define TAG "SemanticSegmentationEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "deeplabv3_mnv2_dm05_pascal_quant.tflite"

/*** Function ***/
int32_t SemanticSegmentationEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
	/* Set model information */
	std::string modelFilename = work_dir + "/model/" + MODEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = "MobilenetV2/MobilenetV2/input";
	inputTensorInfo.tensor_type= TensorInfo::kTensorTypeFp32;
	inputTensorInfo.tensor_dims.batch = 1;
	inputTensorInfo.tensor_dims.width = 513;
	inputTensorInfo.tensor_dims.height = 513;
	inputTensorInfo.tensor_dims.channel = 3;
	inputTensorInfo.data_type = InputTensorInfo::kDataTypeImage;
	inputTensorInfo.normalize.mean[0] = 0.5f;
	inputTensorInfo.normalize.mean[1] = 0.5f;
	inputTensorInfo.normalize.mean[2] = 0.5f;
	inputTensorInfo.normalize.norm[0] = 0.5f;
	inputTensorInfo.normalize.norm[1] = 0.5f;
	inputTensorInfo.normalize.norm[2] = 0.5f;
	m_inputTensorList.push_back(inputTensorInfo);

	/* Set output tensor info */
	m_outputTensorList.clear();
	OutputTensorInfo outputTensorInfo;
	outputTensorInfo.tensor_type = TensorInfo::kTensorTypeFp32;
	outputTensorInfo.name = "ArgMax";
	m_outputTensorList.push_back(outputTensorInfo);

	/* Create and Initialize Inference Helper */
	m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));

	if (!m_inferenceHelper) {
		return kRetErr;
	}
	if (m_inferenceHelper->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
		m_inferenceHelper.reset();
		return kRetErr;
	}
	if (m_inferenceHelper->Initialize(modelFilename, m_inputTensorList, m_outputTensorList) != InferenceHelper::kRetOk) {
		m_inferenceHelper.reset();
		return kRetErr;
	}

	/* Check if input tensor info is set */
	for (const auto& inputTensorInfo : m_inputTensorList) {
		if ((inputTensorInfo.tensor_dims.width <= 0) || (inputTensorInfo.tensor_dims.height <= 0) || inputTensorInfo.tensor_type == TensorInfo::kTensorTypeNone) {
			PRINT_E("Invalid tensor size\n");
			m_inferenceHelper.reset();
			return kRetErr;
		}
	}


	return kRetOk;
}

int32_t SemanticSegmentationEngine::Finalize()
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	m_inferenceHelper->Finalize();
	return kRetOk;
}


int32_t SemanticSegmentationEngine::Process(const cv::Mat& original_mat, Result& result)
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	/*** PreProcess ***/
	const auto& tPreProcess0 = std::chrono::steady_clock::now();
	InputTensorInfo& inputTensorInfo = m_inputTensorList[0];
	/* do resize and color conversion here because some inference engine doesn't support these operations */
	cv::Mat imgSrc;
	cv::resize(original_mat, imgSrc, cv::Size(inputTensorInfo.tensor_dims.width, inputTensorInfo.tensor_dims.height));
#ifndef CV_COLOR_IS_RGB
	cv::cvtColor(imgSrc, imgSrc, cv::COLOR_BGR2RGB);
#endif
	inputTensorInfo.data = imgSrc.data;
	inputTensorInfo.data_type = InputTensorInfo::kDataTypeImage;
	inputTensorInfo.image_info.width = imgSrc.cols;
	inputTensorInfo.image_info.height = imgSrc.rows;
	inputTensorInfo.image_info.channel = imgSrc.channels();
	inputTensorInfo.image_info.crop_x = 0;
	inputTensorInfo.image_info.crop_y = 0;
	inputTensorInfo.image_info.crop_width = imgSrc.cols;
	inputTensorInfo.image_info.crop_height = imgSrc.rows;
	inputTensorInfo.image_info.is_bgr = false;
	inputTensorInfo.image_info.swap_color = false;
	if (m_inferenceHelper->PreProcess(m_inputTensorList) != InferenceHelper::kRetOk) {
		return kRetErr;
	}
	const auto& tPreProcess1 = std::chrono::steady_clock::now();

	/*** Inference ***/
	const auto& tInference0 = std::chrono::steady_clock::now();
	if (m_inferenceHelper->Process(m_outputTensorList) != InferenceHelper::kRetOk) {
		return kRetErr;
	}
	const auto& tInference1 = std::chrono::steady_clock::now();

	/*** PostProcess ***/
	const auto& tPostProcess0 = std::chrono::steady_clock::now();
	/* Retrieve the result */
	int32_t outputWidth = m_outputTensorList[0].tensor_dims.width;
	int32_t outputHeight = m_outputTensorList[0].tensor_dims.height;
	int32_t outputCannel = m_outputTensorList[0].tensor_dims.channel;
	const int64_t* values = static_cast<int64_t*>(m_outputTensorList[0].data);
	cv::Mat maskImage = cv::Mat::zeros(outputHeight, outputWidth, CV_8UC3);
	for (int32_t y = 0; y < outputHeight; y++) {
		for (int32_t x = 0; x < outputWidth; x++) {
		}
	}
	for (int y = 0; y < outputHeight; y++) {
		for (int x = 0; x < outputWidth; x++) {
			int32_t maxChannel = static_cast<int32_t>(values[y * outputWidth + x]);
			float colorRatioB = (maxChannel % 2 + 1) / 2.0f;
			float colorRatioG = (maxChannel % 3 + 1) / 3.0f;
			float colorRatioR = (maxChannel % 4 + 1) / 4.0f;
			maskImage.data[(y * outputWidth + x) * 3 + 0] = static_cast<uint8_t>(255 * colorRatioB);
			maskImage.data[(y * outputWidth + x) * 3 + 1] = static_cast<uint8_t>(255 * colorRatioG);
			maskImage.data[(y * outputWidth + x) * 3 + 2] = static_cast<uint8_t>(255 * (1 - colorRatioR));

		}
	}
	const auto& tPostProcess1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.maskImage = maskImage;
	result.time_pre_process = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.time_inference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
	result.time_post_process = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

	return kRetOk;
}

