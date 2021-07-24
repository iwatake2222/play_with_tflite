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
#include "style_prediction_engine.h"

/*** Macro ***/
#define TAG "StylePredictionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.tflite"

/*** Function ***/
int32_t StylePredictionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
	/* Set model information */
	std::string modelFilename = work_dir + "/model/" + MODEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = "style_image";
	inputTensorInfo.tensor_type= TensorInfo::kTensorTypeFp32;
	inputTensorInfo.tensor_dims.batch = 1;
	inputTensorInfo.tensor_dims.width = 256;
	inputTensorInfo.tensor_dims.height = 256;
	inputTensorInfo.tensor_dims.channel = 3;
	inputTensorInfo.data_type = InputTensorInfo::kDataTypeImage;
	inputTensorInfo.normalize.mean[0] = 0.0f;
	inputTensorInfo.normalize.mean[1] = 0.0f;
	inputTensorInfo.normalize.mean[2] = 0.0f;
	inputTensorInfo.normalize.norm[0] = 1.0f;
	inputTensorInfo.normalize.norm[1] = 1.0f;
	inputTensorInfo.normalize.norm[2] = 1.0f;
	m_inputTensorList.push_back(inputTensorInfo);

	/* Set output tensor info */
	m_outputTensorList.clear();
	OutputTensorInfo outputTensorInfo;
	outputTensorInfo.tensor_type = TensorInfo::kTensorTypeFp32;
	outputTensorInfo.name = "mobilenet_conv/Conv/BiasAdd";
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

int32_t StylePredictionEngine::Finalize()
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	m_inferenceHelper->Finalize();
	return kRetOk;
}


int32_t StylePredictionEngine::Process(const cv::Mat& original_mat, Result& result)
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
	const auto& tPostProcess1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.styleBottleneck = m_outputTensorList[0].GetDataAsFloat();
	result.time_pre_process = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.time_inference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
	result.time_post_process = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

	return kRetOk;
}

