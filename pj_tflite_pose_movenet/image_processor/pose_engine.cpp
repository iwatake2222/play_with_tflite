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
#include "pose_engine.h"

/*** Macro ***/
#define TAG "PoseEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#if 1
/* Official model. https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3 */
#define MODEL_NAME  "lite-model_movenet_singlepose_lightning_3.tflite"
#define INPUT_NAME  "serving_default_input:0"
#define OUTPUT_NAME "StatefulPartitionedCall:0"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#else
/* PINTO_model_zoo. https://github.com/PINTO0309/PINTO_model_zoo */
#define MODEL_NAME  "model_float32.tflite"
// #define MODEL_NAME  "model_weight_quant.tflite"
// #define MODEL_NAME  "model_integer_quant.tflite"
#define INPUT_NAME  "input:0"
#define OUTPUT_NAME "Identity:0"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#endif

/*** Function ***/
int32_t PoseEngine::initialize(const std::string& work_dir, const int32_t num_threads)
{
	/* Set model information */
	std::string modelFilename = work_dir + "/model/" + MODEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = INPUT_NAME;
	inputTensorInfo.tensor_type= TENSORTYPE;
	inputTensorInfo.tensor_dims.batch = 1;
	inputTensorInfo.tensor_dims.width = 192;
	inputTensorInfo.tensor_dims.height = 192;
	inputTensorInfo.tensor_dims.channel = 3;
	inputTensorInfo.data_type = InputTensorInfo::kDataTypeImage;
	// inputTensorInfo.normalize.mean[0] = 0.485f;   	/* https://github.com/onnx/models/tree/master/vision/classification/mobilenet#preprocessing */
	// inputTensorInfo.normalize.mean[1] = 0.456f;
	// inputTensorInfo.normalize.mean[2] = 0.406f;
	// inputTensorInfo.normalize.norm[0] = 0.229f;
	// inputTensorInfo.normalize.norm[1] = 0.224f;
	// inputTensorInfo.normalize.norm[2] = 0.225f;
	/* 0 - 255 (https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3) */
	inputTensorInfo.normalize.mean[0] = 0;
	inputTensorInfo.normalize.mean[1] = 0;
	inputTensorInfo.normalize.mean[2] = 0;
	inputTensorInfo.normalize.norm[0] = 1/255.f;
	inputTensorInfo.normalize.norm[1] = 1/255.f;
	inputTensorInfo.normalize.norm[2] = 1/255.f;
	m_inputTensorList.push_back(inputTensorInfo);

	/* Set output tensor info */
	m_outputTensorList.clear();
	OutputTensorInfo outputTensorInfo;
	outputTensorInfo.tensor_type = TENSORTYPE;
	outputTensorInfo.name = OUTPUT_NAME;
	m_outputTensorList.push_back(outputTensorInfo);

	/* Create and Initialize Inference Helper */
	m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
	// m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
	// m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
	// m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
	// m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));

	if (!m_inferenceHelper) {
		return RET_ERR;
	}
	if (m_inferenceHelper->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}
	if (m_inferenceHelper->Initialize(modelFilename, m_inputTensorList, m_outputTensorList) != InferenceHelper::kRetOk) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}
	/* Check if input tensor info is set */
	for (const auto& inputTensorInfo : m_inputTensorList) {
		if ((inputTensorInfo.tensor_dims.width <= 0) || (inputTensorInfo.tensor_dims.height <= 0) || inputTensorInfo.tensor_type == TensorInfo::kTensorTypeNone) {
			PRINT_E("Invalid tensor size\n");
			m_inferenceHelper.reset();
			return RET_ERR;
		}
	}

	return RET_OK;
}

int32_t PoseEngine::finalize()
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	m_inferenceHelper->Finalize();
	return RET_OK;
}


int32_t PoseEngine::invoke(const cv::Mat& originalMat, RESULT& result)
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	/*** PreProcess ***/
	const auto& tPreProcess0 = std::chrono::steady_clock::now();
	InputTensorInfo& inputTensorInfo = m_inputTensorList[0];
#if 1
	/* do resize and color conversion here because some inference engine doesn't support these operations */
	cv::Mat imgSrc;
	cv::resize(originalMat, imgSrc, cv::Size(inputTensorInfo.tensor_dims.width, inputTensorInfo.tensor_dims.height));
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
#else
	/* Test other input format */
	cv::Mat imgSrc;
	inputTensorInfo.data = originalMat.data;
	inputTensorInfo.data_type = InputTensorInfo::kDataTypeImage;
	inputTensorInfo.image_info.width = originalMat.cols;
	inputTensorInfo.image_info.height = originalMat.rows;
	inputTensorInfo.image_info.channel = originalMat.channels();
	inputTensorInfo.image_info.crop_x = 0;
	inputTensorInfo.image_info.crop_y = 0;
	inputTensorInfo.image_info.crop_width = originalMat.cols;
	inputTensorInfo.image_info.crop_height = originalMat.rows;
	inputTensorInfo.image_info.is_bgr = true;
	inputTensorInfo.image_info.swap_color = true;
#if 0
	InferenceHelper::preProcessByOpenCV(inputTensorInfo, false, imgSrc);
	inputTensorInfo.data_type = InputTensorInfo::DATA_TYPE_BLOB_NHWC;
#else
	InferenceHelper::preProcessByOpenCV(inputTensorInfo, true, imgSrc);
	inputTensorInfo.data_type = InputTensorInfo::DATA_TYPE_BLOB_NCHW;
#endif
	inputTensorInfo.data = imgSrc.data;
#endif
	if (m_inferenceHelper->PreProcess(m_inputTensorList) != InferenceHelper::kRetOk) {
		return RET_ERR;
	}
	const auto& tPreProcess1 = std::chrono::steady_clock::now();

	/*** Inference ***/
	const auto& tInference0 = std::chrono::steady_clock::now();
	if (m_inferenceHelper->Process(m_outputTensorList) != InferenceHelper::kRetOk) {
		return RET_ERR;
	}
	const auto& tInference1 = std::chrono::steady_clock::now();

	/*** PostProcess ***/
	const auto& tPostProcess0 = std::chrono::steady_clock::now();

	/* Retrieve the result */
	float* valFloat = m_outputTensorList[0].GetDataAsFloat();
	std::vector<float> poseKeypointScores;	// z
	std::vector<std::pair<float,float>> poseKeypointCoords;	// x, y
	for (int32_t partIndex = 0; partIndex < m_outputTensorList[0].tensor_dims.width; partIndex++) {
		// PRINT("%f, %f, %f\n", valFloat[1], valFloat[0], valFloat[2]);
		poseKeypointCoords.push_back(std::pair<float,float>(valFloat[1], valFloat[0]));
		poseKeypointScores.push_back(valFloat[2]);
		valFloat += 3;
	}

	/* Find the max score */
	/* note: we have only one body with this model */
	result.poseScores.push_back(1.0);
	result.poseKeypointScores.push_back(poseKeypointScores);
	result.poseKeypointCoords.push_back(poseKeypointCoords);
	const auto& tPostProcess1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.time_pre_process = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.time_inference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
	result.time_post_process = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

	return RET_OK;
}
