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
#include "CommonHelper.h"
#include "InferenceHelper.h"
#include "PoseEngine.h"

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
#define TENSORTYPE  TensorInfo::TENSOR_TYPE_FP32
#else
/* PINTO_model_zoo. https://github.com/PINTO0309/PINTO_model_zoo */
#define MODEL_NAME  "model_float32.tflite"
// #define MODEL_NAME  "model_weight_quant.tflite"
// #define MODEL_NAME  "model_integer_quant.tflite"
#define INPUT_NAME  "input:0"
#define OUTPUT_NAME "Identity:0"
#define TENSORTYPE  TensorInfo::TENSOR_TYPE_FP32
#endif

/*** Function ***/
int32_t PoseEngine::initialize(const std::string& workDir, const int32_t numThreads)
{
	/* Set model information */
	std::string modelFilename = workDir + "/model/" + MODEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = INPUT_NAME;
	inputTensorInfo.tensorType = TENSORTYPE;
	inputTensorInfo.tensorDims.batch = 1;
	inputTensorInfo.tensorDims.width = 192;
	inputTensorInfo.tensorDims.height = 192;
	inputTensorInfo.tensorDims.channel = 3;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
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
	outputTensorInfo.tensorType = TENSORTYPE;
	outputTensorInfo.name = OUTPUT_NAME;
	m_outputTensorList.push_back(outputTensorInfo);

	/* Create and Initialize Inference Helper */
	m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE));
	// m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_XNNPACK));
	// m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_GPU));
	// m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_EDGETPU));
	// m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_NNAPI));

	if (!m_inferenceHelper) {
		return RET_ERR;
	}
	if (m_inferenceHelper->setNumThread(numThreads) != InferenceHelper::RET_OK) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}
	if (m_inferenceHelper->initialize(modelFilename, m_inputTensorList, m_outputTensorList) != InferenceHelper::RET_OK) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}
	/* Check if input tensor info is set */
	for (const auto& inputTensorInfo : m_inputTensorList) {
		if ((inputTensorInfo.tensorDims.width <= 0) || (inputTensorInfo.tensorDims.height <= 0) || inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_NONE) {
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
	m_inferenceHelper->finalize();
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
	cv::resize(originalMat, imgSrc, cv::Size(inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height));
#ifndef CV_COLOR_IS_RGB
	cv::cvtColor(imgSrc, imgSrc, cv::COLOR_BGR2RGB);
#endif
	inputTensorInfo.data = imgSrc.data;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.imageInfo.width = imgSrc.cols;
	inputTensorInfo.imageInfo.height = imgSrc.rows;
	inputTensorInfo.imageInfo.channel = imgSrc.channels();
	inputTensorInfo.imageInfo.cropX = 0;
	inputTensorInfo.imageInfo.cropY = 0;
	inputTensorInfo.imageInfo.cropWidth = imgSrc.cols;
	inputTensorInfo.imageInfo.cropHeight = imgSrc.rows;
	inputTensorInfo.imageInfo.isBGR = false;
	inputTensorInfo.imageInfo.swapColor = false;
#else
	/* Test other input format */
	cv::Mat imgSrc;
	inputTensorInfo.data = originalMat.data;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.imageInfo.width = originalMat.cols;
	inputTensorInfo.imageInfo.height = originalMat.rows;
	inputTensorInfo.imageInfo.channel = originalMat.channels();
	inputTensorInfo.imageInfo.cropX = 0;
	inputTensorInfo.imageInfo.cropY = 0;
	inputTensorInfo.imageInfo.cropWidth = originalMat.cols;
	inputTensorInfo.imageInfo.cropHeight = originalMat.rows;
	inputTensorInfo.imageInfo.isBGR = true;
	inputTensorInfo.imageInfo.swapColor = true;
#if 0
	InferenceHelper::preProcessByOpenCV(inputTensorInfo, false, imgSrc);
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_BLOB_NHWC;
#else
	InferenceHelper::preProcessByOpenCV(inputTensorInfo, true, imgSrc);
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_BLOB_NCHW;
#endif
	inputTensorInfo.data = imgSrc.data;
#endif
	if (m_inferenceHelper->preProcess(m_inputTensorList) != InferenceHelper::RET_OK) {
		return RET_ERR;
	}
	const auto& tPreProcess1 = std::chrono::steady_clock::now();

	/*** Inference ***/
	const auto& tInference0 = std::chrono::steady_clock::now();
	if (m_inferenceHelper->invoke(m_outputTensorList) != InferenceHelper::RET_OK) {
		return RET_ERR;
	}
	const auto& tInference1 = std::chrono::steady_clock::now();

	/*** PostProcess ***/
	const auto& tPostProcess0 = std::chrono::steady_clock::now();

	/* Retrieve the result */
	float* valFloat = m_outputTensorList[0].getDataAsFloat();
	std::vector<float> poseKeypointScores;	// z
	std::vector<std::pair<float,float>> poseKeypointCoords;	// x, y
	for (int32_t partIndex = 0; partIndex < m_outputTensorList[0].tensorDims.width; partIndex++) {
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
	result.timePreProcess = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.timeInference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
	result.timePostProcess = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

	return RET_OK;
}
