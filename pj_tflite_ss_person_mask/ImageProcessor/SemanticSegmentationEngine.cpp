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
#include "SemanticSegmentationEngine.h"

/*** Macro ***/
#define TAG "SemanticSegmentationEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "mobilenet_v3_segm_256.tflite"

/*** Function ***/
int32_t SemanticSegmentationEngine::initialize(const std::string& workDir, const int32_t numThreads)
{
	/* Set model information */
	std::string modelFilename = workDir + "/model/" + MODEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = "input_1";
	inputTensorInfo.tensorDims.batch = -1;	// tensor dims are retrieved from the model file. You can also set [1,224,224,3] here
	inputTensorInfo.tensorDims.width = -1;
	inputTensorInfo.tensorDims.height = -1;
	inputTensorInfo.tensorDims.channel = -1;
	inputTensorInfo.data = nullptr;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.normalize.mean[0] = 0.0f;
	inputTensorInfo.normalize.mean[1] = 0.0f;
	inputTensorInfo.normalize.mean[2] = 0.0f;
	inputTensorInfo.normalize.norm[0] = 1.0f;
	inputTensorInfo.normalize.norm[1] = 1.0f;
	inputTensorInfo.normalize.norm[2] = 1.0f;
#if 1
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif
	m_inputTensorList.push_back(inputTensorInfo);

	/* Set output tensor info */
	m_outputTensorList.clear();
	OutputTensorInfo outputTensorInfo;
	outputTensorInfo.name = "Identity";
	m_outputTensorList.push_back(outputTensorInfo);

	/* Create and Initialize Inference Helper */
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::OPEN_CV));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSOR_RT));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::NCNN));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::MNN));
	m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_EDGETPU));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_GPU));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_XNNPACK));

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

int32_t SemanticSegmentationEngine::finalize()
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	m_inferenceHelper->finalize();
	return RET_OK;
}


int32_t SemanticSegmentationEngine::invoke(const cv::Mat& originalMat, RESULT& result)
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	/*** PreProcess ***/
	const auto& tPreProcess0 = std::chrono::steady_clock::now();
	InputTensorInfo& inputTensorInfo = m_inputTensorList[0];
	/* do resize and color conversion here because some inference engine doesn't support these operations */
	cv::Mat imgSrc;
	cv::resize(originalMat, imgSrc, cv::Size(inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height));
	//cv::cvtColor(imgSrc, imgSrc, cv::COLOR_BGR2RGB);
	inputTensorInfo.data = imgSrc.data;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.imageInfo.width = imgSrc.cols;
	inputTensorInfo.imageInfo.height = imgSrc.rows;
	inputTensorInfo.imageInfo.channel = imgSrc.channels();
	inputTensorInfo.imageInfo.cropX = 0;
	inputTensorInfo.imageInfo.cropY = 0;
	inputTensorInfo.imageInfo.cropWidth = imgSrc.cols;
	inputTensorInfo.imageInfo.cropHeight = imgSrc.rows;
	inputTensorInfo.imageInfo.isBGR = true;
	inputTensorInfo.imageInfo.swapColor = false;
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
	int32_t outputWidth = m_outputTensorList[0].tensorDims.width;
	int32_t outputHeight = m_outputTensorList[0].tensorDims.height;
	int32_t outputCannel = m_outputTensorList[0].tensorDims.channel;
	float_t* values = static_cast<float_t*>(m_outputTensorList[0].data);
	cv::Mat maskImage = cv::Mat(outputHeight, outputWidth, CV_32FC1, values);
	maskImage.convertTo(maskImage, CV_8UC1, 255, 0);
	const auto& tPostProcess1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.maskImage = maskImage;
	result.timePreProcess = static_cast<std::chrono::duration<double_t>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.timeInference = static_cast<std::chrono::duration<double_t>>(tInference1 - tInference0).count() * 1000.0;
	result.timePostProcess = static_cast<std::chrono::duration<double_t>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

	return RET_OK;
}

