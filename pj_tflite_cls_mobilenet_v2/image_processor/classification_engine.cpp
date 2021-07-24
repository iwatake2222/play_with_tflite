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
#include "classification_engine.h"

/*** Macro ***/
#define TAG "ClassificationEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */

#if 1
// FP32
#define MODEL_NAME  "mobilenet_v2_1.0_224.tflite"
#define INPUT_NAME  "input"
#define OUTPUT_NAME "MobilenetV2/Predictions/Reshape_1"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#elif 1
// UIINT8
#define MODEL_NAME  "mobilenet_v2_1.0_224_quant.tflite"
#define INPUT_NAME  "input"
#define OUTPUT_NAME "output"
#define TENSORTYPE  TensorInfo::kTensorTypeUint8
#elif 1
// UIINT8 + EDGETPU
#define MODEL_NAME  "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
#define INPUT_NAME  "input"
#define OUTPUT_NAME "output"
#define TENSORTYPE  TensorInfo::kTensorTypeUint8
#endif

#define LABEL_NAME   "imagenet_labels.txt"


/*** Function ***/
int32_t ClassificationEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
	/* Set model information */
	std::string modelFilename = work_dir + "/model/" + MODEL_NAME;
	std::string labelFilename = work_dir + "/model/" + LABEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = INPUT_NAME;
	inputTensorInfo.tensor_type= TENSORTYPE;
	inputTensorInfo.tensor_dims.batch = 1;
	inputTensorInfo.tensor_dims.width = 224;
	inputTensorInfo.tensor_dims.height = 224;
	inputTensorInfo.tensor_dims.channel = 3;
	inputTensorInfo.data_type = InputTensorInfo::kDataTypeImage;
	inputTensorInfo.normalize.mean[0] = 0.485f;   	/* https://github.com/onnx/models/tree/master/vision/classification/mobilenet#preprocessing */
	inputTensorInfo.normalize.mean[1] = 0.456f;
	inputTensorInfo.normalize.mean[2] = 0.406f;
	inputTensorInfo.normalize.norm[0] = 0.229f;
	inputTensorInfo.normalize.norm[1] = 0.224f;
	inputTensorInfo.normalize.norm[2] = 0.225f;
	m_inputTensorList.push_back(inputTensorInfo);

	/* Set output tensor info */
	m_outputTensorList.clear();
	m_outputTensorList.push_back(OutputTensorInfo(OUTPUT_NAME, TENSORTYPE));

	/* Create and Initialize Inference Helper */
	m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
	// m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
	// m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
	// m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
	// m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));

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

	/* read label */
	if (ReadLabel(labelFilename, m_labelList) != kRetOk) {
		return kRetErr;
	}


	return kRetOk;
}

int32_t ClassificationEngine::Finalize()
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	m_inferenceHelper->Finalize();
	return kRetOk;
}


int32_t ClassificationEngine::Process(const cv::Mat& original_mat, Result& result)
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	/*** PreProcess ***/
	const auto& tPreProcess0 = std::chrono::steady_clock::now();
	InputTensorInfo& inputTensorInfo = m_inputTensorList[0];
#if 1
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
#else
	/* Test other input format */
	cv::Mat imgSrc;
	inputTensorInfo.data = original_mat.data;
	inputTensorInfo.data_type = InputTensorInfo::kDataTypeImage;
	inputTensorInfo.image_info.width = original_mat.cols;
	inputTensorInfo.image_info.height = original_mat.rows;
	inputTensorInfo.image_info.channel = original_mat.channels();
	inputTensorInfo.image_info.crop_x = 0;
	inputTensorInfo.image_info.crop_y = 0;
	inputTensorInfo.image_info.crop_width = original_mat.cols;
	inputTensorInfo.image_info.crop_height = original_mat.rows;
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
	std::vector<float> outputScoreList;
	outputScoreList.resize(m_outputTensorList[0].tensor_dims.width * m_outputTensorList[0].tensor_dims.height * m_outputTensorList[0].tensor_dims.channel);
	const float* valFloat = m_outputTensorList[0].GetDataAsFloat();
	for (int32_t i = 0; i < (int32_t)outputScoreList.size(); i++) {
		outputScoreList[i] = valFloat[i];
	}

	/* Find the max score */
	int32_t maxIndex = (int32_t)(std::max_element(outputScoreList.begin(), outputScoreList.end()) - outputScoreList.begin());
	auto maxScore = *std::max_element(outputScoreList.begin(), outputScoreList.end());
	PRINT("Result = %s (%d) (%.3f)\n", m_labelList[maxIndex].c_str(), maxIndex, maxScore);
	const auto& tPostProcess1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.class_id = maxIndex;
	result.class_name = m_labelList[maxIndex];
	result.score = maxScore;
	result.time_pre_process = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.time_inference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
	result.time_post_process = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

	return kRetOk;
}


int32_t ClassificationEngine::ReadLabel(const std::string& filename, std::vector<std::string>& labelList)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		PRINT_E("Failed to read %s\n", filename.c_str());
		return kRetErr;
	}
	labelList.clear();
	if (with_background_) {
		labelList.push_back("background");
	}
	std::string str;
	while (getline(ifs, str)) {
		labelList.push_back(str);
	}
	return kRetOk;
}
