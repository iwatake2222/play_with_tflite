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
#include "detection_engine.h"

/*** Macro ***/
#define TAG "DetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite"
#define LABEL_NAME   "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.txt"


/*** Function ***/
int32_t DetectionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
	/* Set model information */
	std::string modelFilename = work_dir + "/model/" + MODEL_NAME;
	std::string labelFilename = work_dir + "/model/" + LABEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = "normalized_input_image_tensor";
	inputTensorInfo.tensor_type= TensorInfo::kTensorTypeUint8;
	inputTensorInfo.tensor_dims.batch = 1;
	inputTensorInfo.tensor_dims.width = 300;
	inputTensorInfo.tensor_dims.height = 300;
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
	outputTensorInfo.name = "TFLite_Detection_PostProcess";
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = "TFLite_Detection_PostProcess:1";
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = "TFLite_Detection_PostProcess:2";
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = "TFLite_Detection_PostProcess:3";
	m_outputTensorList.push_back(outputTensorInfo);

	/* Create and Initialize Inference Helper */
	m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
	// m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
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

int32_t DetectionEngine::Finalize()
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	m_inferenceHelper->Finalize();
	return kRetOk;
}


int32_t DetectionEngine::Process(const cv::Mat& original_mat, Result& result)
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
	/* Retrieve result */
	int32_t outputNum = (int32_t)(m_outputTensorList[3].GetDataAsFloat()[0]);
	std::vector<OBJECT> object_list;
	getObject(object_list, m_outputTensorList[0].GetDataAsFloat(), m_outputTensorList[1].GetDataAsFloat(), m_outputTensorList[2].GetDataAsFloat(), outputNum, 0.5, original_mat.cols, original_mat.rows);
	const auto& tPostProcess1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.object_list = object_list;
	result.time_pre_process = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.time_inference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
	result.time_post_process = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

	return kRetOk;
}


int32_t DetectionEngine::ReadLabel(const std::string& filename, std::vector<std::string>& labelList)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		PRINT_E("Failed to read %s\n", filename.c_str());
		return kRetErr;
	}
	labelList.clear();
	std::string str;
	while (getline(ifs, str)) {
		labelList.push_back(str);
	}
	return kRetOk;
}


int32_t DetectionEngine::getObject(std::vector<OBJECT>& object_list, const float *outputBoxList, const float *outputClassList, const float *outputScoreList, const int32_t outputNum,
	const double threshold, const int32_t width, const int32_t height)
{
	for (int32_t i = 0; i < outputNum; i++) {
		int32_t class_id = static_cast<int32_t>(outputClassList[i] + 1);
		float score = outputScoreList[i];
		if (score < threshold) continue;
		float y0 = outputBoxList[4 * i + 0];
		float x0 = outputBoxList[4 * i + 1];
		float y1 = outputBoxList[4 * i + 2];
		float x1 = outputBoxList[4 * i + 3];
		if (width > 0) {
			x0 *= width;
			x1 *= width;
			y0 *= height;
			y1 *= height;
		}
		//PRINT("%d[%.2f]: %.3f %.3f %.3f %.3f\n", class_id, score, x0, y0, x1, y1);
		OBJECT object;
		object.x = x0;
		object.y = y0;
		object.width = x1 - x0;
		object.height = y1 - y0;
		object.class_id = class_id;
		object.label = m_labelList[class_id];
		object.score = score;
		object_list.push_back(object);
	}
	return kRetOk;
}
