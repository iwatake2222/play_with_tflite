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
	std::string model_filename = work_dir + "/model/" + MODEL_NAME;
	std::string labelFilename = work_dir + "/model/" + LABEL_NAME;

	/* Set input tensor info */
	input_tensor_info_list_.clear();
	InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE);
	input_tensor_info.tensor_dims = { 1, 224, 224, 3 };
	input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
	input_tensor_info.normalize.mean[0] = 0.485f;   	/* https://github.com/onnx/models/tree/master/vision/classification/mobilenet#preprocessing */
	input_tensor_info.normalize.mean[1] = 0.456f;
	input_tensor_info.normalize.mean[2] = 0.406f;
	input_tensor_info.normalize.norm[0] = 0.229f;
	input_tensor_info.normalize.norm[1] = 0.224f;
	input_tensor_info.normalize.norm[2] = 0.225f;
	input_tensor_info_list_.push_back(input_tensor_info);

	/* Set output tensor info */
	output_tensor_info_list_.clear();
	output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME, TENSORTYPE));

	/* Create and Initialize Inference Helper */
	inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
	// inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
	// inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
	// inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
	// inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));

	if (!inference_helper_) {
		return kRetErr;
	}
	if (inference_helper_->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
		inference_helper_.reset();
		return kRetErr;
	}
	if (inference_helper_->Initialize(model_filename, input_tensor_info_list_, output_tensor_info_list_) != InferenceHelper::kRetOk) {
		inference_helper_.reset();
		return kRetErr;
	}
	/* Check if input tensor info is set */
	for (const auto& input_tensor_info : input_tensor_info_list_) {
		if ((input_tensor_info.tensor_dims.width <= 0) || (input_tensor_info.tensor_dims.height <= 0) || input_tensor_info.tensor_type == TensorInfo::kTensorTypeNone) {
			PRINT_E("Invalid tensor size\n");
			inference_helper_.reset();
			return kRetErr;
		}
	}

	/* read label */
	if (ReadLabel(labelFilename, label_list_) != kRetOk) {
		return kRetErr;
	}


	return kRetOk;
}

int32_t ClassificationEngine::Finalize()
{
	if (!inference_helper_) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	inference_helper_->Finalize();
	return kRetOk;
}


int32_t ClassificationEngine::Process(const cv::Mat& original_mat, Result& result)
{
	if (!inference_helper_) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	/*** PreProcess ***/
	const auto& t_pre_process0 = std::chrono::steady_clock::now();
	InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
#if 1
	/* do resize and color conversion here because some inference engine doesn't support these operations */
	cv::Mat img_src;
	cv::resize(original_mat, img_src, cv::Size(input_tensor_info.tensor_dims.width, input_tensor_info.tensor_dims.height));
#ifndef CV_COLOR_IS_RGB
	cv::cvtColor(img_src, img_src, cv::COLOR_BGR2RGB);
#endif
	input_tensor_info.data = img_src.data;
	input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
	input_tensor_info.image_info.width = img_src.cols;
	input_tensor_info.image_info.height = img_src.rows;
	input_tensor_info.image_info.channel = img_src.channels();
	input_tensor_info.image_info.crop_x = 0;
	input_tensor_info.image_info.crop_y = 0;
	input_tensor_info.image_info.crop_width = img_src.cols;
	input_tensor_info.image_info.crop_height = img_src.rows;
	input_tensor_info.image_info.is_bgr = false;
	input_tensor_info.image_info.swap_color = false;
#else
	/* Test other input format */
	cv::Mat img_src;
	input_tensor_info.data = original_mat.data;
	input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
	input_tensor_info.image_info.width = original_mat.cols;
	input_tensor_info.image_info.height = original_mat.rows;
	input_tensor_info.image_info.channel = original_mat.channels();
	input_tensor_info.image_info.crop_x = 0;
	input_tensor_info.image_info.crop_y = 0;
	input_tensor_info.image_info.crop_width = original_mat.cols;
	input_tensor_info.image_info.crop_height = original_mat.rows;
	input_tensor_info.image_info.is_bgr = true;
	input_tensor_info.image_info.swap_color = true;
#if 0
	InferenceHelper::PreProcessByOpenCV(input_tensor_info, false, img_src);
	input_tensor_info.data_type = InputTensorInfo::DATA_TYPE_BLOB_NHWC;
#else
	InferenceHelper::PreProcessByOpenCV(input_tensor_info, true, img_src);
	input_tensor_info.data_type = InputTensorInfo::DATA_TYPE_BLOB_NCHW;
#endif
	input_tensor_info.data = img_src.data;
#endif
	if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk) {
		return kRetErr;
	}
	const auto& t_pre_process1 = std::chrono::steady_clock::now();

	/*** Inference ***/
	const auto& t_inference0 = std::chrono::steady_clock::now();
	if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk) {
		return kRetErr;
	}
	const auto& t_inference1 = std::chrono::steady_clock::now();

	/*** PostProcess ***/
	const auto& t_post_process0 = std::chrono::steady_clock::now();
	/* Retrieve the result */
	std::vector<float> output_score_list;
	output_score_list.resize(output_tensor_info_list_[0].tensor_dims.width * output_tensor_info_list_[0].tensor_dims.height * output_tensor_info_list_[0].tensor_dims.channel);
	const float* val_float = output_tensor_info_list_[0].GetDataAsFloat();
	for (int32_t i = 0; i < (int32_t)output_score_list.size(); i++) {
		output_score_list[i] = val_float[i];
	}

	/* Find the max score */
	int32_t max_index = (int32_t)(std::max_element(output_score_list.begin(), output_score_list.end()) - output_score_list.begin());
	auto max_score = *std::max_element(output_score_list.begin(), output_score_list.end());
	PRINT("Result = %s (%d) (%.3f)\n", label_list_[max_index].c_str(), max_index, max_score);
	const auto& t_post_process1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.class_id = max_index;
	result.class_name = label_list_[max_index];
	result.score = max_score;
	result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
	result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
	result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

	return kRetOk;
}


int32_t ClassificationEngine::ReadLabel(const std::string& filename, std::vector<std::string>& label_list)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		PRINT_E("Failed to read %s\n", filename.c_str());
		return kRetErr;
	}
	label_list.clear();
	if (with_background_) {
		label_list.push_back("background");
	}
	std::string str;
	while (getline(ifs, str)) {
		label_list.push_back(str);
	}
	return kRetOk;
}
