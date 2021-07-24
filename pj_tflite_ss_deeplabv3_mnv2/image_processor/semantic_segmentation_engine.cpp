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
	std::string model_filename = work_dir + "/model/" + MODEL_NAME;

	/* Set input tensor info */
	input_tensor_info_list_.clear();
	InputTensorInfo input_tensor_info("MobilenetV2/MobilenetV2/input", TensorInfo::kTensorTypeFp32);
	input_tensor_info.tensor_dims = { 1, 513, 513, 3 };
	input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
	input_tensor_info.normalize.mean[0] = 0.5f;
	input_tensor_info.normalize.mean[1] = 0.5f;
	input_tensor_info.normalize.mean[2] = 0.5f;
	input_tensor_info.normalize.norm[0] = 0.5f;
	input_tensor_info.normalize.norm[1] = 0.5f;
	input_tensor_info.normalize.norm[2] = 0.5f;
	input_tensor_info_list_.push_back(input_tensor_info);

	/* Set output tensor info */
	output_tensor_info_list_.clear();
	output_tensor_info_list_.push_back(OutputTensorInfo("ArgMax", TensorInfo::kTensorTypeFp32));

	/* Create and Initialize Inference Helper */
	inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
	//inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
	//inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
	//inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
	//inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));

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


	return kRetOk;
}

int32_t SemanticSegmentationEngine::Finalize()
{
	if (!inference_helper_) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	inference_helper_->Finalize();
	return kRetOk;
}


int32_t SemanticSegmentationEngine::Process(const cv::Mat& original_mat, Result& result)
{
	if (!inference_helper_) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	/*** PreProcess ***/
	const auto& t_pre_process0 = std::chrono::steady_clock::now();
	InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
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
	int32_t output_width = output_tensor_info_list_[0].tensor_dims.width;
	int32_t output_height = output_tensor_info_list_[0].tensor_dims.height;
	int32_t output_channel = output_tensor_info_list_[0].tensor_dims.channel;
	const int64_t* values = static_cast<int64_t*>(output_tensor_info_list_[0].data);
	cv::Mat image_mask = cv::Mat::zeros(output_height, output_width, CV_8UC3);
	for (int32_t y = 0; y < output_height; y++) {
		for (int32_t x = 0; x < output_width; x++) {
		}
	}
	for (int y = 0; y < output_height; y++) {
		for (int x = 0; x < output_width; x++) {
			int32_t max_channel = static_cast<int32_t>(values[y * output_width + x]);
			float color_ratio_b = (max_channel % 2 + 1) / 2.0f;
			float color_ratio_g = (max_channel % 3 + 1) / 3.0f;
			float color_ratio_r = (max_channel % 4 + 1) / 4.0f;
			image_mask.data[(y * output_width + x) * 3 + 0] = static_cast<uint8_t>(255 * color_ratio_b);
			image_mask.data[(y * output_width + x) * 3 + 1] = static_cast<uint8_t>(255 * color_ratio_g);
			image_mask.data[(y * output_width + x) * 3 + 2] = static_cast<uint8_t>(255 * (1 - color_ratio_r));

		}
	}
	const auto& t_post_process1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.image_mask = image_mask;
	result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
	result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
	result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

	return kRetOk;
}

