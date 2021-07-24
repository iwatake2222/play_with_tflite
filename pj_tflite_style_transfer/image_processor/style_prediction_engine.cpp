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
	std::string model_filename = work_dir + "/model/" + MODEL_NAME;

	/* Set input tensor info */
	input_tensor_info_list_.clear();
	InputTensorInfo input_tensor_info("style_image", TensorInfo::kTensorTypeFp32);
	input_tensor_info.tensor_dims = { 1, 256, 256, 3 };
	input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
	input_tensor_info.normalize.mean[0] = 0.0f;
	input_tensor_info.normalize.mean[1] = 0.0f;
	input_tensor_info.normalize.mean[2] = 0.0f;
	input_tensor_info.normalize.norm[0] = 1.0f;
	input_tensor_info.normalize.norm[1] = 1.0f;
	input_tensor_info.normalize.norm[2] = 1.0f;
	input_tensor_info_list_.push_back(input_tensor_info);

	/* Set output tensor info */
	output_tensor_info_list_.clear();
	output_tensor_info_list_.push_back(OutputTensorInfo("mobilenet_conv/Conv/BiasAdd", TensorInfo::kTensorTypeFp32));

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

int32_t StylePredictionEngine::Finalize()
{
	if (!inference_helper_) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	inference_helper_->Finalize();
	return kRetOk;
}


int32_t StylePredictionEngine::Process(const cv::Mat& original_mat, Result& result)
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
	const auto& t_post_process1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.style_bottleneck = output_tensor_info_list_[0].GetDataAsFloat();
	result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
	result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
	result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

	return kRetOk;
}

