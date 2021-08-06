/* Copyright 2021 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
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
#if 0
/* Official model. https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3 */
#define MODEL_NAME  "lite-model_movenet_singlepose_lightning_3.tflite"
#define INPUT_NAME  "serving_default_input:0"
#define IS_NCHW     false
#define INPUT_DIMS  { 1, 192, 192, 3 }
#define OUTPUT_NAME "StatefulPartitionedCall:0"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#else
/* PINTO_model_zoo. https://github.com/PINTO0309/PINTO_model_zoo */
#define MODEL_NAME  "movenet_lightning.tflite"
#define INPUT_NAME  "serving_default_input_0:0"
#define IS_NCHW     false
#define INPUT_DIMS  { 1, 192, 192, 3 }
#define OUTPUT_NAME "StatefulPartitionedCall_0:0"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#endif

/*** Function ***/
int32_t PoseEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    // input_tensor_info.normalize.mean[0] = 0.485f;   	/* https://github.com/onnx/models/tree/master/vision/classification/mobilenet#preprocessing */
    // input_tensor_info.normalize.mean[1] = 0.456f;
    // input_tensor_info.normalize.mean[2] = 0.406f;
    // input_tensor_info.normalize.norm[0] = 0.229f;
    // input_tensor_info.normalize.norm[1] = 0.224f;
    // input_tensor_info.normalize.norm[2] = 0.225f;
    /* 0 - 255 (https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3) */
    input_tensor_info.normalize.mean[0] = 0;
    input_tensor_info.normalize.mean[1] = 0;
    input_tensor_info.normalize.mean[2] = 0;
    input_tensor_info.normalize.norm[0] = 1/255.f;
    input_tensor_info.normalize.norm[1] = 1/255.f;
    input_tensor_info.normalize.norm[2] = 1/255.f;
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

    return kRetOk;
}

int32_t PoseEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


int32_t PoseEngine::Process(const cv::Mat& original_mat, Result& result)
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
    cv::resize(original_mat, img_src, cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()));
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
    float* val_float = output_tensor_info_list_[0].GetDataAsFloat();
    std::vector<float> pose_keypoint_scores;	// z
    std::vector<std::pair<float,float>> pose_keypoint_coords;	// x, y
    for (int32_t partIndex = 0; partIndex < output_tensor_info_list_[0].tensor_dims[2]; partIndex++) {
        // PRINT("%f, %f, %f\n", val_float[1], val_float[0], val_float[2]);
        pose_keypoint_coords.push_back(std::pair<float,float>(val_float[1], val_float[0]));
        pose_keypoint_scores.push_back(val_float[2]);
        val_float += 3;
    }

    /* Find the max score */
    /* note: we have only one body with this model */
    result.pose_scores.push_back(1.0);
    result.pose_keypoint_scores.push_back(pose_keypoint_scores);
    result.pose_keypoint_coords.push_back(pose_keypoint_coords);
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}
