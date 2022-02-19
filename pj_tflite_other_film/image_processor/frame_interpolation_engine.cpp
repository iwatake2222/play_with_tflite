/* Copyright 2022 iwatake2222

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
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "frame_interpolation_engine.h"

/*** Macro ***/
#define TAG "FrameInterpolationEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define USE_TFLITE
#ifdef USE_TFLITE
#define MODEL_NAME   "film_net_VGG_480x640.tflite"
//#define MODEL_NAME   "film_net_L1_480x640.tflite"
#define INPUT_DIMS   { 1, 480, 640, 3 }
#define INPUT_NAME0  "x0"
#define INPUT_NAME1  "x1"
#define INPUT_NAME2  "time"
#define IS_NCHW      false
#define OUTPUT_NAME  "Identity"
#else
#define MODEL_NAME   "film_net_VGG_480x640.onnx"
//#define MODEL_NAME   "film_net_L1_480x640.onnx"
#define INPUT_DIMS   { 1, 3, 480, 640 }
#define INPUT_NAME0  "x0:0"
#define INPUT_NAME1  "x1:0"
#define INPUT_NAME2  "time:0"
#define IS_NCHW      true
#define OUTPUT_NAME  "tf.__operators__.add_12"
#endif
#define IS_RGB      true
#define TENSORTYPE  TensorInfo::kTensorTypeFp32

/*** Function ***/
int32_t FrameInterpolationEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME0, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    /* Normalize [0.0, 1.0]*/
    input_tensor_info.normalize.mean[0] = 0.0f;
    input_tensor_info.normalize.mean[1] = 0.0f;
    input_tensor_info.normalize.mean[2] = 0.0f;
    input_tensor_info.normalize.norm[0] = 1.0f;
    input_tensor_info.normalize.norm[1] = 1.0f;
    input_tensor_info.normalize.norm[2] = 1.0f;
    input_tensor_info_list_.push_back(input_tensor_info);
    input_tensor_info.name = INPUT_NAME1;
    input_tensor_info_list_.push_back(input_tensor_info);

    InputTensorInfo input_tensor_info_time(INPUT_NAME2, TENSORTYPE, IS_NCHW);
    input_tensor_info_time.tensor_dims = { 1, 1 };
    input_tensor_info_time.data_type = InputTensorInfo::kDataTypeBlobNhwc;
    input_tensor_info_list_.push_back(input_tensor_info_time);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME, TENSORTYPE, IS_NCHW));

    /* Create and Initialize Inference Helper */
#ifdef USE_TFLITE
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
#else
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorrt));
#endif
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

int32_t FrameInterpolationEngine::Finalize()
{

    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


int32_t FrameInterpolationEngine::Process(const cv::Mat& image_0, const cv::Mat& image_1, float time, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    
    /* do resize and color conversion here because some inference engine doesn't support these operations */
    float ratio = static_cast<float>(input_tensor_info_list_[0].GetWidth()) / input_tensor_info_list_[0].GetHeight();
    int32_t crop_x = 0;
    int32_t crop_y = 0;
    int32_t crop_w = image_0.cols;
    int32_t crop_h = image_0.rows;
    cv::Mat img_src_0 = cv::Mat::zeros(input_tensor_info_list_[0].GetHeight(), input_tensor_info_list_[0].GetWidth(), CV_8UC3);
    cv::Mat img_src_1 = cv::Mat::zeros(input_tensor_info_list_[1].GetHeight(), input_tensor_info_list_[1].GetWidth(), CV_8UC3);
    CommonHelper::CropResizeCvt(image_0, img_src_0, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);
    CommonHelper::CropResizeCvt(image_1, img_src_1, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);

    input_tensor_info_list_[0].data = img_src_0.data;
    input_tensor_info_list_[0].data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info_list_[0].image_info.width = img_src_0.cols;
    input_tensor_info_list_[0].image_info.height = img_src_0.rows;
    input_tensor_info_list_[0].image_info.channel = img_src_0.channels();
    input_tensor_info_list_[0].image_info.crop_x = 0;
    input_tensor_info_list_[0].image_info.crop_y = 0;
    input_tensor_info_list_[0].image_info.crop_width = img_src_0.cols;
    input_tensor_info_list_[0].image_info.crop_height = img_src_0.rows;
    input_tensor_info_list_[0].image_info.is_bgr = false;
    input_tensor_info_list_[0].image_info.swap_color = false;
    input_tensor_info_list_[1].data = img_src_1.data;
    input_tensor_info_list_[1].data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info_list_[1].image_info.width = img_src_1.cols;
    input_tensor_info_list_[1].image_info.height = img_src_1.rows;
    input_tensor_info_list_[1].image_info.channel = img_src_1.channels();
    input_tensor_info_list_[1].image_info.crop_x = 0;
    input_tensor_info_list_[1].image_info.crop_y = 0;
    input_tensor_info_list_[1].image_info.crop_width = img_src_1.cols;
    input_tensor_info_list_[1].image_info.crop_height = img_src_1.rows;
    input_tensor_info_list_[1].image_info.is_bgr = false;
    input_tensor_info_list_[1].image_info.swap_color = false;
    input_tensor_info_list_[2].data = &time;
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
    const int32_t output_height = input_tensor_info_list_[0].image_info.height;
    const int32_t output_width = input_tensor_info_list_[0].image_info.width;
    //const std::vector<float> value_list(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_height * output_width * 3);
    //printf("%f, %f, %f\n", value_list[0], value_list[100], value_list[400]);
    cv::Mat mat_out_fp32(cv::Size(output_width, output_height), CV_32FC3, output_tensor_info_list_[0].GetDataAsFloat());
    cv::Mat mat_out;
    mat_out_fp32.convertTo(mat_out, CV_8UC3, 255);
    if (IS_RGB) cv::cvtColor(mat_out, mat_out, cv::COLOR_RGB2BGR);
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.mat_out = mat_out;
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}

