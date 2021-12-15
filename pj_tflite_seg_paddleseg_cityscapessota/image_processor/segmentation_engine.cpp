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
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "segmentation_engine.h"

/*** Macro ***/
#define TAG "SegmentationEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME  "paddleseg_cityscapessota_180x320.tflite"
#define INPUT_DIMS  { 1, 180, 320, 3 }
// #define MODEL_NAME  "paddleseg_cityscapessota_360x640.onnx"
// #define INPUT_DIMS  { 1, 3, 360, 640 }
#define INPUT_NAME  "serving_default_x:0"
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "StatefulPartitionedCall:0"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define OUTPUT_CHANNEL 19

/*** Function ***/
int32_t SegmentationEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.485f;
    input_tensor_info.normalize.mean[1] = 0.456f;
    input_tensor_info.normalize.mean[2] = 0.406f;
    input_tensor_info.normalize.norm[0] = 0.229f;
    input_tensor_info.normalize.norm[1] = 0.224f;
    input_tensor_info.normalize.norm[2] = 0.225f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME, TENSORTYPE, IS_NCHW));

    /* Create and Initialize Inference Helper */
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
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

    return kRetOk;
}

int32_t SegmentationEngine::Finalize()
{

    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


int32_t SegmentationEngine::Process(const cv::Mat& original_mat, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
    /* do resize and color conversion here because some inference engine doesn't support these operations */
    float ratio = static_cast<float>(input_tensor_info.GetWidth()) / input_tensor_info.GetHeight();
    int32_t crop_x = 0;
    int32_t crop_y = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows;
    cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);

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
    const int32_t output_height = input_tensor_info.image_info.height;
    const int32_t output_width = input_tensor_info.image_info.width;
    const std::vector<float> value_list(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_height * output_width * OUTPUT_CHANNEL);
    //printf("%f, %f, %f\n", value_list[0], value_list[100], value_list[400]);

    /* Scores for all the classes */
    std::vector<cv::Mat> mat_separated_list(OUTPUT_CHANNEL);
    for (int32_t c = 0; c < OUTPUT_CHANNEL; c++) {
        mat_separated_list[c] = cv::Mat::zeros(output_height, output_width, CV_32FC1);
    }
 #pragma omp parallel for
    for (int32_t y = 0; y < output_height; y++) {
        for (int32_t x = 0; x < output_width; x++) {
#if 1
            /* Use Score [0.0, 1.0] */
            size_t offset = (size_t)y * output_width * OUTPUT_CHANNEL + (size_t)x * OUTPUT_CHANNEL;
            std::vector<float> score_list(OUTPUT_CHANNEL, 0);
            CommonHelper::SoftMaxFast(value_list.data() + offset, score_list.data(), OUTPUT_CHANNEL);
            for (int32_t c = 0; c < OUTPUT_CHANNEL; c++) {
                mat_separated_list[c].at<float>(cv::Point(x, y)) = score_list[c];
            }
#else
            /* Use Logit */
            size_t offset = (size_t)y * output_width * OUTPUT_CHANNEL + (size_t)x * OUTPUT_CHANNEL;
            for (int32_t c = 0; c < OUTPUT_CHANNEL; c++) {
                float val = value_list[offset + c] / 20;    // experimentally determined
                mat_separated_list[c].at<float>(cv::Point(x, y)) = (std::min)(1.0f, (std::max)(0.0f, val));
            }
#endif
        }
    }

    /* Argmax */
    /* ref: https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.3/paddleseg/core/infer.py#L244 */
    cv::Mat mat_max = cv::Mat::zeros(output_height, output_width, CV_8UC1);
#pragma omp parallel for
    for (int32_t y = 0; y < output_height; y++) {
        for (int32_t x = 0; x < output_width; x++) {
            const auto& current_iter = value_list.begin() + y * output_width * OUTPUT_CHANNEL + x * OUTPUT_CHANNEL;
            const auto& max_iter = std::max_element(current_iter, current_iter + OUTPUT_CHANNEL);
            float max_score = *max_iter;
            auto max_c = std::distance(current_iter, max_iter);
            mat_max.at<uint8_t>(cv::Point(x, y)) = static_cast<uint8_t>(max_c);
        }
    }
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.mat_out_list = mat_separated_list;
    result.mat_out_max = mat_max;
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}

