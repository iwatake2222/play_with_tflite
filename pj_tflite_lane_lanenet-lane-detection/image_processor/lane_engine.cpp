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
#include <numeric>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "lane_engine.h"

/*** Macro ***/
#define TAG "LaneEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define USE_TFLITE
#ifdef USE_TFLITE
#define MODEL_NAME  "lanenet-lane-detection.tflite"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input_tensor"
#define INPUT_DIMS  { 1, 256, 512, 3}
#define IS_NCHW     false
#define IS_RGB      false
#define OUTPUT_NAME_0 "LaneNet/bisenetv2_backend/binary_seg/ArgMax"
#define OUTPUT_NAME_1 "LaneNet/bisenetv2_backend/instance_seg/pix_embedding_conv/pix_embedding_conv"
#else
#define MODEL_NAME  "lanenet-lane-detection.onnx"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input_tensor"
#define INPUT_DIMS  { 1, 256, 512, 3}
#define IS_NCHW     false
#define IS_RGB      false
#define OUTPUT_NAME_0 "binary_seg_ret"
#define OUTPUT_NAME_1 "instance_seg_ret"
#endif

static constexpr int32_t kNumWidth = 512;
static constexpr int32_t kNumHeight = 256;

/*** Function ***/
int32_t LaneEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    /* -1.0 - 1.0 */
    input_tensor_info.normalize.mean[0] = 0.5f;
    input_tensor_info.normalize.mean[1] = 0.5f;
    input_tensor_info.normalize.mean[2] = 0.5f;
    input_tensor_info.normalize.norm[0] = 0.5f;
    input_tensor_info.normalize.norm[1] = 0.5f;
    input_tensor_info.normalize.norm[2] = 0.5f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
#ifdef USE_TFLITE
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TensorInfo::kTensorTypeInt64));
#else
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TensorInfo::kTensorTypeInt32));
#endif
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_1, TENSORTYPE));

    /* Create and Initialize Inference Helper */
#ifdef USE_TFLITE
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
    // inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
    // inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
    // inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));
#else
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOpencv));
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

int32_t LaneEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


int32_t LaneEngine::Process(const cv::Mat& original_mat, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];

    /* do resize and color conversion here because some inference engine doesn't support these operations */
    int32_t crop_x = 0;
    int32_t crop_y = 0; // original_mat.rows * 1 / 3;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows; // *2 / 3;
    cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeCut);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeExpand);

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
#ifdef USE_TFLITE
    std::vector<int64_t> output_binary(static_cast<int64_t*>(output_tensor_info_list_[0].data), static_cast<int64_t*>(output_tensor_info_list_[0].data) + output_tensor_info_list_[0].GetElementNum());
#else
    std::vector<int32_t> output_binary(static_cast<int32_t*>(output_tensor_info_list_[0].data), static_cast<int32_t*>(output_tensor_info_list_[0].data) + output_tensor_info_list_[0].GetElementNum());
#endif
    std::vector<float> output_pixel_embedding(output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat() + output_tensor_info_list_[1].GetElementNum());

    cv::Mat image_binary(kNumHeight, kNumWidth, CV_8UC1);
    for (int32_t i = 0; i < output_binary.size(); i++) {
        image_binary.data[i] = output_binary[i] * 255;
    }
    cv::resize(image_binary, image_binary, cv::Size(crop_w, crop_h));

    result.image_binary_seg = cv::Mat(original_mat.size(), CV_8UC1);
    cv::Rect crop = cv::Rect(crop_x < 0 ? 0 : crop_x, crop_y < 0 ? 0 : crop_y, crop_x < 0 ? original_mat.cols : crop_w, crop_h < 0 ? original_mat.rows : crop_h);
    cv::Mat target = result.image_binary_seg(crop);
    image_binary(cv::Rect(crop_x < 0 ? -crop_x : 0, crop_y < 0 ? -crop_y : 0, crop_x < 0 ? original_mat.cols : crop_w, crop_h < 0 ? original_mat.rows : crop_h)).copyTo(target);


    //cv::Mat image_instance(kNumHeight, kNumWidth, CV_8UC3);
    //for (int32_t i = 0; i < output_binary.size(); i++) {
    //    image_instance.data[i * 3 + 0] = output_pixel_embedding[i * 4 + 0] * 255;
    //    image_instance.data[i * 3 + 1] = std::min(255.f, output_pixel_embedding[i * 4 + 1] * 255 + output_pixel_embedding[i * 4 + 3] * 255);
    //    image_instance.data[i * 3 + 2] = std::min(255.f, output_pixel_embedding[i * 4 + 2] * 255 + output_pixel_embedding[i * 4 + 3] * 255);
    //}
    //cv::imshow("test", image_instance);
    //cv::waitKey(-1);

    const auto& t_post_process1 = std::chrono::steady_clock::now();


    /* Return the results */
    result.crop.x = (std::max)(0, crop_x);
    result.crop.y = (std::max)(0, crop_y);
    result.crop.w = (std::min)(crop_w, original_mat.cols - result.crop.x);
    result.crop.h = (std::min)(crop_h, original_mat.rows - result.crop.y);
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}
