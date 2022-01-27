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
#define USE_TFLITE

#ifdef USE_TFLITE
#define INPUT_NAME  "serving_default_src:0"
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_FGR "StatefulPartitionedCall:1"
#define OUTPUT_NAME_PHA "StatefulPartitionedCall:0"
#else  // ONNX
#define INPUT_NAME  "src"
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_FGR "fgr"
#define OUTPUT_NAME_PHA "pha"
#endif

#ifdef USE_TFLITE
#if 1
#define MODEL_NAME  "rvm_resnet50_720x1280.tflite"
#define INPUT_DIMS  { 1, 720, 1280, 3 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#elif 1
#define MODEL_NAME  "rvm_resnet50_1088x1920.tflite"
#define INPUT_DIMS  { 1, 1088, 1920, 3 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#endif
#else  // ONNX
#if 1
#define MODEL_NAME  "rvm_resnet50_720x1280.onnx"
#define INPUT_DIMS  { 1, 3, 720, 1280 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#elif 1
#define MODEL_NAME  "rvm_resnet50_1088x1920.onnx"
#define INPUT_DIMS  { 1, 3, 1088, 1920 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#endif
#endif


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
#if 0
    input_tensor_info.normalize.mean[0] = 0.485f;   // imagenet
    input_tensor_info.normalize.mean[1] = 0.456f;
    input_tensor_info.normalize.mean[2] = 0.406f;
    input_tensor_info.normalize.norm[0] = 0.229f;
    input_tensor_info.normalize.norm[1] = 0.224f;
    input_tensor_info.normalize.norm[2] = 0.225f;
#elif 1
    input_tensor_info.normalize.mean[0] = 0.0f; // 0.0 - 1.0
    input_tensor_info.normalize.mean[1] = 0.0f;
    input_tensor_info.normalize.mean[2] = 0.0f;
    input_tensor_info.normalize.norm[0] = 1.0f;
    input_tensor_info.normalize.norm[1] = 1.0f;
    input_tensor_info.normalize.norm[2] = 1.0f;
#else
    input_tensor_info.normalize.mean[0] = 0.0f; // 0.0 - 255.0
    input_tensor_info.normalize.mean[1] = 0.0f;
    input_tensor_info.normalize.mean[2] = 0.0f;
    input_tensor_info.normalize.norm[0] = 1.0f / 255.0f;
    input_tensor_info.normalize.norm[1] = 1.0f / 255.0f;
    input_tensor_info.normalize.norm[2] = 1.0f / 255.0f;
#endif
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_FGR, TENSORTYPE, IS_NCHW));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_PHA, TENSORTYPE, IS_NCHW));

    /* Create and Initialize Inference Helper */
#ifdef USE_TFLITE
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));
#else
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOpencv));  // not supporrted
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
    //std::vector<float> fgr_list(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_height * output_width * 3);
    //std::vector<float> pha_list(output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat() + output_height * output_width * 1);
    //printf("FGR: [%f, %f], %f, %f, %f\n", *std::min_element(fgr_list.begin(), fgr_list.end()), *std::max_element(fgr_list.begin(), fgr_list.end()), fgr_list[0], fgr_list[100], fgr_list[400]);
    //printf("PHA: [%f, %f], %f, %f, %f\n", *std::min_element(pha_list.begin(), pha_list.end()), *std::max_element(pha_list.begin(), pha_list.end()), pha_list[0], pha_list[100], pha_list[400]);
    cv::Mat mat_fgr = cv::Mat(output_height, output_width, CV_32FC3, output_tensor_info_list_[0].GetDataAsFloat()).clone();  // need to clone because the data itself is on tensor and will be deleted
    cv::Mat mat_pha = cv::Mat(output_height, output_width, CV_32FC1, output_tensor_info_list_[1].GetDataAsFloat()).clone();
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.mat_fgr = mat_fgr;
    result.mat_pha = mat_pha;
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}

