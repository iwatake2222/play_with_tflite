/* Copyright 2020 iwatake2222

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
#include "camera_calibration_engine.h"

/*** Macro ***/
#define TAG "CameraCalibrationEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
//#define MODEL_TYPE_CLASSIFICATION
#define MODEL_TYPE_REGRESSION

#define MODEL_NAME  "deep_calib_regresion.tflite"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "main_input"
#define INPUT_DIMS  { 1, 299, 299, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_0 "Identity"
#define OUTPUT_NAME_1 "Identity_1"

static constexpr float kDistStart = 0;
static constexpr float kDistEnd = 60 / 50.0f;
static constexpr float kDistInterval = 1.0f / 50.0f;

static constexpr float kFocalStart = 40;
static constexpr float kFocalEnd = 500;
static constexpr float kFocalInterval = 10;


/*** Function ***/
int32_t CameraCalibrationEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.5f;   /* [-1.0, 1.0] */
    input_tensor_info.normalize.mean[1] = 0.5f;
    input_tensor_info.normalize.mean[2] = 0.5f;
    input_tensor_info.normalize.norm[0] = 0.5f;
    input_tensor_info.normalize.norm[1] = 0.5f;
    input_tensor_info.normalize.norm[2] = 0.5f;

    /* !!! It's a strange part, but just follow the original logic !!! */
    /* 1. normalize to [-1.0. 1.0] */
    /* 2. do keras.applications.imagenet_utils.preprocess_input */
    /*     substract [103.939, 116.779, 123.68] */
    /* https://github.com/alexvbogdan/DeepCalib/blob/master/prediction/Regression/Single_net/predict_regressor_dist_focal_to_textfile.py#L94 */
    input_tensor_info.normalize.mean[0] += 0.5f * 103.939f;
    input_tensor_info.normalize.mean[1] += 0.5f * 116.779f;
    input_tensor_info.normalize.mean[2] += 0.5f * 123.68f;

    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_1, TENSORTYPE));

    /* Create and Initialize Inference Helper */
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
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

    for (float i = kDistStart; i < kDistEnd; i += kDistInterval) class_dist_list_.push_back(i);
    for (float i = kFocalStart; i < kFocalEnd + 1; i += kFocalInterval) class_focal_list_.push_back(i);

    return kRetOk;
}

int32_t CameraCalibrationEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


int32_t CameraCalibrationEngine::Process(const cv::Mat& original_mat, Result& result)
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
    int32_t crop_y = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows;
    cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeCut);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeExpand);

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
    std::vector<float> xi_list(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_tensor_info_list_[0].GetElementNum());
    std::vector<float> f_list(output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat() + output_tensor_info_list_[1].GetElementNum());

#if defined(MODEL_TYPE_CLASSIFICATION)
    //for (int32_t i = 0; i < xi_list.size(); i++) {
    //    printf("%d:  %f\n", i, xi_list[i]);
    //}
    //for (int32_t i = 0; i < f_list.size(); i++) {
    //    printf("%d:  %f\n", i, f_list[i]);
    //}
    float xi = class_dist_list_[GetMaxIndex(xi_list)];
    float f = class_focal_list_[GetMaxIndex(f_list)];
    //printf("%f %f\n", xi, f);

#elif defined(MODEL_TYPE_REGRESSION)
    //printf("%f %f\n", xi_list[0], f_list[0]);
    float xi = xi_list[0] * 1.2f;
    float f = f_list[0] * (kFocalEnd + 1.0f - kFocalStart) + kFocalStart;
#else
    error
#endif

    f = f * crop_w / input_tensor_info.GetWidth();      /* Focal length on the original image size */
    PRINT("xi: %f,  f: %f\n", xi, f);

    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.xi = xi;
    result.focal_length = f;
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}


int32_t CameraCalibrationEngine::GetMaxIndex(std::vector<float> value_list)
{
    /* argsort */
    std::vector<size_t> indices(value_list.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&value_list](size_t i1, size_t i2) {
        return value_list[i1] > value_list[i2];
    });

    return static_cast<int32_t>(indices[0]);
}
