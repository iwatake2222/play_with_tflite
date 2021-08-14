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

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "detection_engine.h"

/*** Macro ***/
#define TAG "DetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME  "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite"
#define TENSORTYPE_INPUT  TensorInfo::kTensorTypeUint8
#define INPUT_NAME  "normalized_input_image_tensor"
#define INPUT_DIMS  { 1, 300, 300, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_0 "TFLite_Detection_PostProcess"   /* The number of the detected boxes. */
#define OUTPUT_NAME_1 "TFLite_Detection_PostProcess:1"   /* The scores of the detected boxes. */
#define OUTPUT_NAME_2 "TFLite_Detection_PostProcess:2"   /* The categories of the detected boxes. */
#define OUTPUT_NAME_3 "TFLite_Detection_PostProcess:3"   /* The locations of the detected boxes. */
#define TENSORTYPE_OUTPUT  TensorInfo::kTensorTypeFp32

#define LABEL_NAME   "label_coco_90.txt"


/*** Function ***/
int32_t DetectionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;
    std::string labelFilename = work_dir + "/model/" + LABEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE_INPUT, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.0f;
    input_tensor_info.normalize.mean[1] = 0.0f;
    input_tensor_info.normalize.mean[2] = 0.0f;
    input_tensor_info.normalize.norm[0] = 1.0f / 255.0f;
    input_tensor_info.normalize.norm[1] = 1.0f / 255.0f;
    input_tensor_info.normalize.norm[2] = 1.0f / 255.0f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TENSORTYPE_OUTPUT));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_1, TENSORTYPE_OUTPUT));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_2, TENSORTYPE_OUTPUT));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_3, TENSORTYPE_OUTPUT));

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

    /* read label */
    if (ReadLabel(labelFilename, label_list_) != kRetOk) {
        return kRetErr;
    }

    return kRetOk;
}

int32_t DetectionEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


int32_t DetectionEngine::Process(const cv::Mat& original_mat, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
    /* do crop, resize and color conversion here because some inference engine doesn't support these operations */
    int32_t crop_x = 0;
    int32_t crop_y = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows;
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

    float* num_raw_list = output_tensor_info_list_[3].GetDataAsFloat();
    float* score_raw_list = output_tensor_info_list_[2].GetDataAsFloat();
    float* label_raw_list = output_tensor_info_list_[1].GetDataAsFloat();
    float* bbox_raw_list = output_tensor_info_list_[0].GetDataAsFloat();
    int32_t num_det = static_cast<int32_t>(num_raw_list[0]);

    /* Get boundig box */
    std::vector<BoundingBox> bbox_list;
    for (int32_t i = 0; i < num_det; i++) {
        if (score_raw_list[i] < threshold_confidence_) continue;
        BoundingBox bbox;
        bbox.class_id = static_cast<int32_t>(label_raw_list[i]);
        bbox.label = label_list_[bbox.class_id];
        bbox.score = score_raw_list[i];
        bbox.x = static_cast<int32_t>(bbox_raw_list[i * 4 + 1] * crop_w) + crop_x;
        bbox.y = static_cast<int32_t>(bbox_raw_list[i * 4 + 0] * crop_h) + crop_y;
        bbox.w = static_cast<int32_t>((bbox_raw_list[i * 4 + 3] - bbox_raw_list[i * 4 + 1]) * crop_w);
        bbox.h = static_cast<int32_t>((bbox_raw_list[i * 4 + 2] - bbox_raw_list[i * 4 + 0]) * crop_h);
        bbox_list.push_back(bbox);
    }

    /* NMS */
    std::vector<BoundingBox> bbox_nms_list;
    BoundingBoxUtils::Nms(bbox_list, bbox_nms_list, threshold_nms_iou_);

    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.bbox_list = bbox_nms_list;
    result.crop.x = (std::max)(0, crop_x);
    result.crop.y = (std::max)(0, crop_y);
    result.crop.w = (std::min)(crop_w, original_mat.cols - result.crop.x);
    result.crop.h = (std::min)(crop_h, original_mat.rows - result.crop.y);
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;


    return kRetOk;
}


int32_t DetectionEngine::ReadLabel(const std::string& filename, std::vector<std::string>& label_list)
{
    std::ifstream ifs(filename);
    if (ifs.fail()) {
        PRINT_E("Failed to read %s\n", filename.c_str());
        return kRetErr;
    }
    label_list.clear();
    std::string str;
    while (getline(ifs, str)) {
        label_list.push_back(str);
    }
    return kRetOk;
}

