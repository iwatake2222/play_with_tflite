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
#include "detection_engine.h"

/*** Macro ***/
#define TAG "DetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME  "yolov5_416x416.tflite"
#define INPUT_NAME  "input_1:0"
#define INPUT_DIMS  { 1, 416, 416, 3 }
#define OUTPUT_NAME "Identity:0"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
static constexpr int32_t grid_scale_list[] = { 8, 16, 32 };
static constexpr int32_t grid_channl = 3;;
static constexpr int32_t kNumberOfClass = 80;
static constexpr int32_t kElementNumOfAnchor = kNumberOfClass + 5;    // x, y, w, h, Objectness score, [class probabilities]


#define LABEL_NAME   "label_coco_80.txt"


static constexpr float kThresholdScore = 0.2f;
static constexpr float kThresholdNmsIou = 0.5f;

/*** Function ***/
int32_t DetectionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;
    std::string labelFilename = work_dir + "/model/" + LABEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.0f;     /* 0.0 - 1.0*/
    input_tensor_info.normalize.mean[1] = 0.0f;
    input_tensor_info.normalize.mean[2] = 0.0f;
    input_tensor_info.normalize.norm[0] = 1.0f;
    input_tensor_info.normalize.norm[1] = 1.0f;
    input_tensor_info.normalize.norm[2] = 1.0f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME, TENSORTYPE));

    /* Create and Initialize Inference Helper */
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
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

int32_t DetectionEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


static void GetBoundingBox(const float* data, float scale_x, float  scale_y, int32_t grid_w, int32_t grid_h, std::vector<BoundingBox>& bbox_list)
{
    int32_t index = 0;
    for (int32_t grid_y = 0; grid_y < grid_h; grid_y++) {
        for (int32_t grid_x = 0; grid_x < grid_w; grid_x++) {
            for (int32_t grid_c = 0; grid_c < grid_channl; grid_c++) {
                float box_confidence = data[index + 4];
                if (box_confidence >= kThresholdScore) {
                    int32_t class_id = 0;
                    float confidence = 0;
                    for (int32_t class_index = 0; class_index < kNumberOfClass; class_index++) {
                        float confidence_of_class = data[index + 5 + class_index];
                        if (confidence_of_class > confidence) {
                            confidence = confidence_of_class;
                            class_id = class_index;
                        }
                    }

                    if (confidence >= kThresholdScore) {
                        int32_t cx = static_cast<int32_t>((data[index + 0] + 0) * scale_x);     // no need to + grid_x
                        int32_t cy = static_cast<int32_t>((data[index + 1] + 0) * scale_y);     // no need to + grid_y
                        int32_t w = static_cast<int32_t>(data[index + 2] * scale_x);            // no need to exp
                        int32_t h = static_cast<int32_t>(data[index + 3] * scale_y);            // no need to exp
                        int32_t x = cx - w / 2;
                        int32_t y = cy - h / 2;
                        bbox_list.push_back(BoundingBox(class_id, "", confidence, x, y, w, h));
                    }
                }
                index += kElementNumOfAnchor;
            }
        }
    }
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
    float aspect_ratio_img = static_cast<float>(original_mat.cols) / original_mat.rows;
    float aspect_ratio_tensor = static_cast<float>(input_tensor_info.tensor_dims.width) / input_tensor_info.tensor_dims.height;
    int32_t crop_x = 0;
    int32_t crop_y = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows;
    if (aspect_ratio_img > aspect_ratio_tensor) {
        crop_w = aspect_ratio_tensor * original_mat.rows;
        crop_x = (original_mat.cols - crop_w) / 2;
    } else {
        crop_h = original_mat.cols / aspect_ratio_tensor;
        crop_y = (original_mat.rows - crop_h) / 2;
    }
    cv::Mat img_src = original_mat(cv::Rect(crop_x, crop_y, crop_w, crop_h));

    cv::resize(img_src, img_src, cv::Size(input_tensor_info.tensor_dims.width, input_tensor_info.tensor_dims.height));
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
    /* Get boundig box */
    std::vector<BoundingBox> bbox_list;
    float* output_data = output_tensor_info_list_[0].GetDataAsFloat();
    for (const auto& scale : grid_scale_list) {
        int32_t grid_w = input_tensor_info.tensor_dims.width / scale;
        int32_t grid_h = input_tensor_info.tensor_dims.height / scale;
        int32_t scale_x = input_tensor_info.tensor_dims.width;      // scale to input tensor size
        int32_t scale_y = input_tensor_info.tensor_dims.height;
        GetBoundingBox(output_data, scale_x, scale_y, grid_w, grid_h, bbox_list);
        output_data += grid_w * grid_h * grid_channl * kElementNumOfAnchor;
    }


    for (auto& bbox : bbox_list) {
        bbox.x = (bbox.x * crop_w) / input_tensor_info.tensor_dims.width + crop_x;  // resize to the original image size
        bbox.y = (bbox.y * crop_h) / input_tensor_info.tensor_dims.height + crop_y;
        bbox.w = (bbox.w * crop_w) / input_tensor_info.tensor_dims.width;
        bbox.h = (bbox.h * crop_h) / input_tensor_info.tensor_dims.height;
        bbox.label = label_list_[bbox.class_id];
    }

    /* NMS */
    std::vector<BoundingBox> bbox_nms_list;
    BoundingBoxUtils::Nms(bbox_list, bbox_nms_list, kThresholdNmsIou);

    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.bbox_list = bbox_nms_list;
    result.crop_x = crop_x;
    result.crop_y = crop_y;
    result.crop_w = crop_w;
    result.crop_h = crop_h;
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

