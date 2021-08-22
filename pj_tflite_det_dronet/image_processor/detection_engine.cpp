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
#include "detection_engine.h"

/*** Macro ***/
#define TAG "DetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
//#define MODEL_TYPE_V1
#define MODEL_TYPE_V3

#if defined(MODEL_TYPE_V1)
#define MODEL_NAME  "DroNet_car.cfg"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 3, 608, 608 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "detection_out"
static constexpr int32_t kGridScaleList[] = { 32 };
static constexpr int32_t kGridChannel = 5;
static constexpr float kAnchorRegionW[kGridChannel] = { 1.08f, 3.42f, 6.63f, 9.42f, 16.62f };
static constexpr float kAnchorRegionH[kGridChannel] = { 1.19f, 4.41f, 11.38f, 5.11f, 10.52f };
static constexpr int32_t kNumberOfClass = 1;
static constexpr int32_t kElementNumOfAnchor = kNumberOfClass + 5;    // x, y, w, h, bbox confidence, [class confidence]
#else defined(MODEL_TYPE_V3)
#define MODEL_NAME  "DroNetV3_car.cfg"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 3, 608, 608 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "yolo_15"
#define OUTPUT_NAME_1 "yolo_22"
static constexpr int32_t kGridScaleList[] = { 32, 16 };
static constexpr int32_t kGridChannel = 3;
static constexpr int32_t kNumberOfClass = 2;
static constexpr int32_t kElementNumOfAnchor = kNumberOfClass + 5;    // x, y, w, h, bbox confidence, [class confidence]
#endif


/*** Function ***/
int32_t DetectionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    /* 0.0 - 1.0 */
    input_tensor_info.normalize.mean[0] = 0.0f;
    input_tensor_info.normalize.mean[1] = 0.0f;
    input_tensor_info.normalize.mean[2] = 0.0f;
    input_tensor_info.normalize.norm[0] = 1.0f;
    input_tensor_info.normalize.norm[1] = 1.0f;
    input_tensor_info.normalize.norm[2] = 1.0f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TENSORTYPE));
#if defined(MODEL_TYPE_V3)
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_1, TENSORTYPE));
#endif

    /* Create and Initialize Inference Helper */
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOpencv));


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

int32_t DetectionEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


void DetectionEngine::GetBoundingBox(const float* data, float scale_x, float  scale_y, int32_t grid_w, int32_t grid_h, std::vector<BoundingBox>& bbox_list)
{
    int32_t index = 0;
    for (int32_t grid_y = 0; grid_y < grid_h; grid_y++) {
        for (int32_t grid_x = 0; grid_x < grid_w; grid_x++) {
            for (int32_t grid_c = 0; grid_c < kGridChannel; grid_c++) {
                float box_confidence = data[index + 4];
                //printf("%d %d %d: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", grid_y, grid_x, grid_c, data[index + 0], data[index + 1], data[index + 2], data[index + 3], data[index + 4], data[index + 5]);
                if (box_confidence >= threshold_box_confidence_) {
                    int32_t class_id = 0;
                    float confidence = 0;
                    for (int32_t class_index = 0; class_index < kNumberOfClass; class_index++) {
                        float confidence_of_class = data[index + 5 + class_index];
                        if (confidence_of_class > confidence) {
                            confidence = confidence_of_class;
                            class_id = class_index;
                        }
                    }

                    if (confidence >= threshold_class_confidence_) {
                        int32_t cx = static_cast<int32_t>(data[index + 0] * scale_x);
                        int32_t cy = static_cast<int32_t>(data[index + 1] * scale_y);
                        int32_t w = static_cast<int32_t>(data[index + 2] /* * kAnchorRegionW[grid_c] */ * scale_x);
                        int32_t h = static_cast<int32_t>(data[index + 3] /* * kAnchorRegionH[grid_c] */ * scale_y);
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
    /* Get boundig box */
    std::vector<BoundingBox> bbox_list;
    for (int32_t i = 0; i < sizeof(kGridScaleList) / sizeof(kGridScaleList[0]); i++) {
        const auto& grid_scale = kGridScaleList[i];
        float* output_data = output_tensor_info_list_[i].GetDataAsFloat();
        int32_t grid_w = input_tensor_info.GetWidth() / grid_scale;
        int32_t grid_h = input_tensor_info.GetHeight() / grid_scale;
        float scale_x = static_cast<float>(crop_w);      /* scale to original image */
        float scale_y = static_cast<float>(crop_h);
        GetBoundingBox(output_data, scale_x, scale_y, grid_w, grid_h, bbox_list);
        output_data += grid_w * grid_h * kGridChannel * kElementNumOfAnchor;
    }

    /* Adjust bounding box */
    for (auto& bbox : bbox_list) {
        bbox.x += crop_x;  
        bbox.y += crop_y;
        bbox.label = "";
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

