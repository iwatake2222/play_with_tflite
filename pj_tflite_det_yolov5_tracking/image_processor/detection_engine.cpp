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
static constexpr int32_t kNumberOfAnchor[] = { 10647 ,};
static constexpr int32_t kNumberOfClass = 80;
static constexpr int32_t kElementNumOfAnchor = kNumberOfClass + 5;    // x, y, w, h, Objectness score, [class probabilities]


#define LABEL_NAME   "label_coco_80.txt"


static constexpr float threshold_score = 0.2f;
static constexpr float threshold_nms_iou = 0.5f;

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
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    // inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
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



int32_t DetectionEngine::Process(const cv::Mat& original_mat, Result& result)
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
    /* Get boundig box */
    std::vector<BoundingBox> bbox_list;
    int32_t num_anchor_list = sizeof(kNumberOfAnchor) / sizeof(kNumberOfAnchor[0]);
    for (int32_t index_scale = 0; index_scale < num_anchor_list; index_scale++) {
        float* output_data = output_tensor_info_list_[index_scale].GetDataAsFloat();
        for (int32_t i = 0; i < kNumberOfAnchor[index_scale]; i++) {
            int32_t index_begin = i * kElementNumOfAnchor;
            if (output_data[index_begin + 4] < threshold_score) continue;

            int32_t class_id = 0;
            float confidence = 0;
            for (int32_t class_index = 0; class_index < kNumberOfClass; class_index++) {
                float confidence_of_class = output_data[index_begin + 5 + class_index];
                if (confidence_of_class > confidence) {
                    confidence = confidence_of_class;
                    class_id = class_index;
                }
            }

            if (confidence > threshold_score) {
                int32_t cx = static_cast<int32_t>(output_data[index_begin + 0] * original_mat.cols);
                int32_t cy = static_cast<int32_t>(output_data[index_begin + 1] * original_mat.rows);
                int32_t w = static_cast<int32_t>(output_data[index_begin + 2] * original_mat.cols);
                int32_t h = static_cast<int32_t>(output_data[index_begin + 3] * original_mat.rows);
                int32_t x = cx - w / 2;
                int32_t y = cy - h / 2;

                bbox_list.push_back(BoundingBox(class_id, label_list_[class_id], confidence, x, y, w, h));

            }
        }
    }

    /* NMS */
    std::vector<BoundingBox> bbox_nms_list;
    Nms(bbox_list, bbox_nms_list);

   
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.bbox_list = bbox_nms_list;
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


float DetectionEngine::CalculateIoU(const BoundingBox& obj0, const BoundingBox& obj1)
{
    int32_t interx0 = (std::max)(obj0.x, obj1.x);
    int32_t intery0 = (std::max)(obj0.y, obj1.y);
    int32_t interx1 = (std::min)(obj0.x + obj0.w, obj1.x + obj1.w);
    int32_t intery1 = (std::min)(obj0.y + obj0.h, obj1.y + obj1.h);
    if (interx1 < interx0 || intery1 < intery0) return 0;

    int32_t area0 = obj0.w * obj0.h;
    int32_t area1 = obj1.w * obj1.h;
    int32_t areaInter = (interx1 - interx0) * (intery1 - intery0);
    int32_t areaSum = area0 + area1 - areaInter;

    return static_cast<float>(areaInter) / areaSum;
}


void DetectionEngine::Nms(std::vector<BoundingBox>& bbox_list, std::vector<BoundingBox>& bbox_nms_list)
{
    std::sort(bbox_list.begin(), bbox_list.end(), [](BoundingBox const& lhs, BoundingBox const& rhs) {
        //if (lhs.w * lhs.h > rhs.w * rhs.h) return true;
        if (lhs.score > rhs.score) return true;
        return false;
        });

    std::unique_ptr<bool[]> is_merged(new bool[bbox_list.size()]);
    for (int32_t i = 0; i < bbox_list.size(); i++) is_merged[i] = false;
    for (int32_t index_high_score = 0; index_high_score < bbox_list.size(); index_high_score++) {
        std::vector<BoundingBox> candidates;
        if (is_merged[index_high_score]) continue;
        candidates.push_back(bbox_list[index_high_score]);
        for (int32_t index_low_score = index_high_score + 1; index_low_score < bbox_list.size(); index_low_score++) {
            if (is_merged[index_low_score]) continue;
            if (bbox_list[index_high_score].class_id != bbox_list[index_low_score].class_id) continue;
            if (CalculateIoU(bbox_list[index_high_score], bbox_list[index_low_score]) > threshold_nms_iou) {
                candidates.push_back(bbox_list[index_low_score]);
                is_merged[index_low_score] = true;
            }
        }

        bbox_nms_list.push_back(candidates[0]);
    }
}
