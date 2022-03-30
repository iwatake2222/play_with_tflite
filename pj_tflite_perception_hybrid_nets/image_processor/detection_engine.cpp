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
#include "prior_bbox.h"
#include "detection_engine.h"

/*** Macro ***/
#define TAG "DetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_TYPE_TFLITE
//#define MODEL_TYPE_ONNX

#if defined(MODEL_TYPE_TFLITE)
#define MODEL_NAME  "hybridnets_384x640.tflite"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "serving_default_input:0"
#define INPUT_DIMS  { 1, 384, 640, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_0 "StatefulPartitionedCall:0"
#define OUTPUT_NAME_1 "StatefulPartitionedCall:1"
#define OUTPUT_NAME_2 "StatefulPartitionedCall:2"
#elif defined(MODEL_TYPE_ONNX)
#define MODEL_NAME  "hybridnets_384x640.onnx"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 3, 384, 640 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "segmentation"
#define OUTPUT_NAME_1 "classification"
#define OUTPUT_NAME_2 "regression"
#endif

static const std::vector<std::string> kLabelListDet{ "Car" };
static const std::vector<std::string> kLabelListSeg{ "Background", "Lane", "Line" };


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
    input_tensor_info.normalize.mean[0] = 0.485f;
    input_tensor_info.normalize.mean[1] = 0.456f;
    input_tensor_info.normalize.mean[2] = 0.406f;
    input_tensor_info.normalize.norm[0] = 0.229f;
    input_tensor_info.normalize.norm[1] = 0.224f;
    input_tensor_info.normalize.norm[2] = 0.225f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_1, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_2, TENSORTYPE));

    /* Create and Initialize Inference Helper */
#if defined(MODEL_TYPE_TFLITE)
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));
#elif defined(MODEL_TYPE_ONNX)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOnnxRuntime));
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
    std::vector<float> output_seg_list(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_tensor_info_list_[0].GetElementNum());
    std::vector<float> output_confidence_list(output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat() + output_tensor_info_list_[1].GetElementNum());
    std::vector<float> output_bbox_list(output_tensor_info_list_[2].GetDataAsFloat(), output_tensor_info_list_[2].GetDataAsFloat() + output_tensor_info_list_[2].GetElementNum());

    /* Get Segmentation result. ArgMax */
    cv::Mat mat_seg_max = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC1);
#pragma omp parallel for
    for (int32_t y = 0; y < input_tensor_info.GetHeight(); y++) {
        for (int32_t x = 0; x < input_tensor_info.GetWidth(); x++) {
            if (IS_NCHW) {
                int32_t class_index_max = 0;
                float class_score_max = 0;
                for (int32_t class_index = 0; class_index < kLabelListSeg.size(); class_index++) {
                    float score = output_seg_list[class_index * input_tensor_info.GetHeight() * input_tensor_info.GetWidth() + input_tensor_info.GetWidth() * y + x];
                    if (score > class_score_max) {
                        class_score_max = score;
                        class_index_max = class_index;
                    }
                }
                mat_seg_max.at<uint8_t>(cv::Point(x, y)) = static_cast<uint8_t>(class_index_max);
            } else {
                const auto& current_iter = output_seg_list.begin() + y * input_tensor_info.GetWidth() * kLabelListSeg.size() + x * kLabelListSeg.size();
                const auto& max_iter = std::max_element(current_iter, current_iter + kLabelListSeg.size());
                auto max_c = std::distance(current_iter, max_iter);
                mat_seg_max.at<uint8_t>(cv::Point(x, y)) = static_cast<uint8_t>(max_c);
            }
        }
    }

    /* Get boundig box */
    /* reference: https://github.dev/datvuthanh/HybridNets/blob/c626bb89beb1b52440bacdbcc90ac60f9814c9a2/utils/utils.py#L615-L616 */
    float scale_w = static_cast<float>(crop_w) / input_tensor_info.GetWidth();
    float scale_h = static_cast<float>(crop_h) / input_tensor_info.GetHeight();
    static const size_t kNumPrior = output_bbox_list.size() / 4 / kLabelListDet.size();
    std::vector<BoundingBox> bbox_list;
    for (size_t i = 0; i < kNumPrior; i++) {
        size_t class_index = 0;
        float class_score = output_confidence_list[i];
        size_t prior_index = i * 4;
        if (class_score >= threshold_class_confidence_) {
            /* Prior Box: [0.0, MODEL_SIZE], y0, x0, y1, x1 */
            const float prior_x0 = PRIOR_BBOX::BBOX[prior_index + 1];
            const float prior_y0 = PRIOR_BBOX::BBOX[prior_index + 0];
            const float prior_x1 = PRIOR_BBOX::BBOX[prior_index + 3];
            const float prior_y1 = PRIOR_BBOX::BBOX[prior_index + 2];
            const float prior_cx = (prior_x0 + prior_x1) / 2.0f;
            const float prior_cy = (prior_y0 + prior_y1) / 2.0f;
            const float prior_w = prior_x1 - prior_x0;
            const float prior_h = prior_y1 - prior_y0;

            /* Detected Box */
            float box_cx = output_bbox_list[prior_index + 1];
            float box_cy = output_bbox_list[prior_index + 0];
            float box_w = output_bbox_list[prior_index + 3];
            float box_h = output_bbox_list[prior_index + 2];

            /* Adjust box [0.0, 1.0] */
            float cx = PRIOR_BBOX::VARIANCE[1] * box_cx * prior_w + prior_cx;
            float cy = PRIOR_BBOX::VARIANCE[0] * box_cy * prior_h + prior_cy;
            float w = std::exp(box_w * PRIOR_BBOX::VARIANCE[3]) * prior_w;
            float h = std::exp(box_h * PRIOR_BBOX::VARIANCE[2]) * prior_h;

            /* Store the detected box */
            auto bbox = BoundingBox{
                static_cast<int32_t>(class_index),
                kLabelListDet[class_index],
                class_score,
                static_cast<int32_t>((cx - w / 2.0) * scale_w),
                static_cast<int32_t>((cy - h / 2.0) * scale_h),
                static_cast<int32_t>(w * scale_w),
                static_cast<int32_t>(h * scale_h)
            };
            bbox_list.push_back(bbox);
        }
    }


    /* Adjust bounding box */
    for (auto& bbox : bbox_list) {
        bbox.x += crop_x;  
        bbox.y += crop_y;
    }

    /* NMS */
    std::vector<BoundingBox> bbox_nms_list;
    BoundingBoxUtils::Nms(bbox_list, bbox_nms_list, threshold_nms_iou_);

    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.mat_seg_max = mat_seg_max;
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
