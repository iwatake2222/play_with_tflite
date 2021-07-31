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
#define _USE_MATH_DEFINES
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

/* for meidapipe sub functions */
#include "meidapipe/transpose_conv_bias.h"
#include "meidapipe/ssd_anchors_calculator.h"
#include "meidapipe/tflite_tensors_to_detections_calculator.h"

/* for My modules */
#include "common_helper.h"
#include "inference_helper.h"
#include "palm_detection_engine.h"

/*** Macro ***/
#define TAG "PalmDetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "palm_detection.tflite"

static float CalculateRotation(const Detection& det);
static void Nms(std::vector<Detection>& detection_list, std::vector<Detection>& detection_list_nms, bool use_weight);
static void RectTransformationCalculator(const Detection& det, const float rotation, float& x, float& y, float& width, float& height);
static std::vector<Anchor> s_anchors;

/*** Function ***/
int32_t PalmDetectionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info("input", TensorInfo::kTensorTypeFp32, false);
    input_tensor_info.tensor_dims = { 1, 256, 256, 3 };
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.5f;   	/* normalized to[-1.f, 1.f] (hand_detection_cpu.pbtxt.pbtxt) */
    input_tensor_info.normalize.mean[1] = 0.5f;
    input_tensor_info.normalize.mean[2] = 0.5f;
    input_tensor_info.normalize.norm[0] = 0.5f;
    input_tensor_info.normalize.norm[1] = 0.5f;
    input_tensor_info.normalize.norm[2] = 0.5f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo("regressors", TensorInfo::kTensorTypeFp32));
    output_tensor_info_list_.push_back(OutputTensorInfo("classificators", TensorInfo::kTensorTypeFp32));

    /* Create and Initialize Inference Helper */
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));

    if (!inference_helper_) {
        return kRetErr;
    }
    if (inference_helper_->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }

    std::vector<std::pair<const char*, const void*>> customOps;
    customOps.push_back(std::pair<const char*, const void*>("Convolution2DTransposeBias", (const void*)mediapipe::tflite_operations::RegisterConvolution2DTransposeBias()));
    if (inference_helper_->SetCustomOps(customOps) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }

    if (inference_helper_->Initialize(model_filename, input_tensor_info_list_, output_tensor_info_list_) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }

    /* Call SsdAnchorsCalculator::GenerateAnchors as described in hand_detection_gpu.pbtxt */
    const SsdAnchorsCalculatorOptions options;
    ::mediapipe::GenerateAnchors(&s_anchors, options);

    return kRetOk;
}

int32_t PalmDetectionEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


int32_t PalmDetectionEngine::Process(const cv::Mat& original_mat, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }

    int32_t image_width = original_mat.cols;
    int32_t image_height = original_mat.rows;


    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
    /* do resize and color conversion here because some inference engine doesn't support these operations */
    cv::Mat img_src;
    cv::resize(original_mat, img_src, cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()));
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
    /* Call TfLiteTensorsToDetectionsCalculator::DecodeBoxes as described in hand_detection_gpu.pbtxt */
    std::vector<Detection> detection_list;
    const TfLiteTensorsToDetectionsCalculatorOptions options;
    if (options.num_boxes() != output_tensor_info_list_[0].tensor_dims[1]) {
        return kRetErr;
    }
    if (options.num_coords() != output_tensor_info_list_[0].tensor_dims[2]) {
        return kRetErr;
    }
    if (options.num_classes() != output_tensor_info_list_[1].tensor_dims[2]) {
        return kRetErr;
    }
    const float* raw_boxes = output_tensor_info_list_[0].GetDataAsFloat();
    const float* raw_scores = output_tensor_info_list_[1].GetDataAsFloat();
    mediapipe::Process(options, raw_boxes, raw_scores, s_anchors, detection_list);

    /* Call NonMaxSuppressionCalculator as described in hand_detection_gpu.pbtxt */
    /*  -> use my own NMS */
    std::vector<Detection> detection_list_nms;
    Nms(detection_list, detection_list_nms, false);

    std::vector<PALM> palmList;
    for (auto palmDet : detection_list_nms) {
        /* Convert the coordinate from on (0.0 - 1.0) to on the input image */
        palmDet.x *= image_width;
        palmDet.y *= image_height;
        palmDet.w *= image_width;
        palmDet.h *= image_height;
        for (auto kp : palmDet.keypoints) {
            kp.first *= image_width;
            kp.second *= image_height;
        }

        /* Call DetectionsToRectsCalculator as described in hand_detection_gpu.pbtxt */
        /*  -> use my own calculator */
        float rotation = CalculateRotation(palmDet);
        //printf("%f  %f\n", rotation, rotation * 180 / 3.14);

        /* Call RectTransformationCalculator as described in hand_landmark_cpu.pbtxt */
        /*  -> use my own calculator */
        float x, y, width, height;
        RectTransformationCalculator(palmDet, rotation, x, y, width, height);

        PALM palm = { 0 };
        palm.score = palmDet.score;
        palm.x = (std::min)(image_width * 1.f, (std::max)(x, 0.f));
        palm.y = (std::min)(image_height * 1.f, (std::max)(y, 0.f));
        palm.width = (std::min)(image_width * 1.f - palm.x, (std::max)(width, 0.f));
        palm.height = (std::min)(image_height * 1.f - palm.y, (std::max)(height, 0.f));
        palm.rotation = rotation;
        palmList.push_back(palm);
    }
    const auto& t_post_process1 = std::chrono::steady_clock::now();


    /* Return the results */
    result.palmList = palmList;
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}



static void RectTransformationCalculator(const Detection& det, const float rotation, float& x, float& y, float& width, float& height)
{
    /* Reference: RectTransformationCalculator::TransformRect */
    const float shift_x = 0.0f;
    const float shift_y = -0.5f;
    const float scale_x = 2.6f;
    const float scale_y = 2.6f;

    float x_center = det.x + det.w / 2.f;
    float y_center = det.y + det.h / 2.f;
    if (rotation == 0.f) {
        x_center += det.w * shift_x;
        y_center += det.h * shift_y;
    } else {
        const float x_shift = (det.w * shift_x * std::cos(rotation) - det.h * shift_y * std::sin(rotation));
        const float y_shift = (det.w * shift_x * std::sin(rotation) + det.h * shift_y * std::cos(rotation));
        x_center += x_shift;
        y_center += y_shift;
    }

    const float long_side = (std::max)(det.w, det.h);
    width = long_side * scale_x;
    height = long_side * scale_y;
    x = x_center - width / 2;
    y = y_center - height / 2;
}


static float CalculateRotation(const Detection& det)
{
    /* Reference: ::mediapipe::Status DetectionsToRectsCalculator::ComputeRotation (detections_to_rects_calculator.cc) */
    constexpr int32_t rotation_vector_start_keypoint_index = 0;  // # Center of wrist.
    constexpr int32_t rotation_vector_end_keypoint_index = 2;	// # MCP of middle finger.
    constexpr double rotation_vector_target_angle_degrees = M_PI * 0.5f;

    const float x0 = det.keypoints[rotation_vector_start_keypoint_index].first;
    const float y0 = det.keypoints[rotation_vector_start_keypoint_index].second;
    const float x1 = det.keypoints[rotation_vector_end_keypoint_index].first;
    const float y1 = det.keypoints[rotation_vector_end_keypoint_index].second;

    double rotation;
    rotation = rotation_vector_target_angle_degrees - std::atan2(-(y1 - y0), x1 - x0);
    rotation = rotation - 2 * M_PI * std::floor((rotation - (-M_PI)) / (2 * M_PI));
    return static_cast<float>(rotation);
}

static float CalculateIoU(const Detection& det0, const Detection& det1)
{
    float interx0 = (std::max)(det0.x, det1.x);
    float intery0 = (std::max)(det0.y, det1.y);
    float interx1 = (std::min)(det0.x + det0.w, det1.x + det1.w);
    float intery1 = (std::min)(det0.y + det0.h, det1.y + det1.h);

    float area0 = det0.w * det0.h;
    float area1 = det1.w * det1.h;
    float area_inter = (interx1 - interx0) * (intery1 - intery0);
    float area_sum = area0 + area1 - area_inter;

    return area_inter / area_sum;
}

static void Nms(std::vector<Detection>& detection_list, std::vector<Detection>& detection_list_nms, bool use_weight)
{
    std::sort(detection_list.begin(), detection_list.end(), [](Detection const& lhs, Detection const& rhs) {
        if (lhs.w * lhs.h > rhs.w * rhs.h) return true;
        // if (lhs.score > rhs.score) return true;
        return false;
    });

    bool *is_merged = new bool[detection_list.size()];
    for (int32_t i = 0; i < detection_list.size(); i++) is_merged[i] = false;
    for (int32_t index_high_score = 0; index_high_score < detection_list.size(); index_high_score++) {
        std::vector<Detection> candidates;
        if (is_merged[index_high_score]) continue;
        candidates.push_back(detection_list[index_high_score]);
        for (int32_t index_low_score = index_high_score + 1; index_low_score < detection_list.size(); index_low_score++) {
            if (is_merged[index_low_score]) continue;
            if (detection_list[index_high_score].class_id != detection_list[index_low_score].class_id) continue;
            if (CalculateIoU(detection_list[index_high_score], detection_list[index_low_score]) > 0.5) {
                candidates.push_back(detection_list[index_low_score]);
                is_merged[index_low_score] = true;
            }
        }

        /* weight by score */
        if (use_weight) {
            if (candidates.size() < 3) continue;	// do not use detected object if the number of bbox is small
            Detection merged_box = { 0 };
            merged_box.keypoints.resize(candidates[0].keypoints.size(), std::make_pair<float, float>(0, 0));
            float sum_score = 0;
            for (auto candidate : candidates) {
                sum_score += candidate.score;
                merged_box.score += candidate.score;
                merged_box.x += candidate.x * candidate.score;
                merged_box.y += candidate.y * candidate.score;
                merged_box.w += candidate.w * candidate.score;
                merged_box.h += candidate.h * candidate.score;
                for (int32_t k = 0; k < merged_box.keypoints.size(); k++) {
                    merged_box.keypoints[k].first += candidate.keypoints[k].first * candidate.score;
                    merged_box.keypoints[k].second += candidate.keypoints[k].second * candidate.score;
                }
            }
            merged_box.score /= candidates.size();
            merged_box.x /= sum_score;
            merged_box.y /= sum_score;
            merged_box.w /= sum_score;
            merged_box.h /= sum_score;
            for (int32_t k = 0; k < merged_box.keypoints.size(); k++) {
                merged_box.keypoints[k].first /= sum_score;
                merged_box.keypoints[k].second /= sum_score;
            }
            detection_list_nms.push_back(merged_box);
        } else {
            detection_list_nms.push_back(candidates[0]);
        }

    }
    delete[] is_merged;
}
