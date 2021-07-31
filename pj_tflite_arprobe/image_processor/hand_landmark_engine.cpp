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

/* for My modules */
#include "common_helper.h"
#include "inference_helper.h"
#include "hand_landmark_engine.h"

/*** Macro ***/
#define TAG "HandLandmarkEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "hand_landmark.tflite"

/*** Function ***/
int32_t HandLandmarkEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info("input_1", TensorInfo::kTensorTypeFp32, false);
    input_tensor_info.tensor_dims = { 1, 256, 256, 3 };
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.0f;   	/* normalized to[0.f, 1.f] (hand_landmark_cpu.pbtxt) */
    input_tensor_info.normalize.mean[1] = 0.0f;
    input_tensor_info.normalize.mean[2] = 0.0f;
    input_tensor_info.normalize.norm[0] = 1.0f;
    input_tensor_info.normalize.norm[1] = 1.0f;
    input_tensor_info.normalize.norm[2] = 1.0f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo("ld_21_3d", TensorInfo::kTensorTypeFp32));
    output_tensor_info_list_.push_back(OutputTensorInfo("output_handflag", TensorInfo::kTensorTypeFp32));
    output_tensor_info_list_.push_back(OutputTensorInfo("output_handedness", TensorInfo::kTensorTypeFp32));

    /* Create and Initialize Inference Helper */
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
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

    return kRetOk;
}

int32_t HandLandmarkEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


int32_t HandLandmarkEngine::Process(const cv::Mat& original_mat, int32_t palmX, int32_t palmY, int32_t palmW, int32_t palmH, float palmRotation, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];

    /* Rotate palm image */
    cv::Mat rotated_image;
    cv::RotatedRect rect(cv::Point(palmX + palmW / 2, palmY + palmH / 2), cv::Size(palmW, palmH), palmRotation * 180.f / static_cast<float>(M_PI));
    cv::Mat trans = cv::getRotationMatrix2D(rect.center, rect.angle, 1.0);
    cv::Mat srcRot;
    cv::warpAffine(original_mat, srcRot, trans, original_mat.size());
    cv::getRectSubPix(srcRot, rect.size, rect.center, rotated_image);
    //cv::imshow("rotated_image", rotated_image);

    /* Resize image */
    cv::Mat img_src;
    cv::resize(rotated_image, img_src, cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()));
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

    /* Retrieve the result */
    HAND_LANDMARK& hand_landmark = result.hand_landmark;
    hand_landmark.handflag = output_tensor_info_list_[1].GetDataAsFloat()[0];
    hand_landmark.handedness = output_tensor_info_list_[2].GetDataAsFloat()[0];
    const float *ld21 = output_tensor_info_list_[0].GetDataAsFloat();
    //printf("%f  %f\n", m_outputTensorHandflag->GetDataAsFloat()[0], m_outputTensorHandedness->GetDataAsFloat()[0]);

    for (int32_t i = 0; i < 21; i++) {
        hand_landmark.pos[i].x = ld21[i * 3 + 0] / input_tensor_info.GetWidth();	// 0.0 - 1.0
        hand_landmark.pos[i].y = ld21[i * 3 + 1] / input_tensor_info.GetHeight();	// 0.0 - 1.0
        hand_landmark.pos[i].z = ld21[i * 3 + 2] * 1;	 // Scale Z coordinate as X. (-100 - 100???) todo
        //printf("%f\n", m_outputTensorLd21->GetDataAsFloat()[i]);
        //cv::circle(original_mat, cv::Point(m_outputTensorLd21->GetDataAsFloat()[i * 3 + 0], m_outputTensorLd21->GetDataAsFloat()[i * 3 + 1]), 5, cv::Scalar(255, 255, 0), 1);
    }

    /* Fix landmark rotation */
    for (int32_t i = 0; i < 21; i++) {
        hand_landmark.pos[i].x *= rotated_image.cols;	// coordinate on rotated_image
        hand_landmark.pos[i].y *= rotated_image.rows;
    }
    RotateLandmark(hand_landmark, palmRotation, rotated_image.cols, rotated_image.rows);	// coordinate on thei nput image

    /* Calculate palm rectangle from Landmark */
    TransformLandmarkToRect(hand_landmark);
    hand_landmark.rect.rotation = CalculateRotation(hand_landmark);

    for (int32_t i = 0; i < 21; i++) {
        hand_landmark.pos[i].x += palmX;
        hand_landmark.pos[i].y += palmY;
    }
    hand_landmark.rect.x += palmX;
    hand_landmark.rect.y += palmY;

    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}



void HandLandmarkEngine::RotateLandmark(HAND_LANDMARK& hand_landmark, float rotationRad, int32_t image_width, int32_t image_height)
{
    for (int32_t i = 0; i < 21; i++) {
        float x = hand_landmark.pos[i].x - image_width / 2.f;
        float y = hand_landmark.pos[i].y - image_height / 2.f;

        hand_landmark.pos[i].x = x * std::cos(rotationRad) - y * std::sin(rotationRad) + image_width / 2.f;
        hand_landmark.pos[i].y = x * std::sin(rotationRad) + y * std::cos(rotationRad) + image_height / 2.f;
        //hand_landmark.pos[i].x = std::min(hand_landmark.pos[i].x, 1.f);
        //hand_landmark.pos[i].y = std::min(hand_landmark.pos[i].y, 1.f);
    };
}

float HandLandmarkEngine::CalculateRotation(const HAND_LANDMARK& hand_landmark)
{
    // Reference: mediapipe\graphs\hand_tracking\calculators\hand_detections_to_rects_calculator.cc
    constexpr int32_t kWristJoint = 0;
    constexpr int32_t kMiddleFingerPIPJoint = 12;
    constexpr int32_t kIndexFingerPIPJoint = 8;
    constexpr int32_t kRingFingerPIPJoint = 16;
    constexpr float target_angle_ = static_cast<float>(M_PI) * 0.5f;

    const float x0 = hand_landmark.pos[kWristJoint].x;
    const float y0 = hand_landmark.pos[kWristJoint].y;
    float x1 = (hand_landmark.pos[kMiddleFingerPIPJoint].x + hand_landmark.pos[kMiddleFingerPIPJoint].x) / 2.f;
    float y1 = (hand_landmark.pos[kMiddleFingerPIPJoint].y + hand_landmark.pos[kMiddleFingerPIPJoint].y) / 2.f;
    x1 = (x1 + hand_landmark.pos[kMiddleFingerPIPJoint].x) / 2.f;
    y1 = (y1 + hand_landmark.pos[kMiddleFingerPIPJoint].y) / 2.f;

    float rotation;
    rotation = target_angle_ - std::atan2(-(y1 - y0), x1 - x0);
    rotation = rotation - 2 * static_cast<float>(M_PI) * std::floor((rotation - (-static_cast<float>(M_PI))) / (2 * static_cast<float>(M_PI)));

    return rotation;
}

void HandLandmarkEngine::TransformLandmarkToRect(HAND_LANDMARK &hand_landmark)
{
    constexpr float shift_x = 0.0f;
    constexpr float shift_y = -0.0f;
    constexpr float scale_x = 1.8f;		// tuned parameter by looking
    constexpr float scale_y = 1.8f;

    float width = 0;
    float height = 0;
    float x_center = 0;
    float y_center = 0;

    float xmin = hand_landmark.pos[0].x;
    float xmax = hand_landmark.pos[0].x;
    float ymin = hand_landmark.pos[0].y;
    float ymax = hand_landmark.pos[0].y;

    for (int32_t i = 0; i < 21; i++) {
        if (hand_landmark.pos[i].x < xmin) xmin = hand_landmark.pos[i].x;
        if (hand_landmark.pos[i].x > xmax) xmax = hand_landmark.pos[i].x;
        if (hand_landmark.pos[i].y < ymin) ymin = hand_landmark.pos[i].y;
        if (hand_landmark.pos[i].y > ymax) ymax = hand_landmark.pos[i].y;
    }
    width = xmax - xmin;
    height = ymax - ymin;
    x_center = (xmax + xmin) / 2.f;
    y_center = (ymax + ymin) / 2.f;

    width *= scale_x;
    height *= scale_y;

    float long_side = std::max(width, height);

    /* for hand is closed */
    //float palmDistance = powf(hand_landmark.pos[0].x - hand_landmark.pos[9].x, 2) + powf(hand_landmark.pos[0].y - hand_landmark.pos[9].y, 2);
    //palmDistance = sqrtf(palmDistance);
    //long_side = std::max(long_side, palmDistance);

    hand_landmark.rect.width = (long_side * 1);
    hand_landmark.rect.height = (long_side * 1);
    hand_landmark.rect.x = (x_center - hand_landmark.rect.width / 2);
    hand_landmark.rect.y = (y_center - hand_landmark.rect.height / 2);
}
