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
#include "face_detection_engine.h"

/*** Macro ***/
#define TAG "FaceDetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#if 1
#define MODEL_NAME  "dbface_mbnv2_480x640.tflite"
#define INPUT_NAME  "input_1"
#define IS_NCHW     false
#define INPUT_DIMS  { 1, 480, 640, 3 }
#define OUTPUT_NAME_0 "Identity"    /* key */
#define OUTPUT_NAME_1 "Identity_1"  /* reg */
#define OUTPUT_NAME_2 "Identity_2"  /* hm */
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#else
#define MODEL_NAME  "dbface_mbnv3_480x640.mnn"
#define INPUT_NAME  "input.1"
#define IS_NCHW     true
#define INPUT_DIMS  { 1, 3, 480, 640 }
#define OUTPUT_NAME_0 "1027"     /* key */
#define OUTPUT_NAME_1 "1029"     /* reg */
#define OUTPUT_NAME_2 "1028"     /* hm  */
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#endif


/*** Function ***/
int32_t FaceDetectionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
#if 1
    /* 0 - 255 */
    input_tensor_info.normalize.mean[0] = 0.0f;
    input_tensor_info.normalize.mean[1] = 0.0f;
    input_tensor_info.normalize.mean[2] = 0.0f;
    input_tensor_info.normalize.norm[0] = 1.f / 255.f;
    input_tensor_info.normalize.norm[1] = 1.f / 255.f;
    input_tensor_info.normalize.norm[2] = 1.f / 255.f;
#else
    input_tensor_info.normalize.mean[0] = 0.408f;
    input_tensor_info.normalize.mean[1] = 0.447f;
    input_tensor_info.normalize.mean[2] = 0.470f;
    input_tensor_info.normalize.norm[0] = 0.289f;
    input_tensor_info.normalize.norm[1] = 0.274f;
    input_tensor_info.normalize.norm[2] = 0.278f;
#endif
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_1, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_2, TENSORTYPE));

    /* Create and Initialize Inference Helper */
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
    // inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));
    
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kMnn));

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

int32_t FaceDetectionEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}

int32_t FaceDetectionEngine::Process(const cv::Mat& original_mat, Result& result)
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
    float aspect_ratio_tensor = static_cast<float>(input_tensor_info.GetWidth()) / input_tensor_info.GetHeight();
    int32_t crop_x = 0;
    int32_t crop_y = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows;
    if (aspect_ratio_img > aspect_ratio_tensor) {
        crop_w = static_cast<int32_t>(aspect_ratio_tensor * original_mat.rows);
        crop_x = (original_mat.cols - crop_w) / 2;
    } else {
        crop_h = static_cast<int32_t>(original_mat.cols / aspect_ratio_tensor);
        crop_y = (original_mat.rows - crop_h) / 2;
    }

    cv::Mat img_src;
    cv::Mat img_crop = original_mat(cv::Rect(crop_x, crop_y, crop_w, crop_h));
    cv::resize(img_crop, img_src, cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()));
#ifndef CV_COLOR_IS_RGB
    /* It looks they use BGR */
    //cv::cvtColor(img_src, img_src, cv::COLOR_BGR2RGB);
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

    /* Get output data */
    std::vector<float> key_list(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_tensor_info_list_[0].GetElementNum());
    std::vector<float> reg_list(output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat() + output_tensor_info_list_[1].GetElementNum());
    std::vector<float> hm_list(output_tensor_info_list_[2].GetDataAsFloat(), output_tensor_info_list_[2].GetDataAsFloat() + output_tensor_info_list_[2].GetElementNum());
    
    int32_t hm_w = output_tensor_info_list_[2].tensor_dims[IS_NCHW ? 3 : 2];
    int32_t hm_h = output_tensor_info_list_[2].tensor_dims[IS_NCHW ? 2 : 1];
    int32_t hm_size = hm_w * hm_h;
    float scale_w = crop_w / static_cast<float>(hm_w);
    float scale_h = crop_h / static_cast<float>(hm_h);

    /* Get boundig box */
    std::vector<BoundingBox> bbox_list;
    for (int32_t hm_y = 0; hm_y < hm_h; hm_y++) {
        for (int32_t hm_x = 0; hm_x < hm_w; hm_x++) {
            int32_t index = hm_y * hm_w + hm_x;
            float score = hm_list[index];
            if (score >= threshold_confidence_) {
                float x, y, w, h;
                if (IS_NCHW) {
                    x = reg_list[index + 0 * hm_size];
                    y = reg_list[index + 1 * hm_size];
                    w = reg_list[index + 2 * hm_size];
                    h = reg_list[index + 3 * hm_size];
                } else {
                    int32_t index_anchor = index * 4;
                    x = reg_list[index_anchor + 0];
                    y = reg_list[index_anchor + 1];
                    w = reg_list[index_anchor + 2];
                    h = reg_list[index_anchor + 3];
                }

                BoundingBox bbox;
                bbox.x = static_cast<int32_t>((hm_x - x) * scale_w);
                bbox.y = static_cast<int32_t>((hm_y - y) * scale_h);
                bbox.w = static_cast<int32_t>((x + w) * scale_w);
                bbox.h = static_cast<int32_t>((y + h) * scale_h);
                bbox.score = score;
                bbox.class_id = index;      // use class_id to save index to get correspoinding key point after NMS
                bbox_list.push_back(bbox);
            }
        }
    }

    /* NMS */
    std::vector<BoundingBox> bbox_nms_list;
    BoundingBoxUtils::Nms(bbox_list, bbox_nms_list, threshold_nms_iou_, false);

    /* Get keypoint */
    std::vector<KeyPoint> keypoint_list;
    for (auto& bbox : bbox_nms_list) {
        int32_t anchor_index = bbox.class_id;
        int32_t hm_x = anchor_index % hm_w;
        int32_t hm_y = anchor_index / hm_w;
        KeyPoint keypoint;
        for (int32_t key = 0; key < 5; key++) {
            float x, y;
            if (IS_NCHW) {
                x = key_list[anchor_index + (0 + key) * hm_size] * 4;
                y = key_list[anchor_index + (5 + key) * hm_size] * 4;
            } else {
                x = key_list[anchor_index * 10 + (0 + key)] * 4;
                y = key_list[anchor_index * 10 + (5 + key)] * 4;
            }
            x = (ExpSpecial(x) + hm_x) * scale_w;
            y = (ExpSpecial(y) + hm_y) * scale_h;
            keypoint[key].first = static_cast<int32_t>(x + crop_x);
            keypoint[key].second = static_cast<int32_t>(y + crop_y);
        }
        keypoint_list.push_back(keypoint);
    }

    /* Adjust bounding box */
    for (auto& bbox : bbox_nms_list) {
        bbox.class_id = 0;
        bbox.label = "FACE";
        bbox.x += crop_x;
        bbox.y += crop_y;
        BoundingBoxUtils::FixInScreen(bbox, original_mat.cols, original_mat.rows);
    }
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.bbox_list = bbox_nms_list;
    result.keypoint_list = keypoint_list;
    result.crop.x = crop_x;
    result.crop.y = crop_y;
    result.crop.w = crop_w;
    result.crop.h = crop_h;
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}

/* reference: https://github.com/dlunion/DBFace/blob/master/common.py#L297 */
float FaceDetectionEngine::ExpSpecial(float x)
{
    static constexpr int32_t gate = 1;
    static const float base = std::expf(1);
    if (std::abs(x) < gate) {
        return x * base;
    } else if (x > 0) {
        return std::expf(x);
    } else {
        return -std::expf(-x);
    }
}

