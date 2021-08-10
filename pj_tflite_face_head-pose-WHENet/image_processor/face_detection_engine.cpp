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
#include "face_detection_engine.h"

/*** Macro ***/
#define TAG "FaceDetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_TYPE_128x128_CONCAT
//#define MODEL_TYPE_128x128
//#define MODEL_TYPE_256x256

#if defined(MODEL_TYPE_128x128_CONCAT)
#define MODEL_NAME  "face_detection_front.tflite"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 128, 128, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_0 "classificators"
#define OUTPUT_NAME_1 "regressors"
#elif defined(MODEL_TYPE_128x128)
#define MODEL_NAME  "face_detection_front_128x128_float32.tflite"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 128, 128, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_0 "Identity"
#define OUTPUT_NAME_1 "Identity_2"
#define OUTPUT_NAME_2 "Identity_1"
#define OUTPUT_NAME_3 "Identity_3"
#elif defined(MODEL_TYPE_256x256)
#define MODEL_NAME  "face_detection_back_256x256_float32.tflite"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 256, 256, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_0 "Identity"
#define OUTPUT_NAME_1 "Identity_2"
#define OUTPUT_NAME_2 "Identity_1"
#define OUTPUT_NAME_3 "Identity_3"
#endif
static constexpr int32_t kElementNumOfAnchor = 16;    /* x, y, w, h, [x, y] */
std::array<std::pair<int32_t, int32_t>, 2> kAnchorGridSize = { std::pair<int32_t, int32_t>(16, 16), std::pair < int32_t, int32_t>(8, 8) };
std::array<int32_t, 2> kAnchorNum = { 2, 6 };

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
    input_tensor_info.normalize.mean[0] = 0.5f;     /* -1.0 - 1.0*/
    input_tensor_info.normalize.mean[1] = 0.5f;
    input_tensor_info.normalize.mean[2] = 0.5f;
    input_tensor_info.normalize.norm[0] = 0.5f;
    input_tensor_info.normalize.norm[1] = 0.5f;
    input_tensor_info.normalize.norm[2] = 0.5f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_1, TENSORTYPE));
#if defined(MODEL_TYPE_128x128) || defined(MODEL_TYPE_256x256)
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_2, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_3, TENSORTYPE));
#endif

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

    anchor_list_.clear();
    CreateAnchor(input_tensor_info_list_[0].GetWidth(), input_tensor_info_list_[0].GetHeight(), anchor_list_);

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
    /* Get output data */
#if defined(MODEL_TYPE_128x128_CONCAT)
    std::vector<float> score_list;
    score_list.reserve(output_tensor_info_list_[0].GetElementNum());
    score_list.insert(score_list.end(), output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_tensor_info_list_[0].GetElementNum());

    std::vector<float> regressor_list;
    regressor_list.reserve(output_tensor_info_list_[1].GetElementNum());
    regressor_list.insert(regressor_list.end(), output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat() + output_tensor_info_list_[1].GetElementNum());
#elif defined(MODEL_TYPE_128x128) || defined(MODEL_TYPE_256x256)
    std::vector<float> score_list;
    score_list.reserve(output_tensor_info_list_[0].GetElementNum() + output_tensor_info_list_[2].GetElementNum());
    score_list.insert(score_list.end(), output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_tensor_info_list_[0].GetElementNum());
    score_list.insert(score_list.end(), output_tensor_info_list_[2].GetDataAsFloat(), output_tensor_info_list_[2].GetDataAsFloat() + output_tensor_info_list_[2].GetElementNum());

    std::vector<float> regressor_list;
    regressor_list.reserve(output_tensor_info_list_[1].GetElementNum() + output_tensor_info_list_[3].GetElementNum());
    regressor_list.insert(regressor_list.end(), output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat() + output_tensor_info_list_[1].GetElementNum());
    regressor_list.insert(regressor_list.end(), output_tensor_info_list_[3].GetDataAsFloat(), output_tensor_info_list_[3].GetDataAsFloat() + output_tensor_info_list_[3].GetElementNum());
#endif

    /* Get boundig box */
    std::vector<BoundingBox> bbox_list;
    float score_logit = Logit(threshold_confidence_);
    GetBoundingBox(score_list, regressor_list, anchor_list_, score_logit, static_cast<float>(crop_w) / input_tensor_info.GetWidth(), static_cast<float>(crop_h) / input_tensor_info.GetHeight(), bbox_list);

    /* NMS */
    std::vector<BoundingBox> bbox_nms_list;
    BoundingBoxUtils::Nms(bbox_list, bbox_nms_list, threshold_nms_iou_, false);

    std::vector<KeyPoint> keypoint_list;
    for (auto& bbox : bbox_nms_list) {
        /* Adjust bounding box */
        int32_t anchor_index = bbox.class_id;
        bbox.class_id = 0;
        bbox.label = "FACE";
        bbox.score = Sigmoid(bbox.score);
        bbox.x += crop_x;
        bbox.y += crop_y;
        BoundingBoxUtils::FixInScreen(bbox, original_mat.cols, original_mat.rows);

        /* Get keypoint */
        const float* regressor = &regressor_list[anchor_index * kElementNumOfAnchor];
        KeyPoint keypoint;
        for (int32_t key = 0; key < 6; key++) {
            float x = regressor[4 + 2 * key + 0] + anchor_list_[anchor_index].first;
            float y = regressor[4 + 2 * key + 1] + anchor_list_[anchor_index].second;
            keypoint[key].first = static_cast<int32_t>((x * crop_w) / input_tensor_info.GetWidth() + crop_x);  // resize to the original image size
            keypoint[key].second = static_cast<int32_t>((y * crop_h) / input_tensor_info.GetHeight() + crop_y);
        }
        keypoint_list.push_back(keypoint);
    }
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.bbox_list = bbox_nms_list;
    result.keypoint_list = keypoint_list;
    result.crop.x = (std::max)(0, crop_x);
    result.crop.y = (std::max)(0, crop_y);
    result.crop.w = (std::min)(crop_w, original_mat.cols - result.crop.x);
    result.crop.h = (std::min)(crop_h, original_mat.rows - result.crop.y);
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}


float FaceDetectionEngine::Sigmoid(float x)
{
    if (x >= 0) {
        return 1.0f / (1.0f + std::expf(-x));
    } else {
        return std::expf(x) / (1.0f + std::expf(x));    /* to aovid overflow */
    }
}

float FaceDetectionEngine::Logit(float x)
{
    if (x == 0) {
        return -FLT_MAX;
    } else  if (x == 1) {
        return FLT_MAX;
    } else {
        return std::logf(x / (1.0f - x));
    }
}

/* reference: https://github.com/tensorflow/tfjs-models/blob/master/blazeface/src/face.ts#L62 */
void FaceDetectionEngine::CreateAnchor(int32_t width, int32_t height, std::vector<std::pair<float, float>>& anchor_list)
{
    for (int i = 0; i < kAnchorGridSize.size(); i++) {
        int32_t grid_cols = kAnchorGridSize[i].first;
        int32_t grid_rows = kAnchorGridSize[i].second;
        float  stride_x = static_cast<float>(width) / grid_cols;
        float  stride_y = static_cast<float>(height) / grid_rows;
        int anchor_num = kAnchorNum[i];

        std::pair<float, float> anchor;
        for (int grid_y = 0; grid_y < grid_rows; grid_y++) {
            anchor.second = stride_y * (grid_y + 0.5f);
            for (int grid_x = 0; grid_x < grid_cols; grid_x++) {
                anchor.first = stride_x * (grid_x + 0.5f);
                for (int n = 0; n < anchor_num; n++) {
                    anchor_list.push_back(anchor);
                }
            }
        }
    }
}

/* reference: https://github.com/tensorflow/tfjs-models/blob/master/blazeface/src/face.ts#L86 */
void FaceDetectionEngine::GetBoundingBox(const std::vector<float>& score_list, const std::vector<float>& regressor_list, const std::vector<std::pair<float, float>>& anchor_list, float threshold_score_logit, float scale_x, float  scale_y, std::vector<BoundingBox>& bbox_list)
{
    for (int32_t i = 0; i < anchor_list.size(); i++) {
        if (score_list[i] > threshold_score_logit) {
            int32_t index_regressor = i * kElementNumOfAnchor;
            float cx = regressor_list[index_regressor + 0] + anchor_list[i].first;
            float cy = regressor_list[index_regressor + 1] + anchor_list[i].second;
            float w = regressor_list[index_regressor + 2];
            float h = regressor_list[index_regressor + 3];
            BoundingBox bbox;
            bbox.score = score_list[i];
            bbox.class_id = i;      /* temporary use this fiels to save the index of anchor */
            bbox.x = static_cast<int32_t>((cx - w / 2) * scale_x);
            bbox.y = static_cast<int32_t>((cy - h / 2) * scale_y);
            bbox.w = static_cast<int32_t>(w * scale_x);
            bbox.h = static_cast<int32_t>(h * scale_y);
            bbox_list.push_back(bbox);
        }
    }
}
