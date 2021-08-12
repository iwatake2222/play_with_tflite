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
//#define MODEL_TYPE_TFLITE
#define MODEL_TYPE_ONNX

#ifdef MODEL_TYPE_TFLITE
#define MODEL_NAME  "nanodet_320x320.tflite"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "i"
#define INPUT_DIMS  { 1, 320, 320, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_REG_0 "Identity"
#define OUTPUT_NAME_CLASS_0 "Identity_1"
#define OUTPUT_NAME_REG_1 "Identity_2"
#define OUTPUT_NAME_CLASS_1 "Identity_3"
#define OUTPUT_NAME_REG_2 "Identity_4"
#define OUTPUT_NAME_CLASS_2 "Identity_5"
#else
#define MODEL_NAME  "nanodet_320x320.onnx"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "i"
#define INPUT_DIMS  { 1, 3, 320, 320 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_REG_0 "t"
#define OUTPUT_NAME_CLASS_0 "t.2"
#define OUTPUT_NAME_REG_1 "u"
#define OUTPUT_NAME_CLASS_1 "u.2"
#define OUTPUT_NAME_REG_2 "p"
#define OUTPUT_NAME_CLASS_2 "o"
#endif

static constexpr int32_t kStriceNum = 3;
static constexpr int32_t kStrideList[kStriceNum] = { 32, 16, 8 };
static constexpr int32_t kNumClass = 80;
static constexpr int32_t kRegMax = 7;


#define LABEL_NAME   "label_coco_80.txt"


/*** Function ***/
int32_t DetectionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;
    std::string labelFilename = work_dir + "/model/" + LABEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.408f;
    input_tensor_info.normalize.mean[1] = 0.447f;
    input_tensor_info.normalize.mean[2] = 0.470f;
    input_tensor_info.normalize.norm[0] = 0.289f;
    input_tensor_info.normalize.norm[1] = 0.274f;
    input_tensor_info.normalize.norm[2] = 0.278f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_REG_0, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_CLASS_0, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_REG_1, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_CLASS_1, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_REG_2, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_CLASS_2, TENSORTYPE));

    /* Create and Initialize Inference Helper */
#ifdef MODEL_TYPE_TFLITE
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));
#else
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOpencv));
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
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeCut);
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

    /* Get boundig box */
    std::vector<BoundingBox> bbox_list;
    std::vector<int32_t> feature_num_list(kStriceNum);
    for (int32_t i = 0; i < kStriceNum; i++) {
        feature_num_list[i] = output_tensor_info_list_[i * 2 + 1].GetElementNum() / kNumClass;    // feature num = (the output size of class) / 80
        if (feature_num_list[i] <= 0) feature_num_list[i] = input_tensor_info.GetWidth() * input_tensor_info.GetHeight() / (kStrideList[i] * kStrideList[i]);   /* In case I cannot get GetElementNum (this happens with cv::dnn) */
    }

    std::vector<std::vector<float>> reg_list(kStriceNum);
    std::vector<std::vector<float>> score_list(kStriceNum);
    for (int32_t i = 0; i < kStriceNum; i++) {
        reg_list[i].assign(output_tensor_info_list_[2 * i].GetDataAsFloat(), output_tensor_info_list_[2 * i].GetDataAsFloat() + feature_num_list[i] * (4 * (kRegMax + 1)));
        score_list[i].assign(output_tensor_info_list_[2 * i + 1].GetDataAsFloat(), output_tensor_info_list_[2 * i + 1].GetDataAsFloat() + feature_num_list[i] * kNumClass);
    }
    for (int32_t i = 0; i < kStriceNum; i++) {
        int32_t grid_w = input_tensor_info.GetWidth() / kStrideList[i];
        int32_t grid_h = input_tensor_info.GetHeight() / kStrideList[i];
        DecodeInfer(bbox_list, score_list[i], reg_list[i], threshold_confidence_
            , grid_w, grid_h, static_cast<float>(crop_w) / grid_w, static_cast<float>(crop_h) / grid_h);
    }

    /* NMS */
    std::vector<BoundingBox> bbox_nms_list;
    BoundingBoxUtils::Nms(bbox_list, bbox_nms_list, threshold_nms_iou_);

    /* Adjust bounding box */
    for (auto& bbox : bbox_nms_list) {
        bbox.x = (std::max)(bbox.x, 0) + crop_x;
        bbox.y = (std::max)(bbox.y, 0) + crop_y;
        bbox.w = (std::min)(bbox.w, original_mat.cols - bbox.x);
        bbox.h = (std::min)(bbox.h, original_mat.rows - bbox.y);
    }

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

/* Original code: https://github.com/RangiLyu/nanodet/blob/main/demo_ncnn/nanodet.cpp */
int32_t DetectionEngine::DecodeInfer(std::vector<BoundingBox>& bbox_list, const std::vector<float>& score_list, const std::vector<float>& reg_list, double threshold, int32_t grid_w, int32_t grid_h, float scale_grid2org_w, float scale_grid2org_h)
{
    for (int32_t i = 0; i < grid_w * grid_h; i++) {
        float score_max = 0;
        int32_t class_id_max = 0;
        for (int32_t class_id = 0; class_id < kNumClass; class_id++) {
            float score = score_list[i * kNumClass + class_id];
            if (score > score_max) {
                score_max = score;
                class_id_max = class_id;
            }
        }
        if (score_max > threshold) {
            BoundingBox bbox;
            int32_t grid_x = i % grid_w;
            int32_t grid_y = i / grid_h;
            DisPred2Bbox(bbox, reg_list, i, grid_x, grid_y, scale_grid2org_w, scale_grid2org_h);
            bbox.class_id = class_id_max;
            bbox.label = label_list_[bbox.class_id];
            bbox.score = score_max;
            bbox_list.push_back(bbox);
        }
    }
    return kRetOk;
}

static inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = static_cast<int32_t>((1 << 23) * (1.4426950409 * x + 126.93490512f));
    return v.f;
}

template<typename _Tp>
static int32_t Activation_function_softmax(const _Tp* src, _Tp* dst, int32_t length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int32_t i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int32_t i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

void DetectionEngine::DisPred2Bbox(BoundingBox& bbox, const std::vector<float>& reg_list, int32_t idx, int32_t x, int32_t y, float scale_grid2org_w, float scale_grid2org_h)
{
    float ct_x = (x + 0.5f);
    float ct_y = (y + 0.5f);
    std::vector<float> dis_pred;
    dis_pred.resize(4);


    int32_t pos = idx * ((kRegMax + 1) * 4);  /* idx * 32 */
    for (int32_t i = 0; i < 4; i++) {
        float dis = 0;
        float dis_after_sm[kRegMax + 1];
        Activation_function_softmax(&reg_list[pos + i * (kRegMax + 1)], dis_after_sm, kRegMax + 1);
        for (int32_t j = 0; j < kRegMax + 1; j++) {
            dis += j * dis_after_sm[j];
        }
        dis_pred[i] = dis;
    }

    bbox.x = static_cast<int32_t>((ct_x - dis_pred[0]) * scale_grid2org_w);
    bbox.y = static_cast<int32_t>((ct_y - dis_pred[1]) * scale_grid2org_h);
    bbox.w = static_cast<int32_t>((ct_x + dis_pred[2]) * scale_grid2org_w - bbox.x);
    bbox.h = static_cast<int32_t>((ct_y + dis_pred[3]) * scale_grid2org_h - bbox.y);

    return;
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

