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
#include "inference_helper.h"
#include "detection_engine.h"

/*** Macro ***/
#define TAG "DetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite"
#define LABEL_NAME   "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.txt"


/*** Function ***/
int32_t DetectionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;
    std::string labelFilename = work_dir + "/model/" + LABEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info("normalized_input_image_tensor", TensorInfo::kTensorTypeUint8);
    input_tensor_info.tensor_dims = { 1, 300, 300, 3 };
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.5f;
    input_tensor_info.normalize.mean[1] = 0.5f;
    input_tensor_info.normalize.mean[2] = 0.5f;
    input_tensor_info.normalize.norm[0] = 0.5f;
    input_tensor_info.normalize.norm[1] = 0.5f;
    input_tensor_info.normalize.norm[2] = 0.5f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo("TFLite_Detection_PostProcess", TensorInfo::kTensorTypeFp32));
    output_tensor_info_list_.push_back(OutputTensorInfo("TFLite_Detection_PostProcess:1", TensorInfo::kTensorTypeFp32));
    output_tensor_info_list_.push_back(OutputTensorInfo("TFLite_Detection_PostProcess:2", TensorInfo::kTensorTypeFp32));
    output_tensor_info_list_.push_back(OutputTensorInfo("TFLite_Detection_PostProcess:3", TensorInfo::kTensorTypeFp32));

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
    /* Retrieve result */
    int32_t output_num = (int32_t)(output_tensor_info_list_[3].GetDataAsFloat()[0]);
    std::vector<Object> object_list;
    GetObject(object_list, output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[2].GetDataAsFloat(), output_num, 0.5, original_mat.cols, original_mat.rows);
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.object_list = object_list;
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


int32_t DetectionEngine::GetObject(std::vector<Object>& object_list, const float *output_box_list, const float *output_class_list, const float *output_score_list, const int32_t output_num,
    const double threshold, const int32_t width, const int32_t height)
{
    for (int32_t i = 0; i < output_num; i++) {
        int32_t class_id = static_cast<int32_t>(output_class_list[i] + 1);
        float score = output_score_list[i];
        if (score < threshold) continue;
        float y0 = output_box_list[4 * i + 0];
        float x0 = output_box_list[4 * i + 1];
        float y1 = output_box_list[4 * i + 2];
        float x1 = output_box_list[4 * i + 3];
        if (width > 0) {
            x0 *= width;
            x1 *= width;
            y0 *= height;
            y1 *= height;
        }
        //PRINT("%d[%.2f]: %.3f %.3f %.3f %.3f\n", class_id, score, x0, y0, x1, y1);
        Object object;
        object.x = x0;
        object.y = y0;
        object.width = x1 - x0;
        object.height = y1 - y0;
        object.class_id = class_id;
        object.label = label_list_[class_id];
        object.score = score;
        object_list.push_back(object);
    }
    return kRetOk;
}
