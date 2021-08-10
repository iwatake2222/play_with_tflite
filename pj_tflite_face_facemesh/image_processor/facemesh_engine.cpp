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
#include "facemesh_engine.h"

/*** Macro ***/
#define TAG "FacemeshEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME  "face_landmark.tflite"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input_1"
#define INPUT_DIMS  { 1, 192, 192, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_0 "conv2d_20"
#define OUTPUT_NAME_1 "conv2d_30"

/*** Function ***/
int32_t FacemeshEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
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

    /* Create and Initialize Inference Helper */
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu1));
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

    return kRetOk;
}

int32_t FacemeshEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


int32_t FacemeshEngine::Process(const cv::Mat& original_mat, const std::vector<BoundingBox>& bbox_list, std::vector<Result>& result_list)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }

    result_list.clear();

    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];

    for (const auto& bbox : bbox_list) {
        /*** PreProcess ***/
        const auto& t_pre_process0 = std::chrono::steady_clock::now();
        int32_t cx = bbox.x + bbox.w / 2;
        int32_t cy = bbox.y + bbox.h / 2;
        int32_t face_size = static_cast<int32_t>((std::max)(bbox.w, bbox.h) * 1.7f);   /* expand face bbox */
        int32_t crop_x = (std::max)(0, cx - face_size / 2);
        int32_t crop_y = (std::max)(0, cy - face_size / 2);
        int32_t crop_w = (std::min)(face_size, original_mat.cols - crop_x);
        int32_t crop_h = (std::min)(face_size, original_mat.rows - crop_y);
        cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
        CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);

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
        std::vector<float> landmark_list(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_tensor_info_list_[0].GetElementNum());
        std::vector<float> score_list(output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat() + output_tensor_info_list_[1].GetElementNum());

        float scale_w = static_cast<float>(crop_w) / input_tensor_info.GetWidth();
        float scale_h = static_cast<float>(crop_h) / input_tensor_info.GetHeight();

        /* reference : https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md#output */
        Result result;
        result.score = score_list[0];
        for (int32_t i = 0; i < result.keypoint_list.size(); i++) {
            result.keypoint_list[i].first = static_cast<int32_t>(landmark_list[3 * i + 0] * scale_w + 0.5f + crop_x);
            result.keypoint_list[i].second = static_cast<int32_t>(landmark_list[3 * i + 1] * scale_h + 0.5f + crop_y);
        }

        const auto& t_post_process1 = std::chrono::steady_clock::now();

        result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
        result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
        result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;
        result_list.push_back(result);
    }

    return kRetOk;
}


/* reference: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh.py#L41 */
const std::vector<std::pair<int32_t, int32_t>>& FacemeshEngine::GetConnectionList()
{
    static const std::vector<std::pair<int32_t, int32_t>> connection_list = {
    {61, 146},
    {146, 91},
    {91, 181},
    {181, 84},
    {84, 17},
    {17, 314},
    {314, 405},
    {405, 321},
    {321, 375},
    {375, 291},
    {61, 185},
    {185, 40},
    {40, 39},
    {39, 37},
    {37, 0},
    {0, 267},
    {267, 269},
    {269, 270},
    {270, 409},
    {409, 291},
    {78, 95},
    {95, 88},
    {88, 178},
    {178, 87},
    {87, 14},
    {14, 317},
    {317, 402},
    {402, 318},
    {318, 324},
    {324, 308},
    {78, 191},
    {191, 80},
    {80, 81},
    {81, 82},
    {82, 13},
    {13, 312},
    {312, 311},
    {311, 310},
    {310, 415},
    {415, 308},
    //# Left eye.
    {263, 249},
    {249, 390},
    {390, 373},
    {373, 374},
    {374, 380},
    {380, 381},
    {381, 382},
    {382, 362},
    {263, 466},
    {466, 388},
    {388, 387},
    {387, 386},
    {386, 385},
    {385, 384},
    {384, 398},
    {398, 362},
    //Left eyebrow.
    {276, 283},
    {283, 282},
    {282, 295},
    {295, 285},
    {300, 293},
    {293, 334},
    {334, 296},
    {296, 336},
    //Right eye.
    {33, 7},
    {7, 163},
    {163, 144},
    {144, 145},
    {145, 153},
    {153, 154},
    {154, 155},
    {155, 133},
    {33, 246},
    {246, 161},
    {161, 160},
    {160, 159},
    {159, 158},
    {158, 157},
    {157, 173},
    {173, 133},
    //Right eyebrow.
    {46, 53},
    {53, 52},
    {52, 65},
    {65, 55},
    {70, 63},
    {63, 105},
    {105, 66},
    {66, 107},
    //Face oval.
    {10, 338},
    {338, 297},
    {297, 332},
    {332, 284},
    {284, 251},
    {251, 389},
    {389, 356},
    {356, 454},
    {454, 323},
    {323, 361},
    {361, 288},
    {288, 397},
    {397, 365},
    {365, 379},
    {379, 378},
    {378, 400},
    {400, 377},
    {377, 152},
    {152, 148},
    {148, 176},
    {176, 149},
    {149, 150},
    {150, 136},
    {136, 172},
    {172, 58},
    {58, 132},
    {132, 93},
    {93, 234},
    {234, 127},
    {127, 162},
    {162, 21},
    {21, 54},
    {54, 103},
    {103, 67},
    {67, 109},
    {109, 10},
    };

    return connection_list;
}