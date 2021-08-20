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
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "bounding_box.h"
#include "face_detection_engine.h"
#include "age_gender_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<FaceDetectionEngine> s_facedet_engine;
std::unique_ptr<AgeGenderEngine> s_facemesh_engine;

/*** Function ***/
static void DrawFps(cv::Mat& mat, double time_inference_det, double time_inference_feature, int32_t num_feature, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect = true)
{
    char text[128];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %4.1f, Inference: DET: %4.1f[ms], ATTRIBUTE:%3d x %4.1f[ms]", fps, time_inference_det, num_feature, time_inference_feature / num_feature);
    CommonHelper::DrawText(mat, text, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);
}


int32_t ImageProcessor::Initialize(const ImageProcessor::InputParam& input_param)
{
    if (s_facedet_engine || s_facemesh_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_facedet_engine.reset(new FaceDetectionEngine());
    if (s_facedet_engine->Initialize(input_param.work_dir, input_param.num_threads) != FaceDetectionEngine::kRetOk) {
        s_facedet_engine->Finalize();
        s_facedet_engine.reset();
        return -1;
    }

    s_facemesh_engine.reset(new AgeGenderEngine());
    if (s_facemesh_engine->Initialize(input_param.work_dir, input_param.num_threads) != AgeGenderEngine::kRetOk) {
        s_facemesh_engine->Finalize();
        s_facemesh_engine.reset();
        return -1;
    }

    return 0;
}

int32_t ImageProcessor::Finalize(void)
{
    if (!s_facedet_engine || !s_facemesh_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    if (s_facedet_engine->Finalize() != AgeGenderEngine::kRetOk) {
        return -1;
    }

    if (s_facemesh_engine->Finalize() != AgeGenderEngine::kRetOk) {
        return -1;
    }

    return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
    if (!s_facedet_engine || !s_facemesh_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    switch (cmd) {
    case 0:
    default:
        PRINT_E("command(%d) is not supported\n", cmd);
        return -1;
    }
}


int32_t ImageProcessor::Process(cv::Mat& mat, ImageProcessor::Result& result)
{
    if (!s_facedet_engine || !s_facemesh_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    /* Detect face */
    FaceDetectionEngine::Result det_result;
    if (s_facedet_engine->Process(mat, det_result) != FaceDetectionEngine::kRetOk) {
        return -1;
    }

    /* Display target area  */
    cv::rectangle(mat, cv::Rect(det_result.crop.x, det_result.crop.y, det_result.crop.w, det_result.crop.h), CommonHelper::CreateCvColor(0, 0, 0), 2);

    /* Display detection result and keypoint */
    for (const auto& bbox : det_result.bbox_list) {
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), CommonHelper::CreateCvColor(0, 200, 0), 1);
    }

    //for (const auto& keypoint : det_result.keypoint_list) {
    //    for (const auto& p : keypoint) {
    //        cv::circle(mat, cv::Point(p.first, p.second), 1, CommonHelper::CreateCvColor(0, 255, 0));
    //    }
    //}

    /* Estimate age and gender */
    double time_pre_process_feature = 0;   // [msec]
    double time_inference_feature = 0;    // [msec]
    double time_post_process_feature = 0;  // [msec]
    for (const auto& bbox : det_result.bbox_list) {
        AgeGenderEngine::Result agegender_result;
        if (s_facemesh_engine->Process(mat, bbox, agegender_result) != AgeGenderEngine::kRetOk) {
            return -1;
        }
        
        cv::Scalar color = CommonHelper::CreateCvColor(80, 80, 80);
        if (agegender_result.gender == AgeGenderEngine::kGenderFemale) {
            color = CommonHelper::CreateCvColor(0, 0, 255);
        } else if (agegender_result.gender == AgeGenderEngine::kGenderMale) {
            color = CommonHelper::CreateCvColor(255, 0, 0);
        }
        char text[128];
        snprintf(text, sizeof(text), "%d: %s", agegender_result.age, agegender_result.gender_str.c_str());
        CommonHelper::DrawText(mat, text, cv::Point(bbox.x, bbox.y - 10), 0.4, 1, color, CommonHelper::CreateCvColor(220, 220, 220), true);

        time_pre_process_feature += agegender_result.time_pre_process;
        time_inference_feature += agegender_result.time_inference;
        time_post_process_feature += agegender_result.time_post_process;
    }


    /* Return the results */
    result.time_pre_process = det_result.time_pre_process + time_pre_process_feature;
    result.time_inference = det_result.time_inference + time_inference_feature;
    result.time_post_process = det_result.time_post_process + time_post_process_feature;

    DrawFps(mat, det_result.time_inference, time_inference_feature, static_cast<int32_t>(det_result.bbox_list.size()), cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    return 0;
}

