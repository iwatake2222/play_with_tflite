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
#include "facemesh_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<FaceDetectionEngine> s_facedet_engine;
std::unique_ptr<FacemeshEngine> s_facemesh_engine;

/*** Function ***/
static void DrawFps(cv::Mat& mat, double time_inference, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect = true)
{
    char text[64];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %.1f, Inference: %.1f [ms]", fps, time_inference);
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

    s_facemesh_engine.reset(new FacemeshEngine());
    if (s_facemesh_engine->Initialize(input_param.work_dir, input_param.num_threads) != FacemeshEngine::kRetOk) {
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

    if (s_facedet_engine->Finalize() != FacemeshEngine::kRetOk) {
        return -1;
    }

    if (s_facemesh_engine->Finalize() != FacemeshEngine::kRetOk) {
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

    /* Detect facemesh */
    std::vector<FacemeshEngine::Result> facemesh_result_list;
    if (s_facemesh_engine->Process(mat, det_result.bbox_list, facemesh_result_list) != FacemeshEngine::kRetOk) {
        return -1;
    }

    /* Display result for detected faces */
    const auto& connection_list = FacemeshEngine::GetConnectionList();
    for (const auto& facemesh_result : facemesh_result_list) {
        /* Display wire */
        for (const auto& connection : connection_list) {
            cv::Point p1(facemesh_result.keypoint_list[connection.first].first, facemesh_result.keypoint_list[connection.first].second);
            cv::Point p2(facemesh_result.keypoint_list[connection.second].first, facemesh_result.keypoint_list[connection.second].second);
            cv::line(mat, p1, p2, CommonHelper::CreateCvColor(0, 255, 0));
        }
    
        /* Display point */
        for (const auto& keypoint : facemesh_result.keypoint_list) {
            cv::circle(mat, cv::Point(keypoint.first, keypoint.second), 1, CommonHelper::CreateCvColor(0, 255, 255));
            
        }
    }

    /* Return the results */
    result.time_pre_process = det_result.time_pre_process;
    result.time_inference = det_result.time_inference;
    result.time_post_process = det_result.time_post_process;
    for (const auto& facemesh_result : facemesh_result_list) {
        result.time_pre_process += facemesh_result.time_pre_process;
        result.time_inference += facemesh_result.time_inference;
        result.time_post_process += facemesh_result.time_post_process;
    }

    DrawFps(mat, result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    return 0;
}

