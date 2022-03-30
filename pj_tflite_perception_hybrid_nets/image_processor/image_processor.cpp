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
#include "detection_engine.h"
#include "tracker.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<DetectionEngine> s_engine;
Tracker s_tracker;
CommonHelper::NiceColorGenerator s_nice_color_generator;

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
    if (s_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_engine.reset(new DetectionEngine());
    if (s_engine->Initialize(input_param.work_dir, input_param.num_threads) != DetectionEngine::kRetOk) {
        s_engine->Finalize();
        s_engine.reset();
        return -1;
    }
    return 0;
}

int32_t ImageProcessor::Finalize(void)
{
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    if (s_engine->Finalize() != DetectionEngine::kRetOk) {
        return -1;
    }

    return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
    if (!s_engine) {
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
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    DetectionEngine::Result det_result;
    if (s_engine->Process(mat, det_result) != DetectionEngine::kRetOk) {
        return -1;
    }

    /* Display target area  */
    cv::rectangle(mat, cv::Rect(det_result.crop.x, det_result.crop.y, det_result.crop.w, det_result.crop.h), CommonHelper::CreateCvColor(0, 0, 0), 2);

    /* Draw segmentation image for the class of the highest score */
    cv::Mat& mat_seg_max = det_result.mat_seg_max;
    cv::Mat mat_seg_max_list[] = { mat_seg_max, mat_seg_max, mat_seg_max };
    cv::merge(mat_seg_max_list, 3, mat_seg_max);
    cv::Mat mat_lut = cv::Mat::zeros(256, 1, CV_8UC3);
    mat_lut.at<cv::Vec3b>(0) = cv::Vec3b(0, 0, 0);
    mat_lut.at<cv::Vec3b>(1) = cv::Vec3b(0, 255, 0);
    mat_lut.at<cv::Vec3b>(2) = cv::Vec3b(0, 0, 255);
    cv::LUT(mat_seg_max, mat_lut, mat_seg_max);
    cv::resize(mat_seg_max, mat_seg_max, mat.size(), 0.0, 0.0, cv::INTER_NEAREST);
    cv::Mat mat_masked;
    cv::addWeighted(mat, 0.8, mat_seg_max, 0.5, 0, mat_masked);
    //cv::add(mat_seg_max * kResultMixRatio, mat * (1.0f - kResultMixRatio), mat_masked);
    //cv::hconcat(mat, mat_masked, mat);
    mat = mat_masked;

    /* Display detection result (black rectangle) */
    int32_t num_det = 0;
    for (const auto& bbox : det_result.bbox_list) {
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), CommonHelper::CreateCvColor(0, 0, 0), 1);
        num_det++;
    }

    /* Display tracking result  */
    s_tracker.Update(det_result.bbox_list);
    int32_t num_track = 0;
    auto& track_list = s_tracker.GetTrackList();
    for (auto& track : track_list) {
        if (track.GetDetectedCount() < 2) continue;
        const auto& bbox = track.GetLatestData().bbox;
        /* Use white rectangle for the object which was not detected but just predicted */
        cv::Scalar color = bbox.score == 0 ? CommonHelper::CreateCvColor(255, 255, 255) : s_nice_color_generator.Get(track.GetId());
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), color, 2);
        CommonHelper::DrawText(mat, std::to_string(track.GetId()) + ": " + bbox.label, cv::Point(bbox.x, bbox.y - 13), 0.35, 1, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

        auto& track_history = track.GetDataHistory();
        for (size_t i = 1; i < track_history.size(); i++) {
            cv::Point p0(track_history[i].bbox.x + track_history[i].bbox.w / 2, track_history[i].bbox.y + track_history[i].bbox.h);
            cv::Point p1(track_history[i - 1].bbox.x + track_history[i - 1].bbox.w / 2, track_history[i - 1].bbox.y + track_history[i - 1].bbox.h);
            cv::line(mat, p0, p1, CommonHelper::CreateCvColor(255, 0, 0));
        }
        num_track++;
    }
    CommonHelper::DrawText(mat, "DET: " + std::to_string(num_det) + ", TRACK: " + std::to_string(num_track), cv::Point(0, 20), 0.7, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

    DrawFps(mat, det_result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /* Return the results */
    int32_t bbox_num = 0;
    for (auto& track : track_list) {
        const auto& bbox = track.GetLatestData().bbox;
        result.object_list[bbox_num].class_id = bbox.class_id;
        snprintf(result.object_list[bbox_num].label, sizeof(result.object_list[bbox_num].label), "%s", bbox.label.c_str());
        result.object_list[bbox_num].score = bbox.score;
        result.object_list[bbox_num].x = bbox.x;
        result.object_list[bbox_num].y = bbox.y;
        result.object_list[bbox_num].width = bbox.w;
        result.object_list[bbox_num].height = bbox.h;
        bbox_num++;
        if (bbox_num >= NUM_MAX_RESULT) break;
    }
    result.object_num = bbox_num;

    result.time_pre_process = det_result.time_pre_process;
    result.time_inference = det_result.time_inference;
    result.time_post_process = det_result.time_post_process;

    return 0;
}

