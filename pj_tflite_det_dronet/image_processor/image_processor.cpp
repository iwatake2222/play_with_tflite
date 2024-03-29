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
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <numeric>

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

static cv::Scalar GetColorForId(int32_t id)
{
    static constexpr int32_t kMaxNum = 100;
    static std::vector<cv::Scalar> color_list;
    if (color_list.empty()) {
        std::srand(123);
        for (int32_t i = 0; i < kMaxNum; i++) {
            color_list.push_back(CommonHelper::CreateCvColor(std::rand() % 255, std::rand() % 255, std::rand() % 255));
        }
    }
    return color_list[id % kMaxNum];
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


static void AnalyzeFlow(cv::Mat& mat, std::vector<Track>& track_list)
{

    constexpr int32_t kPastFrameToCalculateVelocity = 10;
    constexpr int32_t kFrameToCalculateAverage = 3;
    constexpr int32_t kLineLength = 200;

    for (auto& track : track_list) {
        const auto& data_list = track.GetDataHistory();
        if (data_list.size() < kPastFrameToCalculateVelocity + kFrameToCalculateAverage + 1) continue;
        BoundingBox bbox_now;
        BoundingBox bbox_past;
        for (int32_t i = 0; i < kFrameToCalculateAverage; i++) {
            bbox_now.x += data_list[data_list.size() - 1 - i].bbox.x;
            bbox_now.y += data_list[data_list.size() - 1 - i].bbox.y;
            bbox_now.w += data_list[data_list.size() - 1 - i].bbox.w;
            bbox_now.h += data_list[data_list.size() - 1 - i].bbox.h;
            bbox_past.x += data_list[data_list.size() - 1 - kPastFrameToCalculateVelocity - i].bbox.x;
            bbox_past.y += data_list[data_list.size() - 1 - kPastFrameToCalculateVelocity - i].bbox.y;
            bbox_past.w += data_list[data_list.size() - 1 - kPastFrameToCalculateVelocity - i].bbox.w;
            bbox_past.h += data_list[data_list.size() - 1 - kPastFrameToCalculateVelocity - i].bbox.h;
        }
        bbox_now.x /= kFrameToCalculateAverage;
        bbox_now.y /= kFrameToCalculateAverage;
        bbox_now.w /= kFrameToCalculateAverage;
        bbox_now.h /= kFrameToCalculateAverage;
        bbox_past.x /= kFrameToCalculateAverage;
        bbox_past.y /= kFrameToCalculateAverage;
        bbox_past.w /= kFrameToCalculateAverage;
        bbox_past.h /= kFrameToCalculateAverage;

        float velocity = static_cast<float>(std::sqrt(std::pow(bbox_now.x - bbox_past.x, 2) + std::pow(bbox_now.y - bbox_past.y, 2)) / kPastFrameToCalculateVelocity);
        float angle = (bbox_now.x == bbox_past.x) ?
            (bbox_now.y > bbox_past.y) ? M_PI / 2 : -M_PI / 2
            : std::atanf((bbox_now.y - bbox_past.y) / static_cast<float>(bbox_now.x - bbox_past.x));
        if (bbox_now.x < bbox_past.x) angle += M_PI;
        if (angle > M_PI) angle -= 2 * M_PI;
        if (angle < -M_PI) angle += 2 * M_PI;
        CommonHelper::DrawText(mat, std::to_string(angle * 180 / M_PI), cv::Point(bbox_now.x, bbox_now.y), 0.3, 1, cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 255));


        velocity = (std::min)(20.0f, velocity / 20.0f);   /* normalize velocity to [0.0 - 20.0] (px/10frames)*/
        int32_t line_length = static_cast<int32_t>(kLineLength * velocity);
        cv::Point p0(bbox_now.x + bbox_now.w / 2, bbox_now.y + bbox_now.h / 2);
        cv::Point p1 = p0 + cv::Point(static_cast<int32_t>(line_length * cos(angle)), static_cast<int32_t>(line_length * sin(angle)));

        /* The following code is just to make nice color */
        float angle_deg = static_cast<float>(angle * 180 / M_PI);
        float right_level = 0;
        if (std::abs(angle_deg) <= 90) right_level = 1.0f - std::abs(angle_deg) / 90.0f;
        float left_level = 0;
        if (std::abs(angle_deg) > 90) left_level = (std::abs(angle_deg) - 90.0f) / 90.0f;
        float up_level = 0;
        if (angle_deg < 0) up_level = 1.0f - std::abs(angle_deg + 90) / 90.0f;
        float down_level = 0;
        if (angle_deg > 0) down_level = 1.0f - std::abs(angle_deg - 90) / 90.0f;

        cv::arrowedLine(mat, p0, p1, CommonHelper::CreateCvColor(
            std::min(255.0f, right_level * 255 + down_level * 128),
            std::min(255.0f, left_level * 255 + down_level * 128),
            std::min(255.0f, up_level * 255)),
            3);

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
        //if (track.GetDetectedCount() < 2) continue;
        const auto& bbox = track.GetLatestData().bbox;
        /* Use white rectangle for the object which was not detected but just predicted */
        cv::Scalar color = bbox.score == 0 ? CommonHelper::CreateCvColor(255, 255, 255) : GetColorForId(track.GetId());
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), color, 2);
        //CommonHelper::DrawText(mat, std::to_string(track.GetId()) + ": " + bbox.label, cv::Point(bbox.x, bbox.y - 13), 0.35, 1, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

        auto& track_history = track.GetDataHistory();
        for (size_t i = 1; i < track_history.size(); i++) {
            cv::Point p0(track_history[i].bbox.x + track_history[i].bbox.w / 2, track_history[i].bbox.y + track_history[i].bbox.h);
            cv::Point p1(track_history[i - 1].bbox.x + track_history[i - 1].bbox.w / 2, track_history[i - 1].bbox.y + track_history[i - 1].bbox.h);
            cv::line(mat, p0, p1, color);
        }
        num_track++;
    }


    AnalyzeFlow(mat, track_list);

    CommonHelper::DrawText(mat, "DET: " + std::to_string(num_det) + ", TRACK: " + std::to_string(num_track), cv::Point(0, 20), 0.7, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));
    DrawFps(mat, det_result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /* Return the results */
    result.time_pre_process = det_result.time_pre_process;
    result.time_inference = det_result.time_inference;
    result.time_post_process = det_result.time_post_process;

    return 0;
}

