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
static cv::Scalar CreateCvColor(int32_t b, int32_t g, int32_t r)
{
#ifdef CV_COLOR_IS_RGB
    return cv::Scalar(r, g, b);
#else
    return cv::Scalar(b, g, r);
#endif
}

static void DrawText(cv::Mat& mat, const std::string& text, cv::Point pos, double fontScale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back)
{
#if 1
    int32_t baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    baseline += thickness;
    pos.y -= textSize.height / 2;
    cv::rectangle(mat, pos + cv::Point(0, baseline), pos + cv::Point(textSize.width, -textSize.height), color_back, -1);
    cv::putText(mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, fontScale, color_front, thickness);
#else
    cv::putText(mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, fontScale, color_back, thickness * 3);
    cv::putText(mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, fontScale, color_front, thickness);
#endif
}

static cv::Scalar GetColorForId(int32_t id)
{
    static constexpr int32_t kMaxNum = 100;
    static std::vector<cv::Scalar> color_list;
    static bool is_first = true;
    if (is_first) {
        is_first = false;

        uint32_t palette[] = { static_cast<uint32_t>(std::pow(2, 11) - 1), static_cast<uint32_t>(std::pow(2, 15) - 1), static_cast<uint32_t>(std::pow(2, 20) - 1) };
        for (int32_t i = 0; i < kMaxNum; i++) {
            color_list.push_back(CreateCvColor((palette[0] * (static_cast<uint32_t>(std::pow(i, 4)) - i + 1)) % 255, (palette[1] * (static_cast<uint32_t>(std::pow(i, 4)) - i + 1)) % 255, (palette[2] * (static_cast<uint32_t>(std::pow(i, 4)) - i + 1)) % 255));
        }
    }
    return color_list[id % kMaxNum];
}

int32_t ImageProcessor::Initialize(const ImageProcessor::InputParam* input_param)
{
    if (s_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_engine.reset(new DetectionEngine());
    if (s_engine->Initialize(input_param->work_dir, input_param->num_threads) != DetectionEngine::kRetOk) {
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



int32_t ImageProcessor::Process(cv::Mat* mat, ImageProcessor::OutputParam* output_param)
{
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    cv::Mat& original_mat = *mat;
    DetectionEngine::Result result;
    if (s_engine->Process(original_mat, result) != DetectionEngine::kRetOk) {
        return -1;
    }

    cv::rectangle(original_mat, cv::Rect(result.crop_x, result.crop_y, result.crop_w, result.crop_h), CreateCvColor(0, 0, 0), 2);
    for (const auto& bbox : result.bbox_list) {
        cv::rectangle(original_mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), CreateCvColor(0, 0, 0), 1);
    }

    std::vector<BoundingBox> bbox_result_list;
    s_tracker.Update(result.bbox_list);
    auto& track_list = s_tracker.GetTrackList();
    for (auto& track : track_list) {
        if (track.GetDetectedCount() < 5) continue;
        
        auto& bbox = track.GetLatestData().bbox;
        cv::Scalar color = bbox.score == 0 ? CreateCvColor(255, 255, 255) : GetColorForId(track.GetId());
        cv::rectangle(original_mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), color, 2);
        DrawText(original_mat, std::to_string(track.GetId()) + ": " + bbox.label, cv::Point(bbox.x, bbox.y), 0.5, 1, CreateCvColor(0, 0, 0), CreateCvColor(220, 220, 220));

        auto& track_history = track.GetDataHistory();
        for (int32_t i = 1; i < track_history.size(); i++) {
            cv::Point p0(track_history[i].bbox.x + track_history[i].bbox.w / 2, track_history[i].bbox.y + track_history[i].bbox.h);
            cv::Point p1(track_history[i - 1].bbox.x + track_history[i - 1].bbox.w / 2, track_history[i - 1].bbox.y + track_history[i - 1].bbox.h);
            cv::line(original_mat, p0, p1, CreateCvColor(255, 0, 0));
        }
    }

    /* Return the results */
    int32_t bbox_num = 0;
    for (const auto& bbox : result.bbox_list) {
        output_param->object_list[bbox_num].class_id = bbox.class_id;
        snprintf(output_param->object_list[bbox_num].label, sizeof(output_param->object_list[bbox_num].label), "%s", bbox.label.c_str());
        output_param->object_list[bbox_num].score = bbox.score;
        output_param->object_list[bbox_num].x = bbox.x;
        output_param->object_list[bbox_num].y = bbox.y;
        output_param->object_list[bbox_num].width = bbox.w;
        output_param->object_list[bbox_num].height = bbox.h;
        bbox_num++;
        if (bbox_num >= NUM_MAX_RESULT) break;
    }
    output_param->object_num = bbox_num;


    output_param->time_pre_process = result.time_pre_process;
    output_param->time_inference = result.time_inference;
    output_param->time_post_process = result.time_post_process;

    return 0;
}

