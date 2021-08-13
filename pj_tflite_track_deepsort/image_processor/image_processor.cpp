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
#include "feature_engine.h"
#include "tracker_deepsort.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

#define USE_DEEPSORT

/*** Global variable ***/
std::unique_ptr<DetectionEngine> s_det_engine;
std::unique_ptr<FeatureEngine> s_feature_engine;
#ifdef USE_DEEPSORT
TrackerDeepSort s_tracker(30);
#else
TrackerDeepSort s_tracker(2);
#endif

/*** Function ***/
static void DrawFps(cv::Mat& mat, double time_inference_det, double time_inference_feature, int32_t num_feature, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect = true)
{
    char text[128];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %4.1f, Inference: DET: %4.1f[ms], FEATURE:%3d x %4.1f[ms]", fps, time_inference_det, num_feature, time_inference_feature / num_feature);
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
    if (s_det_engine || s_feature_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_det_engine.reset(new DetectionEngine(0.4f, 0.2f, 0.5f));
    if (s_det_engine->Initialize(input_param.work_dir, input_param.num_threads) != DetectionEngine::kRetOk) {
        s_det_engine->Finalize();
        s_det_engine.reset();
        return -1;
    }

    s_feature_engine.reset(new FeatureEngine());
    if (s_feature_engine->Initialize(input_param.work_dir, input_param.num_threads) != FeatureEngine::kRetOk) {
        s_feature_engine->Finalize();
        s_feature_engine.reset();
        return -1;
    }
    
    return 0;
}

int32_t ImageProcessor::Finalize(void)
{
    if (!s_det_engine || !s_feature_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    if (s_det_engine->Finalize() != DetectionEngine::kRetOk) {
        return -1;
    }

    if (s_feature_engine->Finalize() != FeatureEngine::kRetOk) {
        return -1;
    }

    return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
    if (!s_det_engine || !s_feature_engine) {
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
    if (!s_det_engine || !s_feature_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    /* Detection */
    DetectionEngine::Result det_result;
    if (s_det_engine->Process(mat, det_result) != DetectionEngine::kRetOk) {
        return -1;
    }

    /* Extract feature for the detected objects */
    std::vector<std::vector<float>> feature_list;
    double time_pre_process_feature = 0;   // [msec]
    double time_inference_feature = 0;    // [msec]
    double time_post_process_feature = 0;  // [msec]
    for (const auto& bbox : det_result.bbox_list) {
#ifdef USE_DEEPSORT
        if (bbox.class_id == 0) {   /* Calculate face feature for person only */
            FeatureEngine::Result feature_result;
            if (s_feature_engine->Process(mat, bbox, feature_result) != DetectionEngine::kRetOk) {
                return -1;
            }
            feature_list.push_back(feature_result.feature);
            time_pre_process_feature += feature_result.time_pre_process;
            time_inference_feature += feature_result.time_inference;
            time_post_process_feature += feature_result.time_post_process;
        } else {
            feature_list.push_back(std::vector<float>());   /* the length of feature is 0. so it's not used in tracker (DeepSORT) */
        }
#else
        feature_list.push_back(std::vector<float>());   /* the length of feature is 0. so it's not used in tracker (DeepSORT) */
#endif
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
    s_tracker.Update(det_result.bbox_list, feature_list);
    int32_t num_track = 0;
    auto& track_list = s_tracker.GetTrackList();
    for (auto& track : track_list) {
        if (track.GetDetectedCount() < 2) continue; /* To decrease FP */
        const auto& bbox = track.GetLatestData().bbox;
        if (bbox.score == 0) continue;  /* the oboject is in tracker, but not detected at the current frame */
        cv::Scalar color = GetColorForId(track.GetId());
        cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), color, 2);
        CommonHelper::DrawText(mat, std::to_string(track.GetId()) + ": " + bbox.label, cv::Point(bbox.x, bbox.y - 13), 0.35, 1, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

        auto& track_history = track.GetDataHistory();
        for (size_t i = 1; i < track_history.size(); i++) {
            cv::Point p0(track_history[i].bbox.x + track_history[i].bbox.w / 2, track_history[i].bbox.y + track_history[i].bbox.h);
            cv::Point p1(track_history[i - 1].bbox.x + track_history[i - 1].bbox.w / 2, track_history[i - 1].bbox.y + track_history[i - 1].bbox.h);
            cv::line(mat, p0, p1, color);
        }
        num_track++;
    }
    CommonHelper::DrawText(mat, "DET: " + std::to_string(num_det) + ", TRACK: " + std::to_string(num_track), cv::Point(0, 20), 0.7, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

    DrawFps(mat, det_result.time_inference, time_inference_feature, static_cast<int32_t>(feature_list.size()), cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /* Return the results */
    result.time_pre_process = det_result.time_pre_process + time_pre_process_feature;
    result.time_inference = det_result.time_inference + time_inference_feature;
    result.time_post_process = det_result.time_post_process + time_post_process_feature;

    return 0;
}

