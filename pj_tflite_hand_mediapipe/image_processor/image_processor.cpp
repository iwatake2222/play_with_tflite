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
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "palm_detection_engine.h"
#include "hand_landmark_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)


/*** Setting ***/
#define INTERVAL_TO_ENFORCE_PALM_DET 5

class Rect {
public:
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
    float rotation;
    Rect fix(int32_t image_width, int32_t image_height) {
        Rect rect;
        rect.x = std::max(0, std::min(image_width, x));
        rect.y = std::max(0, std::min(image_height, y));
        rect.width = std::max(0, std::min(image_width - x, width));
        rect.height = std::max(0, std::min(image_height - y, height));
        rect.rotation = rotation;
        return rect;
    }
};

/*** Global variable ***/
static std::unique_ptr<PalmDetectionEngine> s_palm_detection_engine;
static std::unique_ptr<HandLandmarkEngine> s_hand_landmark_engine;
static int32_t s_frame_cnt;
static Rect s_palm_by_lm;
static bool s_is_palm_by_lm_valid = false;



/*** Function ***/
static void CalcAverageRect(Rect &rect_org, HandLandmarkEngine::HAND_LANDMARK &rect_new, float ratio_pos, float ratio_size);

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
    if (s_palm_detection_engine || s_hand_landmark_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_palm_detection_engine.reset(new PalmDetectionEngine());
    if (s_palm_detection_engine->Initialize(input_param.work_dir, input_param.num_threads) != PalmDetectionEngine::kRetOk) {
        return -1;
    }
    s_hand_landmark_engine.reset(new HandLandmarkEngine());
    if (s_hand_landmark_engine->Initialize(input_param.work_dir, input_param.num_threads) != HandLandmarkEngine::kRetOk) {
        return -1;
    }
    return 0;
}

int32_t ImageProcessor::Finalize(void)
{
    if (!s_palm_detection_engine || !s_hand_landmark_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    if (s_palm_detection_engine->Finalize() != PalmDetectionEngine::kRetOk) {
        return -1;
    }
    if (s_hand_landmark_engine->Finalize() != HandLandmarkEngine::kRetOk) {
        return -1;
    }
    s_palm_detection_engine.reset();
    s_hand_landmark_engine.reset();

    return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
    if (!s_palm_detection_engine || !s_hand_landmark_engine) {
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
    if (!s_palm_detection_engine || !s_hand_landmark_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    s_frame_cnt++;
    
    //bool enforce_palm_det = (s_frame_cnt % INTERVAL_TO_ENFORCE_PALM_DET) == 0;		// to increase accuracy
    bool enforce_palm_det = false;
    bool is_palm_valid = false;
    PalmDetectionEngine::Result palm_result;
    Rect palm = { 0 };
    if (s_is_palm_by_lm_valid == false || enforce_palm_det) {
        /*** Get Palms ***/
        s_palm_detection_engine->Process(mat, palm_result);
        for (const auto& detPalm : palm_result.palmList) {
            s_palm_by_lm.width = 0;	// reset 
            palm.x = (int32_t)(detPalm.x * 1);
            palm.y = (int32_t)(detPalm.y * 1);
            palm.width = (int32_t)(detPalm.width * 1);
            palm.height = (int32_t)(detPalm.height * 1);
            palm.rotation = detPalm.rotation;
            is_palm_valid = true;
            break;	// use only one palm
        }
    } else {
        /* Use the estimated palm position from the previous frame */
        is_palm_valid = true;
        palm.x = s_palm_by_lm.x;
        palm.y = s_palm_by_lm.y;
        palm.width = s_palm_by_lm.width;
        palm.height = s_palm_by_lm.height;
        palm.rotation = s_palm_by_lm.rotation;
    }
    palm = palm.fix(mat.cols, mat.rows);

    /*** Get landmark ***/
    HandLandmarkEngine::Result landmark_result;
    if (is_palm_valid) {
        cv::Scalar color_rect = (s_is_palm_by_lm_valid) ? CommonHelper::CreateCvColor(0, 255, 0) : CommonHelper::CreateCvColor(0, 0, 255);
        cv::rectangle(mat, cv::Rect(palm.x, palm.y, palm.width, palm.height), color_rect, 3);

        /* Get landmark */
        s_hand_landmark_engine->Process(mat, palm.x, palm.y, palm.width, palm.height, palm.rotation, landmark_result);

        if (landmark_result.hand_landmark.handflag >= 0.8) {
            CalcAverageRect(s_palm_by_lm, landmark_result.hand_landmark, 0.6f, 0.4f);
            cv::rectangle(mat, cv::Rect(s_palm_by_lm.x, s_palm_by_lm.y, s_palm_by_lm.width, s_palm_by_lm.height), CommonHelper::CreateCvColor(255, 0, 0), 3);

            /* Display hand landmark */
            for (int32_t i = 0; i < 21; i++) {
                cv::circle(mat, cv::Point((int32_t)landmark_result.hand_landmark.pos[i].x, (int32_t)landmark_result.hand_landmark.pos[i].y), 3, CommonHelper::CreateCvColor(255, 255, 0), 1);
                cv::putText(mat, std::to_string(i), cv::Point((int32_t)landmark_result.hand_landmark.pos[i].x - 10, (int32_t)landmark_result.hand_landmark.pos[i].y - 10), 1, 1, CommonHelper::CreateCvColor(255, 255, 0));
            }
            for (int32_t i = 0; i < 5; i++) {
                for (int32_t j = 0; j < 3; j++) {
                    int32_t indexStart = 4 * i + 1 + j;
                    int32_t indexEnd = indexStart + 1;
                    int32_t color = std::min((int32_t)std::max((landmark_result.hand_landmark.pos[indexStart].z + landmark_result.hand_landmark.pos[indexEnd].z) / 2.0f * -4, 0.f), 255);
                    cv::line(mat, cv::Point((int32_t)landmark_result.hand_landmark.pos[indexStart].x, (int32_t)landmark_result.hand_landmark.pos[indexStart].y), cv::Point((int32_t)landmark_result.hand_landmark.pos[indexEnd].x, (int32_t)landmark_result.hand_landmark.pos[indexEnd].y), CommonHelper::CreateCvColor(color, color, color), 3);
                }
            }
            s_is_palm_by_lm_valid = true;
        } else {
            s_is_palm_by_lm_valid = false;
        }
    }

    DrawFps(mat, palm_result.time_inference + landmark_result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /* Return the results */
    result.time_pre_process = palm_result.time_pre_process + landmark_result.time_pre_process;
    result.time_inference = palm_result.time_inference + landmark_result.time_inference;
    result.time_post_process = palm_result.time_post_process  + landmark_result.time_post_process;

    return 0;
}

static void CalcAverageRect(Rect &rect_org, HandLandmarkEngine::HAND_LANDMARK &rect_new, float ratio_pos, float ratio_size)
{
    if (rect_org.width == 0) {
        // for the first time
        ratio_pos = 1;
        ratio_size = 1;
    }
    rect_org.x = (int32_t)(rect_new.rect.x * ratio_pos + rect_org.x * (1 - ratio_pos));
    rect_org.y = (int32_t)(rect_new.rect.y * ratio_pos + rect_org.y * (1 - ratio_pos));
    rect_org.width = (int32_t)(rect_new.rect.width * ratio_size + rect_org.width * (1 - ratio_size));
    rect_org.height = (int32_t)(rect_new.rect.height * ratio_size + rect_org.height * (1 - ratio_size));
    rect_org.rotation = rect_new.rect.rotation * ratio_size + rect_org.rotation * (1 - ratio_size);
}

