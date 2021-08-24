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
#include "camera_calibration_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<CameraCalibrationEngine> s_engine;

static bool s_update_calib = true;

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

/* reference: https://github.com/alexvbogdan/DeepCalib/blob/master/undistortion/undistSphIm.m */
/* Unified projection model */
static void CreateUndistortMap(cv::Size undist_image_size, float f_undist, float xi, float u0_undist, float v0_undist, float f_dist, float u0_dist, float v0_dist
    , cv::Mat& mapx, cv::Mat& mapy)
{
    cv::Mat grid_x(undist_image_size, CV_32F);
    cv::Mat grid_y(undist_image_size, CV_32F);
#pragma omp parallel for
    for (int32_t y = 0; y < undist_image_size.height; y++) {
        for (int32_t x = 0; x < undist_image_size.width; x++) {
            grid_x.at<float>(y, x) = x + 0.0f;
            grid_y.at<float>(y, x) = y + 0.0f;
        }
    }

    cv::Mat X_Cam(undist_image_size, CV_32F);
    cv::Mat Y_Cam(undist_image_size, CV_32F);
    cv::Mat Z_Cam(undist_image_size, CV_32F);
#pragma omp parallel for
    for (int32_t y = 0; y < undist_image_size.height; y++) {
        for (int32_t x = 0; x < undist_image_size.width; x++) {
            X_Cam.at<float>(y, x) = (grid_x.at<float>(y, x) - u0_undist) / f_undist;
            Y_Cam.at<float>(y, x) = (grid_y.at<float>(y, x) - v0_undist) / f_undist;
            Z_Cam.at<float>(y, x) = 1.0f;
        }
    }

    cv::Mat Alpha_Cam(undist_image_size, CV_32F);
#pragma omp parallel for
    for (int32_t y = 0; y < undist_image_size.height; y++) {
        for (int32_t x = 0; x < undist_image_size.width; x++) {
            Alpha_Cam.at<float>(y, x) = 1 / sqrtf(
                X_Cam.at<float>(y, x) * X_Cam.at<float>(y, x)
                + Y_Cam.at<float>(y, x) * Y_Cam.at<float>(y, x)
                + Z_Cam.at<float>(y, x) * Z_Cam.at<float>(y, x)
            );
        }
    }

    cv::Mat X_Sph(undist_image_size, CV_32F);
    cv::Mat Y_Sph(undist_image_size, CV_32F);
    cv::Mat Z_Sph(undist_image_size, CV_32F);
#pragma omp parallel for
    for (int32_t y = 0; y < undist_image_size.height; y++) {
        for (int32_t x = 0; x < undist_image_size.width; x++) {
            X_Sph.at<float>(y, x) = X_Cam.at<float>(y, x) * Alpha_Cam.at<float>(y, x);
            Y_Sph.at<float>(y, x) = Y_Cam.at<float>(y, x) * Alpha_Cam.at<float>(y, x);
            Z_Sph.at<float>(y, x) = Z_Cam.at<float>(y, x) * Alpha_Cam.at<float>(y, x);
        }
    }

    cv::Mat den(undist_image_size, CV_32F);
#pragma omp parallel for
    for (int32_t y = 0; y < undist_image_size.height; y++) {
        for (int32_t x = 0; x < undist_image_size.width; x++) {
            den.at<float>(y, x) = xi * sqrtf(
                X_Sph.at<float>(y, x) * X_Sph.at<float>(y, x)
                + Y_Sph.at<float>(y, x) * Y_Sph.at<float>(y, x)
                + Z_Sph.at<float>(y, x) * Z_Sph.at<float>(y, x)
            ) + Z_Sph.at<float>(y, x);
        }
    }

    mapx = cv::Mat(undist_image_size, CV_32F);
    mapy = cv::Mat(undist_image_size, CV_32F);
#pragma omp parallel for
    for (int32_t y = 0; y < undist_image_size.height; y++) {
        for (int32_t x = 0; x < undist_image_size.width; x++) {
            mapx.at<float>(y, x) = (X_Sph.at<float>(y, x) * f_dist) / den.at<float>(y, x) + u0_dist;
            mapy.at<float>(y, x) = (Y_Sph.at<float>(y, x) * f_dist) / den.at<float>(y, x) + v0_dist;
        }
    }
}


int32_t ImageProcessor::Initialize(const ImageProcessor::InputParam& input_param)
{
    if (s_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_engine.reset(new CameraCalibrationEngine());
    if (s_engine->Initialize(input_param.work_dir, input_param.num_threads) != CameraCalibrationEngine::kRetOk) {
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

    if (s_engine->Finalize() != CameraCalibrationEngine::kRetOk) {
        return -1;
    }

    s_engine.reset();

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
        s_update_calib = true;
        PRINT_E("Do estimation\n");
        return 0;
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

    int32_t new_image_size_scale = 3;   /* this value should be adjusted according to distortion level */
    static cv::Mat mapx, mapy;      /* save these parameters as static to avoid re - calculate maps */
    CameraCalibrationEngine::Result calib_result;

    if (mapx.empty() || s_update_calib) {
        /*** Predict camera parameters ***/
        if (s_engine->Process(mat, calib_result) != CameraCalibrationEngine::kRetOk) {
            return -1;
        }

        /*** Calibration ***/
        /* Set parameters */
        float xi = calib_result.xi;
        float focal_length = calib_result.focal_length;

        
        cv::Size undist_image_size = mat.size() * new_image_size_scale;

        float f_undist = focal_length;
        float u0_undist = undist_image_size.width / 2.0f;
        float v0_undist = undist_image_size.height / 2.0f;
        float f_dist = focal_length;
        float u0_dist = mat.size().width / 2.0f;
        float v0_dist = mat.size().height / 2.0f;

        /* Calculate undistort map */
        CreateUndistortMap(undist_image_size, f_undist, xi, u0_undist, v0_undist, f_dist, u0_dist, v0_dist, mapx, mapy);

        CommonHelper::DrawText(mat, "Calibration Done", cv::Point(100, 100), 0.5, 2, CommonHelper::CreateCvColor(255, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), false);
        s_update_calib = false;
    }
    
    /* Undistort image */
    cv::Mat image_undistorted;
    cv::remap(mat, image_undistorted, mapx, mapy, cv::INTER_LINEAR);
    cv::resize(image_undistorted, image_undistorted, cv::Size(), 1.0 / new_image_size_scale, 1.0 / new_image_size_scale);

    DrawFps(image_undistorted, calib_result.time_inference, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);

    /* Return the results */
    mat = image_undistorted;
    result.time_pre_process = calib_result.time_pre_process;
    result.time_inference = calib_result.time_inference;
    result.time_post_process = calib_result.time_post_process;

    return 0;
}

