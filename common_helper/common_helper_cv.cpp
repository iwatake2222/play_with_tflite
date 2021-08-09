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

/* for OpenCV */
#include <opencv2/opencv.hpp>

#include "common_helper_cv.h"



void CommonHelper::CropResizeCvt(const cv::Mat& org, cv::Mat& dst, int32_t& crop_x, int32_t& crop_y, int32_t& crop_w, int32_t& crop_h, bool is_rgb, int32_t crop_type, bool resize_by_linear)
{
    const int32_t interpolation_flag = resize_by_linear ? cv::INTER_LINEAR : cv::INTER_NEAREST;

    cv::Mat src = org(cv::Rect(crop_x, crop_y, crop_w, crop_h));

    if (crop_type == kCropTypeStretch) {
        cv::resize(src, dst, dst.size(), 0, 0, interpolation_flag);
    } else if (crop_type == kCropTypeCut) {
        float aspect_ratio_src = static_cast<float>(src.cols) / src.rows;
        float aspect_ratio_dst = static_cast<float>(dst.cols) / dst.rows;
        cv::Rect target_rect(0, 0, src.cols, src.rows);
        if (aspect_ratio_src > aspect_ratio_dst) {
            target_rect.width = static_cast<int32_t>(src.rows * aspect_ratio_dst);
            target_rect.x = (src.cols - target_rect.width) / 2;
        } else {
            target_rect.height = static_cast<int32_t>(src.cols / aspect_ratio_dst);
            target_rect.y = (src.rows - target_rect.height) / 2;
        }
        cv::Mat target = src(target_rect);
        cv::resize(target, dst, dst.size(), 0, 0, interpolation_flag);
        crop_x += target_rect.x;
        crop_y += target_rect.y;
        crop_w = target_rect.width;
        crop_h = target_rect.height;
    } else {
        float aspect_ratio_src = static_cast<float>(src.cols) / src.rows;
        float aspect_ratio_dst = static_cast<float>(dst.cols) / dst.rows;
        cv::Rect target_rect(0, 0, dst.cols, dst.rows);
        if (aspect_ratio_src > aspect_ratio_dst) {
            target_rect.height = static_cast<int32_t>(target_rect.width / aspect_ratio_src);
            target_rect.y = (dst.rows - target_rect.height) / 2;
        } else {
            target_rect.width = static_cast<int32_t>(target_rect.height * aspect_ratio_src);
            target_rect.x = (dst.cols - target_rect.width) / 2;
        }
        cv::Mat target = dst(target_rect);
        cv::resize(src, target, target.size(), 0, 0, interpolation_flag);
        crop_x -= target_rect.x * crop_w / target_rect.width;
        crop_y -= target_rect.y * crop_h / target_rect.height;
        crop_w = dst.cols * crop_w / target_rect.width;
        crop_h = dst.rows * crop_h / target_rect.height;
    }

#ifdef CV_COLOR_IS_RGB
    if (!is_rg) {
        cv::cvtColor(dst, dst, cv::COLOR_RGB2BGR);
    }
#else
    if (is_rgb) {
        cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    }
#endif

}
