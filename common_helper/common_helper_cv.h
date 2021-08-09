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
#ifndef COMMON_HELPER_CV_
#define COMMON_HELPER_CV_

/* for general */
#include <cstdint>
#include <string>
#include <vector>
#include <array>

/* for OpenCV */
#include <opencv2/opencv.hpp>


namespace CommonHelper
{
enum {
    kCropTypeStretch = 0,
    kCropTypeCut,
    kCropTypeExpand,
};

void CropResizeCvt(const cv::Mat& org, cv::Mat& dst, int32_t& crop_x, int32_t& crop_y, int32_t& crop_w, int32_t& crop_h, bool is_rgb = true, int32_t crop_type = kCropTypeStretch, bool resize_by_linear = true);

}

#endif
