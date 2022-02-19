/* Copyright 2022 iwatake2222

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
#include <string>
#include <algorithm>
#include <chrono>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper_cv.h"
#include "image_processor.h"

/*** Macro ***/
static constexpr char kOutputVideoFilename[] = "";
#define WORK_DIR                      RESOURCE_DIR
#define DEFAULT_INPUT_IMAGE_0         RESOURCE_DIR"/frame_interpolation_0.jpg"
#define DEFAULT_INPUT_IMAGE_1         RESOURCE_DIR"/frame_interpolation_1.jpg"
#define LOOP_NUM_FOR_TIME_MEASUREMENT -1

/*** Function ***/
int32_t main(int argc, char* argv[])
{
    /*** Initialize ***/
    /* variables for processing time measurement */
    double total_time_all = 0;
    double total_time_cap = 0;
    double total_time_image_process = 0;
    double total_time_pre_process = 0;
    double total_time_inference = 0;
    double total_time_post_process = 0;

    /* Create trackbar for interpolation time*/
    cv::namedWindow("image_result");
    int32_t trackbar_time = 50; /* 50 % */
    cv::createTrackbar("time", "image_result", &trackbar_time, 100);

    /* Find source image */
    std::string input_name_0 = (argc > 2) ? argv[1] : DEFAULT_INPUT_IMAGE_0;
    std::string input_name_1 = (argc > 2) ? argv[2] : DEFAULT_INPUT_IMAGE_1;
    cv::Mat image_0 = cv::imread(input_name_0);
    cv::Mat image_1 = cv::imread(input_name_1);
    if (image_0.rows > 480) {
        cv::resize(image_0, image_0, cv::Size(), 480.0 / image_0.rows, 480.0 / image_0.rows);
        cv::resize(image_1, image_1, cv::Size(), 480.0 / image_1.rows, 480.0 / image_1.rows);
    }

    /* Create video writer to save output video */
    cv::VideoWriter writer;

    /* Initialize image processor library */
    ImageProcessor::InputParam input_param = { WORK_DIR, 4 };
    if (ImageProcessor::Initialize(input_param) != 0) {
        printf("Initialization Error\n");
        return -1;
    }

    /*** Process for each frame ***/
    int32_t frame_cnt = 0;
    for (frame_cnt = 0; LOOP_NUM_FOR_TIME_MEASUREMENT < 0 || frame_cnt < LOOP_NUM_FOR_TIME_MEASUREMENT; frame_cnt++) {
        const auto& time_all0 = std::chrono::steady_clock::now();
        /* Read image */
        const auto& time_cap0 = std::chrono::steady_clock::now();
        const auto& time_cap1 = std::chrono::steady_clock::now();

        /* Call image processor library */
        const auto& time_image_process0 = std::chrono::steady_clock::now();
        cv::Mat image_result;
        ImageProcessor::Result result;
        ImageProcessor::Process(image_0, image_1, trackbar_time / 100.0f, result, image_result);
        const auto& time_image_process1 = std::chrono::steady_clock::now();

        /* Display result */
        cv::imshow("image_0", image_0);
        cv::imshow("image_1", image_1);
        cv::imshow("image_result", image_result);

        if (frame_cnt == 0 && kOutputVideoFilename[0] != '\0') {
            writer = cv::VideoWriter(kOutputVideoFilename, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 5.0f, image_result.size());
        }
        if (writer.isOpened()) writer.write(image_result);

        /* Input key command */
        int32_t key = cv::waitKey(1) & 0xff;
        if (key == 'q' || key == 27) break;

        /* Print processing time */
        const auto& time_all1 = std::chrono::steady_clock::now();
        double time_all = (time_all1 - time_all0).count() / 1000000.0;
        double time_cap = (time_cap1 - time_cap0).count() / 1000000.0;
        double time_image_process = (time_image_process1 - time_image_process0).count() / 1000000.0;
        printf("Total:               %9.3lf [msec]\n", time_all);
        printf("  Capture:           %9.3lf [msec]\n", time_cap);
        printf("  Image processing:  %9.3lf [msec]\n", time_image_process);
        printf("    Pre processing:  %9.3lf [msec]\n", result.time_pre_process);
        printf("    Inference:       %9.3lf [msec]\n", result.time_inference);
        printf("    Post processing: %9.3lf [msec]\n", result.time_post_process);
        printf("=== Finished %d frame ===\n\n", frame_cnt);

        if (frame_cnt > 0) {    /* do not count the first process because it may include initialize process */
            total_time_all += time_all;
            total_time_cap += time_cap;
            total_time_image_process += time_image_process;
            total_time_pre_process += result.time_pre_process;
            total_time_inference += result.time_inference;
            total_time_post_process += result.time_post_process;
        }
    }
    
    /*** Finalize ***/
    /* Print average processing time */
    if (frame_cnt > 1) {
        frame_cnt--;    /* because the first process was not counted */
        printf("=== Average processing time ===\n");
        printf("Total:               %9.3lf [msec]\n", total_time_all / frame_cnt);
        printf("  Capture:           %9.3lf [msec]\n", total_time_cap / frame_cnt);
        printf("  Image processing:  %9.3lf [msec]\n", total_time_image_process / frame_cnt);
        printf("    Pre processing:  %9.3lf [msec]\n", total_time_pre_process / frame_cnt);
        printf("    Inference:       %9.3lf [msec]\n", total_time_inference / frame_cnt);
        printf("    Post processing: %9.3lf [msec]\n", total_time_post_process / frame_cnt);
    }

    /* Fianlize image processor library */
    ImageProcessor::Finalize();
    cv::waitKey(-1);

    return 0;
}
