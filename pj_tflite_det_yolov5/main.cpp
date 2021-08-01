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

/* for My modules */
#include "image_processor.h"

/*** Macro ***/
#define IMAGE_NAME   RESOURCE_DIR"/kite.jpg"
#define VIDEO_NAME   RESOURCE_DIR"/video.mp4"
#define WORK_DIR     RESOURCE_DIR
#define LOOP_NUM_FOR_TIME_MEASUREMENT 10


int32_t main()
{
    /*** Initialize ***/
    /* Initialize image processor library */
    ImageProcessor::InputParam input_param = { WORK_DIR, 4 };
    ImageProcessor::Initialize(&input_param);

#ifdef SPEED_TEST_ONLY
    /* Read an input image */
    cv::Mat original_image = cv::imread(IMAGE_NAME);

    /* Call image processor library */
    ImageProcessor::OutputParam output_param;
    ImageProcessor::Process(&original_image, &output_param);

    cv::imshow("original_image", original_image);
    cv::waitKey(1);

    /*** (Optional) Measure inference time ***/
    double time_pre_process = 0;
    double time_inference = 0;
    double time_post_process = 0;
    const auto& t0 = std::chrono::steady_clock::now();
    for (int32_t i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
        ImageProcessor::Process(&original_image, &output_param);
        time_pre_process += output_param.time_pre_process;
        time_inference += output_param.time_inference;
        time_post_process += output_param.time_post_process;
    }
    const auto& t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeSpan = t1 - t0;
    printf("PreProcessing time  = %.3lf [msec]\n", time_pre_process / LOOP_NUM_FOR_TIME_MEASUREMENT);
    printf("Inference time  = %.3lf [msec]\n", time_inference / LOOP_NUM_FOR_TIME_MEASUREMENT);
    printf("PostProcessing time  = %.3lf [msec]\n", time_post_process / LOOP_NUM_FOR_TIME_MEASUREMENT);
    printf("Total Image processing time  = %.3lf [msec]\n", timeSpan.count() * 1000.0 / LOOP_NUM_FOR_TIME_MEASUREMENT);
    cv::waitKey(-1);

#else
    /* Initialize camera */
    static cv::VideoCapture cap;
    static cv::VideoWriter writer;
#if 1
    cap = cv::VideoCapture(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    // cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('B', 'G', 'R', '3'));
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
#else
    cap = cv::VideoCapture(VIDEO_NAME);
#endif
    writer = cv::VideoWriter("out.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), (std::max)(10.0, cap.get(cv::CAP_PROP_FPS)), cv::Size(static_cast<int32_t>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int32_t>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));
    while (1) {
        const auto& time_all0 = std::chrono::steady_clock::now();
        /*** Read image ***/
        const auto& time_cap0 = std::chrono::steady_clock::now();
        cv::Mat original_image;
        cap.read(original_image);
        if (original_image.empty()) break;
        const auto& time_cap1 = std::chrono::steady_clock::now();

        /* Call image processor library */
        const auto& time_process0 = std::chrono::steady_clock::now();
        ImageProcessor::OutputParam output_param;
        ImageProcessor::Process(&original_image, &output_param);
        const auto& time_process1 = std::chrono::steady_clock::now();

        if (writer.isOpened()) writer.write(original_image);
        cv::imshow("test", original_image);
        static bool is_pause = false;
        bool is_process_one_frame = false;
        do {
            int32_t key = cv::waitKey(1) & 0xff;
            switch (key) {
            case 'q':
                cap.release();
                break;
            case 'p':
                is_pause = !is_pause;
                break;
            case '>':
                if (is_pause) {
                    is_process_one_frame = true;
                } else {
                    int32_t current_frame = static_cast<int32_t>(cap.get(cv::CAP_PROP_POS_FRAMES));
                    cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<int32_t>(current_frame + 100));
                }
                break;
            case '<':
                int32_t current_frame = static_cast<int32_t>(cap.get(cv::CAP_PROP_POS_FRAMES));
                if (is_pause) {
                    is_process_one_frame = true;
                    cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<int32_t>(current_frame - 2));
                } else {
                    cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<int32_t>(current_frame - 100));
                }
                break;
            }
        } while (is_pause && !is_process_one_frame);

        const auto& time_all1 = std::chrono::steady_clock::now();
        printf("Total time = %.3lf [msec]\n", (time_all1 - time_all0).count() / 1000000.0);
        printf("Capture time = %.3lf [msec]\n", (time_cap1 - time_cap0).count() / 1000000.0);
        printf("Image processing time = %.3lf [msec]\n", (time_process1 - time_process0).count() / 1000000.0);
        printf("========\n");
    }
    if (writer.isOpened()) writer.release();
#endif

    /* Fianlize image processor library */
    ImageProcessor::Finalize();

    return 0;
}
