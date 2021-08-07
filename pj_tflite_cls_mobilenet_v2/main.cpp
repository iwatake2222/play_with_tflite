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
#include <string>
#include <algorithm>
#include <chrono>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "image_processor.h"

/*** Macro ***/
#define WORK_DIR                      RESOURCE_DIR
#define DEFAULT_INPUT_IMAGE           RESOURCE_DIR"/parrot.jpg"
#define LOOP_NUM_FOR_TIME_MEASUREMENT 10

/*** Function ***/
static bool FindSourceImage(const std::string& input_name, cv::VideoCapture& cap)
{
    if (input_name.find(".mp4") != std::string::npos || input_name.find(".avi") != std::string::npos || input_name.find(".webm") != std::string::npos) {
        cap = cv::VideoCapture(input_name);
        if (!cap.isOpened()) {
            printf("Invalid input source: %s\n", input_name.c_str());
            return false;
        }
    } else if (input_name.find(".jpg") != std::string::npos || input_name.find(".png") != std::string::npos || input_name.find(".bmp") != std::string::npos) {
        if (cv::imread(input_name).empty()) {
            printf("Invalid input source: %s\n", input_name.c_str());
            return false;
        }
    } else {
        int32_t cam_id = -1;
        try {
            cam_id = std::stoi(input_name);
        }
        catch (...) {}
        cap = (cam_id >= 0) ? cv::VideoCapture(cam_id): cv::VideoCapture(input_name);     
        if (!cap.isOpened()) {
            printf("Unable to open camera: %s\n", input_name.c_str());
            return false;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }
    return true;
}

static bool InputKeyCommand(cv::VideoCapture& cap)
{
    bool ret_to_quit = false;
    static bool is_pause = false;
    bool is_process_one_frame = false;
    do {
        int32_t key = cv::waitKey(1) & 0xff;
        switch (key) {
        case 'q':
            cap.release();
            ret_to_quit = true;
            break;
        case 'p':
            is_pause = !is_pause;
            break;
        case '>':
            if (is_pause) {
                is_process_one_frame = true;
            } else {
                int32_t current_frame = static_cast<int32_t>(cap.get(cv::CAP_PROP_POS_FRAMES));
                cap.set(cv::CAP_PROP_POS_FRAMES, current_frame + 100);
            }
            break;
        case '<':
            int32_t current_frame = static_cast<int32_t>(cap.get(cv::CAP_PROP_POS_FRAMES));
            if (is_pause) {
                is_process_one_frame = true;
                cap.set(cv::CAP_PROP_POS_FRAMES, current_frame - 2);
            } else {
                cap.set(cv::CAP_PROP_POS_FRAMES, current_frame - 100);
            }
            break;
        }
    } while (is_pause && !is_process_one_frame);

    return ret_to_quit;
}

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

    /* Find source image */
    std::string input_name = (argc > 1) ? argv[1] : DEFAULT_INPUT_IMAGE;
    cv::VideoCapture cap;   /* if cap is not opened, src is still image */
    if (!FindSourceImage(input_name, cap)) {
        return -1;
    }

    /* Create video writer to save output video */
    cv::VideoWriter writer;
    // writer = cv::VideoWriter("out.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), (std::max)(10.0, cap.get(cv::CAP_PROP_FPS)), cv::Size(static_cast<int32_t>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int32_t>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));

    /* Initialize image processor library */
    ImageProcessor::InputParam input_param = { WORK_DIR, 4 };
    ImageProcessor::Initialize(input_param);

    /*** Process for each frame ***/
    int32_t frame_cnt = 0;
    for (frame_cnt = 0; cap.isOpened() || frame_cnt < LOOP_NUM_FOR_TIME_MEASUREMENT; frame_cnt++) {
        const auto& time_all0 = std::chrono::steady_clock::now();
        /* Read image */
        const auto& time_cap0 = std::chrono::steady_clock::now();
        cv::Mat image;
        if (cap.isOpened()) {
            cap.read(image);
        } else {
            image = cv::imread(input_name);
        }
        if (image.empty()) break;
        const auto& time_cap1 = std::chrono::steady_clock::now();

        /* Call image processor library */
        const auto& time_image_process0 = std::chrono::steady_clock::now();
        ImageProcessor::Result result;
        ImageProcessor::Process(image, result);
        const auto& time_image_process1 = std::chrono::steady_clock::now();

        /* Display result */
        if (writer.isOpened()) writer.write(image);
        cv::imshow("test", image);

        /* Input key command */
        if (cap.isOpened()) {
            /* this code needs to be before calculating processing time because cv::waitKey includes image output */
            /* however, when 'q' key is pressed (cap.released()), processing time significantly incraeases. So escape from the loop before calculating time */
            if (InputKeyCommand(cap)) break;
        };

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
    if (writer.isOpened()) writer.release();
    cv::waitKey(-1);

    return 0;
}
