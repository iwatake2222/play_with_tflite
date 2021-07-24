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
#define IMAGE_NAME   RESOURCE_DIR"/hand00.jpg"
#define WORK_DIR     RESOURCE_DIR
#define LOOP_NUM_FOR_TIME_MEASUREMENT 10


int32_t main()
{
	/*** Initialize ***/
	/* Initialize image processor library */
	ImageProcessor::InputParam input_param;
	snprintf(input_param.work_dir, sizeof(input_param.work_dir), WORK_DIR);
	input_param.num_threads = 4;
	ImageProcessor::Initialize(&input_param);

#ifdef SPEED_TEST_ONLY
	/* Read an input image */
	cv::Mat originalImage = cv::imread(IMAGE_NAME);

	/* Call image processor library */
	ImageProcessor::OutputParam output_param;
	ImageProcessor::Process(&originalImage, &output_param);

	cv::imshow("originalImage", originalImage);
	cv::waitKey(1);

	/*** (Optional) Measure inference time ***/
	double time_pre_process = 0;
	double time_inference = 0;
	double time_post_process = 0;
	const auto& t0 = std::chrono::steady_clock::now();
	for (int32_t i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
		ImageProcessor::Process(&originalImage, &output_param);
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
	int32_t originalImageWidth = 1280;
	int32_t originalImageHeight = 720;

	static cv::VideoCapture cap;
	cap = cv::VideoCapture(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, originalImageWidth);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, originalImageHeight);
	// cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('B', 'G', 'R', '3'));
	cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
	while (1) {
		const auto& timeAll0 = std::chrono::steady_clock::now();
		/*** Read image ***/
		const auto& timeCap0 = std::chrono::steady_clock::now();
		cv::Mat originalImage;
		cap.read(originalImage);
		const auto& timeCap1 = std::chrono::steady_clock::now();

		/* Call image processor library */
		const auto& timeProcess0 = std::chrono::steady_clock::now();
		ImageProcessor::OutputParam output_param;
		ImageProcessor::Process(&originalImage, &output_param);
		const auto& timeProcess1 = std::chrono::steady_clock::now();

		cv::imshow("test", originalImage);
		if (cv::waitKey(1) == 'q') break;

		const auto& timeAll1 = std::chrono::steady_clock::now();
		printf("Total time = %.3lf [msec]\n", (timeAll1 - timeAll0).count() / 1000000.0);
		printf("Capture time = %.3lf [msec]\n", (timeCap1 - timeCap0).count() / 1000000.0);
		printf("Image processing time = %.3lf [msec]\n", (timeProcess1 - timeProcess0).count() / 1000000.0);
		printf("========\n");
	}

#endif

	/* Fianlize image processor library */
	ImageProcessor::Finalize();

	return 0;
}
