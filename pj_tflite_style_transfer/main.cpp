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
#include "ImageProcessor.h"

/*** Macro ***/
#define IMAGE_NAME   RESOURCE_DIR"/parrot.jpg"
#define WORK_DIR     RESOURCE_DIR
#define LOOP_NUM_FOR_TIME_MEASUREMENT 10


int32_t main()
{
	/*** Initialize ***/
	/* Initialize image processor library */
	INPUT_PARAM inputParam;
	snprintf(inputParam.workDir, sizeof(inputParam.workDir), WORK_DIR);
	inputParam.numThreads = 4;
	ImageProcessor_initialize(&inputParam);

#ifdef SPEED_TEST_ONLY
	/* Read an input image */
	cv::Mat originalImage = cv::imread(IMAGE_NAME);

	/* Call image processor library */
	OUTPUT_PARAM outputParam;
	ImageProcessor_process(&originalImage, &outputParam);

	cv::imshow("originalImage", originalImage);
	cv::waitKey(1);

	/*** (Optional) Measure inference time ***/
	double timePreProcess = 0;
	double timeInference = 0;
	double timePostProcess = 0;
	const auto& t0 = std::chrono::steady_clock::now();
	for (int32_t i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
		ImageProcessor_process(&originalImage, &outputParam);
		timePreProcess += outputParam.timePreProcess;
		timeInference += outputParam.timeInference;
		timePostProcess += outputParam.timePostProcess;
	}
	const auto& t1 = std::chrono::steady_clock::now();
	std::chrono::duration<double> timeSpan = t1 - t0;
	printf("PreProcessing time  = %.3lf [msec]\n", timePreProcess / LOOP_NUM_FOR_TIME_MEASUREMENT);
	printf("Inference time  = %.3lf [msec]\n", timeInference / LOOP_NUM_FOR_TIME_MEASUREMENT);
	printf("PostProcessing time  = %.3lf [msec]\n", timePostProcess / LOOP_NUM_FOR_TIME_MEASUREMENT);
	printf("Total Image processing time  = %.3lf [msec]\n", timeSpan.count() * 1000.0 / LOOP_NUM_FOR_TIME_MEASUREMENT);
	cv::waitKey(-1);

#else
	/* Initialize camera */
	int32_t originalImageWidth = 640;
	int32_t originalImageHeight = 480;

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
		OUTPUT_PARAM outputParam;
		ImageProcessor_process(&originalImage, &outputParam);
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
	ImageProcessor_finalize();

	return 0;
}
