/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>


/* for OpenCV */
#include <opencv2/opencv.hpp>

#include "StylePrediction.h"
#include "StyleTransfer.h"
#include "ImageProcessor.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#define CV_COLOR_IS_RGB
#include <android/log.h>
#define TAG "MyApp_NDK"
#define _PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define _PRINT(...) printf(__VA_ARGS__)
#endif
#define PRINT(...) _PRINT("[ImageProcessor] " __VA_ARGS__)

#define CHECK(x)                              \
  if (!(x)) {                                                \
	PRINT("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }


/*** Global variable ***/
static StylePrediction s_stylePrediction;
static StyleTransfer s_styleTransfer;
float s_styleBottleneck[StylePrediction::SIZE_STYLE_BOTTLENECK];
std::string s_workDir;

/*** Function ***/
static cv::Scalar createCvColor(int b, int g, int r) {
#ifdef CV_COLOR_IS_RGB
	return cv::Scalar(r, g, b);
#else
	return cv::Scalar(b, g, r);
#endif
}

static int calculateStyleBottleneck(std::string styleFilename)
{
	std::string path = s_workDir + "/../style/" + styleFilename;
	cv::Mat styleImage = cv::imread(path);
	if (styleImage.empty()) {
		PRINT("[error] cannot read %s\n", path.c_str());
		return -1;
	}
	
	StylePrediction::STYLE_PREDICTION_RESULT stylePredictionResult;
	s_stylePrediction.invoke(styleImage, stylePredictionResult);
	for (int i = 0; i < StylePrediction::SIZE_STYLE_BOTTLENECK; i++) s_styleBottleneck[i] = stylePredictionResult.styleBottleneck[i];
	return 0;
}

int ImageProcessor_initialize(const INPUT_PARAM *inputParam)
{
	s_workDir = inputParam->workDir;
	s_stylePrediction.initialize(inputParam->workDir, inputParam->numThreads);
	s_styleTransfer.initialize(inputParam->workDir, inputParam->numThreads);

	ImageProcessor_command(0);

	return 0;
}


int ImageProcessor_finalize(void)
{
	s_stylePrediction.finalize();
	s_styleTransfer.finalize();
	return 0;
}


int ImageProcessor_command(int cmd)
{
	static int s_currentImageFileIndex = 0;
	switch (cmd) {
	case 0:
		s_currentImageFileIndex++;
		if (s_currentImageFileIndex > 30) s_currentImageFileIndex = 30;
		break;
	case 1:
		s_currentImageFileIndex--;
		if (s_currentImageFileIndex < 0) s_currentImageFileIndex = 0;
		break;
	case 2:
		s_currentImageFileIndex = 0;
		break;
	default:
		PRINT("command(%d) is not supported\n", cmd);
		return -1;
	}
	std::string filename = "style" + std::to_string(s_currentImageFileIndex) + ".jpg";
	calculateStyleBottleneck(filename);

	return 0;
}


int ImageProcessor_process(cv::Mat *mat, OUTPUT_PARAM *outputParam)
{
    const int INTERVAL_TO_CALCULATE_CONTENT_BOTTLENECK = 30; // to increase FPS (no need to do this every frame)
	static float s_mergedStyleBottleneck[StylePrediction::SIZE_STYLE_BOTTLENECK];
	static int s_cnt = 0;
	if (s_cnt++ % INTERVAL_TO_CALCULATE_CONTENT_BOTTLENECK == 0) {
		const float ratio = 0.5f;
		StylePrediction::STYLE_PREDICTION_RESULT stylePredictionResult;
		s_stylePrediction.invoke(*mat, stylePredictionResult);
		for (int i = 0; i < StylePrediction::SIZE_STYLE_BOTTLENECK; i++) s_mergedStyleBottleneck[i] = ratio * stylePredictionResult.styleBottleneck[i] + (1 - ratio) * s_styleBottleneck[i];
	}

	StyleTransfer::STYLE_TRANSFER_RESULT styleTransferResult;
	s_styleTransfer.invoke(*mat, s_mergedStyleBottleneck, StylePrediction::SIZE_STYLE_BOTTLENECK, styleTransferResult);

	cv::Mat outMatFp(cv::Size(384, 384), CV_32FC3, styleTransferResult.result);
	cv::Mat outMat;
	outMatFp.convertTo(outMat, CV_8UC3, 255);
	*mat = outMat;

	//cv::Mat outMat(cv::Size(384, 384), CV_8UC3);
	//for (int i = 0; i < 384 * 384 * 3; i++) {
	//	outMat.data[i] = styleTransferResult.result[i] * 255;
	//}
	//cv::imshow("input", *mat);
	//cv::imshow("result", outMat);

	return 0;
}

