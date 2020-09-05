/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>


/* for OpenCV */
#include <opencv2/opencv.hpp>

#include "InferenceHelper.h"
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

/* Model parameters */
#ifdef TFLITE_DELEGATE_EDGETPU
#define MODEL_NAME   "mobilenet_v3_segm_256"
#else
#define MODEL_NAME   "mobilenet_v3_segm_256"
#endif
static const float PIXEL_MEAN[3] = { 0.0f, 0.0f, 0.0f };
static const float PIXEL_STD[3] = { 1.0f,  1.0f, 1.0f };

typedef struct {
	double x;
	double y;
	double w;
	double h;
	int classId;
	std::string classIdName;
	double score;
} BBox;

/*** Global variable ***/
static InferenceHelper *s_inferenceHelper;
static TensorInfo *s_inputTensor;
static TensorInfo *s_outputTensor;

/*** Function ***/
static cv::Scalar createCvColor(int b, int g, int r) {
#ifdef CV_COLOR_IS_RGB
	return cv::Scalar(r, g, b);
#else
	return cv::Scalar(b, g, r);
#endif
}

int ImageProcessor_initialize(const INPUT_PARAM *inputParam)
{
#if defined(TFLITE_DELEGATE_EDGETPU)
	s_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_EDGETPU);
#elif defined(TFLITE_DELEGATE_GPU)
	s_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_GPU);
#elif defined(TFLITE_DELEGATE_XNNPACK)
	s_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_XNNPACK);
#else
	s_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE);
#endif

	std::string modelFilename = std::string(inputParam->workDir) + "/model/" + MODEL_NAME;

	s_inferenceHelper->initialize(modelFilename.c_str(), inputParam->numThreads);
	
	s_inputTensor = new TensorInfo();
	s_outputTensor = new TensorInfo();

	s_inferenceHelper->getTensorByName("input_1", s_inputTensor);
	s_inferenceHelper->getTensorByName("Identity", s_outputTensor);

	return 0;
}

int ImageProcessor_command(int cmd)
{
	switch (cmd) {
	default:
		PRINT("command(%d) is not supported\n", cmd);
		return -1;
	}
}


int ImageProcessor_process(cv::Mat *mat, OUTPUT_PARAM *outputParam)
{
	/*** PreProcess ***/
	cv::Mat inputImage;
	int modelInputWidth = s_inputTensor->dims[2];
	int modelInputHeight = s_inputTensor->dims[1];
	int modelInputChannel = s_inputTensor->dims[3];

	cv::resize(*mat, inputImage, cv::Size(modelInputWidth, modelInputHeight));
	cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
	if (s_inputTensor->type == TensorInfo::TENSOR_TYPE_UINT8) {
		inputImage.convertTo(inputImage, CV_8UC3);
	} else {
		inputImage.convertTo(inputImage, CV_32FC3, 1.0 / 255);
		cv::subtract(inputImage, cv::Scalar(cv::Vec<float, 3>(PIXEL_MEAN)), inputImage);
		cv::divide(inputImage, cv::Scalar(cv::Vec<float, 3>(PIXEL_STD)), inputImage);
	}

	/* Set data to input tensor */
#if 0
	s_inferenceHelper->setBufferToTensorByIndex(s_inputTensor->index, (char*)inputImage.data, (int)(inputImage.total() * inputImage.elemSize()));
#else
	if (s_inputTensor->type == TensorInfo::TENSOR_TYPE_UINT8) {
		memcpy(s_inputTensor->data, inputImage.reshape(0, 1).data, sizeof(uint8_t) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	} else {
		memcpy(s_inputTensor->data, inputImage.reshape(0, 1).data, sizeof(float) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	}
#endif
	
	/*** Inference ***/
	s_inferenceHelper->invoke();

	/*** PostProcess ***/
	/* Retrieve the result */
	int modelOutputWidth = s_outputTensor->dims[2];
	int modelOutputHeight = s_outputTensor->dims[1];
	//int modelOutputChannel = s_outputTensor->dims[3];
	float* output = s_outputTensor->getDataAsFloat();
	cv::Mat maskImage = cv::Mat(modelOutputHeight, modelOutputWidth, CV_32FC1, output);
	maskImage.convertTo(maskImage, CV_8UC1, 255, 0);
	cv::cvtColor(maskImage, maskImage, cv::COLOR_GRAY2BGR);
	
	//cv::imshow("mask", maskImage);
	//cv::waitKey(-1);

	/* Mask the original image */
	cv::resize(maskImage, maskImage, mat->size());
	cv::subtract(*mat, maskImage, *mat);		// Fill out masked area
	cv::multiply(maskImage, cv::Scalar(0, 255, 0), maskImage);	// optional: change mask color
	cv::add(*mat, maskImage, *mat);		// Fill out masked area
	

	/* Return the results */
	outputParam->dummy = 0;
	
	return 0;
}


int ImageProcessor_finalize(void)
{
	s_inferenceHelper->finalize();
	delete s_inputTensor;
	delete s_outputTensor;
	delete s_inferenceHelper;
	return 0;
}
