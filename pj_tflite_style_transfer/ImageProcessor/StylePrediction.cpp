/*** Include ***/
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "StylePrediction.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#define CV_COLOR_IS_RGB
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, "[StylePrediction] " __VA_ARGS__)
#else
#define PRINT(fmt, ...) printf("[StylePrediction] " fmt, __VA_ARGS__)
#endif

/* Model parameters */
#ifdef TFLITE_DELEGATE_EDGETPU
not supported
#else
#define MODEL_NAME "magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1"
#endif

//normalized to[0.f, 1.f] (hand_landmark_cpu.pbtxt)
static const float PIXEL_MEAN[3] = { 0.0f, 0.0f, 0.0f };
static const float PIXEL_STD[3] = { 1.0f,  1.0f, 1.0f };

/*** Function ***/
int StylePrediction::initialize(const char *workDir, const int numThreads)
{
#if defined(TFLITE_DELEGATE_EDGETPU)
	not supported
#elif defined(TFLITE_DELEGATE_GPU)
	m_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_GPU);
#elif defined(TFLITE_DELEGATE_XNNPACK)
	m_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_XNNPACK);
#else
	m_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE);
#endif

	std::string modelFilename = std::string(workDir) + "/" + MODEL_NAME;
	m_inferenceHelper->initialize(modelFilename.c_str(), numThreads);
	m_inputTensor = new TensorInfo();
	m_outputTensor = new TensorInfo();


	m_inferenceHelper->getTensorByName("style_image", m_inputTensor);
	m_inferenceHelper->getTensorByName("mobilenet_conv/Conv/BiasAdd", m_outputTensor);
	return 0;
}

int StylePrediction::finalize()
{
	m_inferenceHelper->finalize();
	delete m_inputTensor;
	delete m_outputTensor;
	delete m_inferenceHelper;
	return 0;
}


int StylePrediction::invoke(cv::Mat &originalMat, STYLE_PREDICTION_RESULT& result)
{
	/*** PreProcess ***/
	int modelInputWidth = m_inputTensor->dims[2];
	int modelInputHeight = m_inputTensor->dims[1];
	int modelInputChannel = m_inputTensor->dims[3];

	/* Resize image */
	/* todo: center crop */
	cv::Mat inputImage;
	cv::resize(originalMat, inputImage, cv::Size(modelInputWidth, modelInputHeight));
#ifndef CV_COLOR_IS_RGB
	cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
#endif
	if (m_inputTensor->type == TensorInfo::TENSOR_TYPE_UINT8) {
		inputImage.convertTo(inputImage, CV_8UC3);
	} else {
		inputImage.convertTo(inputImage, CV_32FC3, 1.0 / 255);
		cv::subtract(inputImage, cv::Scalar(cv::Vec<float, 3>(PIXEL_MEAN)), inputImage);
		cv::divide(inputImage, cv::Scalar(cv::Vec<float, 3>(PIXEL_STD)), inputImage);
	}

	/* Set data to input tensor */
#if 0
	m_inferenceHelper->setBufferToTensorByIndex(m_inputTensor->index, (char*)inputImage.data, (int)(inputImage.total() * inputImage.elemSize()));
#else
	if (m_inputTensor->type == TensorInfo::TENSOR_TYPE_UINT8) {
		memcpy(m_inputTensor->data, inputImage.data, sizeof(uint8_t) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	} else {
		memcpy(m_inputTensor->data, inputImage.data, sizeof(float) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	}
#endif

	/*** Inference ***/
	m_inferenceHelper->invoke();

	/*** PostProcess ***/
	result.styleBottleneck = m_outputTensor->getDataAsFloat();

	return 0;
}

