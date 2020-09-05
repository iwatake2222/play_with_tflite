/*** Include ***/
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>

#include "Classify.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#define CV_COLOR_IS_RGB
#include <android/log.h>
#define TAG "MyApp_NDK"
#define _PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define _PRINT(...) printf(__VA_ARGS__)
#endif
#define PRINT(...) _PRINT("[Classify] " __VA_ARGS__)

/* Model parameters */
#if defined(TFLITE_DELEGATE_EDGETPU)
#define MODEL_NAME   "efficientnet_lite3_int8_2_edgetpu"
#elif defined(TFLITE_DELEGATE_GPU) || defined(TFLITE_DELEGATE_XNNPACK)
#define MODEL_NAME   "efficientnet_lite3_int8_2"
#else
#define MODEL_NAME   "efficientnet_lite3_int8_2"
#endif
#define LABEL_NAME   "imagenet_labels.txt"

//normalized to[0.f, 1.f]
static const float PIXEL_MEAN[3] = { 0.0f, 0.0f, 0.0f };
static const float PIXEL_STD[3] = { 1.0f,  1.0f, 1.0f };

/*** Function ***/
int Classify::initialize(const char *workDir, const int numThreads)
{
#if defined(TFLITE_DELEGATE_EDGETPU)
	not supported
#elif defined(TFLITE_DELEGATE_GPU)
	//m_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_GPU);
	m_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE);
#elif defined(TFLITE_DELEGATE_XNNPACK)
	m_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_XNNPACK);
#else
	m_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE);
#endif

	std::string modelFilename = std::string(workDir) + "/model/" + MODEL_NAME;
	m_inferenceHelper->initialize(modelFilename.c_str(), numThreads);
	m_inputTensor = new TensorInfo();
	m_outputTensor = new TensorInfo();

	m_inferenceHelper->getTensorByName("images", m_inputTensor);
	m_inferenceHelper->getTensorByName("Softmax", m_outputTensor);

	std::string labelFilename = std::string(workDir) + "/model/" + LABEL_NAME;
	readLabel(labelFilename, m_labels);

	return 0;
}

int Classify::finalize()
{
	m_inferenceHelper->finalize();
	delete m_inputTensor;
	delete m_outputTensor;
	return 0;
}


int Classify::invoke(cv::Mat &originalMat, RESULT &result)
{
	/*** PreProcess ***/
	int modelInputWidth = m_inputTensor->dims[2];
	int modelInputHeight = m_inputTensor->dims[1];
	int modelInputChannel = m_inputTensor->dims[3];

	/* Resize image */
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
		memcpy(m_inputTensor->data, inputImage.reshape(0, 1).data, sizeof(uint8_t) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	} else {
		memcpy(m_inputTensor->data, inputImage.reshape(0, 1).data, sizeof(float) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	}
#endif

	/*** Inference ***/
	m_inferenceHelper->invoke();

	/*** PostProcess ***/
	std::vector<float> outputScoreList;
	outputScoreList.resize(m_outputTensor->dims[1]);
	const float* valFloat = m_outputTensor->getDataAsFloat();
	for (int i = 0; i < (int)outputScoreList.size(); i++) {
		outputScoreList[i] = valFloat[i];
	}

	/* Find the max score */
	int maxIndex = (int)(std::max_element(outputScoreList.begin(), outputScoreList.end()) - outputScoreList.begin());
	auto maxScore = *std::max_element(outputScoreList.begin(), outputScoreList.end());
	maxIndex++;	// it looks  background is not included
	PRINT("Result = %s (%d) (%.3f)\n", m_labels[maxIndex].c_str(), maxIndex, maxScore);

	result.labelIndex = maxIndex;
	result.labelName = m_labels[maxIndex];
	result.score = maxScore;

	return 0;
}


void Classify::readLabel(const std::string filename, std::vector<std::string> & labels)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		PRINT("failed to read %s\n", filename.c_str());
		return;
	}
	std::string str;
	while (getline(ifs, str)) {
		labels.push_back(str);
	}
}
