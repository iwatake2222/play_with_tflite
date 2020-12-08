/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "CommonHelper.h"
#include "InferenceHelper.h"
#include "HandLandmarkEngine.h"

/*** Macro ***/
#define TAG "HandLandmarkEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "hand_landmark.tflite"

/*** Function ***/
int32_t HandLandmarkEngine::initialize(const std::string& workDir, const int32_t numThreads)
{
	/* Set model information */
	std::string modelFilename = workDir + "/model/" + MODEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = "input_1";
	inputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
	inputTensorInfo.tensorDims.batch = 1;
	inputTensorInfo.tensorDims.width = 256;
	inputTensorInfo.tensorDims.height = 256;
	inputTensorInfo.tensorDims.channel = 3;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.normalize.mean[0] = 0.0f;   	/* normalized to[0.f, 1.f] (hand_landmark_cpu.pbtxt) */
	inputTensorInfo.normalize.mean[1] = 0.0f;
	inputTensorInfo.normalize.mean[2] = 0.0f;
	inputTensorInfo.normalize.norm[0] = 1.0f;
	inputTensorInfo.normalize.norm[1] = 1.0f;
	inputTensorInfo.normalize.norm[2] = 1.0f;
	m_inputTensorList.push_back(inputTensorInfo);

	/* Set output tensor info */
	m_outputTensorList.clear();
	OutputTensorInfo outputTensorInfo;
	outputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
	outputTensorInfo.name = "ld_21_3d";
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = "output_handflag";
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = "output_handedness";
	m_outputTensorList.push_back(outputTensorInfo);

	/* Create and Initialize Inference Helper */
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::OPEN_CV));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSOR_RT));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::NCNN));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::MNN));
	m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_EDGETPU));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_GPU));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_XNNPACK));

	if (!m_inferenceHelper) {
		return RET_ERR;
	}
	if (m_inferenceHelper->setNumThread(numThreads) != InferenceHelper::RET_OK) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}
	if (m_inferenceHelper->initialize(modelFilename, m_inputTensorList, m_outputTensorList) != InferenceHelper::RET_OK) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}
	/* Check if input tensor info is set */
	for (const auto& inputTensorInfo : m_inputTensorList) {
		if ((inputTensorInfo.tensorDims.width <= 0) || (inputTensorInfo.tensorDims.height <= 0) || inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_NONE) {
			PRINT_E("Invalid tensor size\n");
			m_inferenceHelper.reset();
			return RET_ERR;
		}
	}

	return RET_OK;
}

int32_t HandLandmarkEngine::finalize()
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	m_inferenceHelper->finalize();
	return RET_OK;
}


int32_t HandLandmarkEngine::invoke(const cv::Mat& originalMat, int32_t palmX, int32_t palmY, int32_t palmW, int32_t palmH, float palmRotation, RESULT& result)
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	/*** PreProcess ***/
	const auto& tPreProcess0 = std::chrono::steady_clock::now();
	InputTensorInfo& inputTensorInfo = m_inputTensorList[0];

	/* Rotate palm image */
	cv::Mat rotatedImage;
	cv::RotatedRect rect(cv::Point(palmX + palmW / 2, palmY + palmH / 2), cv::Size(palmW, palmH), palmRotation * 180.f / static_cast<float>(M_PI));
	cv::Mat trans = cv::getRotationMatrix2D(rect.center, rect.angle, 1.0);
	cv::Mat srcRot;
	cv::warpAffine(originalMat, srcRot, trans, originalMat.size());
	cv::getRectSubPix(srcRot, rect.size, rect.center, rotatedImage);
	//cv::imshow("rotatedImage", rotatedImage);

	/* Resize image */
	cv::Mat imgSrc;
	cv::resize(rotatedImage, imgSrc, cv::Size(inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height));
#ifndef CV_COLOR_IS_RGB
	cv::cvtColor(imgSrc, imgSrc, cv::COLOR_BGR2RGB);
#endif
	inputTensorInfo.data = imgSrc.data;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.imageInfo.width = imgSrc.cols;
	inputTensorInfo.imageInfo.height = imgSrc.rows;
	inputTensorInfo.imageInfo.channel = imgSrc.channels();
	inputTensorInfo.imageInfo.cropX = 0;
	inputTensorInfo.imageInfo.cropY = 0;
	inputTensorInfo.imageInfo.cropWidth = imgSrc.cols;
	inputTensorInfo.imageInfo.cropHeight = imgSrc.rows;
	inputTensorInfo.imageInfo.isBGR = false;
	inputTensorInfo.imageInfo.swapColor = false;

	if (m_inferenceHelper->preProcess(m_inputTensorList) != InferenceHelper::RET_OK) {
		return RET_ERR;
	}
	const auto& tPreProcess1 = std::chrono::steady_clock::now();

	/*** Inference ***/
	const auto& tInference0 = std::chrono::steady_clock::now();
	if (m_inferenceHelper->invoke(m_outputTensorList) != InferenceHelper::RET_OK) {
		return RET_ERR;
	}
	const auto& tInference1 = std::chrono::steady_clock::now();

	/*** PostProcess ***/
	const auto& tPostProcess0 = std::chrono::steady_clock::now();

	/* Retrieve the result */
	HAND_LANDMARK& handLandmark = result.handLandmark;
	handLandmark.handflag = m_outputTensorList[1].getDataAsFloat()[0];
	handLandmark.handedness = m_outputTensorList[2].getDataAsFloat()[0];
	const float *ld21 = m_outputTensorList[0].getDataAsFloat();
	//printf("%f  %f\n", m_outputTensorHandflag->getDataAsFloat()[0], m_outputTensorHandedness->getDataAsFloat()[0]);

	for (int32_t i = 0; i < 21; i++) {
		handLandmark.pos[i].x = ld21[i * 3 + 0] / inputTensorInfo.tensorDims.width;	// 0.0 - 1.0
		handLandmark.pos[i].y = ld21[i * 3 + 1] / inputTensorInfo.tensorDims.height;	// 0.0 - 1.0
		handLandmark.pos[i].z = ld21[i * 3 + 2] * 1;	 // Scale Z coordinate as X. (-100 - 100???) todo
		//printf("%f\n", m_outputTensorLd21->getDataAsFloat()[i]);
		//cv::circle(originalMat, cv::Point(m_outputTensorLd21->getDataAsFloat()[i * 3 + 0], m_outputTensorLd21->getDataAsFloat()[i * 3 + 1]), 5, cv::Scalar(255, 255, 0), 1);
	}

	/* Fix landmark rotation */
	for (int32_t i = 0; i < 21; i++) {
		handLandmark.pos[i].x *= rotatedImage.cols;	// coordinate on rotatedImage
		handLandmark.pos[i].y *= rotatedImage.rows;
	}
	rotateLandmark(handLandmark, palmRotation, rotatedImage.cols, rotatedImage.rows);	// coordinate on thei nput image

	/* Calculate palm rectangle from Landmark */
	transformLandmarkToRect(handLandmark);
	handLandmark.rect.rotation = calculateRotation(handLandmark);

	for (int32_t i = 0; i < 21; i++) {
		handLandmark.pos[i].x += palmX;
		handLandmark.pos[i].y += palmY;
	}
	handLandmark.rect.x += palmX;
	handLandmark.rect.y += palmY;

	const auto& tPostProcess1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.timePreProcess = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.timeInference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
	result.timePostProcess = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

	return RET_OK;
}



void HandLandmarkEngine::rotateLandmark(HAND_LANDMARK& handLandmark, float rotationRad, int32_t imageWidth, int32_t imageHeight)
{
	for (int32_t i = 0; i < 21; i++) {
		float x = handLandmark.pos[i].x - imageWidth / 2.f;
		float y = handLandmark.pos[i].y - imageHeight / 2.f;

		handLandmark.pos[i].x = x * std::cos(rotationRad) - y * std::sin(rotationRad) + imageWidth / 2.f;
		handLandmark.pos[i].y = x * std::sin(rotationRad) + y * std::cos(rotationRad) + imageHeight / 2.f;
		//handLandmark.pos[i].x = std::min(handLandmark.pos[i].x, 1.f);
		//handLandmark.pos[i].y = std::min(handLandmark.pos[i].y, 1.f);
	};
}

float HandLandmarkEngine::calculateRotation(const HAND_LANDMARK& handLandmark)
{
	// Reference: mediapipe\graphs\hand_tracking\calculators\hand_detections_to_rects_calculator.cc
	constexpr int32_t kWristJoint = 0;
	constexpr int32_t kMiddleFingerPIPJoint = 12;
	constexpr int32_t kIndexFingerPIPJoint = 8;
	constexpr int32_t kRingFingerPIPJoint = 16;
	constexpr float target_angle_ = static_cast<float>(M_PI) * 0.5f;

	const float x0 = handLandmark.pos[kWristJoint].x;
	const float y0 = handLandmark.pos[kWristJoint].y;
	float x1 = (handLandmark.pos[kMiddleFingerPIPJoint].x + handLandmark.pos[kMiddleFingerPIPJoint].x) / 2.f;
	float y1 = (handLandmark.pos[kMiddleFingerPIPJoint].y + handLandmark.pos[kMiddleFingerPIPJoint].y) / 2.f;
	x1 = (x1 + handLandmark.pos[kMiddleFingerPIPJoint].x) / 2.f;
	y1 = (y1 + handLandmark.pos[kMiddleFingerPIPJoint].y) / 2.f;

	float rotation;
	rotation = target_angle_ - std::atan2(-(y1 - y0), x1 - x0);
	rotation = rotation - 2 * static_cast<float>(M_PI) * std::floor((rotation - (-static_cast<float>(M_PI))) / (2 * static_cast<float>(M_PI)));

	return rotation;
}

void HandLandmarkEngine::transformLandmarkToRect(HAND_LANDMARK &handLandmark)
{
	constexpr float shift_x = 0.0f;
	constexpr float shift_y = -0.0f;
	constexpr float scale_x = 1.8f;		// tuned parameter by looking
	constexpr float scale_y = 1.8f;

	float width = 0;
	float height = 0;
	float x_center = 0;
	float y_center = 0;

	float xmin = handLandmark.pos[0].x;
	float xmax = handLandmark.pos[0].x;
	float ymin = handLandmark.pos[0].y;
	float ymax = handLandmark.pos[0].y;

	for (int32_t i = 0; i < 21; i++) {
		if (handLandmark.pos[i].x < xmin) xmin = handLandmark.pos[i].x;
		if (handLandmark.pos[i].x > xmax) xmax = handLandmark.pos[i].x;
		if (handLandmark.pos[i].y < ymin) ymin = handLandmark.pos[i].y;
		if (handLandmark.pos[i].y > ymax) ymax = handLandmark.pos[i].y;
	}
	width = xmax - xmin;
	height = ymax - ymin;
	x_center = (xmax + xmin) / 2.f;
	y_center = (ymax + ymin) / 2.f;

	width *= scale_x;
	height *= scale_y;

	float long_side = std::max(width, height);

	/* for hand is closed */
	//float palmDistance = powf(handLandmark.pos[0].x - handLandmark.pos[9].x, 2) + powf(handLandmark.pos[0].y - handLandmark.pos[9].y, 2);
	//palmDistance = sqrtf(palmDistance);
	//long_side = std::max(long_side, palmDistance);

	handLandmark.rect.width = (long_side * 1);
	handLandmark.rect.height = (long_side * 1);
	handLandmark.rect.x = (x_center - handLandmark.rect.width / 2);
	handLandmark.rect.y = (y_center - handLandmark.rect.height / 2);
}
