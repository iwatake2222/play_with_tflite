/*** Include ***/
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "HandLandmark.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define PRINT(...) printf(__VA_ARGS__)
#endif

/* Model parameters */
#ifdef TFLITE_DELEGATE_EDGETPU
not supported
#else
#define MODEL_NAME "hand_landmark"
#endif

//normalized to[0.f, 1.f]
static const float PIXEL_MEAN[3] = { 0.0f, 0.0f, 0.0f };
static const float PIXEL_STD[3] = { 1.0f,  1.0f, 1.0f };


int HandLandmark::initialize(const char *workDir, const int numThreads)
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
	m_outputTensorLd21 = new TensorInfo();
	m_outputTensorHandflag = new TensorInfo();
	m_outputTensorHandedness = new TensorInfo();


	m_inferenceHelper->getTensorByName("input_1", m_inputTensor);
	m_inferenceHelper->getTensorByName("ld_21_3d", m_outputTensorLd21);
	m_inferenceHelper->getTensorByName("output_handflag", m_outputTensorHandflag);
	m_inferenceHelper->getTensorByName("output_handedness", m_outputTensorHandedness);
	return 0;
}

int HandLandmark::finalize()
{
	m_inferenceHelper->finalize();
	delete m_inputTensor;
	delete m_outputTensorLd21;
	delete m_outputTensorHandflag;
	delete m_outputTensorHandedness;
	delete m_inferenceHelper;
	return 0;
}


int HandLandmark::invoke(cv::Mat &originalMat, HAND_LANDMARK& handLandmark, int palmX, int palmY, int palmW, int palmH, float palmRotation)
{
	/*** PreProcess ***/
	int modelInputWidth = m_inputTensor->dims[2];
	int modelInputHeight = m_inputTensor->dims[1];
	int modelInputChannel = m_inputTensor->dims[3];

	/* Rotate palm image */
	cv::Mat rotatedImage;
	cv::RotatedRect rect(cv::Point(palmX + palmW / 2, palmY + palmH / 2), cv::Size(palmW, palmH), palmRotation * 180.f / 3.141592654f);
	cv::Mat trans = cv::getRotationMatrix2D(rect.center, rect.angle, 1.0);
	cv::Mat srcRot;
	cv::warpAffine(originalMat, srcRot, trans, originalMat.size());
	cv::getRectSubPix(srcRot, rect.size, rect.center, rotatedImage);
	//cv::imshow("rotatedImage", rotatedImage);

	/* Resize image */
	cv::Mat inputImage;
	cv::resize(rotatedImage, inputImage, cv::Size(modelInputWidth, modelInputHeight));
	cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
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
	/* Retrieve the result */
	handLandmark.handflag = m_outputTensorHandflag->getDataAsFloat()[0];
	handLandmark.handedness = m_outputTensorHandedness->getDataAsFloat()[0];
	//printf("%f  %f\n", m_outputTensorHandflag->getDataAsFloat()[0], m_outputTensorHandedness->getDataAsFloat()[0]);

	for (int i = 0; i < 21; i++) {
		handLandmark.pos[i].x = m_outputTensorLd21->getDataAsFloat()[i * 3 + 0] / modelInputWidth;	// 0.0 - 1.0
		handLandmark.pos[i].y = m_outputTensorLd21->getDataAsFloat()[i * 3 + 1] / modelInputHeight;	// 0.0 - 1.0
		handLandmark.pos[i].z = m_outputTensorLd21->getDataAsFloat()[i * 3 + 2] * 1;	 // Scale Z coordinate as X. (-100 - 100???) todo
		//printf("%f\n", m_outputTensorLd21->getDataAsFloat()[i]);
		//cv::circle(originalMat, cv::Point(m_outputTensorLd21->getDataAsFloat()[i * 3 + 0], m_outputTensorLd21->getDataAsFloat()[i * 3 + 1]), 5, cv::Scalar(255, 255, 0), 1);
	}

	/* Fix landmark rotation */
	for (int i = 0; i < 21; i++) {
		handLandmark.pos[i].x *= rotatedImage.cols;	// coordinate on rotatedImage
		handLandmark.pos[i].y *= rotatedImage.rows;
	}
	rotateLandmark(handLandmark, palmRotation, rotatedImage.cols, rotatedImage.rows);	// coordinate on thei nput image

	/* Calculate palm rectangle from Landmark */
	transformLandmarkToRect(handLandmark);
	handLandmark.rect.rotation = calculateRotation(handLandmark);

	for (int i = 0; i < 21; i++) {
		handLandmark.pos[i].x += palmX;
		handLandmark.pos[i].y += palmY;
	}
	handLandmark.rect.x += palmX;
	handLandmark.rect.y += palmY;
	
	return 0;
}


int HandLandmark::rotateLandmark(HAND_LANDMARK& handLandmark, float rotationRad, int imageWidth, int imageHeight)
{
	for (int i = 0; i < 21; i++) {
		float x = handLandmark.pos[i].x - imageWidth / 2.f;
		float y = handLandmark.pos[i].y - imageHeight / 2.f;

		handLandmark.pos[i].x = x * std::cos(rotationRad) - y * std::sin(rotationRad) + imageWidth / 2.f;
		handLandmark.pos[i].y = x * std::sin(rotationRad) + y * std::cos(rotationRad) + imageHeight / 2.f;
		//handLandmark.pos[i].x = std::min(handLandmark.pos[i].x, 1.f);
		//handLandmark.pos[i].y = std::min(handLandmark.pos[i].y, 1.f);
	};
	return 0;
}

float HandLandmark::calculateRotation(HAND_LANDMARK& handLandmark)
{
	// Reference: mediapipe\graphs\hand_tracking\calculators\hand_detections_to_rects_calculator.cc
#ifndef M_PI
#define M_PI       3.14159265358979323846f   // pi
#endif
	constexpr int kWristJoint = 0;
	constexpr int kMiddleFingerPIPJoint = 12;
	constexpr int kIndexFingerPIPJoint = 8;
	constexpr int kRingFingerPIPJoint = 16;
	const float target_angle_ = M_PI * 0.5f;

	const float x0 = handLandmark.pos[kWristJoint].x;
	const float y0 = handLandmark.pos[kWristJoint].y;
	float x1 = (handLandmark.pos[kMiddleFingerPIPJoint].x + handLandmark.pos[kMiddleFingerPIPJoint].x) / 2.f;
	float y1 = (handLandmark.pos[kMiddleFingerPIPJoint].y + handLandmark.pos[kMiddleFingerPIPJoint].y) / 2.f;
	x1 = (x1 + handLandmark.pos[kMiddleFingerPIPJoint].x) / 2.f;
	y1 = (y1 + handLandmark.pos[kMiddleFingerPIPJoint].y) / 2.f;

	float rotation;
	rotation = target_angle_ - std::atan2(-(y1 - y0), x1 - x0);
	rotation = rotation - 2 * M_PI * std::floor((rotation - (-M_PI)) / (2 * M_PI));

	return rotation;
}

int HandLandmark::transformLandmarkToRect(HAND_LANDMARK &handLandmark)
{
	const float shift_x = 0.0f;
	const float shift_y = -0.0f;
	const float scale_x = 1.8f;
	const float scale_y = 1.8f;

	float width = 0;
	float height = 0;
	float x_center = 0;
	float y_center = 0;

	float xmin = handLandmark.pos[0].x;
	float xmax = handLandmark.pos[0].x;
	float ymin = handLandmark.pos[0].y;
	float ymax = handLandmark.pos[0].y;

	for (int i = 0; i < 21; i++) {
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
	return 0;
}