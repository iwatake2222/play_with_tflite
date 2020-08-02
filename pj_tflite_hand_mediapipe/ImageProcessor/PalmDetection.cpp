/*** Include ***/
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>

#include "meidapipe/transpose_conv_bias.h"
#include "meidapipe/ssd_anchors_calculator.h"
#include "meidapipe/tflite_tensors_to_detections_calculator.h"

#include "PalmDetection.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define PRINT(...) printf(__VA_ARGS__)
#endif

#define CHECK(x)                              \
  if (!(x)) {                                                \
	PRINT("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

/* Model parameters */
#ifdef TFLITE_DELEGATE_EDGETPU
not supported
#else
#define MODEL_NAME "palm_detection"
#endif

//normalized to[0.f, 1.f]
static const float PIXEL_MEAN[3] = { 0.0f, 0.0f, 0.0f };
static const float PIXEL_STD[3] = { 1.0f,  1.0f, 1.0f };

static std::vector<Anchor> s_anchors;

static void nms(std::vector<Detection> &detectionList, std::vector<Detection> &detectionListNMS, bool useWeight = false);
static float calculateIoU(Detection& det0, Detection& det1);
static float calculateRotation(Detection& det);
static void RectTransformationCalculator(Detection& det, float rotation, float *x, float *y, float *width, float *height);


int PalmDetection::PalmDetection::initialize(const char *workDir, const int numThreads)
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

	std::vector<std::pair<const char*, const void*>> customOps;
	customOps.push_back(std::pair<const char*, const void*>("Convolution2DTransposeBias", mediapipe::tflite_operations::RegisterConvolution2DTransposeBias()));
	m_inferenceHelper->initialize(modelFilename.c_str(), numThreads, customOps);

	m_inputTensor = new TensorInfo();
	m_outputTensorBoxes = new TensorInfo();
	m_outputTensorScores = new TensorInfo();


	m_inferenceHelper->getTensorByName("input", m_inputTensor);
	m_inferenceHelper->getTensorByName("regressors", m_outputTensorBoxes);
	m_inferenceHelper->getTensorByName("classificators", m_outputTensorScores);

	/* Call SsdAnchorsCalculator::GenerateAnchors as described in hand_detection_gpu.pbtxt */
	const SsdAnchorsCalculatorOptions options;
	::mediapipe::GenerateAnchors(&s_anchors, options);

	return 0;
}

int PalmDetection::PalmDetection::finalize()
{
	m_inferenceHelper->finalize();
	delete m_inputTensor;
	delete m_outputTensorBoxes;
	delete m_outputTensorScores;
	delete m_inferenceHelper;
	return 0;
}


int PalmDetection::PalmDetection::invoke(cv::Mat &originalMat, std::vector<PALM> &palmList)
{
	int imageWidth = originalMat.cols;
	int imageHeight = originalMat.rows;

	/*** PreProcess ***/
	int modelInputWidth = m_inputTensor->dims[2];
	int modelInputHeight = m_inputTensor->dims[1];
	int modelInputChannel = m_inputTensor->dims[3];

	cv::Mat inputImage;
	cv::resize(originalMat, inputImage, cv::Size(modelInputWidth, modelInputHeight));
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
	/* Call TfLiteTensorsToDetectionsCalculator::DecodeBoxes as described in hand_detection_gpu.pbtxt */
	std::vector<Detection> detectionList;
	const TfLiteTensorsToDetectionsCalculatorOptions options;
	CHECK(options.num_boxes() == m_outputTensorBoxes->dims[1]);
	CHECK(options.num_coords() == m_outputTensorBoxes->dims[2]);
	CHECK(options.num_classes() == m_outputTensorScores->dims[2]);
	const float* raw_boxes = m_outputTensorBoxes->getDataAsFloat();
	const float* raw_scores = m_outputTensorScores->getDataAsFloat();
	mediapipe::Process(options, raw_boxes, raw_scores, s_anchors, detectionList);

	/* Call NonMaxSuppressionCalculator as described in hand_detection_gpu.pbtxt */
	/*  -> use my own NMS */
	std::vector<Detection> detectionListNMS;
	nms(detectionList, detectionListNMS, false);

	for (auto palm : detectionListNMS) {
		/* Convert the coordinate from on (0.0 - 1.0) to on the input image */
		palm.x *= imageWidth;
		palm.y *= imageHeight;
		palm.w *= imageWidth;
		palm.h *= imageHeight;
		for (auto kp : palm.keypoints) {
			kp.first *= imageWidth;
			kp.second *= imageHeight;
		}

		/* Call DetectionsToRectsCalculator as described in hand_detection_gpu.pbtxt */
		/*  -> use my own calculator */
		float rotation = calculateRotation(palm);
		//printf("%f  %f\n", rotation, rotation * 180 / 3.14);

		/* Call RectTransformationCalculator as described in hand_landmark_cpu.pbtxt */
		/*  -> use my own calculator */
		float x, y, width, height;
		RectTransformationCalculator(palm, rotation, &x, &y, &width, &height);

		PALM palm;
		palm.score = palm.score;
		palm.x = std::min(imageWidth * 1.f, std::max(x, 0.f));
		palm.y = std::min(imageHeight * 1.f, std::max(y, 0.f));
		palm.width = std::min(imageWidth * 1.f - palm.x, std::max(width, 0.f));
		palm.height = std::min(imageHeight * 1.f  - palm.y, std::max(height, 0.f));
		palm.rotation = rotation;
		palmList.push_back(palm);
	}

	return 0;
}


static void RectTransformationCalculator(Detection& det, float rotation, float *x, float *y, float *width, float *height)
{
	/* Reference: RectTransformationCalculator::TransformRect */
	const float shift_x = 0.0f;
	const float shift_y = -0.5f;
	const float scale_x = 2.6f;
	const float scale_y = 2.6f;


	float x_center = det.x + det.w / 2.f;
	float y_center = det.y + det.h / 2.f;
	if (rotation == 0.f) {
		x_center += det.w * shift_x;
		y_center += det.h * shift_y;
	} else {
		const float x_shift = (det.w * shift_x * std::cos(rotation) - det.h * shift_y * std::sin(rotation));
		const float y_shift = (det.w * shift_x * std::sin(rotation) + det.h * shift_y * std::cos(rotation));
		x_center += x_shift;
		y_center += y_shift;
	}

	const float long_side = std::max(det.w, det.h);
	*width = long_side * scale_x;
	*height = long_side * scale_y;
	*x = x_center - *width / 2;
	*y = y_center - *height / 2;
}


static float calculateRotation(Detection& det)
{
	/* Reference: ::mediapipe::Status DetectionsToRectsCalculator::ComputeRotation (detections_to_rects_calculator.cc) */
	#define M_PI       3.14159265358979323846   // pi
	const int rotation_vector_start_keypoint_index = 0;  // # Center of wrist.
	const int rotation_vector_end_keypoint_index = 2;	// # MCP of middle finger.
	const float rotation_vector_target_angle_degrees = M_PI * 0.5f;

	const float x0 = det.keypoints[rotation_vector_start_keypoint_index].first;
	const float y0 = det.keypoints[rotation_vector_start_keypoint_index].second;
	const float x1 = det.keypoints[rotation_vector_end_keypoint_index].first;
	const float y1 = det.keypoints[rotation_vector_end_keypoint_index].second;

	float rotation;
	rotation = rotation_vector_target_angle_degrees - std::atan2(-(y1 - y0), x1 - x0);
	rotation = rotation - 2 * M_PI * std::floor((rotation - (-M_PI)) / (2 * M_PI));
	return rotation;
}

static float calculateIoU(Detection& det0, Detection& det1)
{
	float interx0 = std::max(det0.x, det1.x);
	float intery0 = std::max(det0.y, det1.y);
	float interx1 = std::min(det0.x + det0.w, det1.x + det1.w);
	float intery1 = std::min(det0.y + det0.h, det1.y + det1.h);

	float area0 = det0.w * det0.h;
	float area1 = det1.w * det1.h;
	float areaInter = (interx1 - interx0) * (intery1 - intery0);
	float areaSum = area0 + area1 - areaInter;

	return areaInter / areaSum;
}

static void nms(std::vector<Detection> &detectionList, std::vector<Detection> &detectionListNMS, bool useWeight)
{
	std::sort(detectionList.begin(), detectionList.end(), [](auto const& lhs, auto const& rhs) {
		if (lhs.score > rhs.score) return true;
		return false;
	});

	bool *isMerged = new bool[detectionList.size()];
	for (int i = 0; i < detectionList.size(); i++) isMerged[i] = false;
	for (int indexHighScore = 0; indexHighScore < detectionList.size(); indexHighScore++) {
		std::vector<Detection> candidates;
		if (isMerged[indexHighScore]) continue;
		candidates.push_back(detectionList[indexHighScore]);
		for (int indexLowScore = indexHighScore + 1; indexLowScore < detectionList.size(); indexLowScore++) {
			if (isMerged[indexLowScore]) continue;
			if (detectionList[indexHighScore].class_id != detectionList[indexLowScore].class_id) continue;
			if (calculateIoU(detectionList[indexHighScore], detectionList[indexLowScore]) > 0.5) {
				candidates.push_back(detectionList[indexLowScore]);
				isMerged[indexLowScore] = true;
			}
		}

		/* weight by score */
		if (useWeight) {
			if (candidates.size() < 3) continue;	// do not use detected object if the number of bbox is small
			Detection mergedBox = { 0 };
			mergedBox.keypoints.resize(candidates[0].keypoints.size(), std::pair<float, float>(0, 0));
			float sumScore = 0;
			for (auto candidate : candidates) {
				sumScore += candidate.score;
				mergedBox.score += candidate.score;
				mergedBox.x += candidate.x * candidate.score;
				mergedBox.y += candidate.y * candidate.score;
				mergedBox.w += candidate.w * candidate.score;
				mergedBox.h += candidate.h * candidate.score;
				for (int k = 0; k < mergedBox.keypoints.size(); k++) {
					mergedBox.keypoints[k].first += candidate.keypoints[k].first * candidate.score;
					mergedBox.keypoints[k].second += candidate.keypoints[k].second * candidate.score;
				}
			}
			mergedBox.score /= candidates.size();
			mergedBox.x /= sumScore;
			mergedBox.y /= sumScore;
			mergedBox.w /= sumScore;
			mergedBox.h /= sumScore;
			for (int k = 0; k < mergedBox.keypoints.size(); k++) {
				mergedBox.keypoints[k].first /= sumScore;
				mergedBox.keypoints[k].second /= sumScore;
			}
			detectionListNMS.push_back(mergedBox);
		} else {
			detectionListNMS.push_back(candidates[0]);
		}

	}
	delete[] isMerged;
}
