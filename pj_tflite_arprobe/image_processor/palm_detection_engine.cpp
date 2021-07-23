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

/* for meidapipe sub functions */
#include "meidapipe/transpose_conv_bias.h"
#include "meidapipe/ssd_anchors_calculator.h"
#include "meidapipe/tflite_tensors_to_detections_calculator.h"

/* for My modules */
#include "common_helper.h"
#include "inference_helper.h"
#include "PalmDetectionEngine.h"

/*** Macro ***/
#define TAG "PalmDetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "palm_detection.tflite"

static float calculateRotation(const Detection& det);
static void nms(std::vector<Detection>& detectionList, std::vector<Detection>& detectionListNMS, bool useWeight);
static void RectTransformationCalculator(const Detection& det, const float rotation, float& x, float& y, float& width, float& height);
static std::vector<Anchor> s_anchors;

/*** Function ***/
int32_t PalmDetectionEngine::initialize(const std::string& workDir, const int32_t numThreads)
{
	/* Set model information */
	std::string modelFilename = workDir + "/model/" + MODEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = "input";
	inputTensorInfo.tensor_type= TensorInfo::kTensorTypeFp32;
	inputTensorInfo.tensor_dims.batch = 1;
	inputTensorInfo.tensor_dims.width = 256;
	inputTensorInfo.tensor_dims.height = 256;
	inputTensorInfo.tensor_dims.channel = 3;
	inputTensorInfo.data_type = InputTensorInfo::kDataTypeImage;
	inputTensorInfo.normalize.mean[0] = 0.5f;   	/* normalized to[-1.f, 1.f] (hand_detection_cpu.pbtxt.pbtxt) */
	inputTensorInfo.normalize.mean[1] = 0.5f;
	inputTensorInfo.normalize.mean[2] = 0.5f;
	inputTensorInfo.normalize.norm[0] = 0.5f;
	inputTensorInfo.normalize.norm[1] = 0.5f;
	inputTensorInfo.normalize.norm[2] = 0.5f;
	m_inputTensorList.push_back(inputTensorInfo);

	/* Set output tensor info */
	m_outputTensorList.clear();
	OutputTensorInfo outputTensorInfo;
	outputTensorInfo.tensor_type = TensorInfo::kTensorTypeFp32;
	outputTensorInfo.name = "regressors";
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = "classificators";
	m_outputTensorList.push_back(outputTensorInfo);

	/* Create and Initialize Inference Helper */
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::OPEN_CV));
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::TENSOR_RT));
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::NCNN));
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::MNN));
	m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
	//m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
	// m_inferenceHelper.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));

	if (!m_inferenceHelper) {
		return RET_ERR;
	}
	if (m_inferenceHelper->SetNumThreads(numThreads) != InferenceHelper::kRetOk) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}

	std::vector<std::pair<const char*, const void*>> customOps;
	customOps.push_back(std::pair<const char*, const void*>("Convolution2DTransposeBias", (const void*)mediapipe::tflite_operations::RegisterConvolution2DTransposeBias()));
	if (m_inferenceHelper->SetCustomOps(customOps) != InferenceHelper::kRetOk) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}

	if (m_inferenceHelper->Initialize(modelFilename, m_inputTensorList, m_outputTensorList) != InferenceHelper::kRetOk) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}
	/* Check if input tensor info is set */
	for (const auto& inputTensorInfo : m_inputTensorList) {
		if ((inputTensorInfo.tensor_dims.width <= 0) || (inputTensorInfo.tensor_dims.height <= 0) || inputTensorInfo.tensor_type == TensorInfo::kTensorTypeNone) {
			PRINT_E("Invalid tensor size\n");
			m_inferenceHelper.reset();
			return RET_ERR;
		}
	}

	/* Call SsdAnchorsCalculator::GenerateAnchors as described in hand_detection_gpu.pbtxt */
	const SsdAnchorsCalculatorOptions options;
	::mediapipe::GenerateAnchors(&s_anchors, options);

	return RET_OK;
}

int32_t PalmDetectionEngine::finalize()
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	m_inferenceHelper->Finalize();
	return RET_OK;
}


int32_t PalmDetectionEngine::invoke(const cv::Mat& originalMat, RESULT& result)
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}

	int32_t imageWidth = originalMat.cols;
	int32_t imageHeight = originalMat.rows;


	/*** PreProcess ***/
	const auto& tPreProcess0 = std::chrono::steady_clock::now();
	InputTensorInfo& inputTensorInfo = m_inputTensorList[0];
	/* do resize and color conversion here because some inference engine doesn't support these operations */
	cv::Mat imgSrc;
	cv::resize(originalMat, imgSrc, cv::Size(inputTensorInfo.tensor_dims.width, inputTensorInfo.tensor_dims.height));
#ifndef CV_COLOR_IS_RGB
	cv::cvtColor(imgSrc, imgSrc, cv::COLOR_BGR2RGB);
#endif
	inputTensorInfo.data = imgSrc.data;
	inputTensorInfo.data_type = InputTensorInfo::kDataTypeImage;
	inputTensorInfo.image_info.width = imgSrc.cols;
	inputTensorInfo.image_info.height = imgSrc.rows;
	inputTensorInfo.image_info.channel = imgSrc.channels();
	inputTensorInfo.image_info.crop_x = 0;
	inputTensorInfo.image_info.crop_y = 0;
	inputTensorInfo.image_info.crop_width = imgSrc.cols;
	inputTensorInfo.image_info.crop_height = imgSrc.rows;
	inputTensorInfo.image_info.is_bgr = false;
	inputTensorInfo.image_info.swap_color = false;

	if (m_inferenceHelper->PreProcess(m_inputTensorList) != InferenceHelper::kRetOk) {
		return RET_ERR;
	}
	const auto& tPreProcess1 = std::chrono::steady_clock::now();


	/*** Inference ***/
	const auto& tInference0 = std::chrono::steady_clock::now();
	if (m_inferenceHelper->Process(m_outputTensorList) != InferenceHelper::kRetOk) {
		return RET_ERR;
	}
	const auto& tInference1 = std::chrono::steady_clock::now();


	/*** PostProcess ***/
	const auto& tPostProcess0 = std::chrono::steady_clock::now();
	/* Call TfLiteTensorsToDetectionsCalculator::DecodeBoxes as described in hand_detection_gpu.pbtxt */
	std::vector<Detection> detectionList;
	const TfLiteTensorsToDetectionsCalculatorOptions options;
	if (options.num_boxes() != m_outputTensorList[0].tensor_dims.height) {
		return RET_ERR;
	}
	if (options.num_coords() != m_outputTensorList[0].tensor_dims.width) {
		return RET_ERR;
	}
	if (options.num_classes() != m_outputTensorList[1].tensor_dims.width) {
		return RET_ERR;
	}
	const float* raw_boxes = m_outputTensorList[0].GetDataAsFloat();
	const float* raw_scores = m_outputTensorList[1].GetDataAsFloat();
	mediapipe::Process(options, raw_boxes, raw_scores, s_anchors, detectionList);

	/* Call NonMaxSuppressionCalculator as described in hand_detection_gpu.pbtxt */
	/*  -> use my own NMS */
	std::vector<Detection> detectionListNMS;
	nms(detectionList, detectionListNMS, false);

	std::vector<PALM> palmList;
	for (auto palmDet : detectionListNMS) {
		/* Convert the coordinate from on (0.0 - 1.0) to on the input image */
		palmDet.x *= imageWidth;
		palmDet.y *= imageHeight;
		palmDet.w *= imageWidth;
		palmDet.h *= imageHeight;
		for (auto kp : palmDet.keypoints) {
			kp.first *= imageWidth;
			kp.second *= imageHeight;
		}

		/* Call DetectionsToRectsCalculator as described in hand_detection_gpu.pbtxt */
		/*  -> use my own calculator */
		float rotation = calculateRotation(palmDet);
		//printf("%f  %f\n", rotation, rotation * 180 / 3.14);

		/* Call RectTransformationCalculator as described in hand_landmark_cpu.pbtxt */
		/*  -> use my own calculator */
		float x, y, width, height;
		RectTransformationCalculator(palmDet, rotation, x, y, width, height);

		PALM palm = { 0 };
		palm.score = palmDet.score;
		palm.x = (std::min)(imageWidth * 1.f, (std::max)(x, 0.f));
		palm.y = (std::min)(imageHeight * 1.f, (std::max)(y, 0.f));
		palm.width = (std::min)(imageWidth * 1.f - palm.x, (std::max)(width, 0.f));
		palm.height = (std::min)(imageHeight * 1.f - palm.y, (std::max)(height, 0.f));
		palm.rotation = rotation;
		palmList.push_back(palm);
	}
	const auto& tPostProcess1 = std::chrono::steady_clock::now();


	/* Return the results */
	result.palmList = palmList;
	result.timePreProcess = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.timeInference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
	result.timePostProcess = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

	return RET_OK;
}



static void RectTransformationCalculator(const Detection& det, const float rotation, float& x, float& y, float& width, float& height)
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

	const float long_side = (std::max)(det.w, det.h);
	width = long_side * scale_x;
	height = long_side * scale_y;
	x = x_center - width / 2;
	y = y_center - height / 2;
}


static float calculateRotation(const Detection& det)
{
	/* Reference: ::mediapipe::Status DetectionsToRectsCalculator::ComputeRotation (detections_to_rects_calculator.cc) */
	constexpr int32_t rotation_vector_start_keypoint_index = 0;  // # Center of wrist.
	constexpr int32_t rotation_vector_end_keypoint_index = 2;	// # MCP of middle finger.
	constexpr double rotation_vector_target_angle_degrees = M_PI * 0.5f;

	const float x0 = det.keypoints[rotation_vector_start_keypoint_index].first;
	const float y0 = det.keypoints[rotation_vector_start_keypoint_index].second;
	const float x1 = det.keypoints[rotation_vector_end_keypoint_index].first;
	const float y1 = det.keypoints[rotation_vector_end_keypoint_index].second;

	double rotation;
	rotation = rotation_vector_target_angle_degrees - std::atan2(-(y1 - y0), x1 - x0);
	rotation = rotation - 2 * M_PI * std::floor((rotation - (-M_PI)) / (2 * M_PI));
	return static_cast<float>(rotation);
}

static float calculateIoU(const Detection& det0, const Detection& det1)
{
	float interx0 = (std::max)(det0.x, det1.x);
	float intery0 = (std::max)(det0.y, det1.y);
	float interx1 = (std::min)(det0.x + det0.w, det1.x + det1.w);
	float intery1 = (std::min)(det0.y + det0.h, det1.y + det1.h);

	float area0 = det0.w * det0.h;
	float area1 = det1.w * det1.h;
	float areaInter = (interx1 - interx0) * (intery1 - intery0);
	float areaSum = area0 + area1 - areaInter;

	return areaInter / areaSum;
}

static void nms(std::vector<Detection>& detectionList, std::vector<Detection>& detectionListNMS, bool useWeight)
{
	std::sort(detectionList.begin(), detectionList.end(), [](Detection const& lhs, Detection const& rhs) {
		if (lhs.w * lhs.h > rhs.w * rhs.h) return true;
		// if (lhs.score > rhs.score) return true;
		return false;
	});

	bool *isMerged = new bool[detectionList.size()];
	for (int32_t i = 0; i < detectionList.size(); i++) isMerged[i] = false;
	for (int32_t indexHighScore = 0; indexHighScore < detectionList.size(); indexHighScore++) {
		std::vector<Detection> candidates;
		if (isMerged[indexHighScore]) continue;
		candidates.push_back(detectionList[indexHighScore]);
		for (int32_t indexLowScore = indexHighScore + 1; indexLowScore < detectionList.size(); indexLowScore++) {
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
			mergedBox.keypoints.resize(candidates[0].keypoints.size(), std::make_pair<float, float>(0, 0));
			float sumScore = 0;
			for (auto candidate : candidates) {
				sumScore += candidate.score;
				mergedBox.score += candidate.score;
				mergedBox.x += candidate.x * candidate.score;
				mergedBox.y += candidate.y * candidate.score;
				mergedBox.w += candidate.w * candidate.score;
				mergedBox.h += candidate.h * candidate.score;
				for (int32_t k = 0; k < mergedBox.keypoints.size(); k++) {
					mergedBox.keypoints[k].first += candidate.keypoints[k].first * candidate.score;
					mergedBox.keypoints[k].second += candidate.keypoints[k].second * candidate.score;
				}
			}
			mergedBox.score /= candidates.size();
			mergedBox.x /= sumScore;
			mergedBox.y /= sumScore;
			mergedBox.w /= sumScore;
			mergedBox.h /= sumScore;
			for (int32_t k = 0; k < mergedBox.keypoints.size(); k++) {
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
