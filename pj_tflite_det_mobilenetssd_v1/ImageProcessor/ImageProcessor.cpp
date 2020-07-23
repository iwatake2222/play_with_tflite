/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <vector>

/* for OpenCV */
#include <opencv2/opencv.hpp>

#include "InferenceHelper.h"
#include "ImageProcessor.h"

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
#define MODEL_NAME   "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29_edgetpu"
#else
#define MODEL_NAME   "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29"
#endif
#define LABEL_NAME   "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.txt"
static const float PIXEL_MEAN[3] = { 0.5f, 0.5f, 0.5f };
static const float PIXEL_STD[3] = { 0.25f,  0.25f, 0.25f };

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
static std::vector<std::string> s_labels;
static InferenceHelper *s_inferenceHelper;
static TensorInfo *s_inputTensor;
static TensorInfo *s_outputTensorBBox;
static TensorInfo *s_outputTensorClass;
static TensorInfo *s_outputTensorScore;
static TensorInfo *s_outputTensorNum;

/*** Function ***/
static void getBBox(std::vector<BBox> &bboxList, const float *outputBoxList, const float *outputClassList, const float *outputScoreList, const int outputNum, const double threshold, const int imageWidth = 0, const int imageHeight = 0)
{
	for (int i = 0; i < outputNum; i++) {
		int classId = (int)outputClassList[i] + 1;
		float score = outputScoreList[i];
		if (score < threshold) continue;
		float y0 = outputBoxList[4 * i + 0];
		float x0 = outputBoxList[4 * i + 1];
		float y1 = outputBoxList[4 * i + 2];
		float x1 = outputBoxList[4 * i + 3];
		if (imageWidth != 0) {
			x0 *= imageWidth;
			x1 *= imageWidth;
			y0 *= imageHeight;
			y1 *= imageHeight;
		}
		//PRINT("%d[%.2f]: %.3f %.3f %.3f %.3f\n", classId, score, x0, y0, x1, y1);
		BBox bbox;
		bbox.x = x0;
		bbox.y = y0;
		bbox.w = x1 - x0;
		bbox.h = y1 - y0;
		bbox.classId = classId;
		bbox.score = score;
		bboxList.push_back(bbox);
	}
}

static void readLabel(const char* filename, std::vector<std::string> & labels)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		PRINT("failed to read %s\n", filename);
		return;
	}
	std::string str;
	while (getline(ifs, str)) {
		labels.push_back(str);
	}
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

	std::string modelFilename = std::string(inputParam->workDir) + "/" + MODEL_NAME;
	std::string labelFilename = std::string(inputParam->workDir) + "/" + LABEL_NAME;

	s_inferenceHelper->initialize(modelFilename.c_str(), inputParam->numThreads);

	s_inputTensor = new TensorInfo();
	s_outputTensorBBox = new TensorInfo();
	s_outputTensorClass = new TensorInfo();
	s_outputTensorScore = new TensorInfo();
	s_outputTensorNum = new TensorInfo();

	s_inferenceHelper->getTensorByName("normalized_input_image_tensor", s_inputTensor);
	s_inferenceHelper->getTensorByName("TFLite_Detection_PostProcess", s_outputTensorBBox);
	s_inferenceHelper->getTensorByName("TFLite_Detection_PostProcess:1", s_outputTensorClass);
	s_inferenceHelper->getTensorByName("TFLite_Detection_PostProcess:2", s_outputTensorScore);
	s_inferenceHelper->getTensorByName("TFLite_Detection_PostProcess:3", s_outputTensorNum);

	/* read label */
	readLabel(labelFilename.c_str(), s_labels);

	return 0;
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
	s_inferenceHelper->inference();

	/*** PostProcess ***/
	/* Retrieve the result */
	int outputNum = (int)((float*)s_outputTensorNum->data)[0];
	std::vector<BBox> bboxList;
	getBBox(bboxList, s_outputTensorBBox->getDataAsFloat(), s_outputTensorClass->getDataAsFloat(), s_outputTensorScore->getDataAsFloat(), outputNum, 0.5, mat->cols, mat->rows);

	/* Draw the result */
	for (int i = 0; i < (int)bboxList.size(); i++) {
		const BBox bbox = bboxList[i];
		cv::rectangle(*mat, cv::Rect((int)bbox.x, (int)bbox.y, (int)bbox.w, (int)bbox.h), cv::Scalar(0, 255, 0), 3);
		cv::putText(*mat, s_labels[bbox.classId], cv::Point((int)bbox.x, (int)bbox.y), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 0), 5);
		cv::putText(*mat, s_labels[bbox.classId], cv::Point((int)bbox.x, (int)bbox.y), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 255, 0), 2);
	}


	/* Return the results */
	outputParam->resultNum = (int)bboxList.size();
	if (outputParam->resultNum > NUM_MAX_RESULT) outputParam->resultNum = NUM_MAX_RESULT;
	for (int i = 0; i < outputParam->resultNum; i++) {
		const BBox bbox = bboxList[i];
		outputParam->RESULTS[i].classId = bbox.classId;
		snprintf(outputParam->RESULTS[i].label, sizeof(outputParam->RESULTS[i].label), "%s", s_labels[bbox.classId].c_str());
		outputParam->RESULTS[i].score = bbox.score;
		outputParam->RESULTS[i].x = (int)bbox.x;
		outputParam->RESULTS[i].y = (int)bbox.y;
		outputParam->RESULTS[i].width = (int)bbox.w;
		outputParam->RESULTS[i].height = (int)bbox.h;
	}
	
	return 0;
}


int ImageProcessor_finalize(void)
{
	s_inferenceHelper->finalize();
	delete s_inputTensor;
	delete s_outputTensorBBox;
	delete s_outputTensorClass;
	delete s_outputTensorScore;
	delete s_outputTensorNum;
	delete s_inferenceHelper;
	return 0;
}
