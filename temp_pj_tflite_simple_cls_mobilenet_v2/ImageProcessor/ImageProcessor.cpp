/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <chrono>


/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for Tensorflow Lite */
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

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

/* Setting */
static const float PIXEL_MEAN[3] = { 0.5f, 0.5f, 0.5f };
static const float PIXEL_STD[3] = { 0.25f,  0.25f, 0.25f };

/*** Global variable ***/
static std::vector<std::string> s_labels;
static std::unique_ptr<tflite::FlatBufferModel> s_model;
static tflite::ops::builtin::BuiltinOpResolver s_resolver;
static std::unique_ptr<tflite::Interpreter> s_interpreter;


/*** Function ***/
static void displayModelInfo(const tflite::Interpreter* interpreter)
{
	const auto& inputIndices = interpreter->inputs();
	int inputNum = (int)inputIndices.size();
	PRINT("Input num = %d\n", inputNum);
	for (int i = 0; i < inputNum; i++) {
		auto* tensor = interpreter->tensor(inputIndices[i]);
		for (int j = 0; j < tensor->dims->size; j++) {
			PRINT("    tensor[%d]->dims->size[%d]: %d\n", i, j, tensor->dims->data[j]);
		}
		if (tensor->type == kTfLiteUInt8) {
			PRINT("    tensor[%d]->type: quantized\n", i);
			PRINT("    tensor[%d]->params.outputZeroPoint, scale: %d, %f\n", i, tensor->params.zero_point, tensor->params.scale);
		} else {
			PRINT("    tensor[%d]->type: not quantized\n", i);
		}
	}

	const auto& outputIndices = interpreter->outputs();
	int outputNum = (int)outputIndices.size();
	PRINT("Output num = %d\n", outputNum);
	for (int i = 0; i < outputNum; i++) {
		auto* tensor = interpreter->tensor(outputIndices[i]);
		for (int j = 0; j < tensor->dims->size; j++) {
			PRINT("    tensor[%d]->dims->size[%d]: %d\n", i, j, tensor->dims->data[j]);
		}
		if (tensor->type == kTfLiteUInt8) {
			PRINT("    tensor[%d]->type: quantized\n", i);
			PRINT("    tensor[%d]->params.outputZeroPoint, scale: %d, %f\n", i, tensor->params.zero_point, tensor->params.scale);
		} else {
			PRINT("    tensor[%d]->type: not quantized\n", i);
		}
	}
}

static void extractTensorAsFloatVector(tflite::Interpreter *interpreter, const int index, std::vector<float> &output)
{
	const TfLiteTensor* tensor = interpreter->tensor(index);
	int dataNum = 1;
	for (int i = 0; i < tensor->dims->size; i++) {
		dataNum *= tensor->dims->data[i];
	}
	output.resize(dataNum);
	if (tensor->type == kTfLiteUInt8) {
		const auto *valUint8 = interpreter->typed_tensor<uint8_t>(index);
		for (int i = 0; i < dataNum; i++) {
			float valFloat = (valUint8[i] - tensor->params.zero_point) * tensor->params.scale;
			output[i] = valFloat;
		}
	} else {
		const auto *valFloat = interpreter->typed_tensor<float>(index);
		for (int i = 0; i < dataNum; i++) {
			output[i] = valFloat[i];
		}
	}
}

static TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src) {
	if (!src) return nullptr;
	TfLiteFloatArray* ret = static_cast<TfLiteFloatArray*>(
		malloc(TfLiteFloatArrayGetSizeInBytes(src->size)));
	if (!ret) return nullptr;
	ret->size = src->size;
	std::memcpy(ret->data, src->data, src->size * sizeof(float));
	return ret;
}

static void setBufferToTensor(tflite::Interpreter *interpreter, const int index, const char *data, const unsigned int dataSize)
{
	const TfLiteTensor* inputTensor = interpreter->tensor(index);
	const int modelInputHeight = inputTensor->dims->data[1];
	const int modelInputWidth = inputTensor->dims->data[2];
	const int modelInputChannel = inputTensor->dims->data[3];

	if (inputTensor->type == kTfLiteUInt8) {
		CHECK(sizeof(int8_t) * 1 * modelInputHeight * modelInputWidth * modelInputChannel == dataSize);
		/* Need deep copy quantization parameters */
		/* reference: https://github.com/google-coral/edgetpu/blob/master/src/cpp/basic/basic_engine_native.cc */
		/* todo: release them */
		const TfLiteAffineQuantization* inputQuantParams = reinterpret_cast<TfLiteAffineQuantization*>(inputTensor->quantization.params);
		TfLiteQuantization inputQuantClone;
		inputQuantClone = inputTensor->quantization;
		TfLiteAffineQuantization* inputQuantParamsClone = reinterpret_cast<TfLiteAffineQuantization*>(malloc(sizeof(TfLiteAffineQuantization)));
		inputQuantParamsClone->scale = TfLiteFloatArrayCopy(inputQuantParams->scale);
		inputQuantParamsClone->zero_point = TfLiteIntArrayCopy(inputQuantParams->zero_point);
		inputQuantParamsClone->quantized_dimension = inputQuantParams->quantized_dimension;
		inputQuantClone.params = inputQuantParamsClone;

		//memcpy(inputTensor->data.int8, data, sizeof(int8_t) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
		interpreter->SetTensorParametersReadOnly(
			index, inputTensor->type, inputTensor->name,
			std::vector<int>(inputTensor->dims->data, inputTensor->dims->data + inputTensor->dims->size),
			inputQuantClone,	// use copied parameters
			data, dataSize);
	} else {
		CHECK(sizeof(float) * 1 * modelInputHeight * modelInputWidth * modelInputChannel == dataSize);
		//memcpy(inputTensor->data.f, data, sizeof(float) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
		interpreter->SetTensorParametersReadOnly(
			index, inputTensor->type, inputTensor->name,
			std::vector<int>(inputTensor->dims->data, inputTensor->dims->data + inputTensor->dims->size),
			inputTensor->quantization,
			data, sizeof(float) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
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


int ImageProcessor_initialize(const char *modelFilename, INPUT_PARAM *inputParam)
{
	/* Create interpreter */
	s_model = tflite::FlatBufferModel::BuildFromFile((std::string(modelFilename) + ".tflite").c_str());
	CHECK(s_model != nullptr);
	
	tflite::InterpreterBuilder builder(*s_model, s_resolver);
	builder(&s_interpreter);
	CHECK(s_interpreter != nullptr);

	s_interpreter->SetNumThreads(inputParam->numThreads);
#ifdef USE_EDGETPU_DELEGATE
	size_t num_devices;
	std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
	TFLITE_MINIMAL_CHECK(num_devices > 0);
	const auto& device = devices.get()[0];
	auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
	s_interpreter->ModifyGraphWithDelegate({ delegate, edgetpu_free_delegate });
#endif
	CHECK(s_interpreter->AllocateTensors() == kTfLiteOk);


	/* Get model information */
	displayModelInfo(s_interpreter.get());


	/* read label */
	readLabel(inputParam->labelFilename, s_labels);

	return 0;
}


int ImageProcessor_process(cv::Mat *mat, OUTPUT_PARAM *outputParam)
{
	const TfLiteTensor* inputTensor = s_interpreter->input_tensor(0);
	const int modelInputHeight = inputTensor->dims->data[1];
	const int modelInputWidth = inputTensor->dims->data[2];
	const int modelInputChannel = inputTensor->dims->data[3];

	/*** PreProcess for ncnn ***/
	cv::Mat inputImage;

	/*** PreProcess ***/
	cv::resize(*mat, inputImage, cv::Size(modelInputWidth, modelInputHeight));
	cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
	if (inputTensor->type == kTfLiteUInt8) {
		inputImage.convertTo(inputImage, CV_8UC3);
	} else {
		inputImage.convertTo(inputImage, CV_32FC3, 1.0 / 255);
		cv::subtract(inputImage, cv::Scalar(cv::Vec<float, 3>(PIXEL_MEAN)), inputImage);
		cv::divide(inputImage, cv::Scalar(cv::Vec<float, 3>(PIXEL_STD)), inputImage);
	}

	/* Set data to input tensor */
#if 1
	setBufferToTensor(s_interpreter.get(), s_interpreter->inputs()[0], (char*)inputImage.data, (int)(inputImage.total() * inputImage.elemSize()));
#else
	if (inputTensor->type == kTfLiteUInt8) {
		memcpy(s_interpreter->typed_input_tensor<uint8_t>(0), inputImage.reshape(0, 1).data, sizeof(uint8_t) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	} else {
		memcpy(s_interpreter->typed_input_tensor<float>(0), inputImage.reshape(0, 1).data, sizeof(float) * 1 * modelInputWidth * modelInputHeight * modelInputChannel);
	}
#endif
	
	/*** Inference ***/
	CHECK(s_interpreter->Invoke() == kTfLiteOk);

	/*** PostProcess ***/
	/* Retrieve the result */
	std::vector<float> outputScoreList;
	extractTensorAsFloatVector(s_interpreter.get(), s_interpreter->outputs()[0], outputScoreList);

	/* Find the max score */
	int maxIndex = (int)(std::max_element(outputScoreList.begin(), outputScoreList.end()) - outputScoreList.begin());
	auto maxScore = *std::max_element(outputScoreList.begin(), outputScoreList.end());
	PRINT("Result = %s (%d) (%.3f)\n", s_labels[maxIndex].c_str(), maxIndex, maxScore);

	/* Draw the result */
	std::string resultStr;
	resultStr = "Result:" + s_labels[maxIndex] + " (score = " + std::to_string(maxScore) + ")";
	cv::putText(*mat, resultStr, cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 3);
	cv::putText(*mat, resultStr, cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);

	/* Return the results */
	outputParam->classId = maxIndex;
	snprintf(outputParam->label, sizeof(outputParam->label), s_labels[maxIndex].c_str());
	outputParam->score = maxScore;
	
	return 0;
}


int ImageProcessor_finalize(void)
{
	s_model.reset();
	s_interpreter.reset();
	return 0;
}
