/*** Include ***/
/* for general */
#include <stdint.h>
#include <stdio.h>
#include <fstream> 
#include <vector>
#include <string>
#include <chrono>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for Tensorflow Lite */
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

/* for Edge TPU */
#include "edgetpu.h"
#include "edgetpu_c.h"
#include "src/cpp/pipeline/pipelined_model_runner.h"
#include "src/cpp/pipeline/utils.h"

/*** Macro ***/
/* Model parameters */
//#define MODEL_SEGMENTS_NUM 3
//#define MODEL_FILENAME_BASE RESOURCE_DIR"/model/mobilenet_v2_1.0_224_quant"
//#define LABEL_NAME          RESOURCE_DIR"/model/imagenet_labels.txt"
//#define MODEL_WIDTH 224
//#define MODEL_HEIGHT 224

#define MODEL_SEGMENTS_NUM 3
#define MODEL_FILENAME_BASE RESOURCE_DIR"/model/inception_v3_quant"
#define LABEL_NAME          RESOURCE_DIR"/model/imagenet_labels.txt"
#define MODEL_WIDTH 299
#define MODEL_HEIGHT 299

/* Settings */
#define LOOP_NUM_FOR_TIME_MEASUREMENT 100

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }


enum class EdgeTpuType {
	kAny,
	kPciOnly,
	kUsbOnly,
};

/*** Function ***/
/* the following function is from https://github.com/google-coral/edgetpu/blob/master/src/cpp/examples/model_pipelining.cc */
static std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> PrepareEdgeTpuContexts(
	int num_tpus, EdgeTpuType device_type) {
	auto get_available_tpus = [](EdgeTpuType device_type) {
		const auto& all_tpus =
			edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
		if (device_type == EdgeTpuType::kAny) {
			return all_tpus;
		} else {
			std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> result;

			edgetpu::DeviceType target_type;
			if (device_type == EdgeTpuType::kPciOnly) {
				target_type = edgetpu::DeviceType::kApexPci;
			} else if (device_type == EdgeTpuType::kUsbOnly) {
				target_type = edgetpu::DeviceType::kApexUsb;
			} else {
				std::cerr << "Invalid device type" << std::endl;
				return result;
			}

			for (const auto& tpu : all_tpus) {
				if (tpu.type == target_type) {
					result.push_back(tpu);
				}
			}

			return result;
		}
	};

	const auto& available_tpus = get_available_tpus(device_type);
	if (available_tpus.size() < num_tpus) {
		std::cerr << "Not enough Edge TPU detected, expected: " << num_tpus
			<< " actual: " << available_tpus.size();
		return {};
	}

	std::unordered_map<std::string, std::string> options = {
		{"Usb.MaxBulkInQueueLength", "8"},
	};

	std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> edgetpu_contexts(
		num_tpus);
	for (int i = 0; i < num_tpus; ++i) {
		edgetpu_contexts[i] = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
			available_tpus[i].type, available_tpus[i].path, options);
		std::cout << "Device " << available_tpus[i].path << " is selected."
			<< std::endl;
	}

	return edgetpu_contexts;
}

/* the following function is from https://github.com/google-coral/edgetpu/blob/master/src/cpp/examples/model_pipelining.cc */
static std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
	const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* context) {
	tflite::ops::builtin::BuiltinOpResolver resolver;
	resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::InterpreterBuilder interpreter_builder(model.GetModel(), resolver);
	if (interpreter_builder(&interpreter) != kTfLiteOk) {
		std::cerr << "Error in interpreter initialization." << std::endl;
		return nullptr;
	}

	interpreter->SetExternalContext(kTfLiteEdgeTpuContext, context);
	interpreter->SetNumThreads(1);
	if (interpreter->AllocateTensors() != kTfLiteOk) {
		std::cerr << "Failed to allocate tensors." << std::endl;
		return nullptr;
	}

	return interpreter;
}

static std::vector<std::string> getSegmentedModelPathList(std::string modelPathBase, int numSegment)
{
	std::vector<std::string> pathList;
	for (int i = 0; i < numSegment; i++) {
		std::string path;
		path = modelPathBase + "_segment_" + std::to_string(i) + "_of_" + std::to_string(numSegment) + "_edgetpu.tflite";
		pathList.push_back(path);
	}
	return pathList;
}

static coral::PipelineTensor createPipelineTensor(const tflite::Interpreter* interpreter, coral::Allocator* allocator, int index, uint8_t* data, int size)
{
	const auto* inputTensor = interpreter->tensor(index);
	TFLITE_MINIMAL_CHECK(inputTensor->bytes == size);
	TFLITE_MINIMAL_CHECK(inputTensor->type == kTfLiteUInt8);

	coral::PipelineTensor pipelineTensor;
	pipelineTensor.data.data = allocator->alloc(inputTensor->bytes);		// need to be released after consumed
	pipelineTensor.bytes = inputTensor->bytes;
	pipelineTensor.type = inputTensor->type;
	memcpy(pipelineTensor.data.data, data, size);

	return pipelineTensor;
}


static void extractTensorAsFloatVector(const tflite::Interpreter *interpreter, const int index, coral::PipelineTensor &pipelineTensor, std::vector<float> &output)
{
	const TfLiteTensor* tensor = interpreter->tensor(index);
	int dataNum = 1;
	for (int i = 0; i < tensor->dims->size; i++) {
		dataNum *= tensor->dims->data[i];
	}
	output.resize(dataNum);
	if (tensor->type == kTfLiteUInt8) {
		//const auto *valUint8 = interpreter->typed_tensor<uint8_t>(index);
		const auto *valUint8 = pipelineTensor.data.uint8;
		for (int i = 0; i < dataNum; i++) {
			float valFloat = (valUint8[i] - tensor->params.zero_point) * tensor->params.scale;
			output[i] = valFloat;
		}
	} else {
		//const auto *valFloat = interpreter->typed_tensor<float>(index);
		const auto *valFloat = pipelineTensor.data.f;
		for (int i = 0; i < dataNum; i++) {
			output[i] = valFloat[i];
		}
	}
}

static void readLabel(const char* filename, std::vector<std::string> & labels)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		printf("failed to read %s\n", filename);
		return;
	}
	std::string str;
	while (getline(ifs, str)) {
		labels.push_back(str);
	}
}


int main()
{
	/*** Initialize ***/
	/* Prepare Edge TPU contexts */
	auto contexts = PrepareEdgeTpuContexts(MODEL_SEGMENTS_NUM, EdgeTpuType::kAny);
	TFLITE_MINIMAL_CHECK(!contexts.empty());

	/* Build model pipeline runner */
	std::vector<std::string> modelPathList = getSegmentedModelPathList(MODEL_FILENAME_BASE, MODEL_SEGMENTS_NUM);
	std::vector<std::unique_ptr<tflite::Interpreter>> managedInterpreters(MODEL_SEGMENTS_NUM);
	std::vector<tflite::Interpreter*> interpreters(MODEL_SEGMENTS_NUM);
	std::vector<std::unique_ptr<tflite::FlatBufferModel>> models(MODEL_SEGMENTS_NUM);
	for (int i = 0; i < MODEL_SEGMENTS_NUM; i++) {
		models[i] = tflite::FlatBufferModel::BuildFromFile(modelPathList[i].c_str());
		TFLITE_MINIMAL_CHECK(models[i] != nullptr);
		managedInterpreters[i] = BuildEdgeTpuInterpreter(*(models[i]), contexts[i].get());
		TFLITE_MINIMAL_CHECK(managedInterpreters[i] != nullptr);
		interpreters[i] = managedInterpreters[i].get();
	}
	std::unique_ptr<coral::PipelinedModelRunner> runner(new coral::PipelinedModelRunner(interpreters));


	/*** Process for each frame ***/
	/* Define logics */
	auto requestProducer = [&interpreters, &runner]() {
		const tflite::Interpreter* interpreterFirst = interpreters[0];
		for (int i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; ++i) {
			/* Read input image data */
			cv::Mat originalImage = cv::imread(RESOURCE_DIR"/parrot.jpg");
			cv::imshow("test", originalImage); cv::waitKey(1);
			/* Pre-process */
			cv::Mat inputImage;
			cv::cvtColor(originalImage, inputImage, cv::COLOR_BGR2RGB);
			cv::resize(inputImage, inputImage, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
			inputImage.convertTo(inputImage, CV_8UC3);

			/* Set input data to pipeline tensor */
			std::vector<coral::PipelineTensor> request;		// use vector but the size is one because the model has only one input
			request.push_back(createPipelineTensor(interpreterFirst, runner->GetInputTensorAllocator(), interpreterFirst->inputs()[0], inputImage.data, sizeof(uint8_t) * 1 * inputImage.rows  * inputImage.cols * inputImage.channels()));

			/* Push input tensors to be processed by the pipeline. */
			runner->Push(request);
		}

		/* Send signals shutting down the pipeline */
		runner->Push({});
	};

	auto requestConsumer = [&interpreters, &runner]() {
		const tflite::Interpreter* interpreterLast = interpreters[MODEL_SEGMENTS_NUM - 1];
		/* read label */
		std::vector<std::string> labels;
		readLabel(LABEL_NAME, labels);

		std::vector<coral::PipelineTensor> response;
		while (runner->Pop(&response)) {
			/* Retrieve the result */
			std::vector<float> outputScoreList;
			extractTensorAsFloatVector(interpreterLast, interpreterLast->outputs()[0], response[0], outputScoreList);
			int maxIndex = (int)(std::max_element(outputScoreList.begin(), outputScoreList.end()) - outputScoreList.begin());
			auto maxScore = *std::max_element(outputScoreList.begin(), outputScoreList.end());
			printf("%s (%.3f)\n", labels[maxIndex].c_str(), maxScore);

			/* Release memory*/
			coral::FreeTensors(response, runner->GetOutputTensorAllocator());
			response.clear();
		}
	};

	/* Start process for each frame */
	const auto& t0 = std::chrono::steady_clock::now();
	auto producer = std::thread(requestProducer);
	auto consumer = std::thread(requestConsumer);
	producer.join();
	consumer.join();
	const auto& t1 = std::chrono::steady_clock::now();
	std::chrono::duration<double> timeSpan = t1 - t0;
	printf("Total time = %f [msec]\n", timeSpan.count() * 1000.0 / LOOP_NUM_FOR_TIME_MEASUREMENT);

	cv::waitKey(-1);
	cv::destroyAllWindows();

	return 0;
}
