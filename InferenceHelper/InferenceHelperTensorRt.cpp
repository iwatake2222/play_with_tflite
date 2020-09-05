/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>

/* for TensorRT */
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "InferenceHelperTensorRt.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define _PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define _PRINT(...) printf(__VA_ARGS__)
#endif
#define PRINT(...) _PRINT("[InferenceHelperTensorRt] " __VA_ARGS__)

#define CHECK(x)                              \
  if (!(x)) {                                                \
	PRINT("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }


/* very very simple Logger class */
class Logger : public nvinfer1::ILogger
{
public:
	Logger(Severity severity = Severity::kWARNING)
		: reportableSeverity(severity)
	{
	}

	void log(Severity severity, const char* msg) override
	{
		if (severity > reportableSeverity) return;
		PRINT("[ERROR] %s\n", msg);
	}

	Severity reportableSeverity;
};


/*** Function ***/
InferenceHelperTensorRt::InferenceHelperTensorRt()
{
}

int InferenceHelperTensorRt::initialize(const char *modelFilename, int numThreads, std::vector<std::pair<const char*, const void*>> customOps)
{
	PRINT("[WARNING] This method is not supported\n");
	return -1;
}

int InferenceHelperTensorRt::initialize(const char *modelFilename, int numThreads)
{
	bool isOnnxModel = false;
	bool isTrtModel = false;
	std::string trtModelFilename = std::string(modelFilename);
	transform (trtModelFilename.begin(), trtModelFilename.end(), trtModelFilename.begin(), tolower);
	if (trtModelFilename.find(".onnx") != std::string::npos) {
		isOnnxModel = true;
		trtModelFilename = trtModelFilename.replace(trtModelFilename.find(".onnx"), std::string(".onnx").length(), ".trt\0");
	} else if (trtModelFilename.find(".trt") != std::string::npos) {
		isTrtModel = true;
	} else {
		PRINT("[ERROR] unsupoprted file format: %s\n", modelFilename);
	}

	/* create m_runtime and m_engine from model file */
	m_logger = new Logger;
	m_runtime = nvinfer1::createInferRuntime(*m_logger);
	m_engine = NULL;

	if (isOnnxModel) {
		/* create a TensorRT model from the onnx model */
		nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(*m_logger);
		nvinfer1::IBuilderConfig *builderConfig = builder->createBuilderConfig();
#if 0
		/* For older version of JetPack */
		nvinfer1::INetworkDefinition* network = builder->createNetwork();
#else
		const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
#endif
		auto parser = nvonnxparser::createParser(*network, *m_logger);

		if (!parser->parseFromFile(modelFilename, (int)nvinfer1::ILogger::Severity::kWARNING)) {
			PRINT("[ERROR] failed to parse onnx file");
			return -1;
		}

		builder->setMaxBatchSize(1);
		builderConfig->setMaxWorkspaceSize(512 << 20);
		builderConfig->setAvgTimingIterations(4);
		builderConfig->setMinTimingIterations(4) ;
		builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);

		m_engine = builder->buildEngineWithConfig(*network, *builderConfig);

		parser->destroy();
		network->destroy();
		builder->destroy();
		builderConfig->destroy();

	#if 1
		/* save serialized model for next time */
		nvinfer1::IHostMemory* trtModelStream = m_engine->serialize();
		std::ofstream ofs(std::string(trtModelFilename), std::ios::out | std::ios::binary);
		ofs.write((char*)(trtModelStream->data()), trtModelStream->size());
		ofs.close();
		trtModelStream->destroy();
	#endif

	} else if (isTrtModel) {
		std::string buffer;
		std::ifstream stream(modelFilename, std::ios::binary);
		if (stream) {
			stream >> std::noskipws;
			copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), back_inserter(buffer));
		}
		m_engine = m_runtime->deserializeCudaEngine(buffer.data(), buffer.size(), NULL);
		stream.close();
	}

	CHECK(m_engine != NULL);
	m_context = m_engine->createExecutionContext();
	CHECK(m_context != NULL);

	/* Allocate host/device buffers beforehand */
	allocateBuffers();

	return 0;
}

int InferenceHelperTensorRt::finalize(void)
{
	int numOfInOut = m_engine->getNbBindings();
	for (int i = 0; i < numOfInOut; i++) {
		const auto dataType = m_engine->getBindingDataType(i);
		switch (dataType) {
		case nvinfer1::DataType::kFLOAT:
		case nvinfer1::DataType::kHALF:
		case nvinfer1::DataType::kINT32:
			delete[] (float*)(m_bufferListCPUReserved[i].first);
			break;
		case nvinfer1::DataType::kINT8:
			delete[] (int*)(m_bufferListCPUReserved[i].first);
			break;
		default:
			CHECK(false);
		}
	}

	for (auto p : m_bufferListGPU) {
		cudaFree(p);
	}

	m_context->destroy();
	m_engine->destroy();
	m_runtime->destroy();
	delete m_logger;
	
	return 0;
}

int InferenceHelperTensorRt::invoke(void)
{
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	for (int i = 0; i < (int)m_bufferListCPU.size(); i++) {
		if (m_engine->bindingIsInput(i)) {
			cudaMemcpyAsync(m_bufferListGPU[i], m_bufferListCPU[i].first, m_bufferListCPU[i].second, cudaMemcpyHostToDevice, stream);
			printf("m_bufferListCPU[i].first = %p\n", m_bufferListCPU[i].first);
		}
	}
	m_context->enqueue(1, &m_bufferListGPU[0], stream, NULL);
	for (int i = 0; i < (int)m_bufferListCPU.size(); i++) {
		if (!m_engine->bindingIsInput(i)) {
			cudaMemcpyAsync(m_bufferListCPU[i].first, m_bufferListGPU[i], m_bufferListCPU[i].second, cudaMemcpyDeviceToHost, stream);
		}
	}
	cudaStreamSynchronize(stream);

	cudaStreamDestroy(stream);

	return 0;
}


int InferenceHelperTensorRt::getTensorByName(const char *name, TensorInfo *tensorInfo)
{
	int index = m_engine->getBindingIndex(name);
	if (index == -1) {
		PRINT("invalid name: %s\n", name);
		return -1;
	}

	return getTensorByIndex(index, tensorInfo);
}

int InferenceHelperTensorRt::getTensorByIndex(const int index, TensorInfo *tensorInfo)
{
	tensorInfo->index = index;

	const auto dims = m_engine->getBindingDimensions(index);
	for (int i = 0; i < dims.nbDims; i++) {
		tensorInfo->dims.push_back(dims.d[i]);
	}

	const auto dataType = m_engine->getBindingDataType(index);
	switch (dataType) {
	case nvinfer1::DataType::kFLOAT:
		tensorInfo->type = TensorInfo::TENSOR_TYPE_FP32;
		break;
	case nvinfer1::DataType::kHALF:
		tensorInfo->type = TensorInfo::TENSOR_TYPE_FP32;
		break;
	case nvinfer1::DataType::kINT8:
		tensorInfo->type = TensorInfo::TENSOR_TYPE_UINT8;
		tensorInfo->quant.scale = 1.0;			// todo
		tensorInfo->quant.zeroPoint = 0.0;
		break;
	case nvinfer1::DataType::kINT32:
		tensorInfo->type = TensorInfo::TENSOR_TYPE_INT32;
		break;
	default:
		CHECK(false);
	}
	tensorInfo->data = m_bufferListCPU[index].first;
	return 0;
}

int InferenceHelperTensorRt::setBufferToTensorByName(const char *name, void *data, const int dataSize)
{
	PRINT("[WARNING] This method is not tested\n");
	int index = m_engine->getBindingIndex(name);
	if (index == -1) {
		PRINT("invalid name: %s\n", name);
		return -1;
	}

	return setBufferToTensorByIndex(index, data, dataSize);
}

int InferenceHelperTensorRt::setBufferToTensorByIndex(const int index, void *data, const int dataSize)
{
	PRINT("[WARNING] This method is not tested\n");
	CHECK(m_bufferListCPU[index].second == dataSize);
	m_bufferListCPU[index].first = data;
	return 0;
}


void InferenceHelperTensorRt::allocateBuffers()
{
	int numOfInOut = m_engine->getNbBindings();
	PRINT("numOfInOut = %d\n", numOfInOut);

	for (int i = 0; i < numOfInOut; i++) {
		PRINT("tensor[%d]->name: %s\n", i, m_engine->getBindingName(i));
		PRINT("  is input = %d\n", m_engine->bindingIsInput(i));
		int dataSize = 1;
		const auto dims = m_engine->getBindingDimensions(i);
		for (int i = 0; i < dims.nbDims; i++) {
			PRINT("  dims.d[%d] = %d\n", i, dims.d[i]);
			dataSize *= dims.d[i];
		}
		const auto dataType = m_engine->getBindingDataType(i);
		PRINT("  dataType = %d\n", (int)dataType);

		void *bufferCPU = NULL;
		void* bufferGPU = NULL;
		switch (dataType) {
		case nvinfer1::DataType::kFLOAT:
		case nvinfer1::DataType::kHALF:
		case nvinfer1::DataType::kINT32:
			bufferCPU = new float[dataSize];
			CHECK(bufferCPU);
			m_bufferListCPUReserved.push_back(std::pair<void*,int>(bufferCPU, dataSize * 4));
			cudaMalloc(&bufferGPU, dataSize * 4);
			CHECK(bufferGPU);
			m_bufferListGPU.push_back(bufferGPU);
			break;
		case nvinfer1::DataType::kINT8:
			bufferCPU = new int[dataSize];
			CHECK(bufferCPU);
			m_bufferListCPUReserved.push_back(std::pair<void*,int>(bufferCPU, dataSize * 1));
			cudaMalloc(&bufferGPU, dataSize * 1);
			CHECK(bufferGPU);
			m_bufferListGPU.push_back(bufferGPU);
			break;
		default:
			CHECK(false);
		}
	}
	m_bufferListCPU = m_bufferListCPUReserved;
}
