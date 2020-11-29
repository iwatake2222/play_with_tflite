/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <memory>

/* for TensorRT */
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "TensorRT/common.h"

/* for My modules */
#include "CommonHelper.h"
#include "InferenceHelperTensorRt.h"

/*** Macro ***/
#define TAG "InferenceHelperTensorRt"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Setting */
#define USE_FP16
// #define USE_INT8

#define OPT_MAX_WORK_SPACE_SIZE (1 << 30)
#define OPT_AVG_TIMING_ITERATIONS 8
#define OPT_MIN_TIMING_ITERATIONS 4

#ifdef USE_INT8
/* ★ Modify the following (use the same parameter as the model. Also, ppm must be the same size but not normalized.) */
#define CAL_DIR        "/home/pi/play_with_tensorrt/InferenceHelper/TensorRT/calibration/sample_ppm"
#define CAL_LIST_FILE  "list.txt"
#define CAL_INPUT_NAME "data"
#define CAL_BATCH_SIZE 10
#define CAL_NB_BATCHES 2
#define CAL_IMAGE_C    3
#define CAL_IMAGE_H    224
#define CAL_IMAGE_W    224
/* 0 ~ 1.0 */
// #define CAL_SCALE      (1.0 / 255.0)
// #define CAL_BIAS       (0.0)
/* -2.25 ~ 2.25 */
#define CAL_SCALE      (1.0 / (255.0 * 0.225))
#define CAL_BIAS       (0.45 / 0.225)

/* include BatchStream.h after defining parameters */
#include "TensorRT/BatchStream.h"
#include "TensorRT/EntropyCalibrator.h"
#endif

/*** Function ***/
InferenceHelperTensorRt::InferenceHelperTensorRt()
{
	m_numThread = 1;
}

int32_t InferenceHelperTensorRt::setNumThread(const int32_t numThread)
{
	m_numThread = numThread;
	return RET_OK;
}

int32_t InferenceHelperTensorRt::setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps)
{
	PRINT("[WARNING] This method is not supported\n");
	return RET_OK;
}

int32_t InferenceHelperTensorRt::initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	/* check model format */
	bool isTrtModel = false;
	bool isOnnxModel = false;
	// bool isUffModel = false;	// todo
	std::string trtModelFilename = std::string(modelFilename);
	if (modelFilename.find(".onnx") != std::string::npos) {
		isOnnxModel = true;
		trtModelFilename = trtModelFilename.replace(trtModelFilename.find(".onnx"), std::string(".onnx").length(), ".trt\0");
	} else if (trtModelFilename.find(".trt") != std::string::npos) {
		isTrtModel = true;
	} else {
		PRINT_E("unsupoprted file format (%s)\n", modelFilename.c_str());
		return RET_ERR;
	}

	/* create runtime and engine from model file */
	if (isTrtModel) {
		std::string buffer;
		std::ifstream stream(modelFilename, std::ios::binary);
		if (stream) {
			stream >> std::noskipws;
			copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), back_inserter(buffer));
		}
		m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()), samplesCommon::InferDeleter());
		m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size(), NULL), samplesCommon::InferDeleter());
		stream.close();
		if (!m_engine) {
			PRINT_E("Failed to create engine (%s)\n", modelFilename.c_str());
			return RET_ERR;
		}
		m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext(), samplesCommon::InferDeleter());
		if (!m_context) {
			PRINT_E("Failed to create context (%s)\n", modelFilename.c_str());
			return RET_ERR;
		}
	} else if (isOnnxModel) {
		/* create a TensorRT model from another format */
		auto builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()), samplesCommon::InferDeleter());
#if 0
		/* For older version of JetPack */
		auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetwork(), samplesCommon::InferDeleter());
#else
		const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch), samplesCommon::InferDeleter());
#endif
		auto config = std::shared_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig(), samplesCommon::InferDeleter());

		auto parserOnnx = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()), samplesCommon::InferDeleter());
		if (!parserOnnx->parseFromFile(modelFilename.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING)) {
			PRINT_E("Failed to parse onnx file (%s)", modelFilename.c_str());
			return RET_ERR;
		}

		builder->setMaxBatchSize(1);
		config->setMaxWorkspaceSize(OPT_MAX_WORK_SPACE_SIZE);
		config->setAvgTimingIterations(OPT_AVG_TIMING_ITERATIONS);
		config->setMinTimingIterations(OPT_MIN_TIMING_ITERATIONS) ;

#if defined(USE_FP16)
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
		std::vector<std::string> dataDirs;
		dataDirs.push_back(CAL_DIR);
		nvinfer1::DimsNCHW imageDims{CAL_BATCH_SIZE, CAL_IMAGE_C, CAL_IMAGE_H, CAL_IMAGE_W};
		BatchStream calibrationStream(CAL_BATCH_SIZE, CAL_NB_BATCHES, imageDims, CAL_LIST_FILE, dataDirs);
		auto calibrator = std::unique_ptr<nvinfer1::IInt8Calibrator>(new Int8EntropyCalibrator2<BatchStream>(calibrationStream, 0, "my_model", CAL_INPUT_NAME));
		config->setInt8Calibrator(calibrator.get());
#endif 

		m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
		if (!m_engine) {
			PRINT_E("Failed to create engine (%s)\n", modelFilename.c_str());
			return RET_ERR;
		}
		m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext(), samplesCommon::InferDeleter());
		if (!m_context) {
			PRINT_E("Failed to create context (%s)\n", modelFilename.c_str());
			return RET_ERR;
		}
#if 1
		/* save serialized model for next time */
		nvinfer1::IHostMemory* trtModelStream = m_engine->serialize();
		std::ofstream ofs(std::string(trtModelFilename), std::ios::out | std::ios::binary);
		ofs.write((char*)(trtModelStream->data()), trtModelStream->size());
		ofs.close();
		trtModelStream->destroy();
#endif
	}

	/* Allocate host/device buffers and assign to tensor info */
	for (auto& inputTensorInfo : inputTensorInfoList) {
		inputTensorInfo.id = -1;	// not assigned
	}
	for (auto& outputTensorInfo : outputTensorInfoList) {
		outputTensorInfo.id = -1;	// not assigned
	}
	if (allocateBuffers(inputTensorInfoList, outputTensorInfoList) != RET_OK) {
		return RET_ERR;
	}
	/* Check if the tensor is assigned (exists in the model) */
	for (auto& inputTensorInfo : inputTensorInfoList) {
		if (inputTensorInfo.id == -1) {
			PRINT_E("Input tensor doesn't exist in the model (%s)\n", inputTensorInfo.name.c_str());
			return RET_ERR;
		}
	}
	for (auto& outputTensorInfo : outputTensorInfoList) {
		if (outputTensorInfo.id == -1) {
			PRINT_E("Output tensor doesn't exist in the model (%s)\n", outputTensorInfo.name.c_str());
			return RET_ERR;
		}
	}

	return RET_OK;
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
			return RET_ERR;
		}
	}

	for (auto p : m_bufferListGPU) {
		cudaFree(p);
	}

	return RET_OK;
}

int32_t InferenceHelperTensorRt::preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList)
{
	for (const auto& inputTensorInfo : inputTensorInfoList) {
		if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_IMAGE) {
			if ((inputTensorInfo.imageInfo.width != inputTensorInfo.imageInfo.cropWidth) || (inputTensorInfo.imageInfo.height != inputTensorInfo.imageInfo.cropHeight)) {
				PRINT_E("Crop is not supported\n");
				return  RET_ERR;
			}
			if ((inputTensorInfo.imageInfo.cropWidth != inputTensorInfo.tensorDims.width) || (inputTensorInfo.imageInfo.cropHeight != inputTensorInfo.tensorDims.height)) {
				PRINT_E("Resize is not supported\n");
				return  RET_ERR;
			}
			if (inputTensorInfo.imageInfo.channel != inputTensorInfo.tensorDims.channel) {
				PRINT_E("Color conversion is not supported\n");
				return  RET_ERR;
			}

			/* Normalize image */
			if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_FP32) {
				/* convert NHWC to NCHW */
				float_t *dst = (float_t*)(m_bufferListCPU[inputTensorInfo.id].first);
				uint8_t *src = (uint8_t*)(inputTensorInfo.data);
				if (m_bufferListCPU[inputTensorInfo.id].second != 4 * inputTensorInfo.imageInfo.width * inputTensorInfo.imageInfo.height * inputTensorInfo.imageInfo.channel) {
					PRINT_E("Data size doesn't match\n");
					return  RET_ERR;
				}
#pragma omp parallel for num_threads(m_numThread)
				for (int32_t c = 0; c < inputTensorInfo.tensorDims.channel; c++) {
					for (int32_t i = 0; i < inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height; i++) {
#if 1
						dst[c * inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height + i] = 
							(src[i * inputTensorInfo.tensorDims.channel + c] - inputTensorInfo.normalize.mean[c]) * inputTensorInfo.normalize.norm[c];
#else
						dst[c * inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height + i] = 
							(src[i * inputTensorInfo.tensorDims.channel + c] / 255.0f - inputTensorInfo.normalize.mean[c]) / inputTensorInfo.normalize.norm[c];
#endif
					}
				}
			} else if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_UINT8) {
				/* convert NHWC to NCHW */
				uint8_t *dst = (uint8_t*)(m_bufferListCPU[inputTensorInfo.id].first);
				uint8_t *src = (uint8_t*)(inputTensorInfo.data);
				if (m_bufferListCPU[inputTensorInfo.id].second != 1 * inputTensorInfo.imageInfo.width * inputTensorInfo.imageInfo.height * inputTensorInfo.imageInfo.channel) {
					PRINT_E("Data size doesn't match\n");
					return  RET_ERR;
				}
#pragma omp parallel for num_threads(m_numThread)
				for (int32_t c = 0; c < inputTensorInfo.tensorDims.channel; c++) {
					for (int32_t i = 0; i < inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height; i++) {
						dst[c * inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height + i] = src[i * inputTensorInfo.tensorDims.channel + c];
					}
				}
			} else {
				PRINT_E("Unsupported tensorType (%d)\n", inputTensorInfo.tensorType);
				return RET_ERR;
			}

		} else if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NHWC) {
				/* convert NHWC to NCHW */
				uint8_t *dst = (uint8_t*)(m_bufferListCPU[inputTensorInfo.id].first);
				uint8_t *src = (uint8_t*)(inputTensorInfo.data);
#pragma omp parallel for
				for (int32_t c = 0; c < inputTensorInfo.tensorDims.channel; c++) {
					for (int32_t i = 0; i < inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height; i++) {
						dst[c * inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height + i] = src[i * inputTensorInfo.tensorDims.channel + c];
					}
				}
		} else if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NCHW) {
			uint8_t *dst = (uint8_t*)(m_bufferListCPU[inputTensorInfo.id].first);
			uint8_t *src = (uint8_t*)(inputTensorInfo.data);
			memcpy(dst, src, m_bufferListCPU[inputTensorInfo.id].second);
		} else {
			PRINT_E("Unsupported tensorType (%d)\n", inputTensorInfo.tensorType);
			return RET_ERR;
		}

	}
	return RET_OK;
}

int32_t InferenceHelperTensorRt::invoke(std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	for (int i = 0; i < (int)m_bufferListCPU.size(); i++) {
		if (m_engine->bindingIsInput(i)) {
			cudaMemcpyAsync(m_bufferListGPU[i], m_bufferListCPU[i].first, m_bufferListCPU[i].second, cudaMemcpyHostToDevice, stream);
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

	(void)outputTensorInfoList;	// no need to set output data, because the ptr to output data is already set at initialize

	return RET_OK;
}

int32_t InferenceHelperTensorRt::allocateBuffers(std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	int32_t numOfInOut = m_engine->getNbBindings();
	PRINT("numOfInOut = %d\n", numOfInOut);

	for (int32_t i = 0; i < numOfInOut; i++) {
		PRINT("tensor[%d]->name: %s\n", i, m_engine->getBindingName(i));
		PRINT("  is input = %d\n", m_engine->bindingIsInput(i));
		int32_t dataSize = 1;
		const auto dims = m_engine->getBindingDimensions(i);
		for (int32_t i = 0; i < dims.nbDims; i++) {
			PRINT("  dims.d[%d] = %d\n", i, dims.d[i]);
			dataSize *= dims.d[i];
		}
		const auto dataType = m_engine->getBindingDataType(i);
		PRINT("  dataType = %d\n", static_cast<int32_t>(dataType));

		void* bufferCPU = nullptr;
		void* bufferGPU = nullptr;
		switch (dataType) {
		case nvinfer1::DataType::kFLOAT:
		case nvinfer1::DataType::kHALF:
		case nvinfer1::DataType::kINT32:
			bufferCPU = new float_t[dataSize];
			m_bufferListCPU.push_back(std::pair<void*,int32_t>(bufferCPU, dataSize * sizeof(float_t)));
			cudaMalloc(&bufferGPU, dataSize * sizeof(float_t));
			m_bufferListGPU.push_back(bufferGPU);
			break;
		case nvinfer1::DataType::kINT8:
			bufferCPU = new int8_t[dataSize];
			m_bufferListCPU.push_back(std::pair<void*,int32_t>(bufferCPU, dataSize * sizeof(int8_t)));
			cudaMalloc(&bufferGPU, dataSize * sizeof(int8_t));
			m_bufferListGPU.push_back(bufferGPU);
			break;
		default:
			PRINT_E("Unsupported datatype (%d)\n", static_cast<int32_t>(dataType));
			return RET_ERR;
		}

		if(m_engine->bindingIsInput(i)) {
			for (auto& inputTensorInfo : inputTensorInfoList) {
				int32_t id = m_engine->getBindingIndex(inputTensorInfo.name.c_str());
				if (id == i) {
					inputTensorInfo.id = id;
					for (int32_t i = 0; i < dims.nbDims; i++) {
						if (((i == 0) && (inputTensorInfo.tensorDims.batch == dims.d[i]))
							|| ((i == 1) && (inputTensorInfo.tensorDims.channel == dims.d[i]))
							|| ((i == 2) && (inputTensorInfo.tensorDims.height == dims.d[i]))
							|| ((i == 3) && (inputTensorInfo.tensorDims.width == dims.d[i]))) {
							/* OK */
						} else {
							PRINT_E("Input Tensor size doesn't match\n");
							return RET_ERR;
						}
					}
					if (((inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_UINT8) && (dataType == nvinfer1::DataType::kINT8))
						|| ((inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_FP32) && (dataType == nvinfer1::DataType::kFLOAT))
						|| ((inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_INT32) && (dataType == nvinfer1::DataType::kINT32))) {
							/* OK */
					} else {
						PRINT_E("Input Tensor type doesn't match\n");
						return RET_ERR;
					}
				}
			}
		} else {
			for (auto& outputTensorInfo : outputTensorInfoList) {
				int32_t id = m_engine->getBindingIndex(outputTensorInfo.name.c_str());
				if (id == i) {
					outputTensorInfo.id = id;
					for (int32_t i = 0; i < dims.nbDims; i++) {
						if (i == 0) outputTensorInfo.tensorDims.batch = dims.d[i];
						if (i == 1) outputTensorInfo.tensorDims.channel = dims.d[i];
						if (i == 2) outputTensorInfo.tensorDims.height = dims.d[i];
						if (i == 3) outputTensorInfo.tensorDims.width = dims.d[i];
					}
					if (((outputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_UINT8) && (dataType == nvinfer1::DataType::kINT8))
						|| ((outputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_FP32) && (dataType == nvinfer1::DataType::kFLOAT))
						|| ((outputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_INT32) && (dataType == nvinfer1::DataType::kINT32))) {
							/* OK */
					} else {
						PRINT_E("Output Tensor type doesn't match\n");
						return RET_ERR;
					}
					if (dataType == nvinfer1::DataType::kINT8) {
						outputTensorInfo.quant.scale = 1.0;			// todo
						outputTensorInfo.quant.zeroPoint = 0.0;
					}
					outputTensorInfo.data = bufferCPU;
				}
			}
		}
	}

	return RET_OK;
}
