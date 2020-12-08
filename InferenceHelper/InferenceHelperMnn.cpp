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

/* for MNN */
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/AutoTime.hpp>

/* for My modules */
#include "CommonHelper.h"
#include "InferenceHelperMnn.h"

/*** Macro ***/
#define TAG "InferenceHelperMnn"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Function ***/
InferenceHelperMnn::InferenceHelperMnn()
{
	m_numThread = 1;
}

InferenceHelperMnn::~InferenceHelperMnn()
{
}

int32_t InferenceHelperMnn::setNumThread(const int32_t numThread)
{
	m_numThread = numThread;
	return RET_OK;
}

int32_t InferenceHelperMnn::setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps)
{
	PRINT("[WARNING] This method is not supported\n");
	return RET_OK;
}

int32_t InferenceHelperMnn::initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	/*** Create network ***/
	m_net.reset(MNN::Interpreter::createFromFile(modelFilename.c_str()));
	if (!m_net) {
		PRINT_E("Failed to load model file (%s)\n", modelFilename.c_str());
		return RET_ERR;
	}

	MNN::ScheduleConfig scheduleConfig;
	scheduleConfig.type = MNN_FORWARD_AUTO;
	scheduleConfig.numThread = m_numThread;
	// BackendConfig bnconfig;
	// bnconfig.precision = BackendConfig::Precision_Low;
	// config.backendConfig = &bnconfig;
	m_session = m_net->createSession(scheduleConfig);
	if (!m_session) {
		PRINT_E("Failed to create session\n");
		return RET_ERR;
	}

	/* Check tensor info fits the info from model */
	for (auto& inputTensorInfo : inputTensorInfoList) {
		auto inputTensor = m_net->getSessionInput(m_session, inputTensorInfo.name.c_str());
		if (inputTensor == nullptr) {
			PRINT_E("Invalid input name (%s)\n", inputTensorInfo.name.c_str());
			return RET_ERR;
		}
		if ((inputTensor->getType().code == halide_type_float) && (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_FP32)) {
			/* OK */
		} else if ((inputTensor->getType().code == halide_type_uint) && (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_UINT8)) {
			/* OK */
		} else {
			PRINT_E("Incorrect input tensor type (%d, %d)\n", inputTensor->getType().code, inputTensorInfo.tensorType);
			return RET_ERR;
		}
		if ((inputTensor->channel() != -1) && (inputTensor->height() != -1) && (inputTensor->width() != -1)) {
			if (inputTensorInfo.tensorDims.channel != -1) {
				if ((inputTensor->channel() == inputTensorInfo.tensorDims.channel) && (inputTensor->height() == inputTensorInfo.tensorDims.height) && (inputTensor->width() == inputTensorInfo.tensorDims.width)) {
					/* OK */
				} else {
					PRINT_E("Incorrect input tensor size\n");
					return RET_ERR;
				}
			} else {
				PRINT("Input tensor size is set from the model\n");
				inputTensorInfo.tensorDims.channel = inputTensor->channel();
				inputTensorInfo.tensorDims.height = inputTensor->height();
				inputTensorInfo.tensorDims.width = inputTensor->width();
			}
		} else {
			if (inputTensorInfo.tensorDims.channel != -1) {
				PRINT("Input tensor size is resized\n");
				/* In case the input size  is not fixed */
				m_net->resizeTensor(inputTensor, { 1, inputTensorInfo.tensorDims.channel, inputTensorInfo.tensorDims.height, inputTensorInfo.tensorDims.width });
				m_net->resizeSession(m_session);
			} else {
				PRINT_E("Model input size is not set\n");
				return RET_ERR;
			}
		}
	}
	for (const auto& outputTensorInfo : outputTensorInfoList) {
		auto outputTensor = m_net->getSessionOutput(m_session, outputTensorInfo.name.c_str());
		if (outputTensor == nullptr) {
			PRINT_E("Invalid output name (%s)\n", outputTensorInfo.name.c_str());
			return RET_ERR;
		}
		/* Output size is set when run inference later */
	}

	/* Convert normalize parameter to speed up */
	for (auto& inputTensorInfo : inputTensorInfoList) {
		convertNormalizeParameters(inputTensorInfo);
	}


	return RET_OK;
};


int32_t InferenceHelperMnn::finalize(void)
{
	m_net->releaseSession(m_session);
	m_net->releaseModel();
	m_net.reset();
	m_outMatList.clear();
	return RET_ERR;
}

int32_t InferenceHelperMnn::preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList)
{
	for (const auto& inputTensorInfo : inputTensorInfoList) {
		auto inputTensor = m_net->getSessionInput(m_session, inputTensorInfo.name.c_str());
		if (inputTensor == nullptr) {
			PRINT_E("Invalid input name (%s)\n", inputTensorInfo.name.c_str());
			return RET_ERR;
		}
		if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_IMAGE) {
			/* Crop */
			if ((inputTensorInfo.imageInfo.width != inputTensorInfo.imageInfo.cropWidth) || (inputTensorInfo.imageInfo.height != inputTensorInfo.imageInfo.cropHeight)) {
				PRINT_E("Crop is not supported\n");
				return RET_ERR;
			}

			MNN::CV::ImageProcess::Config imageProcessconfig;
			/* Convert color type */
			if ((inputTensorInfo.imageInfo.channel == 3) && (inputTensorInfo.tensorDims.channel == 3)) {
				imageProcessconfig.sourceFormat = (inputTensorInfo.imageInfo.isBGR) ? MNN::CV::BGR : MNN::CV::RGB;
				if (inputTensorInfo.imageInfo.swapColor) {
					imageProcessconfig.destFormat = (inputTensorInfo.imageInfo.isBGR) ? MNN::CV::RGB : MNN::CV::BGR;
				} else {
					imageProcessconfig.destFormat = (inputTensorInfo.imageInfo.isBGR) ? MNN::CV::BGR : MNN::CV::RGB;
				}
			} else if ((inputTensorInfo.imageInfo.channel == 1) && (inputTensorInfo.tensorDims.channel == 1)) {
				imageProcessconfig.sourceFormat = MNN::CV::GRAY;
				imageProcessconfig.destFormat = MNN::CV::GRAY;
			} else if ((inputTensorInfo.imageInfo.channel == 3) && (inputTensorInfo.tensorDims.channel == 1)) {
				imageProcessconfig.sourceFormat = (inputTensorInfo.imageInfo.isBGR) ? MNN::CV::BGR : MNN::CV::RGB;
				imageProcessconfig.destFormat = MNN::CV::GRAY;
			} else if ((inputTensorInfo.imageInfo.channel == 1) && (inputTensorInfo.tensorDims.channel == 3)) {
				imageProcessconfig.sourceFormat = MNN::CV::GRAY;
				imageProcessconfig.destFormat = MNN::CV::BGR;
			} else {
				PRINT_E("Unsupported color conversion (%d, %d)\n", inputTensorInfo.imageInfo.channel, inputTensorInfo.tensorDims.channel);
				return RET_ERR;
			}

			/* Normalize image */
			std::memcpy(imageProcessconfig.mean, inputTensorInfo.normalize.mean, sizeof(imageProcessconfig.mean));
			std::memcpy(imageProcessconfig.normal, inputTensorInfo.normalize.norm, sizeof(imageProcessconfig.normal));
			
			/* Resize image */
			imageProcessconfig.filterType = MNN::CV::BILINEAR;
			MNN::CV::Matrix trans;
			trans.setScale(static_cast<float>(inputTensorInfo.imageInfo.cropWidth) / inputTensorInfo.tensorDims.width, static_cast<float>(inputTensorInfo.imageInfo.cropHeight) / inputTensorInfo.tensorDims.height);

			/* Do pre-process */
			std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(imageProcessconfig));
			pretreat->setMatrix(trans);
			pretreat->convert(static_cast<uint8_t*>(inputTensorInfo.data), inputTensorInfo.imageInfo.cropWidth, inputTensorInfo.imageInfo.cropHeight, 0, inputTensor);
		} else if ( (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NHWC) || (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NCHW) ) {
			std::unique_ptr<MNN::Tensor> tensor;
			if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NHWC) {
				tensor.reset(new MNN::Tensor(inputTensor, MNN::Tensor::TENSORFLOW));
			} else {
				tensor.reset(new MNN::Tensor(inputTensor, MNN::Tensor::CAFFE));
			}
			if (tensor->getType().code == halide_type_float) {
				for (int32_t i = 0; i < inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height * inputTensorInfo.tensorDims.channel; i++) {
					tensor->host<float>()[i] = static_cast<float*>(inputTensorInfo.data)[i];
				}
			} else {
				for (int32_t i = 0; i < inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height * inputTensorInfo.tensorDims.channel; i++) {
					tensor->host<uint8_t>()[i] = static_cast<uint8_t*>(inputTensorInfo.data)[i];
				}
			}
			inputTensor->copyFromHostTensor(tensor.get());
		} else {
			PRINT_E("Unsupported data type (%d)\n", inputTensorInfo.dataType);
			return RET_ERR;
		}
	}
	return RET_OK;
}

int32_t InferenceHelperMnn::invoke(std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	m_net->runSession(m_session);

	m_outMatList.clear();
	for (auto& outputTensorInfo : outputTensorInfoList) {
		auto outputTensor = m_net->getSessionOutput(m_session, outputTensorInfo.name.c_str());
		if (outputTensor == nullptr) {
			PRINT_E("Invalid output name (%s)\n", outputTensorInfo.name.c_str());
			return RET_ERR;
		}

		auto dimType = outputTensor->getDimensionType();
		std::unique_ptr<MNN::Tensor> outputUser(new MNN::Tensor(outputTensor, dimType));
		outputTensor->copyToHostTensor(outputUser.get());
		auto type = outputUser->getType();
		if (type.code == halide_type_float) {
			outputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
			outputTensorInfo.data = outputUser->host<float>();
		} else if (type.code == halide_type_uint && type.bytes() == 1) {
			outputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_UINT8;
			outputTensorInfo.data = outputUser->host<uint8_t>();
		} else {
			PRINT_E("Unexpected data type\n");
			return RET_ERR;
		}
		
		outputTensorInfo.tensorDims.batch = (std::max)(outputUser->batch(), 1);
		outputTensorInfo.tensorDims.channel = (std::max)(outputUser->channel(), 1);
		outputTensorInfo.tensorDims.height = (std::max)(outputUser->height(), 1);
		outputTensorInfo.tensorDims.width = (std::max)(outputUser->width(), 1);
		m_outMatList.push_back(std::move(outputUser));	// store data in member variable so that data keep exist
	}

	return RET_OK;
}

void InferenceHelperMnn::convertNormalizeParameters(InputTensorInfo& inputTensorInfo)
{
	if (inputTensorInfo.dataType != InputTensorInfo::DATA_TYPE_IMAGE) return;

#if 0
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm  = src * 1 / (255 * norm) - (mean / norm) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] /= inputTensorInfo.normalize.norm[i];
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif
#if 1
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif
}