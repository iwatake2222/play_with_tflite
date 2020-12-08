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

/* for ncnn */
#include "net.h"

/* for My modules */
#include "CommonHelper.h"
#include "InferenceHelperNcnn.h"

/*** Macro ***/
#define TAG "InferenceHelperNcnn"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Function ***/
InferenceHelperNcnn::InferenceHelperNcnn()
{
	m_numThread = 1;
}

InferenceHelperNcnn::~InferenceHelperNcnn()
{
}

int32_t InferenceHelperNcnn::setNumThread(const int32_t numThread)
{
	m_numThread = numThread;
	return RET_OK;
}

int32_t InferenceHelperNcnn::setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps)
{
	PRINT("[WARNING] This method is not supported\n");
	return RET_OK;
}

int32_t InferenceHelperNcnn::initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	/*** Create network ***/
	m_net.reset(new ncnn::Net());
	m_net->opt.use_fp16_arithmetic = true;
	m_net->opt.use_fp16_packed = true;
	m_net->opt.use_fp16_storage = true;

	std::string binFilename = modelFilename;
	if (modelFilename.find(".param") == std::string::npos) {
		PRINT_E("Invalid model param filename (%s)\n", modelFilename.c_str());
		return RET_ERR;
	}
	binFilename = binFilename.replace(binFilename.find(".param"), std::string(".param").length(), ".bin\0");
	if (m_net->load_param(modelFilename.c_str()) != 0) {
		PRINT_E("Failed to load model param file (%s)\n", modelFilename.c_str());
		return RET_ERR;
	}
	if (m_net->load_model(binFilename.c_str()) != 0) {
		PRINT_E("Failed to load model bin file (%s)\n", binFilename.c_str());
		return RET_ERR;
	}

	/* Convert normalize parameter to speed up */
	for (auto& inputTensorInfo : inputTensorInfoList) {
		convertNormalizeParameters(inputTensorInfo);
	}

	return RET_OK;
};


int32_t InferenceHelperNcnn::finalize(void)
{
	m_net.reset();
	m_inMatList.clear();
	m_outMatList.clear();
	return RET_ERR;
}

int32_t InferenceHelperNcnn::preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList)
{
	m_inMatList.clear();
	for (const auto& inputTensorInfo : inputTensorInfoList) {
		ncnn::Mat ncnnMat;
		if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_IMAGE) {
			/* Crop */
			if ((inputTensorInfo.imageInfo.width != inputTensorInfo.imageInfo.cropWidth) || (inputTensorInfo.imageInfo.height != inputTensorInfo.imageInfo.cropHeight)) {
				PRINT_E("Crop is not supported\n");
				return RET_ERR;
			}
			/* Convert color type */
			int32_t pixelType = 0;
			if ((inputTensorInfo.imageInfo.channel == 3) && (inputTensorInfo.tensorDims.channel == 3)) {
				pixelType = (inputTensorInfo.imageInfo.isBGR) ? ncnn::Mat::PIXEL_BGR : ncnn::Mat::PIXEL_RGB;
				if (inputTensorInfo.imageInfo.swapColor) {
					pixelType = (inputTensorInfo.imageInfo.isBGR) ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_RGB2BGR;
				}
			} else if ((inputTensorInfo.imageInfo.channel == 1) && (inputTensorInfo.tensorDims.channel == 1)) {
				pixelType = ncnn::Mat::PIXEL_GRAY;
			} else if ((inputTensorInfo.imageInfo.channel == 3) && (inputTensorInfo.tensorDims.channel == 1)) {
				pixelType = (inputTensorInfo.imageInfo.isBGR) ? ncnn::Mat::PIXEL_BGR2GRAY : ncnn::Mat::PIXEL_RGB2GRAY;
			} else if ((inputTensorInfo.imageInfo.channel == 1) && (inputTensorInfo.tensorDims.channel == 3)) {
				pixelType = ncnn::Mat::PIXEL_GRAY2RGB;
			} else {
				PRINT_E("Unsupported color conversion (%d, %d)\n", inputTensorInfo.imageInfo.channel, inputTensorInfo.tensorDims.channel);
				return RET_ERR;
			}
			
			if (inputTensorInfo.imageInfo.cropWidth == inputTensorInfo.tensorDims.width && inputTensorInfo.imageInfo.cropHeight == inputTensorInfo.tensorDims.height) {
				/* Convert to blob */
				ncnnMat = ncnn::Mat::from_pixels((uint8_t*)inputTensorInfo.data, pixelType, inputTensorInfo.imageInfo.width, inputTensorInfo.imageInfo.height);
			} else {
				/* Convert to blob with resize */
				ncnnMat = ncnn::Mat::from_pixels_resize((uint8_t*)inputTensorInfo.data, pixelType, inputTensorInfo.imageInfo.width, inputTensorInfo.imageInfo.height, inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height);
			}
			/* Normalize image */
			ncnnMat.substract_mean_normalize(inputTensorInfo.normalize.mean, inputTensorInfo.normalize.norm);
		} else if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NHWC) {
			PRINT_E("[ToDo] Unsupported data type (%d)\n", inputTensorInfo.dataType);
			ncnnMat = ncnn::Mat::from_pixels((uint8_t*)inputTensorInfo.data, inputTensorInfo.tensorDims.channel == 3 ? ncnn::Mat::PIXEL_RGB : ncnn::Mat::PIXEL_GRAY, inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height);
		} else if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NCHW) {
			ncnnMat = ncnn::Mat(inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height, inputTensorInfo.tensorDims.channel, inputTensorInfo.data);
		} else {
			PRINT_E("Unsupported data type (%d)\n", inputTensorInfo.dataType);
			return RET_ERR;
		}
		m_inMatList.push_back(std::pair<std::string, ncnn::Mat>(inputTensorInfo.name, ncnnMat));
	}
	return RET_OK;
}

int32_t InferenceHelperNcnn::invoke(std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	ncnn::Extractor ex = m_net->create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(m_numThread);
	for (const auto& inputMat : m_inMatList) {
		if (ex.input(inputMat.first.c_str(), inputMat.second) != 0) {
			PRINT_E("Input mat error (%s)\n", inputMat.first.c_str());
			return RET_ERR;
		}
	}

	m_outMatList.clear();
	for (auto& outputTensorInfo : outputTensorInfoList) {
		ncnn::Mat ncnnOut;
		if (ex.extract(outputTensorInfo.name.c_str(), ncnnOut) != 0) {
			PRINT_E("Output mat error (%s)\n", outputTensorInfo.name.c_str());
			return RET_ERR;
		}
		m_outMatList.push_back(ncnnOut);	// store ncnn mat in member variable so that data keep exist
		outputTensorInfo.data = ncnnOut.data;
		outputTensorInfo.tensorDims.batch = 1;
		outputTensorInfo.tensorDims.channel = ncnnOut.c;
		outputTensorInfo.tensorDims.height = ncnnOut.h;
		outputTensorInfo.tensorDims.width = ncnnOut.w;
	}

	return RET_OK;
}

void InferenceHelperNcnn::convertNormalizeParameters(InputTensorInfo& inputTensorInfo)
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

