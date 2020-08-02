#include <jni.h>
#include <string>

#include <opencv2/opencv.hpp>
#include "ImageProcessor.h"

#define WORK_DIR    "/sdcard/models/"

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroidtflite_MainActivity_ImageProcessorInitialize(
        JNIEnv* env,
        jobject /* this */) {

    int ret = 0;
    INPUT_PARAM inputParam;
    snprintf(inputParam.workDir, sizeof(inputParam.workDir), WORK_DIR);
    inputParam.numThreads = 4;
    ret = ImageProcessor_initialize(&inputParam);
    return ret;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroidtflite_MainActivity_ImageProcessorProcess(
        JNIEnv* env,
        jobject, /* this */
        jlong   objMat) {

    int ret = 0;
    cv::Mat* mat = (cv::Mat*) objMat;
    OUTPUT_PARAM outputParam;
    ret = ImageProcessor_process(mat, &outputParam);
    return ret;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroidtflite_MainActivity_ImageProcessorFinalize(
        JNIEnv* env,
        jobject /* this */) {

    int ret = 0;
    ret = ImageProcessor_finalize();
    return ret;
}
