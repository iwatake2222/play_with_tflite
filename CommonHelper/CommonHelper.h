
#ifndef COMMON_HELPER_
#define COMMON_HELPER_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>


#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define COMMON_HELPER_NDK_TAG "MyApp_NDK"
#define COMMON_HELPER_PRINT_(...) __android_log_print(ANDROID_LOG_INFO, COMMON_HELPER_NDK_TAG, __VA_ARGS__)
#else
#define COMMON_HELPER_PRINT_(...) printf(__VA_ARGS__)
#endif

#define COMMON_HELPER_PRINT(COMMON_HELPER__PRINT_TAG, ...) do { \
	COMMON_HELPER_PRINT_("[" COMMON_HELPER__PRINT_TAG "][%d] ", __LINE__); \
	COMMON_HELPER_PRINT_(__VA_ARGS__); \
} while(0);

#define COMMON_HELPER_PRINT_E(COMMON_HELPER__PRINT_TAG, ...) do { \
	COMMON_HELPER_PRINT_("[ERR: " COMMON_HELPER__PRINT_TAG "][%d] ", __LINE__); \
	COMMON_HELPER_PRINT_(__VA_ARGS__); \
} while(0);

#endif
