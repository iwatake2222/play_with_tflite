/* Copyright 2021 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>

#include "common_helper.h"

float CommonHelper::Sigmoid(float x)
{
    if (x >= 0) {
        return 1.0f / (1.0f + std::exp(-x));
    } else {
        return std::exp(x) / (1.0f + std::exp(x));    /* to aovid overflow */
    }
}

float CommonHelper::Logit(float x)
{
    if (x == 0) {
        return static_cast<float>(INT32_MIN);
    } else  if (x == 1) {
        return static_cast<float>(INT32_MAX);
    } else {
        return std::log(x / (1.0f - x));
    }
}