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
/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

/* for My modules */
#include "kalman_filter.h"

template<typename T>
void KalmanFilter<T>::Initialize(T start_value, float start_deviation, float deviation_true, float deviation_noise)
{
    start_deviation_ = start_deviation;
    deviation_true_ = deviation_true;
    deviation_noise_ = deviation_noise;

    P_prev_ = start_deviation_;
    x_prev_ = static_cast<float>(start_value);
    K_ = P_prev_ / (P_prev_ + deviation_noise_);
    P_ = deviation_noise_ * P_prev_ / (P_prev_ + deviation_noise_);
    x_ = x_prev_ + K_ * (start_value - x_prev_);
}

template<typename T>
T KalmanFilter<T>::Predict(void) const
{
    //return x_ + K_ * (x_ - x_prev_);
    return static_cast<T>(x_ + (x_ - x_prev_));
}

template<typename T>
T KalmanFilter<T>::Update(T observation_value)
{
    P_prev_ = P_ + deviation_true_;
    x_prev_ = x_;
    K_ = P_prev_ / (P_prev_ + deviation_noise_);
    P_ = deviation_noise_ * P_prev_ / (P_prev_ + deviation_noise_);
    x_ = x_prev_ + K_ * (observation_value - x_prev_);

    return static_cast<T>(x_);
}


/* Generate implementation */
template void KalmanFilter<int32_t>::Initialize(int32_t start_value, float start_deviation, float deviation_true, float deviation_noise);
template int32_t KalmanFilter<int32_t>::Predict(void) const;
template int32_t KalmanFilter<int32_t>::Update(int32_t observation_value);
