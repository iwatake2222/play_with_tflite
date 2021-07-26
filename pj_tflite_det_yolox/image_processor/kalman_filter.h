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
#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include <cstdint>
#include <string>
#include <vector>

template<typename T>
class KalmanFilter {
public:
    KalmanFilter()
        : x_prev_(0), P_prev_(0), K_(0), P_(0), x_(0), start_deviation_(0), deviation_true_(0), deviation_noise_(0)
    {}

    ~KalmanFilter() {}
    void Initialize(T start_value, float start_deviation, float deviation_true, float deviation_noise);
    T Predict(void) const;
    T Update(T observation_value);

private:
    float x_prev_;
    float P_prev_;
    float K_;
    float P_;
    float x_;

    float start_deviation_;
    float deviation_true_;
    float deviation_noise_;
};


#endif
