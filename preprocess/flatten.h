#pragma once

#include "data.h"
#include <cmath>

namespace Math {

size_t q_cnt(std::span <double> data, size_t _Init = 0, size_t _Step = 1) {
    assert(data.size() == kTimes);
    size_t cnt = 0;
    for (auto i = _Init; i < kCount; i += _Step)
        if (data[i] != -1) cnt++;
    return cnt;
}

double q_sum(std::span <double> data, size_t _Init = 0, size_t _Step = 1) {
    assert(data.size() == kTimes);
    double sum = 0;
    for (auto i = _Init; i < kCount; i += _Step)
        if (data[i] != -1) sum += data[i];
    return sum;
}

double q_sqr(std::span <double> data, size_t _Init = 0, size_t _Step = 1) {
    assert(data.size() == kTimes);
    double sum = 0;
    for (auto i = _Init; i < kCount; i += _Step)
        if (data[i] != -1) sum += data[i] * data[i];
    return sum;
}

struct statistic {
    size_t cnt;
    double avg;
    double var;

    auto init(std::span <double> data, size_t _Init = 0, size_t _Step = 1) {
        cnt = q_cnt(data);
        avg = q_sum(data) / cnt;
        var = q_sqr(data) / cnt - avg * avg;
        var = std::sqrt(var);
        return *this;
    }
};

} // namespace _Math
