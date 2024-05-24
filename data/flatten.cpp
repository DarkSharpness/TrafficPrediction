#include "data.h"
#include <vector>
#include <fstream>
#include <array>
#include <cmath>

constexpr size_t kCOUNT = 10000;
constexpr size_t kTIMES = 20000;

size_t appear[kCOUNT];
std::vector <double> train[kCOUNT];

void read_meta() {
    std::ifstream in(Path::meta_csv);
    assert(in.is_open());
    std::string str;
    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto index = reader.read<size_t>();
        auto count = reader.read<size_t>();
        appear[index] = count;
        train[index].resize(kTIMES, -1);
    }
}

void read_train() {
    read_meta();

    std::ifstream in(Path::train_csv);
    assert(in.is_open());
    std::string str;

    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto iu_ac = reader.read<size_t>();
        auto times = reader.read<size_t>();
        auto value = reader.read<double>();
        train[iu_ac][times] = value;
    }
}

void flat_data() {
    std::ofstream out(Path::flat_dat_csv);
    assert(out.is_open());
    for (size_t i = 0; i < kCOUNT; i++) {
        if (appear[i] == 0) continue;
        for (size_t j = 0; j < kTIMES; j++)
            if (train[i][j] != -1)
                out << train[i][j] << '\n';
    }
}

constexpr size_t kAmount = 24 + 1;

/** Whether the 25 data is available. */
bool is_available(size_t i, size_t j) {
    for (size_t k = 0; k < kAmount; k++)
        if (train[i][j + k] == -1)
            return false;
    return true;
}

size_t normal_count = 0;

void flat_index() {
    std::ofstream out(Path::flat_idx_csv);
    assert(out.is_open());
    size_t index = 0;
    size_t avail = 0;
    for (size_t i = 0; i < kCOUNT; i++) {
        if (appear[i] == 0) continue;
        for (size_t j = 0; j < kTIMES - kAmount; j++) {
            if (train[i][j] == -1) continue;
            if (is_available(i, j))
                out << index << '\n', avail++;
            index++;
        }
    }
    ::normal_count = index;
}

// Flatten the data for training
void make_flat() {
    flat_data();
    flat_index();
}

size_t q_cnt(std::span <double> data, size_t _Init = 0, size_t _Step = 1) {
    assert(data.size() == kTIMES);
    size_t cnt = 0;
    for (auto i = _Init; i < kCOUNT; i += _Step)
        if (data[i] != -1) cnt++;
    return cnt;
}

double q_sum(std::span <double> data, size_t _Init = 0, size_t _Step = 1) {
    assert(data.size() == kTIMES);
    double sum = 0;
    for (auto i = _Init; i < kCOUNT; i += _Step)
        if (data[i] != -1) sum += data[i];
    return sum;
}

double q_sqr(std::span <double> data, size_t _Init = 0, size_t _Step = 1) {
    assert(data.size() == kTIMES);
    double sum = 0;
    for (auto i = _Init; i < kCOUNT; i += _Step)
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

size_t abnormal_count = 0;

// Find abnormal peeks in the data, and filter them out.
void filter_data() {
    constexpr size_t kPeriod = 24 * 7;
    static statistic __data[kPeriod];

    std::vector <double> cached;

    constexpr size_t kWindow = 2;

    size_t abnormal = 0;

    for (size_t i = 0; i < kCOUNT; i++) {
        if (appear[i] == 0) continue;

        auto train = ::train[i];
        auto data  = statistic{}.init(train);

        if (data.cnt < 500) continue;

        // for (size_t i = 0 ; i != kPeriod ; ++i)
        //     __data[i].init(train, i, kPeriod);

        cached = train;

        for (size_t i = kWindow ; i < kTIMES - kWindow; i++) {
            if (train[i] == -1) continue;

            // All the data in the window should be available
            bool flag {};
            for (size_t j = i - kWindow; j <= i + kWindow; j++)
                if (train[j] == -1) flag = true;
            if (flag) continue;

            // Look up the nearby window area
            double window_sum = 0;
            for (size_t j = i - kWindow; j <= i + kWindow ; j++)
                window_sum += train[j];
            window_sum /= 2 * kWindow + 1;

            // If the data is too far from the nearby sum.
            if (std::abs(train[i] - window_sum) > 5 * data.var)
                cached[i] = -1, abnormal++;
        }

        cached.swap(train);
    }

    abnormal_count = abnormal;
}

signed main() {
    read_train();
    filter_data();
    make_flat();
    std::cerr << "Flatten data successfully." << std::endl;
    std::cerr << "Abnormal proportion: " <<
        double(abnormal_count) / normal_count << std::endl;
    return 0;
}
