#include "data.h"
#include "flatten.h"
#include <vector>
#include <fstream>
#include <array>

constexpr size_t kAmount = 24 + 1;
size_t normal_count = 0;
size_t abnormal_count = 0;

// Whether the data is available.
bool is_available(size_t i, size_t j) {
    for (size_t k = 0; k < kAmount; k++)
        if (train[i][j + k] == -1)
            return false;
    return true;
}

void flatten_data() {
    std::ofstream out(Path::flat_dat_csv);
    assert(out.is_open());
    for (size_t i = 0 ; i < kCount ; i++) {
        if (train[i].size() == 0) continue;
        for (size_t j = 0 ; j < kTimes ; j++)
            if (train[i][j] != -1)
                out << train[i][j] << '\n';
    }
}

void flatten_and_finetune_index() {
    std::ofstream out(Path::flat_idx_csv);
    std::ofstream tune(Path::finetune_csv);
    assert(out.is_open());
    size_t index = 0;
    size_t avail = 0;
    std::string line;

    struct Guard {
        std::ofstream &out;
        std::string &str;
        ~Guard() { out << str << '\n'; }
    };

    for (size_t i = 0; i < kCount; i++) {
        Guard guard { tune, line };

        line.clear();
        line += std::format("{}:", i);

        if (train[i].size() == 0) continue;

        for (size_t j = 0; j < kTimes - kAmount; j++) {
            if (train[i][j] == -1) continue;
            if (is_available(i, j)) {
                avail++;
                out << index << '\n';
                line += std::format("{},", index);
            }
            index++;
        }

        if (line.back() == ',') line.pop_back();
    }

    ::normal_count = index;
}

// Flatten the data for training
void flatten_train_data() {
    flatten_data();
    flatten_and_finetune_index();
}

// Find abnormal peeks in the data, and filter them out.
void filter_peek() {
    constexpr size_t kPeriod = 24 * 7;
    [[maybe_unused]]
    static Math::statistic __data[kPeriod];

    std::vector <double> cached;

    constexpr size_t kWindow = 2;

    size_t abnormal = 0;

    for (size_t i = 0; i < kCount; i++) {
        if (train[i].size() == 0) continue;

        auto train = ::train[i];
        auto data  = Math::statistic{}.init(train);

        if (data.cnt < 500) continue;

        // for (size_t i = 0 ; i != kPeriod ; ++i)
        //     __data[i].init(train, i, kPeriod);

        cached = train;

        for (size_t i = kWindow ; i < kTimes - kWindow; i++) {
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

void debug_print() {
    std::cerr << "Abnormal proportion: " <<
        double(abnormal_count) / normal_count << std::endl;
}

signed main() {
    Function::read_train();
    // filter_peek();
    flatten_train_data();
    // debug_print();
    std::cerr << "Flatten data (pre-train | finetune) successfully." << std::endl;
}
