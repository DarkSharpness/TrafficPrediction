#include "data.h"

struct Statistic {
    constexpr static size_t _Nm = 7 * 24;

    double  sum[_Nm];
    size_t  cnt[_Nm];

    consteval static size_t size() { return _Nm; }

    auto process() {
        double sum_24[24] {};
        size_t cnt_24[24] {};

        double sum_all {};
        size_t cnt_all {};

        for (size_t i = 0 ; i < size() ; ++i) {
            size_t index = i % 24;
            sum_24[index] += sum[i];
            cnt_24[index] += cnt[i];
            sum_all += sum[i];
            cnt_all += cnt[i];
        }

        assert(cnt_all != 0, sum_all);

        size_t tmp {};
        size_t day {};
        size_t all {};

        for (size_t i = 0 ; i < size() ; ++i) {
            if (cnt[i] != 0)
                ++tmp,
                sum[i] /= cnt[i];
            else if (cnt_24[i % 24] != 0)
                ++day,
                sum[i] = sum_24[i % 24] / cnt_24[i % 24];
            else
                ++all,
                sum[i] = sum_all / cnt_all;
        }

        return std::make_tuple(tmp, day, all);
    }
};

Statistic global[kCount];

void collect_data() {
    for (size_t i = 0 ; i < kCount ; ++i) {
        auto &train = ::train[i];
        if (train.empty()) continue;
        for (size_t j = 0 ; j < train.size() ; ++j) {
            if (train[j] == -1) continue;
            size_t index = j % global[i].size();
            global[i].sum[index] += train[j];
            global[i].cnt[index] += 1;
        }
    }
}

constexpr double kMagic = 6000;

void rewrite_data() {
    size_t norm {};
    size_t days {};
    size_t alls {};

    for (size_t i = 0 ; i < kCount ; ++i) {
        auto &train = ::train[i];
        if (train.empty()) continue;
        auto [tmp, day, all] = global[i].process();
        days += day; alls += all; norm += tmp;

        for (size_t j = 0 ; j < train.size() ; ++j) {
            if (train[j] == -1) continue;
            size_t index = j % global[i].size();
            train[j] -= global[i].sum[index];
            train[j] += kMagic;
            assert(train[j] >= 0, train[j]);
        }
    }

    std::cerr << std::format("week: {}, day: {}, all: {}\n", norm, days, alls);
}

void write_train() {
    std::ofstream out(Path::train_csv);
    assert(out.is_open(), "file not open");
    for (size_t i = 0 ; i < kCount ; ++i) {
        auto &train = ::train[i];
        if (train.empty()) continue;
        for (size_t j = 0 ; j < train.size() ; ++j) {
            if (train[j] == -1) continue;
            out << std::format("{},{},{:.2f}\n", i, j, train[j]);
        }
    }
}

signed main() {
    Function::read_train();
    collect_data();
    rewrite_data();
    write_train();
    return 0;
}
