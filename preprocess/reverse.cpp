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

void rewrite_predict(std::span <double> data) {
    for (size_t i = 0 ; i < kCount ; ++i) {
        if (train[i].empty()) continue;
        global[i].process();
    }

    assert(data.size() == prediction.size(), data.size());
    size_t which = 0;
    for (auto [id, times] : prediction) {
        size_t index = times % global[id].size();
        data[which] += global[id].sum[index];
        data[which] -= kMagic;
        void(data.at(which++));
    }
}

std::vector <double> read_real(std::string path) {
    std::ifstream input {path};
    assert(input.is_open(), path);
    assert(input.good(), path);

    std::vector <double> real;

    std::string line;
    size_t number = 0;
    std::getline(input, line);
    assert(line == "id,estimate_q");

    while (std::getline(input, line)) {
        auto reader = Reader {line};
        size_t which = reader.read <size_t>();
        assert(++number == which);
        double value = reader.read <double>();
        real.push_back(value);
    }

    return real;
}

void write_updated(std::span <double> data, std::string path) {
    std::ofstream output {path};
    assert(output.is_open(), path);
    output << "id,estimate_q\n";
    for (size_t i = 0 ; i < data.size() ; ++i)
        output << std::format("{},{:.2f}\n", i + 1, data[i]);
}

signed main(int argc, const char *argv[]) {
    assert(argc == 3, argc);
    Function::read_train();
    Function::read_pred();
    collect_data();
    auto vec = read_real(argv[1]);
    rewrite_predict(vec);
    write_updated(vec, argv[2]);
    return 0;
}
