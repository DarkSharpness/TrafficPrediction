/**
 * 思路:
 *  类似 kNN, 根据前后 7 天的数据进行预测
 *  时间点可以是当前时间点前后
 *  同时，参考地理距离最近的 5 个点的同一时间点的数据
 * 
*/

#include "../data.h"
#include <vector>
#include <fstream>

constexpr size_t kCOUNT = 10000;
constexpr size_t kTIMES = 20000;
size_t appear[kCOUNT];
std::vector <double> train[kCOUNT];

void read_meta() {
    std::ifstream in("meta.csv");
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

    std::ifstream in("train.csv");
    assert(in.is_open());
    std::string str;

    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto iu_ac = reader.read<size_t>();
        auto times = reader.read<size_t>();
        auto value = reader.read<double>();
        // auto stamp = TimeStamp { times };
        train[iu_ac][times] = value;
    }
}

constexpr size_t
    kWEEK = 7 * 24,
    kDAYS  = 24,
    kHOUR = 1;

inline bool is_valid(size_t time) {
    return time >= 0 && time < kTIMES;
}

std::pair <double, size_t> sum_hour(size_t iu_ac, size_t times) {
    double val = 0; 
    size_t cnt = 0;
    for (size_t i = -1 ; i <= 1 ; ++i) {
        auto time = times + i * kHOUR;
        if (is_valid(time) && train[iu_ac][time] != -1) {
            val += train[iu_ac][time];
            cnt += 1;
        }
    }
    return {val, cnt};
}

std::pair <double, size_t> sum_days(size_t iu_ac, size_t times) {
    double val = 0;
    size_t cnt = 0;
    for (size_t i = times % kDAYS ; i < kTIMES ; i += kDAYS) {
        if (train[iu_ac][i] != -1) {
            val += train[iu_ac][i];
            cnt += 1;
        }
    }
    return {val, cnt};
}

std::pair <double, size_t> sum_week(size_t iu_ac, size_t times) {
    double val = 0; 
    size_t cnt = 0;
    for (size_t i = times % kWEEK ; i < kTIMES ; i += kWEEK) {
        if (train[iu_ac][i] != -1) {
            val += train[iu_ac][i];
            cnt += 1;
        }
    }
    return {val, cnt}; 
}

double make_pred(std::string_view line) {
    auto reader = Reader {line};
    auto iu_ac = reader.read<size_t>();
    auto times = reader.read<size_t>();

    if (appear[iu_ac] == 0) {
        std::cerr <<
            std::format("iu_ac: {} not found\n", iu_ac);
        return -1;
    }

    // auto pair_0 = sum_hour(iu_ac, times);
    // auto pair_1 = sum_days(iu_ac, times);
    // auto pair_2 = sum_week(iu_ac, times);

    // std::pair <double, size_t> pairs[3] = {
    //     pair_0, pair_1, pair_2
    // };
    // const double factor[3] = {
    //     0.0, 10.0, 0.0
    // };
    // const double weight[3] = {
    //     factor[0] / pair_0.second,
    //     factor[1] / pair_1.second,
    //     factor[2] / pair_2.second,
    // };

    // double sum = 0;
    // double div = 0;
    // for (size_t i = 0; i < 3; ++i) 
    //     if (pairs[i].second != 0) {
    //         sum += pairs[i].first * weight[i];
    //         div += factor[i];
    //     }

    auto [sum, div] = sum_week(iu_ac, times);
    // auto [sum_1, div_1] = sum_week(iu_ac, times);

    // if (div_0 + div_1 == 0) return -1;
    // if (div_0 == 0) return sum_1 / div_1;
    // if (div_1 == 0) return sum_0 / div_0;
    // auto sum = sum_0 / div_0 + sum_1 / div_1;
    if (div == 0) return -1;

    return sum / div;
}

std::vector <double> base;

void read_base() {
    std::ifstream in("loop_sensor_test_baseline.csv");
    assert(in.is_open());
    base.resize(439300, -1);

    std::string str;
    std::getline(in, str);
    assert(str == "id,estimate_q");
    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto index = reader.read<size_t>();
        auto value = reader.read<double>();
        base[index] = value;
    }
}

std::vector <double> result;

void read_pred() {
    std::ifstream in("pred.csv");
    assert(in.is_open());
    std::string str;
    std::ofstream out("result.csv");
    out << "id,estimate_q\n";
    result.resize(439300, -1);
    size_t cnt = 0;
    while (std::getline(in, str)) {
        result[++cnt] = make_pred(str);
        if (!(result[cnt] > 0)) result[cnt] = 0;
    }
}

signed main() {
    read_train();
    read_base();
    std::cerr << "read train done\n";
    read_pred();

    double sum = 0;
    for (size_t i = 1; i <= 439300; ++i) {
        if (base[i] == -1) continue;
        sum += (base[i] - result[i]) * (base[i] - result[i]);
    }

    std::cerr << "sum: " << sum << '\n';
    return 0;
}
