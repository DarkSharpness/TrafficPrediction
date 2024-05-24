#include "../data.h"
#include <fstream>
#include <vector>

constexpr size_t kCOUNT = 10000;
constexpr size_t kTIMES = 20000;

size_t appear[kCOUNT];
std::vector <double> train[kCOUNT];


struct Table {
    static constexpr size_t kWeek = 24 * 7;
    static constexpr size_t kHour = 24;
    std::array <double, kWeek> week_data;
    std::array <double, kHour> hour_data;
    double global;
    bool init {};

    void try_init(std::span <double> arr) {
        assert(init == 0);
        init = true;

        assert(arr.size() == kTIMES);
        size_t week_count[kWeek] {};
        size_t hour_count[kHour] {};
        size_t global_count {};

        for (size_t i = 0 ; i < kTIMES ; ++i) {
            week_data[i % kWeek] += arr[i];
            hour_data[i % kHour] += arr[i];
            week_count[i % kWeek] += 1;
            hour_count[i % kHour] += 1;
            global += arr[i];
            global_count += 1;
        }

        for (size_t i = 0 ; i < kWeek ; ++i)
            week_data[i] /= week_count[i];
        for (size_t i = 0 ; i < kHour ; ++i)
            hour_data[i] /= hour_count[i];
        global /= global_count;
    }

    double get_average(size_t times) const {
        assert(init);

        size_t week = times % kWeek;
        if (week_data[week] != -1) return week_data[week];

        size_t hour = times % kHour;
        if (hour_data[hour] != -1) return hour_data[hour];

        return global;
    }
};

Table table[kCOUNT];

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

    for (size_t i = 0 ; i < kCOUNT ; ++i)
        if (!train[i].empty())
            table[i].try_init(train[i]);
}


constexpr size_t kWindow = 24;
size_t count_of_missing[kWindow + 1];
using _Format_Pred_t = std::array <double, kWindow>;

#include <unordered_set>

void fill_pred(_Format_Pred_t &arr, size_t index, size_t times) {
    // Can no fill up the prediction
    if (train[index].empty()) {
        static std::unordered_set <size_t> set;
        if (set.insert(index).second)
            std::cerr << std::format("No data for index {}\n", index);
        return;
    }

    assert(times >= kWindow);

    auto start = times - kWindow;
    size_t missing = 0;

    for (size_t i = 0 ; i < kWindow ; ++i) {
        arr[i] = train[index][start + i];
        if (arr[i] != -1) continue;
        ++missing;
        arr[i] = table[index].get_average(start + i);
    }

    ++count_of_missing[missing];
}

void read_pred() {
    std::ifstream in(Path::pred_csv);
    std::ofstream out(Path::pred_fmt_csv);

    assert(in.is_open());
    std::string str;
    _Format_Pred_t pred;

    std::string buf;
    buf.reserve(7 * kWindow + 1);

    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto index = reader.read<size_t>();
        auto times = reader.read<size_t>();

        fill_pred(pred, index, times);

        buf.clear();
        for (size_t i = 0 ; i < kWindow ; ++i)
            buf += std::format("{:.1f},", pred[i]);
        buf.back() = '\n';
        out << buf;
    }
}

signed main() {
    read_train();
    read_pred();

    size_t count_of_prediction = 0;
    for (size_t i = 0 ; i <= kWindow ; i++)
        count_of_prediction += count_of_missing[i];

    double prev {}; // Maybe +=
    for (size_t i = 0 ; i <= kWindow ; i++)
        std::cerr << std::format("Missing {} times: {}\n",
            i, (prev = count_of_missing[i]) / count_of_prediction * 100.0);

    return 0;
}
