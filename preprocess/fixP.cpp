#include "data.h"
#include <unordered_set>

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

        assert(arr.size() == kTimes);
        size_t week_count[kWeek] {};
        size_t hour_count[kHour] {};
        size_t global_count {};

        for (size_t i = 0 ; i < kTimes ; ++i) {
            // Invalid.
            if (arr[i] < 0) continue;
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

Table table[kCount];
constexpr size_t kWindow = 24;
size_t count_of_missing[kWindow + 1];
using _Format_Pred_t = std::array <double, kWindow>;

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

void process_train() {
    Function::read_train();
    for (size_t i = 0 ; i < kCount ; ++i)
        if (!train[i].empty())
            table[i].try_init(train[i]);
}


void make_pred_idx() {
    static std::pair <size_t, size_t> temp[kCount];

    std::ofstream idx(Path::predict_idx_csv);

    size_t last     = prediction[0].index;
    size_t front    = 0;

    for (size_t i = 0 ; i < prediction.size() ; ++i) {
        auto [index, times] = prediction[i];
        if (index != last) {
            // last : [front, i - 1]
            temp[last] = { front + 1, i - 1 + 1 };
            // idx << std::format("{},{},{}\n", last, front + 1, i - 1 + 1);
            front = i;
            last = index;
        }
    }

    temp[last] = { front + 1, prediction.size() - 1 + 1 };

    for (size_t i = 0 ; i < kCount ; ++i)
        idx << std::format("{},{},{}\n", i, temp[i].first, temp[i].second);
}

void process_pred() {
    Function::read_pred();

    std::ofstream out(Path::pred_fmt_csv);
    _Format_Pred_t pred;

    std::string buf;
    buf.reserve(7 * kWindow + 1);

    for (auto [index, times] : prediction) {
        fill_pred(pred, index, times);

        buf.clear();
        for (size_t i = 0 ; i < kWindow ; ++i)
            buf += std::format("{:.2f},", pred[i]);
        buf.back() = '\n';

        out << buf;
    }

    make_pred_idx();
}

void debug_print() {
    size_t count_of_prediction = 0;
    for (size_t i = 0 ; i <= kWindow ; i++)
        count_of_prediction += count_of_missing[i];

    double prev {}; // Maybe +=
    for (size_t i = 0 ; i <= kWindow ; i++)
        std::cerr << std::format("Missing {} times: {}\n",
            i, (prev = count_of_missing[i]) / count_of_prediction * 100.0);
}


signed main() {
    process_train();
    process_pred();
    debug_print();
    return 0;
}
