#include "../data.h"
#include <fstream>
#include <vector>

constexpr size_t kCOUNT = 10000;
constexpr size_t kTIMES = 20000;

size_t appear[kCOUNT];
std::vector <double> train[kCOUNT];

constexpr size_t kTable = 24 * 7;

struct Table {
    std::array <double, kTable> data;
    double global;
    bool init {};

    void try_init(std::span <double> arr) {
        if (init) return;
        init = true;

        assert(arr.size() == kTIMES);
        size_t count[kTable] {};

        for (size_t i = 0 ; i < kTIMES ; ++i)
            data[i % kTable] += arr[i],
            count[i % kTable] += 1;

        size_t total = 0;
        for (size_t i = 0 ; i < kTable ; ++i) {
            global += data[i];
            total += count[i];
            if (count[i] == 0)
                data[i] = -1;
            else
                data[i] /= count[i];
        }

        assert(total != 0);
        global /= total;
    }

    double get_average(size_t times) const {
        assert(init);
        size_t entry = times % kTable;
        return data[entry] == -1 ? global : data[entry];
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

void fill_pred(_Format_Pred_t &arr, size_t index, size_t times) {
    // Can no fill up the prediction
    if (train[index].empty()) return;

    times -= (kWindow + 1);

    for (size_t i = 0; i < kWindow; i++)
        arr[i] = train[index].at(times + i);

    size_t missing = 0;
    for (size_t i = 0 ; i < kWindow ; ++i) {
        if (arr[i] != -1) continue;
        ++missing;
        auto value = table[index].get_average(times + i);
        arr[i] = value;
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
        for (size_t i = 0 ; i < 24 ; ++i)
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
