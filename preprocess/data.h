#pragma once

#include <iostream>
#include <format>
#include <source_location>

template <typename T, typename U = const char *>
void assert(T &&x, U &&str = "",
    std::source_location loc = std::source_location::current()) {
    if (x) return;
    std::cerr << std::format("Assertion failed: {}:{}:{} | {}\n",
            loc.file_name(), loc.line(), loc.column(), str);
    std::exit(1);
}

struct TimeStamp {
    static constexpr size_t kMONTH[] = {
        0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366
    };

    size_t inner;
    explicit TimeStamp(size_t inner) : inner(inner) {}

    static size_t magic_trans(size_t times) {
        if (times >= 15985) return times + 1;
        if (times >= 10778) return times + 2;
        if (times >= 8761)  return times + 1;
        return times;
    }

    // format as below:
    // 2022-01-12 00:00:00
    // Timestamp relative to 2022-01-01 00:00:00
    // Consider the corner case of 2022-02-29
    static TimeStamp parse(std::string_view view) {
        constexpr int is_digit_pos[] = {
            0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15
        };
        assert(view.size() == 19);
        for (auto i : is_digit_pos) assert(std::isdigit(view[i]));
        assert(view.substr(13) == ":00:00");

        size_t year = (view[0] - '0') * 1000 + (view[1] - '0') * 100 + (view[2] - '0') * 10 + (view[3] - '0');
        size_t month = (view[5] - '0') * 10 + (view[6] - '0');
        size_t day = (view[8] - '0') * 10 + (view[9] - '0');
        size_t hour = (view[11] - '0') * 10 + (view[12] - '0');

        using _Self = TimeStamp;

        assert(year >= 2022 && year <= 2024);
        assert(month >= 1 && month <= 12);
        assert(day >= 1 && day <= 31);
        assert(hour >= 0 && hour <= 23);

        size_t input_time = (kMONTH[month - 1] + day - 1 + (year - 2022) * 365) * 24 + hour;

        return _Self { magic_trans(input_time) };
    }

    // std::string to_string() const {
    //     size_t hour = inner % 24;
    //     size_t day = inner / 24;
    //     size_t year = 2022 + day / 365;
    //     day %= 365;
    //     size_t month = 0;
    //     while (day >= kMONTH[month + 1]) ++month;
    //     return std::format("{}-{:02d}-{:02d} {:02d}:00:00", year, month + 1, day - kMONTH[month] + 1, hour);
    // }
};

struct Reader {
    std::string_view line;
    template <typename _Tp>
    _Tp read();
    void skip(size_t cnt = 1) { line.remove_prefix(cnt); }
    std::string_view split(char delim = ';') {
        auto pos = line.find(delim);
        auto ret = line.substr(0, pos);
        line.remove_prefix(pos + 1);
        return ret;
    }
};

template <>
TimeStamp Reader::read <TimeStamp> () {
    auto view = this->line.substr(0, 19);
    line.remove_prefix(20);
    return TimeStamp::parse(view);
}

template <>
size_t Reader::read <size_t> () {
    size_t x = 0;
    while (!std::isdigit(line[0])) line.remove_prefix(1);
    while (std::isdigit(line[0])) {
        x = x * 10 + line[0] - '0';
        line.remove_prefix(1);
    }
    return x;
}

template <>
double Reader::read <double> () {
    double x = 0;
    while (!std::isdigit(line[0]))
        line.remove_prefix(1);
    while (std::isdigit(line[0])) {
        x = x * 10 + line[0] - '0';
        line.remove_prefix(1);
    }
    if (line[0] == '.') {
        line.remove_prefix(1);
        double t = 0.1;
        while (std::isdigit(line[0])) {
            x += t * (line[0] - '0');
            t *= 0.1;
            line.remove_prefix(1);
        }
    }
    return x;
}

namespace Path {

using _Path_t = const char *;

inline const _Path_t
    train_csv       = "__exe__/train.csv",      // training data
    pred_csv        = "__exe__/pred.csv",       // prediction task
    all0_csv        = "__exe__/all0.csv",       // Prefix 0
    mid0_csv        = "__exe__/mid0.csv",       // Consecutive 0
    list_csv        = "__exe__/list.csv",       // List of all the index used.
    pred_map_csv    = "__exe__/pred.map.csv",   // Index of the prediction
    raw_train_csv   = "data/loop_sensor_train.csv",
    raw_pred_csv    = "data/loop_sensor_test_x.csv",
    baseline_csv    = "data/loop_sensor_test_baseline.csv",
    flat_raw_csv    = "__exe__/flat.raw.csv",
    flat_idx_csv    = "__exe__/flat.idx.csv",
    pred_fmt_csv    = "__exe__/pred.fmt.csv",
    geo_raw_csv     = "data/geo_reference.csv", // Geo reference
    graph_out       = "data/graph.out",         // Graph description
    geo_near_csv    = "__exe__/near.csv",       // k nearest
    geo_stream_csv  = "__exe__/stream.csv",     // up & down stream
    geo_train_path  = "__exe__/geo_training",   // training by geo
    geo_pred_path   = "__exe__/geo_predict",    // prediction by geo
    geo_which_csv   = "__exe__/geo_which.csv",  // which geo
    geo_merge_csv   = "__exe__/geo_merge.csv",  // merge geo
    tune_idx_csv    = "__exe__/finetune.idx.csv";   // finetune data index

} // namespace Path

#include <fstream>
#include <vector>


struct Prediction {
    size_t index;
    size_t times;
};

// Count of all the trains.
constexpr size_t kCount = 10000;
// Maximum timestamp.
constexpr size_t kTimes = 20000;

// Raw training data.
inline static std::vector <double> train[kCount];
// Prediction data.
inline static std::vector <Prediction> prediction;

namespace Function {

// Read the prediction data.
inline static void read_pred() {
    std::ifstream in(Path::pred_csv);
    assert(in.is_open());

    std::string str;
    prediction.reserve(2000); // Enough count by hacking :)
    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto index = reader.read<size_t>();
        auto times = reader.read<size_t>();
        prediction.emplace_back(index, times);
    }
}

// Read the training data.
inline static void read_train() {
    std::ifstream in(Path::train_csv);
    assert(in.is_open());
    std::string str;
    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto iu_ac = reader.read<size_t>();
        auto times = reader.read<size_t>();
        auto value = reader.read<double>();

        if (train[iu_ac].empty())
            train[iu_ac].resize(kTimes, -1);

        train[iu_ac][times] = value;
    }
}

} // namespace Function
