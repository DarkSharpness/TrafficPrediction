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

struct TimeDiff {
    size_t inner;
    explicit TimeDiff(size_t inner) : inner(inner) {}
};
struct TimeStamp {
    static constexpr size_t kMONTH[] = {
        0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366
    };

    size_t inner;
    explicit TimeStamp(size_t inner) : inner(inner) {}

    TimeStamp &operator += (TimeDiff td) {
        inner += td.inner;
        return *this;
    }
    TimeStamp &operator -= (TimeDiff td) {
        inner -= td.inner;
        return *this;
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

        assert(year >= 2022 && year <= 2025);
        assert(month >= 1 && month <= 12);
        assert(day >= 1 && day <= 31);
        assert(hour >= 0 && hour <= 23);
        return _Self { (kMONTH[month - 1] + day - 1 + (year - 2022) * 365) * 24 + hour };
    }

    size_t get_hour() const { return inner % 24; }

    std::string to_string() const {
        size_t hour = inner % 24;
        size_t day = inner / 24;
        size_t year = 2022 + day / 365;
        day %= 365;
        size_t month = 0;
        while (day >= kMONTH[month + 1]) ++month;
        return std::format("{}-{:02d}-{:02d} {:02d}:00:00", year, month + 1, day - kMONTH[month] + 1, hour);
    }

};

TimeDiff operator "" _h(unsigned long long x) {
    return TimeDiff(x);
}
TimeDiff operator "" _d(unsigned long long x) {
    return TimeDiff(x * 24);
}

TimeStamp operator + (TimeStamp ts, TimeDiff td) {
    return TimeStamp(ts.inner + td.inner);
}
TimeStamp operator + (TimeDiff td, TimeStamp ts) {
    return TimeStamp(ts.inner + td.inner);
}
TimeStamp operator - (TimeStamp ts, TimeDiff td) {
    return TimeStamp(ts.inner - td.inner);
}
TimeDiff operator - (TimeStamp ts1, TimeStamp ts2) {
    return TimeDiff(ts1.inner - ts2.inner);
}

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
    train_csv       = "__exe__/train.csv",  // training data
    pred_csv        = "__exe__/pred.csv",   // prediction task
    meta_csv        = "__exe__/meta.csv",   // meta data, occurance data
    all0_csv        = "__exe__/all0.csv",   // Prefix 0
    mid0_csv        = "__exe__/mid0.csv",   // Consecutive 0
    raw_train_csv   = "loop_sensor_train.csv",
    raw_pred_csv    = "loop_sensor_test_x.csv",
    baseline_csv    = "loop_sensor_test_baseline.csv",
    raw_result_csv  = "raw.csv",
    final_result_csv= "__exe__/result.csv",
    flat_dat_csv    = "__exe__/flat.dat.csv",
    flat_idx_csv    = "__exe__/flat.idx.csv",
    pred_fmt_csv    = "__exe__/pred.fmt.csv";

} // namespace Path
