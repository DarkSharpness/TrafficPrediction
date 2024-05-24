#include "../data.h"
#include <vector>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <optional>
#include <filesystem>

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

struct Prediction {
    size_t index;
    size_t times;
};

// Prediction input
std::vector <Prediction> prediction;

void read_pred() {
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

// Prediction result
std::vector <double> result;

void read_result() {
    std::ifstream in(Path::raw_result_csv);
    assert(in.is_open());

    std::string str;
    std::getline(in, str); // Skip header
    assert(str == "id,estimate_q");

    size_t cnt = 0;
    result.reserve(2000); // Enough count by hacking :)
    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto which = reader.read<size_t>();
        assert(which == ++cnt);
        auto value = reader.read<double>();
        result.push_back(value);
    }
}

// From [beg , end]
struct Range {
    size_t beg; size_t end;
    friend auto operator <=> (const Range &lhs, const Range &rhs) = default;
};

// Fixed zero set
std::unordered_set <size_t> zero_set;

std::optional <double> try_update(size_t index, size_t pos) {
    if (zero_set.contains(index)) return 0;
    auto value = train[index][pos];
    return value == -1 ? std::nullopt : std::make_optional(value);
}

void read_zero() {
    std::ifstream in(Path::all0_csv);
    assert(in.is_open());
    while (in) {
        size_t index;
        in >> index;
        zero_set.insert(index);
    }

    in.close();
    in.open(Path::mid0_csv);
    assert(in.is_open());

    std::string str;
    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto index  = reader.read<size_t>();
        auto beg    = reader.read<size_t>();
        auto end    = reader.read<size_t>();
        for (size_t i = beg; i <= end; i++)
            train[index][i] = 0;
    }
}

std::string next_name(std::string_view str) {
    if (!std::filesystem::exists(str)) {
        return std::string(str);
    } else {
        size_t i = 0;
        std::string name;
        do {
            name = std::format("{}_{}", str, i++);
        } while(std::filesystem::exists(name));
        return name;
    }
}

void rewrite_result() {
    std::ofstream out(next_name(Path::final_result_csv));
    assert(out.is_open());
    out << "id,estimate_q\n";
    for (size_t i = 0; i < prediction.size(); i++) {
        auto [index, times] = prediction[i];
        auto value = try_update(index, times).value_or(result[i]);
        out << std::format("{},{:.1f}\n", i + 1, value);
    }
}

signed main() {
    read_train();
    read_pred();
    read_zero();
    read_result();
    assert(prediction.size() == result.size());
    rewrite_result();
    return 0;
}
