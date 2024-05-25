#include "data.h"
#include <vector>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <optional>
#include <filesystem>

// Prediction result
std::vector <double> result;
// Fixed zero set
std::unordered_set <size_t> zero_set;

void read_raw_result(const char *raw_path) {
    std::ifstream in(raw_path);
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

std::optional <double> try_update(size_t index, size_t pos) {
    if (zero_set.contains(index)) return 0;
    auto value = train[index][pos];
    return value == -1 ? std::nullopt : std::make_optional(value);
}

void read_zero() {
    std::ifstream in(Path::all0_csv);
    assert(in.is_open());
    while (in) {
        size_t index; in >> index;
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

void rewrite_result(const char *out_path) {
    std::ofstream out(out_path);
    assert(out.is_open());
    out << "id,estimate_q\n";
    for (size_t i = 0; i < prediction.size(); i++) {
        auto [index, times] = prediction[i];
        auto value = try_update(index, times).value_or(result[i]);
        out << std::format("{},{:.1f}\n", i + 1, value);
    }
}

signed main(const int argc, const char *argv[]) {
    assert(argc == 3, argv[0]);

    auto raw_path = argv[1];
    auto out_path = argv[2];

    Function::read_train();
    Function::read_pred();
    read_zero();
    read_raw_result(raw_path);
    assert(prediction.size() == result.size());
    rewrite_result(out_path);
    return 0;
}
