#include "data.h"

std::vector <size_t> predict_which[kCount];

void read_which() {
    std::ifstream input(Path::geo_which_csv);
    assert(input.is_open());
    std::string line;
    while (std::getline(input, line)) {
        auto reader = Reader { line };
        size_t index = reader.read<size_t>();
        size_t count = reader.read<size_t>();
        predict_which[index].reserve(count);
        for (size_t i = 0 ; i < count ; ++i)
            predict_which[index].push_back(reader.read<size_t>());
    }
}

std::vector <double> best_output;

void read_best(const char *path) {
    std::ifstream input(path);
    assert(input.is_open());
    std::string line;
    std::getline(input, line);
    assert(line == "id,estimate_q");
    best_output.push_back(-1);      // 1-based
    best_output.reserve(439300);    // Hack
    while (std::getline(input, line)) {
        auto reader = Reader { line };
        reader.read<size_t>();  // Useless
        best_output.push_back(reader.read<double>());
    }
}

void read_geo_and_merge(const char *path) {
    for (size_t i = 0 ; i < kCount ; ++i) {
        auto &whichs = predict_which[i];
        if (whichs.empty()) continue;
        std::ifstream input(std::format("{}/{}.csv", path, i));
        assert(input.is_open(), i);
        std::string line;
        std::getline(input, line);
        assert(line == "id,estimate_q");
        size_t idx = 0;
        while (std::getline(input, line)) {
            auto reader = Reader { line };
            reader.read<size_t>();  // Useless
            auto estimate_q = reader.read<double>();
            best_output[whichs.at(idx++)] = estimate_q;
        }
    }
}

void write_merged() {
    std::ofstream output(Path::geo_merge_csv);
    assert(output.is_open());
    output << "id,estimate_q\n";
    for (size_t i = 1 ; i < best_output.size() ; ++i)
        output << std::format("{},{:.2f}\n", i, best_output[i]);
}

signed main(int argc, const char *argv[]) {
    assert(argc == 3, "Usage: ./geo_merge <best_output_csv> <geo_predict_path>");
    const char *best_output_csv  = argv[1];
    const char *geo_predict_path = argv[2];
    read_which();
    read_best(best_output_csv);
    read_geo_and_merge(geo_predict_path);
    write_merged();
}
