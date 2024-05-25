#include "data.h"
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>

struct IndexPack {
    size_t times;
    double value;
};

std::vector <size_t> visited;

// Just add to visit
void split_train(size_t index, std::span <IndexPack> data) {
    std::ofstream out(std::format("train/{}.csv", index));
    for (auto [times, value] : data)
        out << std::format("{},{}\n", times, value);
    assert(out.is_open());
    out.close();
    visited.push_back(index);
}

void read_train() {
    std::ifstream in(Path::train_csv);
    assert(in.is_open());
    std::string str;

    std::vector <IndexPack> data;

    auto last = size_t { 1655 }; // Hack from data

    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto iu_ac = reader.read<size_t>();
        auto times = reader.read<size_t>();
        auto value = reader.read<double>();

        if (iu_ac != last) {
            split_train(last, data);
            last = iu_ac;
            data.clear();
        }

        data.push_back({times, value});
    }

    split_train(last, data);
}

void write_list() {
    std::ofstream out(Path::list_csv);
    assert(out.is_open());
    for (auto index : visited)
        out << std::format("{} ", index);
}

// void split_pred(size_t index) {
// }

void read_pred() {
    // Function::read_pred();
}

signed main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::filesystem::create_directory("train");
    // std::filesystem::create_directory("predict");
    read_train();
    read_pred();
    write_list();
    auto finish = std::chrono::high_resolution_clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << "ms\n";
    return 0;
}
