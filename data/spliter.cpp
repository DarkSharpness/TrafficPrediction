#include "data.h"
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>

struct IndexPack {
    size_t times;
    double value;
};

void split_work(size_t index, std::span <IndexPack> data) {
    std::ofstream out(std::format("index/{}.csv", index));
    for (auto [times, value] : data)
        out << std::format("{},{}\n", times, value);
    assert(out.is_open());
    out.close();
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
            split_work(last, data);
            last = iu_ac;
            data.clear();
        }

        data.push_back({times, value});
    }
}

signed main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::filesystem::create_directory("index");
    read_train();
    auto finish = std::chrono::high_resolution_clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << "ms\n";
    return 0;
}
