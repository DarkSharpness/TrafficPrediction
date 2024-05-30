#include "data.h"
#include <chrono>
#include <fstream>
#include <filesystem>
#include <map>

struct Packed {
    TimeStamp   stamp;
    size_t      index;
    size_t      barre; 
    double      value;
};

inline Packed process_raw(std::string_view line) {
    std::string_view view = line;
    size_t iuac_end = view.find(',');
    auto iuac_str = view.substr(0, iuac_end);

    size_t t1h_end = view.find(',', iuac_end + 1);
    auto t1h_str = view.substr(iuac_end + 1, t1h_end - iuac_end - 1);

    size_t etatbarre = view.find(',', t1h_end + 1);
    auto barre_str = view.substr(t1h_end + 1, etatbarre - t1h_end - 1);
    assert(barre_str.size() == 1);
    auto barre = barre_str[0] - '0';

    auto q_str = view.substr(etatbarre + 1);
    auto stamp = TimeStamp::parse(t1h_str);

    auto index = size_t {};
    std::from_chars(iuac_str.data(), iuac_str.data() + iuac_str.size(), index);

    auto value = double {};
    std::from_chars(q_str.data(), q_str.data() + q_str.size(), value);

    return Packed { stamp, index, (size_t)barre , value };
}

constexpr size_t kCOUNT = 10000;

signed main() {
    if (std::filesystem::exists(Path::train_csv)) {
        std::cerr << "Train file already exists\n";
        return 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream input(Path::raw_train_csv);

    std::string line;
    std::getline(input, line);
    assert(line == "iu_ac,t_1h,etat_barre,q");

    std::ofstream output(Path::train_csv, std::ios::trunc);

    while (std::getline(input, line)) {
        auto [timestamp, index, barre, value] = process_raw(line);
        output << std::format("{}, {}, {}\n", index, timestamp.inner, value);
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << "ms\n";
    return 0;
}
