#include "../data.h"
#include <vector>
#include <fstream>
#include <array>

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

void clear_prefix_0();
void clear_middle_0();
void fix_prefix_0();
void fix_middle_0();

// Some of the roud is not open at the beginning
// So we need to clean the data
void clean_data() {
    clear_prefix_0();
    fix_prefix_0();
    clear_middle_0();
    fix_middle_0();
}

void rewrite_meta();
void rewrite_train();

void rewrite_data() {
    rewrite_meta();
    rewrite_train();
}

signed main() {
    read_train();
    std::cerr << "Read data successfully.\n";
    clean_data();
    std::cerr << "Clean data successfully.\n";
    rewrite_data();
    std::cerr << "Rewrite data successfully.\n";
    return 0;
}

void clear_prefix_0() {
    std::ofstream out(Path::all0_csv, std::ios::app);

    assert(out.is_open());

    // Prefix 0 is not allowed.
    // According to observation, if prefix 0 of a data exceeds 5,
    // it will either be a all-0 data, or not used in the prediction.
    // Since these data accounts for only a small proportion of the
    // whole data, we can safely remove them.
    // We record all these all-0 index, and then remove data.

    for (size_t i = 0; i < kCOUNT; i++) {
        if (appear[i] > 0) {
            size_t cnt {};
            size_t last_0 {};
            for (size_t j = 0; j < kTIMES; j++) {
                if (train[i][j] == -1) continue;
                if (train[i][j] != 0) {
                    break;
                } else { // Prefix 0
                    last_0 = j;
                    ++cnt;
                }
            }

            // Normal case.
            if (cnt < 5) continue;

            if (cnt == appear[i]) {
                std::cerr << std::format("{}: All Zero.\n", i);
                out << i << '\n';
            } else if (cnt != 0) {
                std::cerr <<
                    std::format("{}: Zero until {}.  \n", i, last_0);
            }

            train[i].clear();
            train[i].shrink_to_fit();
            appear[i] = 0;
        }
    }
}

void clear_middle_0() {
    std::ofstream out(Path::mid0_csv, std::ios::app);
    for (size_t i = 0 ; i < kCOUNT ; i++) {
        if (appear[i] > 0) {
            size_t cnt {}; // count of consecutive 0
            size_t beg {}; // First 0 position
            size_t pos {}; // Last position 
            for (size_t j = 0; j < kTIMES; j++) {
                if (train[i][j] == -1) continue;
                if (train[i][j] != 0) {
                    // Consecutive 0 in [beg, pos]
                    if (cnt >= 10) {
                        out << std::format("{},{},{}\n", i, beg, pos);
                        std::cerr <<
                            std::format("Consecutive {} zero in [{},{}]\n", cnt, beg, pos);
                        for (size_t k = beg; k <= pos; k++)
                            train[i][k] = -1;
                    }
                    cnt = 0;
                } else {
                    // First 0 position
                    if (cnt++ == 0) beg = j;
                    pos = j;
                }
            }
        }
    }
}

void fix_prefix_0() {
    std::fstream all0(Path::all0_csv, std::ios::in);

    std::string str;
    std::vector <size_t> list;
    while (std::getline(all0, str)) {
        auto index = std::stoull(str);
        list.push_back(index);
    }

    std::sort(list.begin(), list.end());
    list.resize(std::unique(list.begin(), list.end()) - list.begin());

    all0.close();
    all0.open(Path::all0_csv, std::ios::out | std::ios::trunc);

    for (auto i : list) all0 << i << '\n';
}

void fix_middle_0() {
    std::fstream mid0(Path::mid0_csv, std::ios::in);

    std::string str;
    using _Tuple = std::tuple <size_t, unsigned, unsigned>;
    std::vector <_Tuple> list;
    while (std::getline(mid0, str)) {
        auto reader = Reader {str};
        auto index  = reader.read<size_t>();
        auto beg    = reader.read<size_t>();
        auto end    = reader.read<size_t>();
        list.emplace_back(index, beg, end);
    }

    std::sort(list.begin(), list.end());
    list.resize(std::unique(list.begin(), list.end()) - list.begin());

    mid0.close();
    mid0.open(Path::mid0_csv, std::ios::out | std::ios::trunc);

    for (auto [index, beg, end] : list)
        mid0 << std::format("{},{},{}\n", index, beg, end);
}

void rewrite_meta() {
    std::ofstream out(Path::meta_csv);
    for (size_t i = 0; i < kCOUNT; i++) {
        if (appear[i] == 0) continue;

        size_t appear = 0;
        for (size_t j = 0; j < kTIMES; j++)
            if (train[i][j] != -1) ++appear;

        out << std::format("{},{}\n", i, appear);
        ::appear[i] = appear;
    }
}

void rewrite_train() {
    std::ofstream out(Path::train_csv);
    for (size_t i = 0; i < kCOUNT; i++) {
        if (appear[i] == 0) continue;
        for (size_t j = 0; j < kTIMES; j++) {
            if (train[i][j] == -1) continue;
            out << std::format("{},{},{}\n", i, j, train[i][j]);
        }
    }
}
