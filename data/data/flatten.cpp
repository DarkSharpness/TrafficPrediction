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

void flat_data() {
    std::ofstream out(Path::flat_dat_csv);
    assert(out.is_open());
    for (size_t i = 0; i < kCOUNT; i++) {
        if (appear[i] == 0) continue;
        for (size_t j = 0; j < kTIMES; j++)
            if (train[i][j] != -1)
                out << train[i][j] << '\n';
    }
}

constexpr size_t kAmount = 24 + 1;

/** Whether the 25 data is available. */
bool is_available(size_t i, size_t j) {
    for (size_t k = 0; k < kAmount; k++)
        if (train[i][j + k] == -1)
            return false;
    return true;
}

void flat_index() {
    std::ofstream out(Path::flat_idx_csv);
    assert(out.is_open());
    size_t index = 0;
    size_t avail = 0;
    for (size_t i = 0; i < kCOUNT; i++) {
        if (appear[i] == 0) continue;
        for (size_t j = 0; j < kTIMES - kAmount; j++) {
            if (train[i][j] == -1) continue;
            if (is_available(i, j))
                out << index << '\n', avail++;
            index++;
        }
    }
}

// Flatten the data for training
void make_flat() {
    flat_data();
    flat_index();
}


signed main() {
    read_train();
    make_flat();
    return 0;
}
