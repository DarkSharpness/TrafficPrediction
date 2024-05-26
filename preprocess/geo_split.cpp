#include "data.h"
#include <fstream>
#include <vector>
#include <filesystem>

struct Neighbor_Set {
    size_t neighbor[7]; // At most 7 neighbors
    size_t count = {};  // Count of neighbors
    void push_back(size_t __n) {
        assert(count < 7);
        neighbor[count++] = __n;
    }
    size_t operator[](size_t __x) const {
        assert(__x < count);
        return neighbor[__x];
    }
    size_t size() const { return count; }

    const size_t *begin() const { return neighbor; }
    const size_t *end() const { return neighbor + count; }
};

Neighbor_Set neighbor[kCount];

void read_stream() {
    std::ifstream in(Path::geo_stream_csv);
    assert(in.is_open());

    std::string line;
    while (std::getline(in, line)) {
        auto reader = Reader {line};
        auto index = reader.read<size_t>();
        auto count = reader.read<size_t>();
        while (count--)
            neighbor[index].push_back(reader.read<size_t>());
    }
}

constexpr size_t kRange = 24;

bool train_available(size_t index, size_t times) {
    for (size_t i = 1 ; i <= kRange; i++)
        if (train[index][times - i] == -1)
            return false;
    return true;
}

bool can_infer_precondition(size_t index) {
    // Must be available
    if (train[index].size() == 0) return false;
    // No neighbor
    if (neighbor[index].size() == 0) return false;
    // All neighbor must be available
    for (auto neigh : neighbor[index])
        if (train[neigh].size() == 0) return false;
    // Otherwise, OK
    return true;
}

bool can_infer_from_neighbor(size_t index, size_t times) {
    if (times < kRange) return false;
    if (!train_available(index, times)) return false;
    for (size_t neigh : neighbor[index])
        if (!train_available(neigh, times)) return false;
    return true;
}

bool can_infer_safe(size_t index, size_t times) {
    return can_infer_precondition(index) &&
        can_infer_from_neighbor(index, times);
}

std::vector <size_t> inferable_predict_times[kCount];
std::vector <size_t> inferable_train_times[kCount];
std::vector <size_t> inferable_predict_which[kCount];

void count_infer() {
    size_t can_infer = 0;
    size_t which = 0;

    for (auto [index, times] : prediction) {
        ++which;
        if (can_infer_safe(index, times)) {
            can_infer++;
            inferable_predict_times[index].push_back(times);
            inferable_predict_which[index].push_back(which);
        }
    }

    for (size_t i = 0; i < kCount; i++) {
        if (inferable_predict_times[i].size() == 0) continue;
        assert(can_infer_precondition(i));
        auto &train = ::train[i];
        assert(train.size() == kTimes);
        for (size_t j = 0; j < kTimes; j++)
            if (train[j] != -1 && can_infer_from_neighbor(i, j))
                inferable_train_times[i].push_back(j);
    }

    std::ofstream out(Path::geo_which_csv);
    for (size_t i = 0; i < kCount; i++) {
        if (inferable_predict_times[i].size() == 0) continue;
        out << i << ' ' << inferable_predict_times[i].size() << ' ';
        for (auto which : inferable_predict_which[i])
            out << which << ' ';
        out << '\n';
    }
}

void append_infer(std::string &line, size_t index, size_t times) {
    for (size_t i = 24 ; i > 0; i--)
        line += std::format("{},", train[index][times - i]);
}

enum class Pack_Type {
    Training,
    Prediction
};

template <Pack_Type _Type>
void pack_infer(std::span <const size_t> infer, size_t index) {
    assert(infer.size() > 0);
    const char *path = (_Type == Pack_Type::Training) ?
        Path::geo_train_path : Path::geo_pred_path;
    std::ofstream out { std::format("{}/{}.csv", path, index) };
    std::string line;
    for (auto times : infer) {
        line.clear();
        append_infer(line, index, times);
        for (auto neigh : neighbor[index])
            append_infer(line, neigh, times);

        if constexpr (_Type == Pack_Type::Training) {
            line += std::format("{}\n",train[index][times]);
        } else {
            line.back() = '\n';
        }

        out << line;
    }
}

// If there's too few data, return false.
// We only train when there's enough data.
bool make_infer_train(size_t index) {
    auto &infer = ::inferable_train_times[index];

    constexpr size_t kThreshold = 500;
    if (infer.size() < kThreshold) return false;

    pack_infer <Pack_Type::Training> (infer, index);
    return true;
}

void make_infer_pred(size_t index) {
    auto &infer = ::inferable_predict_times[index];
    assert(infer.size() > 0);
    pack_infer <Pack_Type::Prediction> (infer, index);
}

void write_infer() {
    std::filesystem::create_directories(Path::geo_train_path);
    std::filesystem::create_directories(Path::geo_pred_path);

    // Debug use only
    std::ofstream list("__exe__/geo_list.csv");
    size_t can_infer {};

    for (size_t i = 0 ; i < kCount; i++) {
        size_t count = inferable_predict_times[i].size();
        if (count == 0) continue;
        if (make_infer_train(i)) {
            can_infer += count;
            make_infer_pred(i);
            list << i << '\n'; // Add to can infer list.
        }
    }

    std::cerr << "Inferable in prediction: " <<
        std::format("{:.4f}%", 100.0 * can_infer / prediction.size()) << std::endl;
}

signed main() {
    Function::read_train();
    Function::read_pred();
    read_stream();
    count_infer();
    write_infer();
}
