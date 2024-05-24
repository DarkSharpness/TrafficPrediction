#include "data.h"
#include <fstream>
#include <vector>

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

bool can_infer_from_neighbor(size_t index, size_t times) {
    if (neighbor[index].size() == 0)
        return false;

    if (!Function::train_available(index, times))
        return false;

    for (size_t neigh : neighbor[index])
        if (!Function::train_available(neigh, times))
            return false;

    return true;
}

bool can_infer_range(size_t index, size_t times, size_t range) {
    for (size_t i = 1; i <= range; i++)
        if (!can_infer_from_neighbor(index, times - i))
            return false;
    return true;
}

void debug_print() {
    size_t can_infer {};

    for (auto [index, times] : prediction) {
        if (can_infer_range(index, times, 24))
            can_infer++;
    }

    std::cerr << "Inferable: " <<
        std::format("{:.4f}%", 100.0 * can_infer / prediction.size()) << std::endl;
}

signed main() {
    Function::read_train();
    Function::read_pred();
    read_stream();
    debug_print();
}
