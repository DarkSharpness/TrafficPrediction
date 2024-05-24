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



signed main() {
    Function::read_train();
    read_stream();
}
