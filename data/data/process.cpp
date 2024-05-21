/**
 * 思路:
 *  类似 kNN, 根据前后 7 天的数据进行预测
 *  时间点可以是当前时间点前后
 *  同时，参考地理距离最近的 5 个点的同一时间点的数据
 * 
*/

#include "../data.h"
#include <fstream>

constexpr size_t kCOUNT = 10000;
size_t appear[kCOUNT];

void read_meta() {
    std::ifstream in("meta.csv");
    std::string str;
    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto index = reader.read<size_t>();
        auto count = reader.read<size_t>();
        appear[index] = count;
        // Store something here.
    }
}

void read_train() {
    read_meta();

    std::ifstream in("train.csv");
    std::string str;

    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto iu_ac = reader.read<size_t>();
        auto times = reader.read<size_t>();
        auto value = reader.read<double>();
        auto stamp = TimeStamp { times };
        // do something here
    }
}

void make_pred() {
    std::ifstream in("pred.csv");
    std::string str;
    while (std::getline(in, str)) {
        auto reader = Reader {str};
        auto iu_ac = reader.read<size_t>();
        auto times = reader.read<size_t>();
        // Predict the value

    }
}

signed main() {

    return 0;
}
