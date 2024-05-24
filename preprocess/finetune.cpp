#include "data.h"
#include <filesystem>

void split_file() {
    std::filesystem::create_directory("finetune");
    size_t last = prediction[0].index;
    std::vector <size_t> time_list;
    for (auto [index, times] : prediction) {
        if (index == last) {
            time_list.push_back(times);
            continue;
        }





        last = index;
        time_list.clear();
    }
}


signed main() {
    Function::read_train();
    Function::read_pred();

}
