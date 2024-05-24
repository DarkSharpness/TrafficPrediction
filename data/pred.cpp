#include "../data.h"
#include <fstream>
#include <chrono>

auto process_raw(std::string_view line) {
    struct _Result_t {
        size_t iu_ac;
        TimeStamp timestamp;
    };

    auto reader = Reader {line};
    // id, useless
    reader.read <size_t> ();
    auto iu_ac = reader.read <size_t> ();
    reader.skip();
    auto t_1h = reader.split(',');
    auto timestamp = TimeStamp::parse(t_1h);

    return _Result_t { iu_ac, timestamp };
}

signed main() {
    auto start = std::chrono::high_resolution_clock::now();

    std::ifstream in(Path::raw_pred_csv);
    std::string str;
    assert(std::getline(in, str));
    assert(str == "id,iu_ac,t_1h,etat_barre");
    std::ofstream out(Path::pred_csv);

    while (std::getline(in, str)) {
        auto [iu_ac, timestamp] = process_raw(str);
        out << std::format("{}, {}\n", iu_ac, timestamp.inner);
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << "ms\n";
    return 0;
}
