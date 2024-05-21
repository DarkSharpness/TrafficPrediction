#include "../data.h"
#include <fstream>
#include <print>

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
    std::ifstream in("loop_sensor_test_x.csv");
    std::string str;
    assert(std::getline(in, str));
    assert(str == "id,iu_ac,t_1h,etat_barre");
    std::ofstream out("pred.csv");

    while (std::getline(in, str)) {
        auto [iu_ac, timestamp] = process_raw(str);
        std::print(out, "{}, {}\n", iu_ac, timestamp.inner);
    }

    return 0;
}
