#include <iostream>
#include <fstream>
#include <format>
#include <thread>
#include <source_location>

template <typename T, typename U = const char *>
void assert(T &&x, U &&str = "", std::source_location loc = std::source_location::current()) {
    if (!x) {
        std::cerr <<
            std::format("Assertion failed: {}:{}:{} : {}\n",
                loc.file_name(), loc.line(), loc.column(), str);
        exit(1);
    }
}

signed main() {
    std::string path = "loop_sensor_train.csv";
    std::ifstream file(path);
    assert(file.is_open());

    std::string line;
    std::getline(file, line);
    assert(line == "iu_ac,t_1h,etat_barre,q");

    auto start = std::chrono::high_resolution_clock::now();

    size_t count = 0;
    while (file) {
        ++count;
        bool result { std::getline(file, line) };
        if (!file) {
            std::cout << std::boolalpha;
            std::cout << file.eof() << '\n'
                      << file.bad() << '\n';
            std::cout << count << '\n';
            break;
        }
        assert(result, line);

        std::string_view view = line;
        size_t iuac = view.find(',');
        auto iuac_str = view.substr(0, iuac);

        size_t t1h = view.find(',', iuac + 1);
        auto t1h_str = view.substr(iuac + 1, t1h - iuac - 1);

        size_t etatbarre = view.find(',', t1h + 1);

        auto q_str = view.substr(etatbarre + 1);
    }

    auto finish = std::chrono::high_resolution_clock::now();

    std::cout <<
        std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << "ms\n";

    return 0;
}