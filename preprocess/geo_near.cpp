#include "data.h"
#include <fstream>
#include <vector>
#include <algorithm>

// iu_ac;date_debut;date_fin;libelle;iu_nd_aval;
// libelle_nd_aval;iu_nd_amont;libelle_nd_amont;geo_point_2d;geo_shape

// amont: upstream
// aval: downstream

// libelle: name
// iu_nd: ID of the node
// iu_ac: ID of camera

struct point {
    double x;
    double y;
    std::vector <size_t> nearest;
};

constexpr size_t kCOUNT = 10000;
point array[kCOUNT];

double dist(const point& a, const point& b) {
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

std::pair <double, size_t> distance[kCOUNT];

void read_line(std::string_view str) {
    auto reader = Reader { str };
    auto index = reader.read <size_t> ();
    [[maybe_unused]]
    auto index_str = reader.split();

    reader.split(); // date_debut
    reader.split(); // date_fin
    reader.split(); // libelle

    // ID of the downstream node
    [[maybe_unused]]
    auto iu_nd_aval = reader.split();

    // libelle_nd_aval
    reader.split();

    // ID of the upstream node
    [[maybe_unused]]
    auto iu_nd_amont = reader.split();

    // libelle_nd_amont
    reader.split();

    // auto geo_point_2d   = reader.split();

    // auto reader = Reader { geo_point_2d };
    auto x = reader.read <double> ();
    reader.skip();
    auto y = reader.read <double> ();

    array[index] = { x, y };

    reader.split();
    [[maybe_unused]]
    auto geo_shape      = reader.split();
}

constexpr size_t kNEAR = 10;

void find_near(point &self) {
    for (auto j = size_t{}; j < kCOUNT; j++) {
        distance[j] = { dist(self, array[j]), j };

        std::sort(distance, distance + kCOUNT);
        assert(distance[0].first == 0);
        self.nearest.resize(kNEAR + 1);
        for (size_t i = 0 ; i < kNEAR + 1 ; ++i)
            self.nearest[i] = distance[i].second;
    }
}

void write_geo() {
    std::ofstream out(Path::geo_near_csv);
    std::string line;

    for (auto &self : array) {
        if (self.x == 0 && self.y == 0) continue;
        auto view = std::span(self.nearest.data(), kNEAR + 1);

        line.clear();
        line += std::format("{}: [", view[0]);

        for (auto &j : view.subspan(1))
            line += std::format("{},", j);

        line.back() = ']';
        line.push_back('\n');

        out << line;
    }
}

signed main() {
    std::string str;
    std::ifstream in(Path::geo_raw_csv);
    std::getline(in, str);

    while (std::getline(in, str))
        read_line(str);

    for (auto &self : array)
        find_near(self);

    write_geo();
    return 0;
}
