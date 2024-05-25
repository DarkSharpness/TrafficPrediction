#include "data.h"
#include <unordered_set>
#include <map>
#include <format>
#include <fstream>
#include <iostream>

struct Node {
    std::unordered_set <size_t> in_;
    std::unordered_set <size_t> out;
};

Node nodes[kCount];

void add_edge(size_t fr, size_t to) {
    nodes[fr].out.insert(to);
    nodes[to].in_.insert(fr);
}

void print_graph() {
    for (size_t i = 0 ; i < kCount ; ++i) {
        if (nodes[i].in_.size() == 0
         && nodes[i].out.size() == 0) continue;

        std::cerr << std::format("Node {}. ", i);

        if (nodes[i].in_.size() > 0) {
            std::cerr << "In: ";
            for (size_t j : nodes[i].in_)
                std::cerr << j << ' ';
        }

        if (nodes[i].out.size() > 0) {
            std::cerr << "Out: ";
            for (size_t j : nodes[i].out) {
                std::cerr << j << ' ';
            }
        }

        std::cerr << std::endl;
    }

    std::cerr << std::format("{:-^50}\n", ' ');
}

void write_graph() {
    // Read the list of nodes that may appear in the graph
    std::unordered_set <size_t> in_list = []() {
        if (std::ifstream input(Path::list_csv); input.fail()) {
            std::cerr << "Can't open list.csv\n";
            std::exit(1);
        } else {
            std::unordered_set <size_t> list;
            for (size_t x; input >> x;) list.insert(x);
            return list;
        }
    }();

    std::ofstream output(Path::geo_stream_csv);
    std::unordered_set <size_t> visited;
    std::map <size_t, size_t> count;

    std::string line;
    for (size_t i = 0 ; i < kCount ; ++i) {
        if (in_list.count(i) == 0) continue;

        visited.clear();
        for (size_t j : nodes[i].in_)
            if (in_list.count(j)) visited.insert(j);
        for (size_t j : nodes[i].out) 
            if (in_list.count(j)) visited.insert(j);
        if (visited.empty()) continue;

        ++count[visited.size()];

        line.clear();
        line += std::format("[{};{}]: [", i, visited.size());
        for (size_t j : visited) line += std::format("{},", j);

        line.back() = ']';
        line += '\n';
        output << line;
    }

    for (const auto& [k, v] : count)
        std::cerr << std::format("In + out = {} : {}\n", k, v);
}


signed main() {
    std::ifstream input(Path::graph_out);

    size_t n;
    input >> n;

    for (size_t i = 0 ; i < n ; ++i) {
        size_t x, y;
        input >> x >> y;
        add_edge(x, y);
    }

    // print_graph();
    write_graph();

    return 0;
}
