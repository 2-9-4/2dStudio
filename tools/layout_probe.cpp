// Layout quality probe. Loads a Reaction Studio project, runs layoutGraph on a
// subgraph a few times, and reports how well single-consumer links align.
//
// Usage:
//   layout_probe <project.reaction> [--subgraph <id>] [--iterations N]
//                [--height H] [--width W] [--dump]
//
// A node with exactly one consumer is "exact" when it sits on that consumer's
// row; "packed" when |dy| <= one small node height (it visually shares the
// row); anything farther is reported as an offender with its dy. Run with
// multiple iterations: layout reads current positions as a tie-breaker, so a
// few passes can converge to a better fixed point than a single pass.
#include "reaction/core/layout.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Offender {
    std::string label;
    std::string consumer;
    float dy;
};

struct Report {
    int exact = 0;
    int packed = 0;
    int offenders = 0;
    int total = 0;
    std::vector<Offender> list;
    float minY = 0.0F;
    float maxY = 0.0F;
};

Report assess(const reaction::GraphBody& body,
              const std::unordered_map<reaction::NodeId, std::string>& labels,
              const std::unordered_map<reaction::NodeId, float>& halfHeights) {
    std::unordered_map<reaction::NodeId, float> centerY;
    for (const auto& node : body.nodes())
        centerY[node.id] = node.position.y + (halfHeights.contains(node.id) ? halfHeights.at(node.id) : 60.0F);
    Report report;
    for (const auto& node : body.nodes()) {
        report.minY = std::min(report.minY, node.position.y);
        report.maxY = std::max(report.maxY, node.position.y + 120.0F);
        std::vector<reaction::NodeId> consumers;
        for (const auto& link : body.links())
            if (link.fromNode == node.id) consumers.push_back(link.toNode);
        if (consumers.size() != 1) continue;
        const auto consumerId = consumers.front();
        const float dy = centerY.at(node.id) - centerY.at(consumerId);
        ++report.total;
        const float ownHeight = halfHeights.contains(node.id) ? halfHeights.at(node.id) * 2.0F : 120.0F;
        const float oneRow = ownHeight + 80.0F;
        if (dy == 0.0F) {
            ++report.exact;
        } else if (std::fabs(dy) <= oneRow) {
            ++report.packed;
        } else {
            ++report.offenders;
            report.list.push_back({labels.at(node.id), labels.at(consumerId), dy});
        }
    }
    return report;
}

void printReport(const std::string& tag, const Report& report) {
    std::printf("%-10s exact=%-4d packed=%-4d off=%-4d span=%.0f..%.0f\n",
                tag.c_str(), report.exact, report.packed, report.offenders,
                report.minY, report.maxY);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <project.reaction> [--subgraph ID] [--iterations N] "
                     "[--height H] [--width W] [--dump]\n",
                     argv[0]);
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    nlohmann::json root = nlohmann::json::parse(in);

    std::string subgraphId;
    int iterations = 4;
    float height = 0.0F;
    float width = 120.0F;
    bool dump = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--subgraph" && i + 1 < argc) subgraphId = argv[++i];
        else if (arg == "--iterations" && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) height = std::atof(argv[++i]);
        else if (arg == "--width" && i + 1 < argc) width = std::atof(argv[++i]);
        else if (arg == "--dump") dump = true;
    }
    if (iterations < 1) iterations = 1;

    for (const auto& sub : root.at("subgraphs")) {
        if (!subgraphId.empty() && sub.value("id", "") != subgraphId) continue;
        reaction::GraphBody body;
        std::unordered_map<int, reaction::NodeId> ids;
        std::unordered_map<reaction::NodeId, std::string> labels;
        std::unordered_map<reaction::NodeId, float> halfHeights;
        for (const auto& n : sub.at("nodes")) {
            const int src = n.at("id").get<int>();
            const auto id = body.addNode(n.at("type").get<std::string>(),
                                         {n.at("position").at(0).get<float>(),
                                          n.at("position").at(1).get<float>()});
            ids[src] = id;
            labels[id] = n.value("label", n.at("type").get<std::string>());
        }
        for (const auto& l : sub.at("links")) {
            body.addLink(ids.at(l.at("from").at("node").get<int>()),
                         l.at("from").at("socket").get<std::string>(),
                         ids.at(l.at("to").at("node").get<int>()),
                         l.at("to").at("socket").get<std::string>());
        }

        std::unordered_map<reaction::NodeId, reaction::Vec2> sizes;
        if (height > 0.0F)
            for (const auto& node : body.nodes())
                sizes[node.id] = {width, height};
        for (const auto& node : body.nodes())
            halfHeights[node.id] = height > 0.0F ? height * 0.5F : 60.0F;

        std::printf("subgraph %s (%zu nodes, %zu links)\n",
                    sub.value("name", sub.value("id", "?")).c_str(),
                    body.nodes().size(), body.links().size());
        for (int iter = 1; iter <= iterations; ++iter) {
            reaction::layoutGraph(body, height > 0.0F ? &sizes : nullptr);
            printReport("iter " + std::to_string(iter), assess(body, labels, halfHeights));
        }

        const Report final = assess(body, labels, halfHeights);
        if (dump) {
            std::printf("id\tlabel\tx\ty\n");
            for (const auto& node : body.nodes())
                std::printf("n%zu\t%s\t%.0f\t%.0f\n", static_cast<std::size_t>(node.id),
                            labels.at(node.id).c_str(), node.position.x, node.position.y);
        }
        std::printf("offenders (single-consumer links not sharing the consumer row):\n");
        for (const auto& o : final.list)
            std::printf("  %-32s -> %-28s dy=%+.0f\n", o.label.c_str(), o.consumer.c_str(), o.dy);
        std::printf("\n");
    }
    return 0;
}