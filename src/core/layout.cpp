#include "reaction/core/layout.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <unordered_map>
#include <vector>

namespace reaction {
namespace {

constexpr float kColumnGap = 100.0F;
constexpr float kRowGap = 80.0F;
constexpr float kDefaultWidth = 200.0F;
constexpr float kDefaultHeight = 120.0F;

} // namespace

void layoutGraph(GraphBody& body, const std::unordered_map<NodeId, Vec2>* nodeSizes) {
    if (body.nodes().empty()) return;

    std::unordered_map<NodeId, Vec2> size;
    for (const auto& node : body.nodes()) {
        if (nodeSizes != nullptr) {
            const auto found = nodeSizes->find(node.id);
            size[node.id] = found != nodeSizes->end() ? found->second
                                                      : Vec2{kDefaultWidth, kDefaultHeight};
        } else {
            size[node.id] = {kDefaultWidth, kDefaultHeight};
        }
    }

    std::unordered_map<NodeId, std::vector<NodeId>> successors;
    std::unordered_map<NodeId, std::vector<NodeId>> predecessors;
    std::unordered_map<NodeId, int> indegree;
    for (const auto& node : body.nodes()) {
        successors[node.id] = {};
        predecessors[node.id] = {};
        indegree[node.id] = 0;
    }
    for (const auto& link : body.links()) {
        if (body.findNode(link.fromNode) == nullptr || body.findNode(link.toNode) == nullptr) continue;
        successors[link.fromNode].push_back(link.toNode);
        predecessors[link.toNode].push_back(link.fromNode);
    }
    for (auto& [id, list] : successors) {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }
    for (auto& [id, list] : predecessors) {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
        indegree[id] = static_cast<int>(list.size());
    }

    // Layer nodes as late as possible: every node hugs its consumers so short
    // side branches terminate next to the merge they feed instead of spanning
    // the whole width. Nodes on a longest source-to-sink path keep their
    // earliest layer, so chains stay straight. Nodes in a cycle are left at the
    // rightmost column so the layout still terminates on a broken graph.
    auto remaining = indegree;
    std::priority_queue<NodeId, std::vector<NodeId>, std::greater<>> ready;
    for (const auto& [id, degree] : indegree)
        if (degree == 0) ready.push(id);
    std::vector<NodeId> topo;
    while (!ready.empty()) {
        const auto id = ready.top();
        ready.pop();
        topo.push_back(id);
        for (const auto next : successors[id])
            if (--remaining[next] == 0) ready.push(next);
    }

    std::unordered_map<NodeId, int> sinkDepth;
    for (const auto& node : body.nodes()) sinkDepth[node.id] = 0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const auto id = *it;
        for (const auto succ : successors[id])
            sinkDepth[id] = std::max(sinkDepth[id], sinkDepth[succ] + 1);
    }
    int globalDepth = 0;
    for (const auto& [id, depth] : sinkDepth) globalDepth = std::max(globalDepth, depth);

    std::unordered_map<NodeId, int> layer;
    for (const auto& node : body.nodes()) layer[node.id] = globalDepth - sinkDepth[node.id];
    for (const auto& node : body.nodes())
        if (predecessors[node.id].empty() && successors[node.id].empty())
            layer[node.id] = 0;

    // Depth from the nearest source: used below to keep a node on its longest
    // upstream line instead of being dragged around by short side inputs.
    std::unordered_map<NodeId, int> sourceDepth;
    for (const auto& node : body.nodes()) sourceDepth[node.id] = 0;
    for (const auto id : topo)
        for (const auto pred : predecessors[id])
            sourceDepth[id] = std::max(sourceDepth[id], sourceDepth[pred] + 1);

    std::map<int, std::vector<NodeId>> byLayer;
    for (const auto& node : body.nodes()) byLayer[layer[node.id]].push_back(node.id);

    // Seed each layer's order from the existing layout so re-running is stable,
    // then reduce crossings with a few barycenter sweeps.
    for (auto& [l, ids] : byLayer) {
        std::stable_sort(ids.begin(), ids.end(), [&](NodeId a, NodeId b) {
            const auto* first = body.findNode(a);
            const auto* second = body.findNode(b);
            if (first->position.y != second->position.y) return first->position.y < second->position.y;
            if (first->position.x != second->position.x) return first->position.x < second->position.x;
            return a < b;
        });
    }

    std::unordered_map<NodeId, int> orderInLayer;
    const auto rebuildOrder = [&] {
        for (const auto& [l, ids] : byLayer)
            for (std::size_t i = 0; i < ids.size(); ++i) orderInLayer[ids[i]] = static_cast<int>(i);
    };
    rebuildOrder();

    const int minLayer = byLayer.begin()->first;
    const int maxLayer = byLayer.rbegin()->first;
    for (int pass = 0; pass < 8; ++pass) {
        for (int l = minLayer + 1; l <= maxLayer; ++l) {
            auto& ids = byLayer[l];
            std::vector<std::pair<double, NodeId>> scored;
            scored.reserve(ids.size());
            for (const auto id : ids) {
                double sum = 0.0;
                int count = 0;
                for (const auto pred : predecessors[id])
                    if (layer[pred] == l - 1) {
                        sum += static_cast<double>(orderInLayer[pred]);
                        ++count;
                    }
                const double score = count > 0 ? sum / static_cast<double>(count)
                                               : static_cast<double>(orderInLayer[id]);
                scored.emplace_back(score, id);
            }
            std::stable_sort(scored.begin(), scored.end(),
                             [](const auto& a, const auto& b) { return a.first < b.first; });
            for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = scored[i].second;
        }
        rebuildOrder();
        for (int l = maxLayer - 1; l >= minLayer; --l) {
            auto& ids = byLayer[l];
            std::vector<std::pair<double, NodeId>> scored;
            scored.reserve(ids.size());
            for (const auto id : ids) {
                double sum = 0.0;
                int count = 0;
                for (const auto succ : successors[id])
                    if (layer[succ] == l + 1) {
                        sum += static_cast<double>(orderInLayer[succ]);
                        ++count;
                    }
                const double score = count > 0 ? sum / static_cast<double>(count)
                                               : static_cast<double>(orderInLayer[id]);
                scored.emplace_back(score, id);
            }
            std::stable_sort(scored.begin(), scored.end(),
                             [](const auto& a, const auto& b) { return a.first < b.first; });
            for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = scored[i].second;
        }
        rebuildOrder();
    }

    // Horizontal position: each layer is its own column, sized by the widest
    // node in it so columns never crowd each other.
    std::map<int, float> columnX;
    float xCursor = 0.0F;
    for (const auto& [l, ids] : byLayer) {
        columnX[l] = xCursor;
        float widest = 0.0F;
        for (const auto id : ids) widest = std::max(widest, size[id].x);
        xCursor += widest + kColumnGap;
    }

    // Vertical position: follow the dominant input line. Each node sits at the
    // average of its longest-running feeders, so a chain of single-consumer
    // nodes stays straight while a real merge between two equally deep lines
    // lands halfway between them. Short side inputs are ignored. Nodes in the
    // same layer are then compacted to guarantee a minimum gap.
    std::unordered_map<NodeId, float> centerY;
    for (int l = minLayer; l <= maxLayer; ++l) {
        const auto& ids = byLayer[l];
        std::vector<std::pair<float, NodeId>> desired;
        desired.reserve(ids.size());
        for (const auto id : ids) {
            int deepest = -1;
            float sum = 0.0F;
            int count = 0;
            for (const auto pred : predecessors[id]) {
                if (sourceDepth[pred] > deepest) {
                    deepest = sourceDepth[pred];
                    sum = centerY[pred];
                    count = 1;
                } else if (sourceDepth[pred] == deepest) {
                    sum += centerY[pred];
                    ++count;
                }
            }
            const float target = count > 0
                ? sum / static_cast<float>(count)
                : (static_cast<float>(orderInLayer[id]) -
                   (static_cast<float>(ids.size()) - 1.0F) * 0.5F) * (kDefaultHeight + kRowGap);
            desired.emplace_back(target, id);
        }
        std::stable_sort(desired.begin(), desired.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        float cursor = std::numeric_limits<float>::lowest();
        for (const auto& [target, id] : desired) {
            const float height = size[id].y;
            float top = target - height * 0.5F;
            if (top < cursor + kRowGap) top = cursor + kRowGap;
            centerY[id] = top + height * 0.5F;
            cursor = top + height;
        }
    }

    for (auto& node : body.nodes()) {
        node.position.x = columnX[layer[node.id]];
        node.position.y = centerY[node.id] - size[node.id].y * 0.5F;
    }
}

} // namespace reaction
