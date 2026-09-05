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
        auto& out = successors[link.fromNode];
        if (std::find(out.begin(), out.end(), link.toNode) == out.end())
            out.push_back(link.toNode);
        auto& in = predecessors[link.toNode];
        if (std::find(in.begin(), in.end(), link.fromNode) == in.end())
            in.push_back(link.fromNode);
    }
    for (const auto& [id, list] : predecessors)
        indegree[id] = static_cast<int>(list.size());

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

    // Depth from the nearest source: identifies a node's longest-running input
    // line so short side inputs don't drag a chain off its lane.
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

    // Vertical position: lay out each connected component independently and
    // stack the components vertically, so unrelated branches never interleave.
    // Within a component, a node with a single consumer follows its longest
    // input line (keeping sequential nodes horizontally aligned), while
    // junctions average their inputs so a genuine merge lands halfway between
    // its source lines. Sources get a fresh row, then each layer is compacted.
    std::unordered_map<NodeId, int> componentOf;
    std::vector<std::vector<NodeId>> components;
    for (const auto& node : body.nodes()) {
        if (componentOf.count(node.id)) continue;
        const int id = static_cast<int>(components.size());
        std::vector<NodeId> stack{node.id};
        componentOf[node.id] = id;
        std::vector<NodeId> comp;
        while (!stack.empty()) {
            const auto current = stack.back();
            stack.pop_back();
            comp.push_back(current);
            for (const auto succ : successors[current])
                if (!componentOf.count(succ)) { componentOf[succ] = id; stack.push_back(succ); }
            for (const auto pred : predecessors[current])
                if (!componentOf.count(pred)) { componentOf[pred] = id; stack.push_back(pred); }
        }
        components.push_back(std::move(comp));
    }

    std::sort(components.begin(), components.end(), [&](const auto& a, const auto& b) {
        int leftA = std::numeric_limits<int>::max();
        int leftB = std::numeric_limits<int>::max();
        for (const auto id : a) leftA = std::min(leftA, layer[id]);
        for (const auto id : b) leftB = std::min(leftB, layer[id]);
        if (leftA != leftB) return leftA < leftB;
        int topA = std::numeric_limits<int>::max();
        int topB = std::numeric_limits<int>::max();
        for (const auto id : a) if (layer[id] == leftA) topA = std::min(topA, orderInLayer[id]);
        for (const auto id : b) if (layer[id] == leftB) topB = std::min(topB, orderInLayer[id]);
        return topA < topB;
    });

    std::unordered_map<NodeId, float> centerY;
    float stackOffset = 0.0F;
    for (const auto& comp : components) {
        const int compId = componentOf[comp.front()];
        float compTop = std::numeric_limits<float>::max();
        float compBottom = std::numeric_limits<float>::lowest();
        for (int l = minLayer; l <= maxLayer; ++l) {
            std::vector<NodeId> ids;
            for (const auto id : byLayer[l])
                if (componentOf[id] == compId) ids.push_back(id);
            if (ids.empty()) continue;
            std::vector<std::pair<float, NodeId>> desired;
            desired.reserve(ids.size());
            for (std::size_t i = 0; i < ids.size(); ++i) {
                const auto id = ids[i];
                float target = 0.0F;
                if (predecessors[id].empty()) {
                    target = (static_cast<float>(i) -
                              (static_cast<float>(ids.size()) - 1.0F) * 0.5F) *
                             (kDefaultHeight + kRowGap);
                } else if (successors[id].size() == 1) {
                    NodeId chosen = predecessors[id].front();
                    for (const auto pred : predecessors[id]) {
                        const bool predDedicated = successors[pred].size() == 1;
                        const bool chosenDedicated = successors[chosen].size() == 1;
                        if (sourceDepth[pred] > sourceDepth[chosen] ||
                            (sourceDepth[pred] == sourceDepth[chosen] &&
                             predDedicated && !chosenDedicated)) {
                            chosen = pred;
                        }
                    }
                    target = centerY[chosen];
                } else {
                    float sum = 0.0F;
                    for (const auto pred : predecessors[id]) sum += centerY[pred];
                    target = sum / static_cast<float>(predecessors[id].size());
                }
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
                compTop = std::min(compTop, top);
                compBottom = std::max(compBottom, top + height);
            }
        }
        for (const auto id : comp) centerY[id] += stackOffset - compTop;
        stackOffset += (compBottom - compTop) + kRowGap * 2.0F;
    }

    for (auto& node : body.nodes()) {
        node.position.x = columnX[layer[node.id]];
        node.position.y = centerY[node.id] - size[node.id].y * 0.5F;
    }
}

} // namespace reaction
