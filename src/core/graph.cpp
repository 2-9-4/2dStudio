#include "reaction/core/graph.hpp"

#include <algorithm>
#include <queue>
#include <set>
#include <unordered_set>

namespace reaction {
namespace {

const SocketDescriptor* socket(const NodeDescriptor& node, std::string_view key,
                               SocketDirection direction) {
    const auto it = std::ranges::find_if(node.sockets, [&](const auto& value) {
        return value.key == key && value.direction == direction;
    });
    return it == node.sockets.end() ? nullptr : &*it;
}

} // namespace

NodeId GraphBody::addNode(std::string type, Vec2 position) {
    const auto id = nextNodeId++;
    NodeRecord record;
    record.id = id;
    record.type = std::move(type);
    record.position = position;
    nodes_.push_back(std::move(record));
    return id;
}

bool GraphBody::removeNode(NodeId id) {
    const auto oldSize = nodes_.size();
    std::erase_if(nodes_, [id](const NodeRecord& node) { return node.id == id; });
    std::erase_if(links_, [id](const LinkRecord& link) {
        return link.fromNode == id || link.toNode == id;
    });
    return nodes_.size() != oldSize;
}

LinkId GraphBody::addLink(NodeId fromNode, std::string fromSocket,
                          NodeId toNode, std::string toSocket) {
    std::erase_if(links_, [&](const LinkRecord& link) {
        return link.toNode == toNode && link.toSocket == toSocket;
    });
    const auto id = nextLinkId++;
    links_.push_back(LinkRecord{id, fromNode, std::move(fromSocket),
                                toNode, std::move(toSocket)});
    return id;
}

bool GraphBody::removeLink(LinkId id) {
    const auto oldSize = links_.size();
    std::erase_if(links_, [id](const LinkRecord& link) { return link.id == id; });
    return links_.size() != oldSize;
}

void GraphBody::clear() {
    nodes_.clear();
    links_.clear();
    nextNodeId = 1;
    nextLinkId = 1;
}

const NodeRecord* GraphBody::findNode(NodeId id) const {
    const auto it = std::ranges::find(nodes_, id, &NodeRecord::id);
    return it == nodes_.end() ? nullptr : &*it;
}

NodeRecord* GraphBody::findNode(NodeId id) {
    const auto it = std::ranges::find(nodes_, id, &NodeRecord::id);
    return it == nodes_.end() ? nullptr : &*it;
}

bool Graph::removeNode(NodeId id) {
    const bool removed = GraphBody::removeNode(id);
    if (removed && activeOutput == id) activeOutput = 0;
    if (removed) pruneOrphanedSubgraphs();
    return removed;
}

void Graph::pruneOrphanedSubgraphs() {
    std::unordered_set<std::string> referenced;
    for (const auto& node : nodes_)
        if (node.type == "subgraph" && !node.subgraphId.empty())
            referenced.insert(node.subgraphId);
    std::erase_if(subgraphs_, [&](const SubgraphDefinition& definition) {
        return !referenced.contains(definition.id);
    });
}

void Graph::clear() {
    GraphBody::clear();
    subgraphs_.clear();
    activeOutput = 0;
}

const SubgraphDefinition* Graph::findSubgraph(std::string_view id) const {
    const auto it = std::ranges::find(subgraphs_, id, &SubgraphDefinition::id);
    return it == subgraphs_.end() ? nullptr : &*it;
}

SubgraphDefinition* Graph::findSubgraph(std::string_view id) {
    const auto it = std::ranges::find(subgraphs_, id, &SubgraphDefinition::id);
    return it == subgraphs_.end() ? nullptr : &*it;
}

CompileResult Graph::compile(const NodeRegistry& registry) const {
    CompileResult result;
    std::unordered_map<NodeId, int> indegree;
    std::unordered_map<NodeId, std::vector<NodeId>> outgoing;
    std::unordered_set<std::string> occupiedInputs;

    for (const auto& node : nodes_) {
        indegree[node.id] = 0;
        NodeDescriptor descriptorStorage;
        if (!resolveDescriptor(*this, node, registry, descriptorStorage)) {
            if (node.type == "subgraph")
                result.errors.push_back("Missing subgraph definition '" + node.subgraphId + "'");
            else if (!node.missing)
                result.errors.push_back("Unknown node type '" + node.type + "'");
        }
    }

    for (const auto& link : links_) {
        const auto* from = findNode(link.fromNode);
        const auto* to = findNode(link.toNode);
        if (!from || !to) {
            result.errors.push_back("Link " + std::to_string(link.id) + " has a missing endpoint");
            continue;
        }
        NodeDescriptor fromStorage, toStorage;
        const auto* fromDesc = resolveDescriptor(*this, *from, registry, fromStorage);
        const auto* toDesc = resolveDescriptor(*this, *to, registry, toStorage);
        if (!fromDesc || !toDesc) {
            result.errors.push_back("Link " + std::to_string(link.id) + " touches a missing node type");
            continue;
        }
        const auto* output = socket(*fromDesc, link.fromSocket, SocketDirection::Output);
        const auto* input = socket(*toDesc, link.toSocket, SocketDirection::Input);
        if (!output || !input) {
            result.errors.push_back("Link " + std::to_string(link.id) + " names an unknown socket");
            continue;
        }
        if (!areSocketTypesCompatible(output->type, input->type)) {
            result.errors.push_back("Link " + std::to_string(link.id) + " has incompatible socket types");
            continue;
        }
        const auto inputKey = std::to_string(link.toNode) + ":" + link.toSocket;
        if (!occupiedInputs.insert(inputKey).second) {
            result.errors.push_back("Input has more than one link: " + inputKey);
            continue;
        }
        ++indegree[link.toNode];
        outgoing[link.fromNode].push_back(link.toNode);
    }

    std::priority_queue<NodeId, std::vector<NodeId>, std::greater<>> ready;
    for (const auto& [id, degree] : indegree) if (degree == 0) ready.push(id);
    while (!ready.empty()) {
        const auto id = ready.top();
        ready.pop();
        result.order.push_back(id);
        for (const auto next : outgoing[id]) if (--indegree[next] == 0) ready.push(next);
    }
    if (result.order.size() != nodes_.size()) result.errors.push_back("Graph contains a cycle");

    for (const auto id : result.order) {
        const auto* node = findNode(id);
        NodeDescriptor descriptorStorage;
        const auto* descriptor = node ? resolveDescriptor(*this, *node, registry, descriptorStorage) : nullptr;
        if (!descriptor) continue;
        ValueType inferred = ValueType::Float;
        if (descriptor->producedField) inferred = ValueType::Image2D;
        for (const auto& port : descriptor->sockets) {
            if (port.direction == SocketDirection::Output && port.type == ValueType::Image2D) {
                inferred = ValueType::Image2D;
            } else if (port.direction == SocketDirection::Output &&
                       port.type == ValueType::Vec2 && inferred != ValueType::Image2D) {
                inferred = ValueType::Vec2;
            }
        }
        for (const auto& link : links_) {
            if (link.toNode != id) continue;
            if (result.inferredOutputs[link.fromNode] == ValueType::Image2D) inferred = ValueType::Image2D;
            else if (result.inferredOutputs[link.fromNode] == ValueType::Vec2 &&
                     inferred != ValueType::Image2D) inferred = ValueType::Vec2;
        }
        result.inferredOutputs[id] = inferred;
    }

    std::unordered_set<std::string> validatedSubgraphs;
    for (const auto& node : nodes_) {
        if (node.type != "subgraph" || !validatedSubgraphs.insert(node.subgraphId).second)
            continue;
        const auto* definition = resolveSubgraph(*this, node.subgraphId);
        if (!definition) continue; // The missing-definition error was added above.
        for (const auto& error : validateSubgraph(*definition, registry)) {
            result.errors.push_back("Subgraph '" + definition->name + "': " + error);
        }
    }

    if (activeOutput != 0) {
        const auto* output = findNode(activeOutput);
        if (!output) result.errors.push_back("Active output node does not exist");
        else if (output->type != "output") result.errors.push_back("Active output must reference an Output node");
    }
    result.valid = result.errors.empty();
    return result;
}

} // namespace reaction
