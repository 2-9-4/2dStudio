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

CompileResult compileGraphBody(
    const GraphBody& body, const BodyDescriptorResolver& descriptorResolver,
    const MissingDescriptorMessage& missingDescriptorMessage) {
    CompileResult result;
    std::unordered_map<NodeId, int> indegree;
    std::unordered_map<NodeId, std::vector<NodeId>> outgoing;
    std::unordered_set<std::string> occupiedInputs;

    for (const auto& node : body.nodes()) {
        indegree[node.id] = 0;
        NodeDescriptor descriptorStorage;
        if (!descriptorResolver(node, descriptorStorage)) {
            if (missingDescriptorMessage) {
                const auto message = missingDescriptorMessage(node);
                if (!message.empty()) result.errors.push_back(message);
            } else if (!node.missing) {
                result.errors.push_back("Unknown node type '" + node.type + "'");
            }
        }
    }

    for (const auto& link : body.links()) {
        const auto* from = body.findNode(link.fromNode);
        const auto* to = body.findNode(link.toNode);
        if (!from || !to) {
            result.errors.push_back("Link " + std::to_string(link.id) + " has a missing endpoint");
            continue;
        }
        NodeDescriptor fromStorage, toStorage;
        const auto* fromDesc = descriptorResolver(*from, fromStorage);
        const auto* toDesc = descriptorResolver(*to, toStorage);
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
        const auto inputKey = std::to_string(link.toNode) + ":" + link.toSocket;
        if (!occupiedInputs.insert(inputKey).second) {
            result.errors.push_back("Input has more than one link: " + inputKey);
            continue;
        }
        ++indegree[link.toNode];
        outgoing[link.fromNode].push_back(link.toNode);
    }

    std::priority_queue<NodeId, std::vector<NodeId>, std::greater<>> ready;
    for (const auto& [id, degree] : indegree)
        if (degree == 0) ready.push(id);
    while (!ready.empty()) {
        const auto id = ready.top();
        ready.pop();
        result.order.push_back(id);
        for (const auto next : outgoing[id])
            if (--indegree[next] == 0) ready.push(next);
    }
    if (result.order.size() != body.nodes().size())
        result.errors.push_back("Graph contains a cycle");

    const auto incoming = [&](NodeId node, std::string_view key) -> const LinkRecord* {
        const auto found = std::ranges::find_if(body.links(), [&](const LinkRecord& link) {
            return link.toNode == node && link.toSocket == key;
        });
        return found == body.links().end() ? nullptr : &*found;
    };

    for (const auto id : result.order) {
        const auto* node = body.findNode(id);
        NodeDescriptor descriptorStorage;
        const auto* descriptor =
            node ? descriptorResolver(*node, descriptorStorage) : nullptr;
        if (!descriptor) continue;

        std::unordered_map<std::string, ValueType> inputTypes;
        for (const auto& port : descriptor->sockets) {
            if (port.direction != SocketDirection::Input) continue;
            ValueType resolved = port.fieldDefault
                ? fieldTypeForWidth(componentCount(disconnectedType(port.contract)))
                : disconnectedType(port.contract);
            if (const auto* link = incoming(id, port.key)) {
                const auto source = result.socketType(link->fromNode, link->fromSocket);
                if (source) resolved = *source;
                if (source && !contractAccepts(port.contract, *source)) {
                    result.errors.push_back(
                        "Cannot connect " + toString(*source) + " to " +
                        descriptor->displayName + "." + port.key + "; socket accepts " +
                        toString(port.contract) + " (semantically inadmissible)");
                } else if (source && port.requiresImage && !isFieldType(*source)) {
                    result.errors.push_back(descriptor->displayName + "." + port.key +
                                            " requires a field image");
                }
            }
            inputTypes[port.key] = resolved;
            result.resolvedSockets[{id, port.key, SocketDirection::Input}] =
                {port.contract, resolved};
        }

        for (const auto& port : descriptor->sockets) {
            if (port.direction != SocketDirection::Output) continue;
            const auto resolved = resolveOutputType(
                *descriptor, port,
                [&](std::string_view key) -> std::optional<ValueType> {
                    const auto found = inputTypes.find(std::string(key));
                    return found == inputTypes.end()
                        ? std::nullopt
                        : std::optional(found->second);
                });
            result.resolvedSockets[{id, port.key, SocketDirection::Output}] =
                {port.contract, resolved};
        }

        for (const auto& [key, sourceType] : inputTypes) {
            const auto inputKey =
                CompileResult::SocketKey{id, key, SocketDirection::Input};
            const auto targetType = resolveInputType(
                *descriptor, key, sourceType,
                [&](std::string_view outputKey) {
                    return result.socketType(id, outputKey,
                                             SocketDirection::Output);
                });
            if (auto found = result.resolvedSockets.find(inputKey);
                found != result.resolvedSockets.end())
                found->second.concreteType = targetType;
        }
    }

    for (const auto& link : body.links()) {
        const auto source = result.socketType(link.fromNode, link.fromSocket);
        const auto target =
            result.socketType(link.toNode, link.toSocket, SocketDirection::Input);
        if (!source || !target) continue;
        const auto coercion = coercionBetween(*source, *target);
        if (!coercion) {
            result.errors.push_back(
                "Link " + std::to_string(link.id) + " cannot convert " +
                toString(*source) + " to " + toString(*target));
            continue;
        }
        result.resolvedEdges[link.id] = {
            {link.fromNode, link.fromSocket, SocketDirection::Output},
            {link.toNode, link.toSocket, SocketDirection::Input},
            *source, *target, *coercion};
    }

    result.valid = result.errors.empty();
    return result;
}

CompileResult Graph::compile(const NodeRegistry& registry) const {
    auto result = compileGraphBody(
        *this,
        [&](const NodeRecord& node, NodeDescriptor& storage) {
            return resolveDescriptor(*this, node, registry, storage);
        },
        [&](const NodeRecord& node) {
            if (node.type == "subgraph")
                return "Missing subgraph definition '" + node.subgraphId + "'";
            return node.missing ? std::string{}
                                : "Unknown node type '" + node.type + "'";
        });

    std::unordered_set<std::string> validatedSubgraphs;
    for (const auto& node : nodes_) {
        if (node.type != "subgraph" ||
            !validatedSubgraphs.insert(node.subgraphId).second)
            continue;
        const auto* definition = resolveSubgraph(*this, node.subgraphId);
        if (!definition) continue;
        for (const auto& error : validateSubgraph(*definition, registry)) {
            result.errors.push_back("Subgraph '" + definition->name + "': " + error);
        }
    }

    if (activeOutput != 0) {
        const auto* output = findNode(activeOutput);
        if (!output)
            result.errors.push_back("Active output node does not exist");
        else if (output->type != "output")
            result.errors.push_back(
                "Active output must reference an Output node");
    }
    result.valid = result.errors.empty();
    return result;
}

} // namespace reaction
