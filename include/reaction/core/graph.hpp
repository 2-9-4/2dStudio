#pragma once

#include "reaction/core/node.hpp"

#include <optional>
#include <unordered_map>

namespace reaction {

struct NodeRecord {
    NodeId id = 0;
    std::string type;
    int typeVersion = 1;
    Vec2 position;
    nlohmann::json parameters = nlohmann::json::object();
    bool missing = false;
    nlohmann::json preservedJson;
};

struct LinkRecord {
    LinkId id = 0;
    NodeId fromNode = 0;
    std::string fromSocket;
    NodeId toNode = 0;
    std::string toSocket;
};

struct ProjectSettings {
    int width = 1024;
    int height = 1024;
    int targetFps = 60;
};

struct CompileResult {
    bool valid = false;
    std::vector<NodeId> order;
    std::vector<std::string> errors;
    std::unordered_map<NodeId, ValueType> inferredOutputs;
};

class Graph {
public:
    NodeId addNode(std::string type, Vec2 position = {});
    bool removeNode(NodeId id);
    LinkId addLink(NodeId fromNode, std::string fromSocket,
                   NodeId toNode, std::string toSocket);
    bool removeLink(LinkId id);
    void clear();

    [[nodiscard]] CompileResult compile(const NodeRegistry& registry) const;
    [[nodiscard]] const NodeRecord* findNode(NodeId id) const;
    [[nodiscard]] NodeRecord* findNode(NodeId id);
    [[nodiscard]] const std::vector<NodeRecord>& nodes() const { return nodes_; }
    [[nodiscard]] std::vector<NodeRecord>& nodes() { return nodes_; }
    [[nodiscard]] const std::vector<LinkRecord>& links() const { return links_; }
    [[nodiscard]] std::vector<LinkRecord>& links() { return links_; }

    ProjectSettings settings;
    NodeId activeOutput = 0;
    NodeId nextNodeId = 1;
    LinkId nextLinkId = 1;

private:
    std::vector<NodeRecord> nodes_;
    std::vector<LinkRecord> links_;
};

} // namespace reaction

