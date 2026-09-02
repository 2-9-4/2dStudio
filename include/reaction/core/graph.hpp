#pragma once

#include "reaction/core/node.hpp"

#include <optional>
#include <unordered_map>

namespace reaction {

enum class SubgraphExecution { Pipeline, Simulation };
enum class SubgraphInterfaceKind { Input, Slider, Output };

struct SubgraphInterfaceItem {
    std::string key;
    std::string label;
    SubgraphInterfaceKind kind = SubgraphInterfaceKind::Input;
    ValueType type = ValueType::Float;
    bool optional = false;
    float defaultValue = 0.0F;
    float minimum = 0.0F;
    float maximum = 1.0F;
    ParameterDescriptor::Control control = ParameterDescriptor::Control::Float;
    std::string role;

    SubgraphInterfaceItem() = default;
    SubgraphInterfaceItem(std::string keyValue, std::string labelValue,
                          SubgraphInterfaceKind kindValue, ValueType typeValue,
                          bool optionalValue = false, float defaultValueValue = 0.0F,
                          float minimumValue = 0.0F, float maximumValue = 1.0F,
                          ParameterDescriptor::Control controlValue = ParameterDescriptor::Control::Float,
                          std::string roleValue = {})
        : key(std::move(keyValue)), label(std::move(labelValue)), kind(kindValue), type(typeValue),
          optional(optionalValue), defaultValue(defaultValueValue), minimum(minimumValue),
          maximum(maximumValue), control(controlValue), role(std::move(roleValue)) {}
};

// Kernel nodes are deliberately data rather than C++ classes: editable simulation
// subgraphs can be serialized without introducing a plug-in ABI. Inputs contains
// stable node keys (or interface keys) and properties contains operation settings.
struct SubgraphKernelNode {
    std::string key;
    std::string operation;
    std::vector<std::string> inputs;
    nlohmann::json properties = nlohmann::json::object();
    Vec2 position;
};

struct SubgraphDefinition {
    std::string id;
    int version = 1;
    std::string name;
    std::string category = "Subgraphs";
    SubgraphExecution execution = SubgraphExecution::Pipeline;
    bool immutable = false;
    std::vector<SubgraphInterfaceItem> interface;
    std::vector<SubgraphKernelNode> kernel;
};

struct NodeRecord {
    NodeId id = 0;
    std::string type;
    std::string subgraphId;
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
    [[nodiscard]] const SubgraphDefinition* findSubgraph(std::string_view id) const;
    [[nodiscard]] SubgraphDefinition* findSubgraph(std::string_view id);
    [[nodiscard]] const std::vector<NodeRecord>& nodes() const { return nodes_; }
    [[nodiscard]] std::vector<NodeRecord>& nodes() { return nodes_; }
    [[nodiscard]] const std::vector<LinkRecord>& links() const { return links_; }
    [[nodiscard]] std::vector<LinkRecord>& links() { return links_; }
    [[nodiscard]] const std::vector<SubgraphDefinition>& subgraphs() const { return subgraphs_; }
    [[nodiscard]] std::vector<SubgraphDefinition>& subgraphs() { return subgraphs_; }

    ProjectSettings settings;
    NodeId activeOutput = 0;
    NodeId nextNodeId = 1;
    LinkId nextLinkId = 1;

private:
    std::vector<NodeRecord> nodes_;
    std::vector<LinkRecord> links_;
    std::vector<SubgraphDefinition> subgraphs_;
};

[[nodiscard]] const std::vector<SubgraphDefinition>& builtInSubgraphs();
[[nodiscard]] const SubgraphDefinition* resolveSubgraph(const Graph& graph, std::string_view id);
[[nodiscard]] NodeDescriptor describeSubgraph(const SubgraphDefinition& definition);
[[nodiscard]] const NodeDescriptor* resolveDescriptor(const Graph& graph,
                                                      const NodeRecord& node,
                                                      const NodeRegistry& registry,
                                                      NodeDescriptor& storage);
[[nodiscard]] std::vector<std::string> validateSubgraph(const SubgraphDefinition& definition);

} // namespace reaction
