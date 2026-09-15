#include "reaction/core/persistence.hpp"
#include "persistence_internal.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace reaction {
namespace {

SocketContract socketContract(const std::string& value) {
    if (value == "vec2") return SocketContract::Vec2Only;
    if (value == "scalar_field") return SocketContract::ScalarFieldOnly;
    if (value == "vector_field") return SocketContract::VectorFieldOnly;
    if (value == "color_image" || value == "image2d")
        return SocketContract::ColorImageOnly;
    if (value == "numeric") return SocketContract::Numeric;
    if (value == "vector" || value == "vector_numeric")
        return SocketContract::VectorNumeric;
    if (value == "any_field") return SocketContract::AnyField;
    if (value == "any_image_value") return SocketContract::AnyImageValue;
    return SocketContract::FloatOnly;
}

std::optional<ValueType> persistedValueType(std::string_view value) {
    for (const auto type : {ValueType::Float, ValueType::Vec2, ValueType::ScalarField,
                            ValueType::VectorField, ValueType::ColorImage})
        if (toString(type) == value) return type;
    return std::nullopt;
}

nlohmann::json serializeNode(const NodeRecord& node) {
    auto value = node.missing && node.preservedJson.is_object()
        ? node.preservedJson
        : nlohmann::json{{"id", node.id}, {"type", node.type},
            {"typeVersion", node.typeVersion},
            {"position", {node.position.x, node.position.y}},
            {"parameters", node.parameters}};
    value["id"] = node.id;
    value["position"] = {node.position.x, node.position.y};
    if (!node.label.empty()) value["label"] = node.label;
    else value.erase("label");
    if (!node.subgraphId.empty()) value["subgraphId"] = node.subgraphId;
    return value;
}

nlohmann::json serializeLink(const LinkRecord& link) {
    return {{"id", link.id},
            {"from", {{"node", link.fromNode}, {"socket", link.fromSocket}}},
            {"to", {{"node", link.toNode}, {"socket", link.toSocket}}}};
}

NodeRecord deserializeNode(const nlohmann::json& value) {
    NodeRecord node;
    node.id = value.at("id").get<NodeId>();
    node.type = value.at("type").get<std::string>();
    node.subgraphId = value.value("subgraphId", std::string{});
    node.typeVersion = value.value("typeVersion", 1);
    node.label = value.value("label", std::string{});
    const auto& position = value.at("position");
    node.position = {position.at(0).get<float>(), position.at(1).get<float>()};
    // Older builds could save a null parameter block for a node created from a
    // plain search-menu entry.  Treat it as an empty object so loading remains
    // safe for all parameter widgets and runtime nodes.
    const auto parameters = value.value("parameters", nlohmann::json::object());
    node.parameters = parameters.is_object() ? parameters : nlohmann::json::object();
    return node;
}

LinkRecord deserializeLink(const nlohmann::json& value) {
    return {value.at("id").get<LinkId>(),
            value.at("from").at("node").get<NodeId>(),
            value.at("from").at("socket").get<std::string>(),
            value.at("to").at("node").get<NodeId>(),
            value.at("to").at("socket").get<std::string>()};
}

void deserializeBody(GraphBody& body, const nlohmann::json& nodes,
                     const nlohmann::json& links) {
    for (const auto& value : nodes) {
        body.nodes().push_back(deserializeNode(value));
        body.nextNodeId = std::max(body.nextNodeId, body.nodes().back().id + 1);
    }
    for (const auto& value : links) {
        body.links().push_back(deserializeLink(value));
        body.nextLinkId = std::max(body.nextLinkId, body.links().back().id + 1);
    }
}

void markLegacyPreviousStateWiring(SubgraphDefinition& definition) {
    for (auto& node : definition.body.nodes()) {
        if (node.type != "simulation_previous_state") continue;
        node.needsAttention = std::ranges::any_of(
            definition.body.links(), [&](const LinkRecord& link) {
                return link.fromNode == node.id &&
                       (link.fromSocket == "a" || link.fromSocket == "b");
            });
    }
}

// Format-4 editable reaction graphs used one implicit RG feedback texture. Turn
// that representation into the explicit Vector Field slot without changing the
// math or the A/B presentation outputs.
nlohmann::json serializeSubgraph(const SubgraphDefinition& definition) {
    nlohmann::json interface = nlohmann::json::array();
    for (const auto& item : definition.interface) {
        const char* kind = item.kind == SubgraphInterfaceKind::Input ? "input" : "output";
        const char* control = item.control == ParameterDescriptor::Control::Integer ? "integer" :
                              item.control == ParameterDescriptor::Control::Boolean ? "boolean" :
                              item.control == ParameterDescriptor::Control::Enum ? "enum" : "float";
        interface.push_back({{"key", item.key}, {"label", item.label}, {"kind", kind},
            {"type", toString(item.contract)}, {"optional", item.optional},
            {"default", item.defaultValue}, {"minimum", item.minimum}, {"maximum", item.maximum},
            {"control", control}, {"options", item.enumOptions}, {"role", item.role}});
    }
    nlohmann::json stateSlots = nlohmann::json::array();
    for (const auto& slot : definition.stateSlots)
        stateSlots.push_back({{"key", slot.key}, {"label", slot.label}, {"type", toString(slot.type)}});
    nlohmann::json nodes = nlohmann::json::array();
    for (const auto& node : definition.body.nodes()) nodes.push_back(serializeNode(node));
    nlohmann::json links = nlohmann::json::array();
    for (const auto& link : definition.body.links()) links.push_back(serializeLink(link));
    return {{"id", definition.id}, {"version", definition.version}, {"name", definition.name},
            {"category", definition.category},
            {"execution", definition.execution == SubgraphExecution::Simulation ? "simulation" : "pipeline"},
            {"interface", std::move(interface)}, {"stateSlots", std::move(stateSlots)}, {"nodes", std::move(nodes)},
            {"links", std::move(links)}};
}

SubgraphDefinition deserializeSubgraph(const nlohmann::json& value,
                                       const NodeRegistry& registry) {
    SubgraphDefinition result;
    result.id = value.at("id").get<std::string>();
    result.version = value.value("version", 1);
    result.name = value.at("name").get<std::string>();
    result.category = value.value("category", std::string("Subgraphs"));
    result.execution = value.value("execution", std::string("pipeline")) == "simulation"
        ? SubgraphExecution::Simulation : SubgraphExecution::Pipeline;
    for (const auto& entry : value.value("interface", nlohmann::json::array())) {
        SubgraphInterfaceItem item;
        item.key = entry.at("key").get<std::string>();
        item.label = entry.value("label", item.key);
        const auto kind = entry.value("kind", std::string("input"));
        item.kind = kind == "output" ? SubgraphInterfaceKind::Output
                                      : SubgraphInterfaceKind::Input;
        item.contract = socketContract(entry.value("type", std::string("float")));
        item.optional = item.optional || entry.value("optional", false);
        item.defaultValue = entry.value("default", 0.0F);
        item.minimum = entry.value("minimum", 0.0F);
        item.maximum = entry.value("maximum", 1.0F);
        const auto control = entry.value("control", std::string("float"));
        item.control = control == "integer" ? ParameterDescriptor::Control::Integer :
                       control == "boolean" ? ParameterDescriptor::Control::Boolean :
                       control == "enum" ? ParameterDescriptor::Control::Enum :
                       ParameterDescriptor::Control::Float;
        if (entry.contains("options") && entry["options"].is_array())
            for (const auto& option : entry["options"])
                if (option.is_string()) item.enumOptions.push_back(option.get<std::string>());
        item.role = entry.value("role", std::string{});
        result.interface.push_back(std::move(item));
    }
    for (const auto& entry : value.value("stateSlots", nlohmann::json::array())) {
        SimulationStateSlot slot;
        slot.key = entry.at("key").get<std::string>();
        slot.label = entry.value("label", slot.key);
        const auto type = entry.value("type", std::string("scalar_field"));
        slot.type = persistedValueType(type).value_or(ValueType::ScalarField);
        result.stateSlots.push_back(std::move(slot));
    }
    const auto& nodes = value.at("nodes");
    deserializeBody(result.body, nodes, value.value("links", nlohmann::json::array()));
    markLegacyPreviousStateWiring(result);
    for (std::size_t index = 0; index < result.body.nodes().size(); ++index) {
        auto& node = result.body.nodes()[index];
        NodeDescriptor descriptorStorage;
        node.missing = resolveSubgraphBodyDescriptor(result, node, registry,
                                                     descriptorStorage) == nullptr;
        if (node.missing) node.preservedJson = nodes[index];
    }
    return result;
}

} // namespace

nlohmann::json serializeProject(const Graph& graph) {
    nlohmann::json nodes = nlohmann::json::array();
    for (const auto& node : graph.nodes()) nodes.push_back(serializeNode(node));

    nlohmann::json links = nlohmann::json::array();
    for (const auto& link : graph.links()) links.push_back(serializeLink(link));
    nlohmann::json subgraphs = nlohmann::json::array();
    for (const auto& definition : graph.subgraphs()) subgraphs.push_back(serializeSubgraph(definition));
    return {{"formatVersion", kProjectFormatVersion},
            {"project", {{"width", graph.settings.width},
                         {"height", graph.settings.height},
                         {"targetFps", graph.settings.targetFps}}},
            {"subgraphs", std::move(subgraphs)}, {"nodes", std::move(nodes)}, {"links", std::move(links)},
            {"activeOutput", graph.activeOutput}};
}

Graph persistence_internal::deserializeCurrentProject(
    const nlohmann::json& document, const NodeRegistry& registry) {
    if (!document.is_object() ||
        document.value("formatVersion", 0) != kProjectFormatVersion) {
        throw std::runtime_error("Current project parser requires formatVersion " +
                                 std::to_string(kProjectFormatVersion));
    }

    Graph graph;
    const auto& project = document.at("project");
    graph.settings.width = project.value("width", 1024);
    graph.settings.height = project.value("height", 1024);
    graph.settings.targetFps = project.value("targetFps", 60);
    if (graph.settings.width < 16 || graph.settings.height < 16 ||
        graph.settings.targetFps < 1 || graph.settings.targetFps > 240) {
        throw std::runtime_error("Project settings are out of range");
    }

    for (const auto& value : document.value("subgraphs", nlohmann::json::array()))
        graph.subgraphs().push_back(deserializeSubgraph(value, registry));

    for (const auto& value : document.at("nodes")) {
        auto node = deserializeNode(value);
        NodeDescriptor descriptorStorage;
        node.missing = resolveDescriptor(graph, node, registry, descriptorStorage) == nullptr;
        if (node.missing) node.preservedJson = value;
        graph.nodes().push_back(std::move(node));
        graph.nextNodeId = std::max(graph.nextNodeId, graph.nodes().back().id + 1);
    }
    for (const auto& value : document.at("links")) {
        auto link = deserializeLink(value);
        graph.links().push_back(std::move(link));
        graph.nextLinkId = std::max(graph.nextLinkId, graph.links().back().id + 1);
    }
    graph.activeOutput = document.value("activeOutput", NodeId{0});
    return graph;
}

Graph deserializeProject(const nlohmann::json& document, const NodeRegistry& registry) {
    return persistence_internal::deserializeCurrentProject(
        migrateProjectJson(document, registry), registry);
}

void saveProjectAtomic(const Graph& graph, const std::filesystem::path& path) {
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot open temporary project file");
        output << serializeProject(graph).dump(2) << '\n';
        if (!output) throw std::runtime_error("Failed writing project file");
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
    }
    if (error) throw std::runtime_error("Cannot replace project file: " + error.message());
}

Graph loadProject(const std::filesystem::path& path, const NodeRegistry& registry) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open project file");
    nlohmann::json document;
    input >> document;
    return deserializeProject(document, registry);
}

} // namespace reaction
