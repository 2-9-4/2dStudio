#include "reaction/core/persistence.hpp"

#include <fstream>
#include <stdexcept>

namespace reaction {
namespace {

std::string valueTypeName(ValueType value) { return toString(value); }
ValueType valueType(const std::string& value) {
    if (value == "image2d") return ValueType::Image2D;
    if (value == "numeric") return ValueType::AnyNumeric;
    return ValueType::Float;
}

nlohmann::json serializeSubgraph(const SubgraphDefinition& definition) {
    nlohmann::json interface = nlohmann::json::array();
    for (const auto& item : definition.interface) {
        const char* kind = item.kind == SubgraphInterfaceKind::Input ? "input" :
                           item.kind == SubgraphInterfaceKind::Slider ? "slider" : "output";
        const char* control = item.control == ParameterDescriptor::Control::Integer ? "integer" :
                              item.control == ParameterDescriptor::Control::Boolean ? "boolean" : "float";
        interface.push_back({{"key", item.key}, {"label", item.label}, {"kind", kind},
            {"type", valueTypeName(item.type)}, {"optional", item.optional},
            {"default", item.defaultValue}, {"minimum", item.minimum}, {"maximum", item.maximum},
            {"control", control}, {"role", item.role}});
    }
    nlohmann::json kernel = nlohmann::json::array();
    for (const auto& item : definition.kernel) {
        kernel.push_back({{"key", item.key}, {"operation", item.operation}, {"inputs", item.inputs},
                          {"properties", item.properties},
                          {"position", {item.position.x, item.position.y}}});
    }
    return {{"id", definition.id}, {"version", definition.version}, {"name", definition.name},
            {"category", definition.category},
            {"execution", definition.execution == SubgraphExecution::Simulation ? "simulation" : "pipeline"},
            {"interface", std::move(interface)}, {"kernel", std::move(kernel)}};
}

SubgraphDefinition deserializeSubgraph(const nlohmann::json& value) {
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
        item.kind = kind == "slider" ? SubgraphInterfaceKind::Slider :
                    kind == "output" ? SubgraphInterfaceKind::Output : SubgraphInterfaceKind::Input;
        item.type = valueType(entry.value("type", std::string("float")));
        item.optional = entry.value("optional", false);
        item.defaultValue = entry.value("default", 0.0F);
        item.minimum = entry.value("minimum", 0.0F);
        item.maximum = entry.value("maximum", 1.0F);
        const auto control = entry.value("control", std::string("float"));
        item.control = control == "integer" ? ParameterDescriptor::Control::Integer :
                       control == "boolean" ? ParameterDescriptor::Control::Boolean :
                       ParameterDescriptor::Control::Float;
        item.role = entry.value("role", std::string{});
        result.interface.push_back(std::move(item));
    }
    for (const auto& entry : value.value("kernel", nlohmann::json::array())) {
        SubgraphKernelNode item;
        item.key = entry.at("key").get<std::string>();
        item.operation = entry.at("operation").get<std::string>();
        item.inputs = entry.value("inputs", std::vector<std::string>{});
        item.properties = entry.value("properties", nlohmann::json::object());
        const auto position = entry.value("position", nlohmann::json::array({0, 0}));
        item.position = {position.at(0).get<float>(), position.at(1).get<float>()};
        result.kernel.push_back(std::move(item));
    }
    return result;
}

} // namespace

nlohmann::json serializeProject(const Graph& graph) {
    nlohmann::json nodes = nlohmann::json::array();
    for (const auto& node : graph.nodes()) {
        auto value = node.missing && node.preservedJson.is_object()
            ? node.preservedJson
            : nlohmann::json{{"id", node.id}, {"type", node.type},
                {"typeVersion", node.typeVersion},
                {"position", {node.position.x, node.position.y}},
                {"parameters", node.parameters}};
        value["id"] = node.id;
        value["position"] = {node.position.x, node.position.y};
        if (!node.subgraphId.empty()) value["subgraphId"] = node.subgraphId;
        nodes.push_back(std::move(value));
    }

    nlohmann::json links = nlohmann::json::array();
    for (const auto& link : graph.links()) {
        links.push_back({{"id", link.id},
                         {"from", {{"node", link.fromNode}, {"socket", link.fromSocket}}},
                         {"to", {{"node", link.toNode}, {"socket", link.toSocket}}}});
    }
    nlohmann::json subgraphs = nlohmann::json::array();
    for (const auto& definition : graph.subgraphs()) subgraphs.push_back(serializeSubgraph(definition));
    return {{"formatVersion", kProjectFormatVersion},
            {"project", {{"width", graph.settings.width},
                         {"height", graph.settings.height},
                         {"targetFps", graph.settings.targetFps}}},
            {"subgraphs", std::move(subgraphs)}, {"nodes", std::move(nodes)}, {"links", std::move(links)},
            {"activeOutput", graph.activeOutput}};
}

Graph deserializeProject(const nlohmann::json& document, const NodeRegistry& registry) {
    const int format = document.value("formatVersion", 0);
    if (!document.is_object() || (format != 1 && format != kProjectFormatVersion)) {
        throw std::runtime_error("Unsupported or missing project formatVersion");
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

    if (format >= 2) {
        for (const auto& value : document.value("subgraphs", nlohmann::json::array()))
            graph.subgraphs().push_back(deserializeSubgraph(value));
    }
    for (const auto& value : document.at("nodes")) {
        NodeRecord node;
        node.id = value.at("id").get<NodeId>();
        node.type = value.at("type").get<std::string>();
        node.subgraphId = value.value("subgraphId", std::string{});
        node.typeVersion = value.value("typeVersion", 1);
        const auto& position = value.at("position");
        node.position = {position.at(0).get<float>(), position.at(1).get<float>()};
        node.parameters = value.value("parameters", nlohmann::json::object());
        NodeDescriptor descriptorStorage;
        node.missing = resolveDescriptor(graph, node, registry, descriptorStorage) == nullptr;
        if (node.missing) node.preservedJson = value;
        graph.nodes().push_back(std::move(node));
        graph.nextNodeId = std::max(graph.nextNodeId, graph.nodes().back().id + 1);
    }
    for (const auto& value : document.at("links")) {
        LinkRecord link{value.at("id").get<LinkId>(),
                        value.at("from").at("node").get<NodeId>(),
                        value.at("from").at("socket").get<std::string>(),
                        value.at("to").at("node").get<NodeId>(),
                        value.at("to").at("socket").get<std::string>()};
        graph.links().push_back(std::move(link));
        graph.nextLinkId = std::max(graph.nextLinkId, graph.links().back().id + 1);
    }
    graph.activeOutput = document.value("activeOutput", NodeId{0});
    return graph;
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
