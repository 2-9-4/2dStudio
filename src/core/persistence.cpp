#include "reaction/core/persistence.hpp"

#include <fstream>
#include <stdexcept>

namespace reaction {

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
        nodes.push_back(std::move(value));
    }

    nlohmann::json links = nlohmann::json::array();
    for (const auto& link : graph.links()) {
        links.push_back({{"id", link.id},
                         {"from", {{"node", link.fromNode}, {"socket", link.fromSocket}}},
                         {"to", {{"node", link.toNode}, {"socket", link.toSocket}}}});
    }
    return {{"formatVersion", kProjectFormatVersion},
            {"project", {{"width", graph.settings.width},
                         {"height", graph.settings.height},
                         {"targetFps", graph.settings.targetFps}}},
            {"nodes", std::move(nodes)}, {"links", std::move(links)},
            {"activeOutput", graph.activeOutput}};
}

Graph deserializeProject(const nlohmann::json& document, const NodeRegistry& registry) {
    if (!document.is_object() || document.value("formatVersion", 0) != kProjectFormatVersion) {
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

    for (const auto& value : document.at("nodes")) {
        NodeRecord node;
        node.id = value.at("id").get<NodeId>();
        node.type = value.at("type").get<std::string>();
        node.typeVersion = value.value("typeVersion", 1);
        const auto& position = value.at("position");
        node.position = {position.at(0).get<float>(), position.at(1).get<float>()};
        node.parameters = value.value("parameters", nlohmann::json::object());
        node.missing = !registry.contains(node.type);
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
