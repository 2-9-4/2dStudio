#include "reaction/core/persistence.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <filesystem>

namespace reaction {
namespace {

class DummyNode final : public NodeInstance {
public:
    explicit DummyNode(NodeDescriptor descriptor) : descriptor_(std::move(descriptor)) {}
    const NodeDescriptor& descriptor() const override { return descriptor_; }
    void evaluate(EvaluationContext&, std::span<const Value>, std::span<Value>) override {}
    nlohmann::json parameters() const override { return values_; }
    void setParameters(const nlohmann::json& values) override { values_ = values; }
private:
    NodeDescriptor descriptor_;
    nlohmann::json values_;
};

void add(NodeRegistry& registry, NodeDescriptor descriptor) {
    const auto copy = descriptor;
    registry.add(std::move(descriptor), [copy] { return std::make_unique<DummyNode>(copy); });
}

NodeRegistry registry() {
    NodeRegistry result;
    add(result, {"float", 1, "Float", "Test",
        {{"out", "Out", ValueType::Float, SocketDirection::Output}}, {}});
    add(result, {"image", 1, "Image", "Test",
        {{"out", "Out", ValueType::Image2D, SocketDirection::Output}}, {}});
    add(result, {"math", 1, "Math", "Test",
        {{"a", "A", ValueType::AnyNumeric, SocketDirection::Input, true},
         {"out", "Out", ValueType::AnyNumeric, SocketDirection::Output}}, {}});
    add(result, {"output", 1, "Output", "Test",
        {{"in", "In", ValueType::Image2D, SocketDirection::Input},
         {"out", "Out", ValueType::Image2D, SocketDirection::Output}}, {}});
    return result;
}

} // namespace

TEST_CASE("graph compiles in dependency order") {
    auto nodes = registry();
    Graph graph;
    const auto source = graph.addNode("image");
    const auto math = graph.addNode("math");
    const auto output = graph.addNode("output");
    graph.addLink(source, "out", math, "a");
    graph.addLink(math, "out", output, "in");
    graph.activeOutput = output;
    const auto result = graph.compile(nodes);
    REQUIRE(result.valid);
    REQUIRE(result.order == std::vector<NodeId>{source, math, output});
    REQUIRE(result.inferredOutputs.at(math) == ValueType::Image2D);
}

TEST_CASE("cycles are rejected without mutating the graph") {
    auto nodes = registry();
    Graph graph;
    const auto a = graph.addNode("math");
    const auto b = graph.addNode("math");
    graph.addLink(a, "out", b, "a");
    graph.addLink(b, "out", a, "a");
    const auto result = graph.compile(nodes);
    REQUIRE_FALSE(result.valid);
    REQUIRE(result.errors.front().find("cycle") != std::string::npos);
    REQUIRE(graph.links().size() == 2);
}

TEST_CASE("socket errors and duplicate inputs are reported") {
    auto nodes = registry();
    Graph graph;
    const auto scalar = graph.addNode("float");
    const auto image = graph.addNode("image");
    const auto secondImage = graph.addNode("image");
    const auto output = graph.addNode("output");
    graph.links().push_back({1, image, "out", output, "in"});
    graph.links().push_back({2, secondImage, "out", output, "in"});
    graph.links().push_back({3, scalar, "out", output, "in"});
    const auto result = graph.compile(nodes);
    REQUIRE_FALSE(result.valid);
    REQUIRE(result.errors.size() >= 2);
}

TEST_CASE("project JSON round trips graph state") {
    auto nodes = registry();
    Graph graph;
    graph.settings = {640, 480, 30};
    const auto image = graph.addNode("image", {12.5F, 40.0F});
    const auto output = graph.addNode("output", {300.0F, 40.0F});
    graph.findNode(image)->parameters["seed"] = 42;
    graph.addLink(image, "out", output, "in");
    graph.activeOutput = output;

    const auto restored = deserializeProject(serializeProject(graph), nodes);
    REQUIRE(restored.settings.width == 640);
    REQUIRE(restored.settings.targetFps == 30);
    REQUIRE(restored.nodes().size() == 2);
    REQUIRE(restored.links().size() == 1);
    REQUIRE(restored.activeOutput == output);
    REQUIRE(restored.findNode(image)->parameters.at("seed") == 42);
}

TEST_CASE("unknown nodes preserve their original JSON") {
    auto nodes = registry();
    const nlohmann::json document = {
        {"formatVersion", 1}, {"project", {{"width", 128}, {"height", 128}, {"targetFps", 60}}},
        {"nodes", {{{"id", 9}, {"type", "plugin.future"}, {"typeVersion", 7},
                    {"position", {1, 2}}, {"parameters", {{"special", "value"}}}, {"vendorData", 123}}}},
        {"links", nlohmann::json::array()}, {"activeOutput", 0}};
    const auto graph = deserializeProject(document, nodes);
    REQUIRE(graph.nodes().front().missing);
    const auto serialized = serializeProject(graph);
    REQUIRE(serialized["nodes"][0]["vendorData"] == 123);
    REQUIRE(serialized["nodes"][0]["parameters"]["special"] == "value");
}

TEST_CASE("unsupported schema and invalid project settings fail clearly") {
    auto nodes = registry();
    const nlohmann::json unsupported = {{"formatVersion", 99}};
    REQUIRE_THROWS_WITH(deserializeProject(unsupported, nodes),
                        "Unsupported or missing project formatVersion");
    auto document = nlohmann::json{{"formatVersion", 1},
        {"project", {{"width", 0}, {"height", 128}, {"targetFps", 60}}},
        {"nodes", nlohmann::json::array()}, {"links", nlohmann::json::array()}};
    REQUIRE_THROWS_WITH(deserializeProject(document, nodes), "Project settings are out of range");
}

TEST_CASE("built-in discrete reaction exposes a stable dynamic interface") {
    auto nodes = registry();
    REQUIRE(validateSubgraph(builtInSubgraphs().front()).empty());
    Graph graph;
    const auto id = graph.addNode("subgraph");
    graph.findNode(id)->subgraphId = "builtin.reaction_diffusion.discrete";
    NodeDescriptor storage;
    const auto* descriptor = resolveDescriptor(graph, *graph.findNode(id), nodes, storage);
    REQUIRE(descriptor != nullptr);
    REQUIRE(descriptor->displayName == "Reaction Diffusion (Discrete)");
    REQUIRE(descriptor->stateful);
    REQUIRE(std::ranges::count_if(descriptor->sockets, [](const auto& socket) {
        return socket.direction == SocketDirection::Output;
    }) == 3);
    REQUIRE(graph.compile(nodes).valid);
}

TEST_CASE("custom shared subgraphs round trip once with per-instance values") {
    auto nodes = registry();
    Graph graph;
    auto definition = builtInSubgraphs().front();
    definition.id = "project.gray_scott.custom";
    definition.name = "My Gray Scott";
    definition.immutable = false;
    graph.subgraphs().push_back(definition);
    const auto first = graph.addNode("subgraph");
    const auto second = graph.addNode("subgraph");
    graph.findNode(first)->subgraphId = definition.id;
    graph.findNode(second)->subgraphId = definition.id;
    graph.findNode(first)->parameters["feed"] = .04F;
    graph.findNode(second)->parameters["feed"] = .07F;

    const auto json = serializeProject(graph);
    REQUIRE(json["formatVersion"] == 2);
    REQUIRE(json["subgraphs"].size() == 1);
    const auto restored = deserializeProject(json, nodes);
    REQUIRE(restored.subgraphs().size() == 1);
    REQUIRE(restored.findNode(first)->subgraphId == definition.id);
    REQUIRE(restored.findNode(first)->parameters["feed"] != restored.findNode(second)->parameters["feed"]);
    REQUIRE(restored.compile(nodes).valid);
}

TEST_CASE("subgraph labels may change without changing serialized socket keys") {
    auto definition = builtInSubgraphs().front();
    const auto originalKey = definition.interface.front().key;
    definition.interface.front().label = "Painted Feed";
    const auto descriptor = describeSubgraph(definition);
    REQUIRE(descriptor.sockets.front().key == originalKey);
    REQUIRE(descriptor.sockets.front().label == "Painted Feed");
}

TEST_CASE("subgraph validation catches ranges cycles endpoints and nesting") {
    auto definition = builtInSubgraphs().front();
    definition.interface[3].minimum = 1.0F;
    definition.interface[3].maximum = 0.0F;
    definition.kernel.push_back({"cycleA", "add", {"cycleB", "a0"}});
    definition.kernel.push_back({"cycleB", "add", {"cycleA", "b0"}});
    definition.kernel.push_back({"nested", "subgraph", {}});
    const auto errors = validateSubgraph(definition);
    REQUIRE(std::ranges::any_of(errors, [](const auto& error) { return error.find("range") != std::string::npos; }));
    REQUIRE(std::ranges::any_of(errors, [](const auto& error) { return error.find("cycle") != std::string::npos; }));
    REQUIRE(std::ranges::any_of(errors, [](const auto& error) { return error.find("Nested") != std::string::npos; }));
}

TEST_CASE("missing subgraph definitions fail graph compilation") {
    auto nodes = registry();
    Graph graph;
    const auto id = graph.addNode("subgraph");
    graph.findNode(id)->subgraphId = "project.missing";
    const auto result = graph.compile(nodes);
    REQUIRE_FALSE(result.valid);
    REQUIRE(result.errors.front().find("Missing subgraph") != std::string::npos);
}

} // namespace reaction
