#include "reaction/core/layout.hpp"
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
        {{"out", "Out", ValueType::Float, SocketDirection::Output},
         {"value", "Value", ValueType::Float, SocketDirection::Output}}, {}});
    add(result, {"image", 1, "Image", "Test",
        {{"out", "Out", ValueType::Image2D, SocketDirection::Output}}, {}});
    add(result, {"math", 1, "Math", "Test",
        {{"a", "A", ValueType::AnyNumeric, SocketDirection::Input, true},
         {"b", "B", ValueType::AnyNumeric, SocketDirection::Input, true},
         {"c", "C", ValueType::AnyNumeric, SocketDirection::Input, true},
         {"out", "Out", ValueType::AnyNumeric, SocketDirection::Output},
         {"result", "Result", ValueType::AnyNumeric, SocketDirection::Output}},
        {{"operation", "Operation", 0, 0, 11}, {"a", "A", 0, -10, 10},
         {"b", "B", 0, -10, 10}, {"c", "C", 1, -10, 10}}});
    add(result, {"threshold", 1, "Threshold", "Test",
        {{"value", "Value", ValueType::AnyNumeric, SocketDirection::Input, true},
         {"result", "Result", ValueType::AnyNumeric, SocketDirection::Output}},
        {{"value", "Value", .5F, 0, 1}, {"threshold", "Threshold", .5F, 0, 1}}});
    add(result, {"select", 1, "Select", "Test",
        {{"condition", "Condition", ValueType::AnyNumeric, SocketDirection::Input, true},
         {"ifTrue", "If True", ValueType::AnyNumeric, SocketDirection::Input, true},
         {"ifFalse", "If False", ValueType::AnyNumeric, SocketDirection::Input, true},
         {"result", "Result", ValueType::AnyNumeric, SocketDirection::Output}},
        {{"condition", "Condition", 0.0F, -10.0F, 10.0F},
         {"ifTrue", "If True", 1.0F, -10.0F, 10.0F},
         {"ifFalse", "If False", 0.0F, -10.0F, 10.0F}}});
    add(result, {"coordinates", 1, "Canvas Coordinates", "Input",
        {{"x", "X", ValueType::Image2D, SocketDirection::Output},
         {"y", "Y", ValueType::Image2D, SocketDirection::Output}}, {}});
    add(result, {"laplacian", 1, "Laplacian", "Filter",
        {{"value", "Value", ValueType::AnyVector, SocketDirection::Input},
         {"scale", "Scale", ValueType::Float, SocketDirection::Input, true},
         {"result", "Result", ValueType::AnyVector, SocketDirection::Output}},
        {{"scale", "Scale", 1.0F, .25F, 8.0F}}});
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

TEST_CASE("Vec2 is confined to AnyVector paths and excluded from numeric math") {
    auto nodes = registry();
    add(nodes, {"vec2_source", 1, "Vec2", "Test",
        {{"state", "State", ValueType::Vec2, SocketDirection::Output}}, {}});
    REQUIRE(toString(ValueType::Vec2) == "vec2");
    REQUIRE(toString(ValueType::AnyVector) == "vector");

    Graph vectorGraph;
    const auto state = vectorGraph.addNode("vec2_source");
    const auto laplacian = vectorGraph.addNode("laplacian");
    vectorGraph.addLink(state, "state", laplacian, "value");
    const auto vectorResult = vectorGraph.compile(nodes);
    REQUIRE(vectorResult.valid);
    REQUIRE(vectorResult.inferredOutputs.at(laplacian) == ValueType::Vec2);

    Graph numericGraph;
    const auto numericState = numericGraph.addNode("vec2_source");
    const auto math = numericGraph.addNode("math");
    numericGraph.addLink(numericState, "state", math, "a");
    REQUIRE_FALSE(numericGraph.compile(nodes).valid);
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

TEST_CASE("unknown nodes inside subgraphs preserve their original JSON") {
    auto nodes = registry();
    Graph graph;
    auto definition = builtInSubgraphs().front();
    definition.id = "project.future_body";
    definition.immutable = false;
    graph.subgraphs().push_back(std::move(definition));
    auto document = serializeProject(graph);
    document["subgraphs"][0]["nodes"][0]["type"] = "plugin.future.body";
    document["subgraphs"][0]["nodes"][0]["vendorData"] = 321;

    const auto restored = deserializeProject(document, nodes);
    REQUIRE(restored.subgraphs().front().body.nodes().front().missing);
    const auto serialized = serializeProject(restored);
    REQUIRE(serialized["subgraphs"][0]["nodes"][0]["vendorData"] == 321);
    REQUIRE(serialized["subgraphs"][0]["nodes"][0]["type"] == "plugin.future.body");
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
    REQUIRE(validateSubgraph(builtInSubgraphs().front(), nodes).empty());
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
    REQUIRE(descriptor->sockets[4].label == "Chemical A");
    REQUIRE(descriptor->sockets[5].label == "Chemical B");
    REQUIRE(graph.compile(nodes).valid);
}

TEST_CASE("custom shared subgraphs round trip once with per-instance values") {
    auto nodes = registry();
    Graph graph;
    auto definition = builtInSubgraphs().front();
    definition.id = "project.gray_scott.custom";
    definition.name = "My Gray Scott";
    definition.immutable = false;
    const NodeId labeledBodyNode = definition.body.nodes().front().id;
    definition.body.nodes().front().label = "Custom Coordinate Source";
    graph.subgraphs().push_back(definition);
    const auto first = graph.addNode("subgraph");
    const auto second = graph.addNode("subgraph");
    graph.findNode(first)->subgraphId = definition.id;
    graph.findNode(second)->subgraphId = definition.id;
    graph.findNode(first)->parameters["feed"] = .04F;
    graph.findNode(second)->parameters["feed"] = .07F;

    const auto json = serializeProject(graph);
    REQUIRE(json["formatVersion"] == 3);
    REQUIRE(json["subgraphs"].size() == 1);
    REQUIRE(json["subgraphs"][0].contains("nodes"));
    REQUIRE(json["subgraphs"][0].contains("links"));
    REQUIRE_FALSE(json["subgraphs"][0].contains("kernel"));
    const auto restored = deserializeProject(json, nodes);
    REQUIRE(restored.subgraphs().size() == 1);
    REQUIRE(restored.subgraphs().front().body.nodes().size() == definition.body.nodes().size());
    REQUIRE(restored.subgraphs().front().body.links().size() == definition.body.links().size());
    REQUIRE(restored.subgraphs().front().body.findNode(labeledBodyNode)->label ==
            "Custom Coordinate Source");
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
    const auto cycleA = definition.body.addNode("math");
    const auto cycleB = definition.body.addNode("math");
    definition.body.addLink(cycleA, "result", cycleB, "a");
    definition.body.addLink(cycleB, "result", cycleA, "a");
    definition.body.addNode("subgraph");
    auto nodes = registry();
    const auto errors = validateSubgraph(definition, nodes);
    REQUIRE(std::ranges::any_of(errors, [](const auto& error) { return error.find("range") != std::string::npos; }));
    REQUIRE(std::ranges::any_of(errors, [](const auto& error) { return error.find("cycle") != std::string::npos; }));
    REQUIRE(std::ranges::any_of(errors, [](const auto& error) { return error.find("Nested") != std::string::npos; }));
}

TEST_CASE("graph bodies use stable node and link IDs with root graph mutation semantics") {
    GraphBody body;
    const auto first = body.addNode("float", {10, 20});
    const auto second = body.addNode("math", {30, 40});
    const auto link = body.addLink(first, "value", second, "a");
    REQUIRE(first != second);
    REQUIRE(body.findNode(first)->position.x == 10.0F);
    REQUIRE(body.links().front().id == link);

    const auto replacement = body.addNode("float");
    body.addLink(replacement, "value", second, "a");
    REQUIRE(body.links().size() == 1);
    REQUIRE(body.links().front().fromNode == replacement);
    REQUIRE(body.removeNode(replacement));
    REQUIRE(body.links().empty());
    REQUIRE(body.findNode(first) != nullptr);
}

TEST_CASE("simulation bodies reuse registered normal node descriptors") {
    auto nodes = registry();
    auto definition = builtInSubgraphs().front();
    definition.body.addNode("float");
    definition.body.addNode("select");
    for (const auto* type : {"float", "math", "threshold", "select",
                             "coordinates", "laplacian"}) {
        const auto normalNode = std::ranges::find(definition.body.nodes(), std::string(type),
                                                  &NodeRecord::type);
        REQUIRE(normalNode != definition.body.nodes().end());
        NodeDescriptor storage;
        const auto* resolved = resolveSubgraphBodyDescriptor(definition, *normalNode,
                                                             nodes, storage);
        REQUIRE(resolved == nodes.descriptor(type));
    }
    const auto channel = std::ranges::find(definition.body.nodes(),
                                           std::string("simulation_channel"),
                                           &NodeRecord::type);
    REQUIRE(channel != definition.body.nodes().end());
    NodeDescriptor channelStorage;
    const auto* channelDescriptor = resolveSubgraphBodyDescriptor(
        definition, *channel, nodes, channelStorage);
    REQUIRE(channelDescriptor != nullptr);
    REQUIRE(channelDescriptor->sockets.front().type == ValueType::Vec2);
    REQUIRE(channelDescriptor->sockets[1].type == ValueType::Float);
}

TEST_CASE("simulation topology accepts sources created after their targets") {
    auto nodes = registry();
    auto definition = builtInSubgraphs().front();
    definition.id = "project.out_of_order";
    definition.immutable = false;
    const auto target = definition.body.addNode("math");
    const auto source = definition.body.addNode("float");
    definition.body.addLink(source, "value", target, "a");
    REQUIRE(source > target);
    REQUIRE(validateSubgraph(definition, nodes).empty());
}

TEST_CASE("Vec2 Laplacian results require a simulation channel split before math") {
    auto nodes = registry();
    auto definition = builtInSubgraphs().front();
    const auto laplacian = std::ranges::find(definition.body.nodes(),
                                             std::string("laplacian"),
                                             &NodeRecord::type);
    REQUIRE(laplacian != definition.body.nodes().end());
    const auto laplacianId = laplacian->id;
    const auto math = definition.body.addNode("math");
    definition.body.addLink(laplacianId, "result", math, "a");
    const auto errors = validateSubgraph(definition, nodes);
    REQUIRE(std::ranges::any_of(errors, [](const std::string& error) {
        return error.find("resolved value types") != std::string::npos;
    }));
}

TEST_CASE("format 2 kernel subgraphs migrate to normal node and link bodies") {
    auto nodes = registry();
    const nlohmann::json interface = nlohmann::json::array({
        {{"key", "image"}, {"label", "Image"}, {"kind", "output"}, {"type", "image2d"}},
        {{"key", "a"}, {"label", "Chemical A"}, {"kind", "output"}, {"type", "image2d"}},
        {{"key", "b"}, {"label", "Chemical B"}, {"kind", "output"}, {"type", "image2d"}},
    });
    const nlohmann::json kernel = nlohmann::json::array({
        {{"key", "initialA"}, {"operation", "constant"}, {"properties", {{"value", 1.0F}}}},
        {{"key", "initialB"}, {"operation", "constant"}, {"properties", {{"value", 0.0F}}}},
        {{"key", "initial"}, {"operation", "pack2"}, {"inputs", {"initialA", "initialB"}},
         {"properties", {{"role", "initial"}}}},
        {{"key", "state"}, {"operation", "previous_state"}},
        {{"key", "a0"}, {"operation", "swizzle"}, {"inputs", {"state"}},
         {"properties", {{"channel", 0}}}},
        {{"key", "b0"}, {"operation", "swizzle"}, {"inputs", {"state"}},
         {"properties", {{"channel", 1}}}},
        {{"key", "value"}, {"operation", "constant"}, {"properties", {{"value", .25F}}}},
        {{"key", "inMin"}, {"operation", "constant"}, {"properties", {{"value", 0.0F}}}},
        {{"key", "inMax"}, {"operation", "constant"}, {"properties", {{"value", 1.0F}}}},
        {{"key", "outMin"}, {"operation", "constant"}, {"properties", {{"value", -1.0F}}}},
        {{"key", "outMax"}, {"operation", "constant"}, {"properties", {{"value", 1.0F}}}},
        {{"key", "remapped"}, {"operation", "remap"},
         {"inputs", {"value", "inMin", "inMax", "outMin", "outMax"}}},
        {{"key", "edgeBase"}, {"operation", "constant"}, {"properties", {{"value", .1F}}}},
        {{"key", "edgeOffset"}, {"operation", "constant"}, {"properties", {{"value", .1F}}}},
        {{"key", "dynamicEdge"}, {"operation", "add"}, {"inputs", {"edgeBase", "edgeOffset"}}},
        {{"key", "stepped"}, {"operation", "step"}, {"inputs", {"remapped", "dynamicEdge"}}},
        {{"key", "chosen"}, {"operation", "select"},
         {"inputs", {"stepped", "remapped", "a0"}}},
        {{"key", "next"}, {"operation", "pack2"}, {"inputs", {"chosen", "b0"}},
         {"properties", {{"role", "next"}}}},
        {{"key", "imageOut"}, {"operation", "output"}, {"inputs", {"b0"}},
         {"properties", {{"key", "image"}}}},
        {{"key", "aOut"}, {"operation", "output"}, {"inputs", {"chosen"}},
         {"properties", {{"key", "a"}}}},
        {{"key", "bOut"}, {"operation", "output"}, {"inputs", {"b0"}},
         {"properties", {{"key", "b"}}}},
    });
    const nlohmann::json document = {
        {"formatVersion", 2},
        {"project", {{"width", 128}, {"height", 128}, {"targetFps", 60}}},
        {"subgraphs", nlohmann::json::array({
            {{"id", "project.legacy"}, {"name", "Legacy"}, {"execution", "simulation"},
             {"interface", interface}, {"kernel", kernel}}
        })},
        {"nodes", nlohmann::json::array()}, {"links", nlohmann::json::array()},
        {"activeOutput", 0}
    };

    const auto restored = deserializeProject(document, nodes);
    REQUIRE(restored.subgraphs().size() == 1);
    const auto& migrated = restored.subgraphs().front();
    REQUIRE_FALSE(migrated.body.nodes().empty());
    REQUIRE_FALSE(migrated.body.links().empty());
    REQUIRE(std::ranges::count(migrated.body.nodes(), std::string("select"),
                               &NodeRecord::type) == 1);
    REQUIRE(std::ranges::count(migrated.body.nodes(), std::string("simulation_channel"),
                               &NodeRecord::type) >= 1);
    REQUIRE(std::ranges::count(migrated.body.nodes(), std::string("threshold"),
                               &NodeRecord::type) == 1);
    REQUIRE(std::ranges::count(migrated.body.nodes(), std::string("math"),
                               &NodeRecord::type) >= 8);
    const auto migratedSelect = std::ranges::find(migrated.body.nodes(), std::string("select"),
                                                  &NodeRecord::type);
    REQUIRE(std::ranges::count(migrated.body.links(), migratedSelect->id,
                               &LinkRecord::toNode) == 3);
    REQUIRE(std::ranges::none_of(migrated.body.nodes(), [](const NodeRecord& node) {
        return node.type == "math" &&
               static_cast<int>(node.parameters.value("operation", -1.0F)) == 11;
    }));
    const auto dynamicEdge = std::ranges::find(migrated.body.nodes(),
                                               std::string("Dynamic Edge"),
                                               &NodeRecord::label);
    REQUIRE(dynamicEdge != migrated.body.nodes().end());
    REQUIRE(std::ranges::any_of(migrated.body.links(), [&](const LinkRecord& link) {
        return link.fromNode == dynamicEdge->id;
    }));
    const auto migrationErrors = validateSubgraph(migrated, nodes);
    INFO(nlohmann::json(migrationErrors).dump());
    REQUIRE(migrationErrors.empty());
    const auto serialized = serializeProject(restored);
    REQUIRE(serialized["formatVersion"] == 3);
    REQUIRE(serialized["subgraphs"][0].contains("nodes"));
    REQUIRE_FALSE(serialized["subgraphs"][0].contains("kernel"));
}

TEST_CASE("legacy previous-state channel wiring is flagged and remains invalid") {
    auto nodes = registry();
    Graph graph;
    auto definition = builtInSubgraphs().front();
    definition.id = "project.legacy_previous_state";
    definition.immutable = false;
    const auto previous = std::ranges::find(definition.body.nodes(),
                                            std::string("simulation_previous_state"),
                                            &NodeRecord::type);
    REQUIRE(previous != definition.body.nodes().end());
    auto outgoing = std::ranges::find_if(definition.body.links(), [&](const LinkRecord& link) {
        return link.fromNode == previous->id && link.fromSocket == "state";
    });
    REQUIRE(outgoing != definition.body.links().end());
    outgoing->fromSocket = "a";
    graph.subgraphs().push_back(std::move(definition));

    const auto restored = deserializeProject(serializeProject(graph), nodes);
    const auto& loaded = restored.subgraphs().front();
    const auto loadedPrevious = std::ranges::find(loaded.body.nodes(),
        std::string("simulation_previous_state"), &NodeRecord::type);
    REQUIRE(loadedPrevious != loaded.body.nodes().end());
    REQUIRE(loadedPrevious->needsAttention);
    REQUIRE_FALSE(validateSubgraph(loaded, nodes).empty());
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

TEST_CASE("auto-layout keeps sequential nodes in a line") {
    GraphBody body;
    const auto a = body.addNode("float", {0, 0});
    const auto b = body.addNode("math", {500, 700});
    const auto c = body.addNode("math", {900, 300});
    body.addLink(a, "value", b, "a");
    body.addLink(b, "result", c, "a");
    layoutGraph(body);
    REQUIRE(body.findNode(b)->position.y == body.findNode(a)->position.y);
    REQUIRE(body.findNode(c)->position.y == body.findNode(a)->position.y);
    REQUIRE(body.findNode(b)->position.x > body.findNode(a)->position.x);
    REQUIRE(body.findNode(c)->position.x > body.findNode(b)->position.x);
}

TEST_CASE("auto-layout places merges at the average of their source lines") {
    GraphBody body;
    const auto s1 = body.addNode("float", {0, 0});
    const auto s2 = body.addNode("float", {0, 1000});
    const auto m = body.addNode("math", {500, 0});
    body.addLink(s1, "value", m, "a");
    body.addLink(s2, "value", m, "b");
    layoutGraph(body);
    const float first = body.findNode(s1)->position.y;
    const float second = body.findNode(s2)->position.y;
    REQUIRE(first < second);
    REQUIRE(body.findNode(m)->position.y == (first + second) / 2.0F);
}

TEST_CASE("auto-layout leaves nodes in disjoint components on distinct rows") {
    GraphBody body;
    const auto a = body.addNode("float", {0, 0});
    const auto b = body.addNode("float", {0, 0});
    const auto isolated = body.addNode("float", {0, 0});
    body.addLink(a, "value", b, "a");
    layoutGraph(body);
    REQUIRE(body.findNode(a)->position.x == body.findNode(isolated)->position.x);
    REQUIRE(body.findNode(a)->position.y != body.findNode(isolated)->position.y);
    REQUIRE(body.findNode(b)->position.x > body.findNode(a)->position.x);
}

TEST_CASE("auto-layout layers nodes fed by several links from one source") {
    GraphBody body;
    const auto s = body.addNode("float", {0, 0});
    const auto m = body.addNode("math", {0, 0});
    body.addLink(s, "value", m, "a");
    body.addLink(s, "value", m, "b");
    layoutGraph(body);
    REQUIRE(body.findNode(m)->position.x > body.findNode(s)->position.x);
}

} // namespace reaction
