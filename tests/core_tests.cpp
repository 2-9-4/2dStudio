#include "reaction/core/layout.hpp"
#include "reaction/core/math.hpp"
#include "reaction/core/persistence.hpp"
#include "reaction/core/vector_math.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <filesystem>
#include <stdexcept>

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
    add(result, {"vector", 1, "Vector", "Test",
        {{"value", "Vector", ValueType::Vec2, SocketDirection::Output}},
        {{"x", "X", 0.0F, -10.0F, 10.0F}, {"y", "Y", 0.0F, -10.0F, 10.0F}}});
    NodeDescriptor combine{"combine_vector", 1, "Combine Vector", "Test",
        {{"x", "X", SocketContract::Numeric, SocketDirection::Input, true},
         {"y", "Y", SocketContract::Numeric, SocketDirection::Input, true},
         {"value", "Vector", SocketContract::VectorNumeric, SocketDirection::Output}}, {}};
    combine.sockets.back().typePolicy = SocketDescriptor::TypePolicy::VectorPromotion;
    combine.sockets.back().typeInputs = {"x", "y"};
    combine.lowerable = true;
    add(result, std::move(combine));
    add(result, {"image", 1, "Image", "Test",
        {{"out", "Out", ValueType::ScalarField, SocketDirection::Output}}, {}});
    add(result, {"math", 1, "Math", "Test",
        {{"a", "A", SocketContract::Numeric, SocketDirection::Input, true},
         {"b", "B", SocketContract::Numeric, SocketDirection::Input, true},
         {"c", "C", SocketContract::Numeric, SocketDirection::Input, true},
         {"out", "Out", SocketContract::Numeric, SocketDirection::Output},
         {"result", "Result", SocketContract::Numeric, SocketDirection::Output}},
        {{"operation", "Operation", 0, 0, 11}, {"a", "A", 0, -10, 10},
         {"b", "B", 0, -10, 10}, {"c", "C", 1, -10, 10}}});
    add(result, {"threshold", 1, "Threshold", "Test",
        {{"value", "Value", SocketContract::Numeric, SocketDirection::Input, true},
         {"result", "Result", SocketContract::Numeric, SocketDirection::Output}},
        {{"value", "Value", .5F, 0, 1}, {"threshold", "Threshold", .5F, 0, 1}}});
    NodeDescriptor select{"select", 1, "Select", "Test",
        {{"condition", "Condition", SocketContract::Numeric, SocketDirection::Input, true},
         {"ifTrue", "If True", SocketContract::AnyImageValue, SocketDirection::Input, true},
         {"ifFalse", "If False", SocketContract::AnyImageValue, SocketDirection::Input, true},
         {"result", "Result", SocketContract::AnyImageValue, SocketDirection::Output}},
        {{"condition", "Condition", 0.0F, -10.0F, 10.0F},
         {"ifTrue", "If True", 1.0F, -10.0F, 10.0F},
         {"ifFalse", "If False", 0.0F, -10.0F, 10.0F}}};
    select.sockets.back().typePolicy = SocketDescriptor::TypePolicy::WidestValue;
    select.sockets.back().typeInputs = {"ifTrue", "ifFalse"};
    select.sockets.back().fieldInputs = {"condition", "ifTrue", "ifFalse"};
    add(result, std::move(select));
    add(result, {"coordinates", 1, "Canvas Coordinates", "Input",
        {{"coordinates", "Coordinates", SocketContract::VectorNumeric, SocketDirection::Output}},
        {{"pixels", "Pixels", 0.0F, 0.0F, 1.0F,
          ParameterDescriptor::Control::Boolean}}});
    add(result, {"laplacian", 1, "Laplacian", "Filter",
        {{"value", "Value", SocketContract::VectorNumeric, SocketDirection::Input},
         {"scale", "Scale", ValueType::Float, SocketDirection::Input, true},
         {"result", "Result", SocketContract::VectorNumeric, SocketDirection::Output}},
        {{"scale", "Scale", 1.0F, .25F, 8.0F}}});
    add(result, {"output", 1, "Output", "Test",
        {{"in", "In", SocketContract::AnyField, SocketDirection::Input},
         {"out", "Out", SocketContract::AnyField, SocketDirection::Output}}, {}});
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
    REQUIRE(result.socketType(math, "out") == ValueType::ScalarField);
}

TEST_CASE("registry rejects incomplete enum metadata") {
    NodeRegistry nodes;
    REQUIRE_THROWS_AS(add(nodes, {"bad_enum", 1, "Bad Enum", "Test", {},
        {{"mode", "Mode", 0.0F, 0.0F, 2.0F, ParameterDescriptor::Control::Enum,
          {"Only one label"}}}}), std::invalid_argument);
}

TEST_CASE("registry validates generated field and neighborhood declarations") {
    NodeRegistry nodes;
    auto generator = NodeDescriptor{"bad_generator", 1, "Bad Generator", "Test",
        {{"out", "Out", SocketContract::Numeric, SocketDirection::Output}}, {}};
    generator.producedField = true;
    REQUIRE_THROWS_AS(add(nodes, generator), std::invalid_argument);

    auto filter = NodeDescriptor{"bad_filter", 1, "Bad Filter", "Test",
        {{"source", "Source", ValueType::ColorImage, SocketDirection::Input},
         {"out", "Out", ValueType::ColorImage, SocketDirection::Output}}, {}};
    filter.lowerable = true;
    filter.neighborhoodSocket = "source";
    REQUIRE_THROWS_AS(add(nodes, filter), std::invalid_argument);
}

TEST_CASE("producedField makes a Numeric generator infer a field output") {
    NodeRegistry nodes;
    auto generator = NodeDescriptor{"field_generator", 1, "Field Generator", "Test",
        {{"out", "Out", SocketContract::Numeric, SocketDirection::Output}}, {}};
    generator.lowerable = true;
    generator.producedField = true;
    add(nodes, generator);

    Graph graph;
    const auto node = graph.addNode("field_generator");
    const auto compiled = graph.compile(nodes);
    REQUIRE(compiled.valid);
    REQUIRE(compiled.socketType(node, "out") == ValueType::ScalarField);
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

TEST_CASE("Vec2 is confined to VectorNumeric paths and excluded from numeric math") {
    auto nodes = registry();
    add(nodes, {"vec2_source", 1, "Vec2", "Test",
        {{"state", "State", ValueType::Vec2, SocketDirection::Output}}, {}});
    REQUIRE(toString(ValueType::Vec2) == "vec2");
    REQUIRE(toString(SocketContract::VectorNumeric) == "vector_numeric");

    Graph vectorGraph;
    const auto state = vectorGraph.addNode("vec2_source");
    const auto laplacian = vectorGraph.addNode("laplacian");
    vectorGraph.addLink(state, "state", laplacian, "value");
    const auto vectorResult = vectorGraph.compile(nodes);
    REQUIRE(vectorResult.valid);
    REQUIRE(vectorResult.socketType(laplacian, "result") == ValueType::Vec2);

    Graph numericGraph;
    const auto numericState = numericGraph.addNode("vec2_source");
    const auto math = numericGraph.addNode("math");
    numericGraph.addLink(numericState, "state", math, "a");
    REQUIRE_FALSE(numericGraph.compile(nodes).valid);
}

TEST_CASE("semantic contracts are distinct from the widening table") {
    constexpr std::array allTypes{ValueType::Float, ValueType::Vec2,
        ValueType::ScalarField, ValueType::VectorField, ValueType::ColorImage};
    for (const auto expected : allTypes) {
        for (const auto actual : allTypes)
            REQUIRE(contractAccepts(exactContract(expected), actual) == (expected == actual));
        REQUIRE(contractAccepts(SocketContract::AnyImageValue, expected));
        REQUIRE(contractAccepts(SocketContract::AnyField, expected) == isFieldType(expected));
        REQUIRE(coercionBetween(expected, expected) == Coercion::Identity);
    }
    REQUIRE(contractAccepts(SocketContract::Numeric, ValueType::Float));
    REQUIRE(contractAccepts(SocketContract::Numeric, ValueType::ScalarField));
    REQUIRE_FALSE(contractAccepts(SocketContract::Numeric, ValueType::Vec2));
    REQUIRE_FALSE(contractAccepts(SocketContract::Numeric, ValueType::VectorField));
    REQUIRE_FALSE(contractAccepts(SocketContract::VectorNumeric, ValueType::Float));
    REQUIRE_FALSE(contractAccepts(SocketContract::VectorNumeric, ValueType::ScalarField));
    REQUIRE(contractAccepts(SocketContract::VectorNumeric, ValueType::Vec2));
    REQUIRE(contractAccepts(SocketContract::VectorNumeric, ValueType::VectorField));

    REQUIRE(coercionBetween(ValueType::Float, ValueType::ScalarField) ==
            Coercion::FloatToScalarField);
    REQUIRE(coercionBetween(ValueType::Float, ValueType::Vec2) ==
            Coercion::FloatToVec2);
    REQUIRE(coercionBetween(ValueType::Float, ValueType::VectorField) ==
            Coercion::FloatToVectorField);
    REQUIRE(coercionBetween(ValueType::Float, ValueType::ColorImage) ==
            Coercion::FloatToColorImage);
    REQUIRE(coercionBetween(ValueType::Vec2, ValueType::VectorField) ==
            Coercion::Vec2ToVectorField);
    REQUIRE(coercionBetween(ValueType::Vec2, ValueType::ColorImage) ==
            Coercion::Vec2ToColorImage);
    REQUIRE(coercionBetween(ValueType::ScalarField, ValueType::VectorField) ==
            Coercion::ScalarFieldToVectorField);
    REQUIRE(coercionBetween(ValueType::ScalarField, ValueType::ColorImage) ==
            Coercion::ScalarFieldToColorImage);
    REQUIRE(coercionBetween(ValueType::VectorField, ValueType::ColorImage) ==
            Coercion::VectorFieldToColorImage);
    REQUIRE_FALSE(coercionBetween(ValueType::Vec2, ValueType::ScalarField));
    REQUIRE_FALSE(coercionBetween(ValueType::VectorField, ValueType::ScalarField));
    REQUIRE_FALSE(coercionBetween(ValueType::ColorImage, ValueType::VectorField));
}

TEST_CASE("widest-value inference is per socket and records edge coercions") {
    NodeRegistry nodes;
    add(nodes, {"float_source", 1, "Float", "Test",
        {{"value", "Value", ValueType::Float, SocketDirection::Output}}, {}});
    add(nodes, {"scalar_source", 1, "Scalar", "Test",
        {{"value", "Value", ValueType::ScalarField, SocketDirection::Output}}, {}});
    add(nodes, {"color_source", 1, "Color", "Test",
        {{"value", "Value", ValueType::ColorImage, SocketDirection::Output}}, {}});
    auto widest = NodeDescriptor{"widest", 1, "Widest", "Test",
        {{"a", "A", SocketContract::AnyImageValue, SocketDirection::Input, true},
         {"b", "B", SocketContract::AnyImageValue, SocketDirection::Input, true},
         {"factor", "Factor", SocketContract::Numeric, SocketDirection::Input, true},
         {"value", "Value", SocketContract::AnyImageValue, SocketDirection::Output}}, {}};
    widest.sockets.back().typePolicy = SocketDescriptor::TypePolicy::WidestValue;
    widest.sockets.back().typeInputs = {"a", "b"};
    widest.sockets.back().fieldInputs = {"a", "b", "factor"};
    add(nodes, widest);

    Graph graph;
    const auto scalar = graph.addNode("scalar_source");
    const auto color = graph.addNode("color_source");
    const auto factor = graph.addNode("scalar_source");
    const auto operation = graph.addNode("widest");
    const auto scalarLink = graph.addLink(scalar, "value", operation, "a");
    graph.addLink(color, "value", operation, "b");
    graph.addLink(factor, "value", operation, "factor");
    const auto compiled = graph.compile(nodes);
    INFO(nlohmann::json(compiled.errors).dump());
    REQUIRE(compiled.valid);
    REQUIRE(compiled.socketType(operation, "value") == ValueType::ColorImage);
    REQUIRE(compiled.socketType(operation, "a", SocketDirection::Input) ==
            ValueType::ColorImage);
    REQUIRE(compiled.socketType(operation, "factor", SocketDirection::Input) ==
            ValueType::ScalarField);
    REQUIRE(compiled.resolvedEdges.at(scalarLink).coercion ==
            Coercion::ScalarFieldToColorImage);

    Graph spatialFactorGraph;
    const auto a = spatialFactorGraph.addNode("float_source");
    const auto b = spatialFactorGraph.addNode("float_source");
    const auto spatialFactor = spatialFactorGraph.addNode("scalar_source");
    const auto spatialOperation = spatialFactorGraph.addNode("widest");
    spatialFactorGraph.addLink(a, "value", spatialOperation, "a");
    spatialFactorGraph.addLink(b, "value", spatialOperation, "b");
    spatialFactorGraph.addLink(spatialFactor, "value", spatialOperation, "factor");
    const auto spatialCompiled = spatialFactorGraph.compile(nodes);
    REQUIRE(spatialCompiled.valid);
    REQUIRE(spatialCompiled.socketType(spatialOperation, "value") ==
            ValueType::ScalarField);
}

TEST_CASE("multi-output and preserve policies do not leak types between sockets") {
    NodeRegistry nodes;
    add(nodes, {"multi", 1, "Multi", "Test",
        {{"scalar", "Scalar", ValueType::ScalarField, SocketDirection::Output},
         {"vector", "Vector", ValueType::VectorField, SocketDirection::Output}}, {}});
    auto preserve = NodeDescriptor{"preserve", 1, "Preserve", "Test",
        {{"source", "Source", SocketContract::AnyField, SocketDirection::Input},
         {"value", "Value", SocketContract::AnyField, SocketDirection::Output}}, {}};
    preserve.sockets.back().typePolicy = SocketDescriptor::TypePolicy::PreserveInput;
    preserve.sockets.back().typeInputs = {"source"};
    add(nodes, preserve);
    Graph graph;
    const auto multi = graph.addNode("multi");
    const auto filter = graph.addNode("preserve");
    graph.addLink(multi, "vector", filter, "source");
    const auto compiled = graph.compile(nodes);
    REQUIRE(compiled.valid);
    REQUIRE(compiled.socketType(multi, "scalar") == ValueType::ScalarField);
    REQUIRE(compiled.socketType(multi, "vector") == ValueType::VectorField);
    REQUIRE(compiled.socketType(filter, "value") == ValueType::VectorField);
}

TEST_CASE("Vector Math descriptors expose only active semantic sockets") {
    const auto normalize = vectorMathDescriptor(VectorMathOperation::Normalize);
    REQUIRE(normalize.sockets.size() == 2);
    REQUIRE(normalize.sockets[0].key == "a");
    REQUIRE(normalize.sockets[0].contract == SocketContract::VectorNumeric);
    REQUIRE(normalize.sockets[1].key == "result");
    REQUIRE(normalize.sockets[1].contract == SocketContract::VectorNumeric);

    const auto dot = vectorMathDescriptor(VectorMathOperation::Dot);
    REQUIRE(dot.sockets.size() == 3);
    REQUIRE(dot.sockets[0].key == "a");
    REQUIRE(dot.sockets[1].key == "b");
    REQUIRE(dot.sockets[2].contract == SocketContract::Numeric);

    const auto scale = vectorMathDescriptor(VectorMathOperation::Scale);
    REQUIRE(scale.sockets.size() == 3);
    REQUIRE(scale.sockets[0].contract == SocketContract::VectorNumeric);
    REQUIRE(scale.sockets[1].key == "scalar");
    REQUIRE(scale.sockets[1].contract == SocketContract::Numeric);
    REQUIRE(scale.sockets[2].contract == SocketContract::VectorNumeric);

}

TEST_CASE("Vector Math scalar results do not feed vector-only operations") {
    auto nodes = registry();
    add(nodes, {"vec2_source", 1, "Vec2", "Test",
        {{"value", "Value", ValueType::Vec2, SocketDirection::Output}}, {}});

    Graph graph;
    const auto source = graph.addNode("vec2_source");
    const auto dot = graph.addNode("vector_math");
    graph.findNode(dot)->parameters["operation"] = static_cast<float>(VectorMathOperation::Dot);
    const auto laplacian = graph.addNode("laplacian");
    graph.addLink(source, "value", dot, "a");
    graph.addLink(source, "value", dot, "b");
    graph.addLink(dot, "result", laplacian, "value");
    REQUIRE_FALSE(graph.compile(nodes).valid);
}

TEST_CASE("Vector Math field controls promote vector results without changing width") {
    auto nodes = registry();
    Graph graph;
    const auto vector = graph.addNode("vector");
    const auto scalarField = graph.addNode("image");
    const auto scale = graph.addNode("vector_math");
    graph.findNode(scale)->parameters["operation"] =
        static_cast<float>(VectorMathOperation::Scale);
    graph.addLink(vector, "value", scale, "a");
    graph.addLink(scalarField, "out", scale, "scalar");
    const auto compiled = graph.compile(nodes);
    INFO(nlohmann::json(compiled.errors).dump());
    REQUIRE(compiled.valid);
    REQUIRE(compiled.socketType(scale, "result") == ValueType::VectorField);
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

TEST_CASE("legacy color narrowing migrates through explicit channel nodes") {
    auto nodes = registry();
    add(nodes, {"legacy_color", 1, "Legacy Color", "Test",
        {{"value", "Value", ValueType::ColorImage, SocketDirection::Output}}, {}});
    add(nodes, {"color_r", 1, "Color to R", "Test",
        {{"color", "Color", ValueType::ColorImage, SocketDirection::Input},
         {"value", "R", ValueType::ScalarField, SocketDirection::Output}}, {}});
    add(nodes, {"color_rg", 1, "Color to RG", "Test",
        {{"color", "Color", ValueType::ColorImage, SocketDirection::Input},
         {"value", "RG", ValueType::VectorField, SocketDirection::Output}}, {}});
    const nlohmann::json document = {
        {"formatVersion", 3},
        {"project", {{"width", 128}, {"height", 128}, {"targetFps", 60}}},
        {"nodes", {{{"id", 1}, {"type", "legacy_color"}, {"position", {0, 0}},
                     {"parameters", nlohmann::json::object()}},
                    {{"id", 2}, {"type", "math"}, {"position", {200, 0}},
                     {"parameters", nlohmann::json::object()}}}},
        {"links", {{{"id", 7}, {"from", {{"node", 1}, {"socket", "value"}}},
                     {"to", {{"node", 2}, {"socket", "a"}}}}}},
        {"activeOutput", 0}};

    const auto restored = deserializeProject(document, nodes);
    REQUIRE(restored.nodes().size() == 3);
    const auto conversion = std::ranges::find(restored.nodes(), std::string("color_r"),
                                               &NodeRecord::type);
    REQUIRE(conversion != restored.nodes().end());
    REQUIRE(restored.links().size() == 2);
    const auto preserved = std::ranges::find(restored.links(), LinkId{7}, &LinkRecord::id);
    REQUIRE(preserved != restored.links().end());
    REQUIRE(preserved->toNode == conversion->id);
    REQUIRE(restored.compile(nodes).valid);
}

TEST_CASE("null node parameters load as an empty object") {
    auto nodes = registry();
    const nlohmann::json document = {
        {"formatVersion", 1}, {"project", {{"width", 128}, {"height", 128}, {"targetFps", 60}}},
        {"nodes", {{{"id", 9}, {"type", "math"}, {"position", {1, 2}}, {"parameters", nullptr}}}},
        {"links", nlohmann::json::array()}, {"activeOutput", 0}};

    const auto graph = deserializeProject(document, nodes);
    REQUIRE(graph.nodes().front().parameters.is_object());
    REQUIRE(serializeProject(graph)["nodes"][0]["parameters"].is_object());
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
    const auto validationErrors = validateSubgraph(builtInSubgraphs().front(), nodes);
    INFO(nlohmann::json(validationErrors).dump());
    REQUIRE(validationErrors.empty());
    Graph graph;
    const auto id = graph.addNode("subgraph");
    graph.findNode(id)->subgraphId = "builtin.reaction_diffusion.discrete";
    NodeDescriptor storage;
    const auto* descriptor = resolveDescriptor(graph, *graph.findNode(id), nodes, storage);
    REQUIRE(descriptor != nullptr);
    REQUIRE(descriptor->displayName == "Reaction Diffusion (Discrete)");
    REQUIRE(descriptor->stateful);
    REQUIRE(std::ranges::count_if(descriptor->sockets, [](const auto& socket) {
        return socket.direction == SocketDirection::Input;
    }) == 11);
    REQUIRE(std::ranges::count_if(descriptor->sockets, [](const auto& socket) {
        return socket.direction == SocketDirection::Output;
    }) == 3);
    REQUIRE(descriptor->sockets[11].label == "Image");
    REQUIRE(descriptor->sockets[12].label == "Chemical A");
    REQUIRE(descriptor->sockets[13].label == "Chemical B");
    Graph persisted;
    persisted.subgraphs().push_back(builtInSubgraphs().front());
    const auto serialized = serializeProject(persisted);
    REQUIRE(serialized["subgraphs"][0]["interface"][2]["type"] == "scalar_field");
    REQUIRE(serialized["subgraphs"][0]["interface"][11]["type"] == "scalar_field");
    REQUIRE(graph.compile(nodes).valid);
}

TEST_CASE("generic cellular automata is an editable Life-like scalar simulation") {
    auto nodes = registry();
    NodeDescriptor convolutionDescriptor{"convolution", 1, "Convolution", "Filter",
        {{"image", "Image", SocketContract::AnyField, SocketDirection::Input},
         {"image", "Image", SocketContract::AnyField, SocketDirection::Output}},
        {{"scale", "Scale", 1.0F, 0.01F, 128.0F}}};
    convolutionDescriptor.lowerable = true;
    convolutionDescriptor.sockets[1].typePolicy = SocketDescriptor::TypePolicy::PreserveInput;
    convolutionDescriptor.sockets[1].typeInputs = {"image"};
    add(nodes, std::move(convolutionDescriptor));
    add(nodes, {"bit_test", 1, "Bit Test / Integer Mask", "Math",
        {{"mask", "Mask", ValueType::Float, SocketDirection::Input, true},
         {"bit", "Bit", SocketContract::Numeric, SocketDirection::Input, true},
         {"result", "Result", SocketContract::Numeric, SocketDirection::Output}}, {}});
    NodeDescriptor table{"table", 1, "Table", "Math",
        {{"index", "Index", SocketContract::Numeric, SocketDirection::Input, true},
         {"result", "Result", SocketContract::Numeric, SocketDirection::Output}}, {}};
    table.lowerable = true;
    add(nodes, std::move(table));
    const auto builtIn = std::ranges::find_if(builtInSubgraphs(), [](const SubgraphDefinition& definition) {
        return definition.id == "builtin.cellular_automata.generic";
    });
    REQUIRE(builtIn != builtInSubgraphs().end());
    const auto errors = validateSubgraph(*builtIn, nodes);
    INFO(nlohmann::json(errors).dump());
    REQUIRE(errors.empty());

    const auto descriptor = describeSubgraph(*builtIn);
    REQUIRE(descriptor.stateful);
    REQUIRE(descriptor.timeDependent);
    REQUIRE(descriptor.sockets.size() == 7);
    REQUIRE(descriptor.sockets[0].key == "initialState");
    REQUIRE(descriptor.sockets[1].key == "birthMask");
    REQUIRE(descriptor.sockets[2].key == "survivalMask");
    REQUIRE(descriptor.sockets[3].key == "preset");
    REQUIRE(descriptor.sockets[4].key == "reset");
    REQUIRE(descriptor.sockets[5].key == "iterations");
    REQUIRE(descriptor.sockets[6].key == "state");
    REQUIRE(descriptor.parameters[0].defaultValue == 8.0F);
    REQUIRE(descriptor.parameters[1].defaultValue == 12.0F);
    REQUIRE(descriptor.parameters[2].control == ParameterDescriptor::Control::Enum);
    REQUIRE(descriptor.parameters[2].enumOptions.front() == "Custom");

    Graph persisted;
    persisted.subgraphs().push_back(*builtIn);
    const auto serialized = serializeProject(persisted);
    REQUIRE(serialized["subgraphs"][0]["interface"][3]["control"] == "enum");
    REQUIRE(serialized["subgraphs"][0]["interface"][3]["options"][2] == "HighLife");

    const auto convolution = std::ranges::find(builtIn->body.nodes(), "convolution",
                                                &NodeRecord::type);
    REQUIRE(convolution != builtIn->body.nodes().end());
    REQUIRE(convolution->parameters["kernel"] == nlohmann::json::array(
        {1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F}));
    REQUIRE(std::ranges::count(builtIn->body.nodes(), "bit_test", &NodeRecord::type) == 2);
    REQUIRE(std::ranges::count(builtIn->body.nodes(), "table", &NodeRecord::type) == 3);
    REQUIRE(std::ranges::any_of(builtIn->body.nodes(), [](const NodeRecord& node) {
        return node.type == "select";
    }));
}

TEST_CASE("flood fill is an editable scalar region simulation") {
    auto nodes = registry();
    NodeDescriptor offsetSample{"state_input_sample_offset", 1,
        "State/Input Sample at Offset", "Coordinates",
        {{"source", "Source", SocketContract::AnyField, SocketDirection::Input},
         {"offset", "Offset Pixels", SocketContract::VectorNumeric, SocketDirection::Input, true},
         {"sampled", "Sampled", SocketContract::AnyField, SocketDirection::Output}}, {}};
    offsetSample.sockets[0].requiresImage = true;
    offsetSample.sockets[2].typePolicy = SocketDescriptor::TypePolicy::PreserveInput;
    offsetSample.sockets[2].typeInputs = {"source"};
    offsetSample.neighborhoodSocket = "source";
    offsetSample.lowerable = true;
    add(nodes, std::move(offsetSample));

    const auto builtIn = std::ranges::find_if(builtInSubgraphs(), [](const SubgraphDefinition& definition) {
        return definition.id == "builtin.flood_fill";
    });
    REQUIRE(builtIn != builtInSubgraphs().end());
    const auto errors = validateSubgraph(*builtIn, nodes);
    INFO(nlohmann::json(errors).dump());
    REQUIRE(errors.empty());

    const auto descriptor = describeSubgraph(*builtIn);
    REQUIRE(descriptor.stateful);
    REQUIRE(descriptor.sockets.size() == 6);
    REQUIRE(descriptor.sockets[0].key == "passableMask");
    REQUIRE(descriptor.sockets[1].key == "seedMask");
    REQUIRE(descriptor.sockets[2].key == "initialState");
    REQUIRE(descriptor.sockets[2].optional);
    REQUIRE(descriptor.sockets[3].key == "connectivity");
    REQUIRE(descriptor.parameters[0].control == ParameterDescriptor::Control::Enum);
    REQUIRE(descriptor.parameters[0].enumOptions == std::vector<std::string>{"4", "8"});
    REQUIRE(descriptor.sockets[5].key == "region");
    REQUIRE(std::ranges::count(builtIn->body.nodes(), "state_input_sample_offset",
                               &NodeRecord::type) == 8);
}

TEST_CASE("skeletonization is an editable Zhang-Suen thinning simulation") {
    auto nodes = registry();
    NodeDescriptor offsetSample{"state_input_sample_offset", 1,
        "State/Input Sample at Offset", "Coordinates",
        {{"source", "Source", SocketContract::AnyField, SocketDirection::Input},
         {"offset", "Offset Pixels", SocketContract::VectorNumeric, SocketDirection::Input, true},
         {"sampled", "Sampled", SocketContract::AnyField, SocketDirection::Output}}, {}};
    offsetSample.sockets[0].requiresImage = true;
    offsetSample.sockets[2].typePolicy = SocketDescriptor::TypePolicy::PreserveInput;
    offsetSample.sockets[2].typeInputs = {"source"};
    offsetSample.neighborhoodSocket = "source";
    offsetSample.lowerable = true;
    add(nodes, std::move(offsetSample));

    const auto builtIn = std::ranges::find_if(builtInSubgraphs(), [](const SubgraphDefinition& definition) {
        return definition.id == "builtin.skeletonization";
    });
    REQUIRE(builtIn != builtInSubgraphs().end());
    const auto errors = validateSubgraph(*builtIn, nodes);
    INFO(nlohmann::json(errors).dump());
    REQUIRE(errors.empty());

    const auto descriptor = describeSubgraph(*builtIn);
    REQUIRE(descriptor.stateful);
    REQUIRE(descriptor.sockets.size() == 4);
    REQUIRE(descriptor.sockets[0].key == "initialState");
    REQUIRE(descriptor.sockets[1].key == "iterations");
    REQUIRE(descriptor.sockets[3].key == "skeleton");
    REQUIRE(builtIn->stateSlots.size() == 2);
    REQUIRE(builtIn->stateSlots[0].type == ValueType::ScalarField);
    REQUIRE(builtIn->stateSlots[1].key == "phase");
    REQUIRE(std::ranges::count(builtIn->body.nodes(), "state_input_sample_offset",
                               &NodeRecord::type) == 8);
}

TEST_CASE("distance from nearest white pixel is an editable Euclidean distance simulation") {
    auto nodes = registry();
    NodeDescriptor offsetSample{"state_input_sample_offset", 1,
        "State/Input Sample at Offset", "Coordinates",
        {{"source", "Source", SocketContract::AnyField, SocketDirection::Input},
         {"offset", "Offset Pixels", SocketContract::VectorNumeric, SocketDirection::Input, true},
         {"sampled", "Sampled", SocketContract::AnyField, SocketDirection::Output}}, {}};
    offsetSample.sockets[0].requiresImage = true;
    offsetSample.sockets[2].typePolicy = SocketDescriptor::TypePolicy::PreserveInput;
    offsetSample.sockets[2].typeInputs = {"source"};
    offsetSample.neighborhoodSocket = "source";
    offsetSample.lowerable = true;
    add(nodes, std::move(offsetSample));

    const auto builtIn = std::ranges::find_if(builtInSubgraphs(), [](const SubgraphDefinition& definition) {
        return definition.id == "builtin.distance_from_nearest_white_pixel";
    });
    REQUIRE(builtIn != builtInSubgraphs().end());
    const auto errors = validateSubgraph(*builtIn, nodes);
    INFO(nlohmann::json(errors).dump());
    REQUIRE(errors.empty());

    const auto descriptor = describeSubgraph(*builtIn);
    REQUIRE(descriptor.stateful);
    REQUIRE(descriptor.sockets.size() == 5);
    REQUIRE(descriptor.sockets[0].key == "whiteMask");
    REQUIRE(descriptor.sockets[1].key == "maxDistance");
    REQUIRE(descriptor.sockets[2].key == "iterations");
    REQUIRE(descriptor.sockets[4].key == "distance");
    REQUIRE(builtIn->stateSlots.size() == 2);
    REQUIRE(builtIn->stateSlots[0].type == ValueType::VectorField);
    REQUIRE(std::ranges::count(builtIn->body.nodes(), "state_input_sample_offset",
                               &NodeRecord::type) == 8);
    REQUIRE(std::ranges::count_if(builtIn->body.nodes(), [](const NodeRecord& node) {
        return node.type == "select" && node.parameters.value("ifTrue", 1.0F) == 0.0F;
    }) == 1);
}

TEST_CASE("SDF generator is an editable signed-distance simulation") {
    auto nodes = registry();
    NodeDescriptor offsetSample{"state_input_sample_offset", 1,
        "State/Input Sample at Offset", "Coordinates",
        {{"source", "Source", SocketContract::AnyField, SocketDirection::Input},
         {"offset", "Offset Pixels", SocketContract::VectorNumeric, SocketDirection::Input, true},
         {"sampled", "Sampled", SocketContract::AnyField, SocketDirection::Output}}, {}};
    offsetSample.sockets[0].requiresImage = true;
    offsetSample.sockets[2].typePolicy = SocketDescriptor::TypePolicy::PreserveInput;
    offsetSample.sockets[2].typeInputs = {"source"};
    offsetSample.neighborhoodSocket = "source";
    offsetSample.lowerable = true;
    add(nodes, std::move(offsetSample));
    const auto builtIn = std::ranges::find_if(builtInSubgraphs(), [](const SubgraphDefinition& item) {
        return item.id == "builtin.sdf_generator";
    });
    REQUIRE(builtIn != builtInSubgraphs().end());
    INFO(nlohmann::json(validateSubgraph(*builtIn, nodes)).dump());
    REQUIRE(validateSubgraph(*builtIn, nodes).empty());
    const auto descriptor = describeSubgraph(*builtIn);
    REQUIRE(descriptor.stateful);
    REQUIRE(descriptor.sockets.back().key == "distance");
    REQUIRE(builtIn->stateSlots.size() == 3);
    REQUIRE(std::ranges::count(builtIn->body.nodes(), "state_input_sample_offset",
                               &NodeRecord::type) == 16);
}

TEST_CASE("Lenia is an editable scalar growth simulation") {
    auto nodes = registry();
    NodeDescriptor convolution{"convolution", 1, "Convolution", "Filter",
        {{"image", "Image", SocketContract::AnyField, SocketDirection::Input},
         {"image", "Image", SocketContract::AnyField, SocketDirection::Output}},
        {{"scale", "Scale", 1.0F, 0.01F, 128.0F}}};
    convolution.lowerable = true;
    convolution.neighborhoodSocket = "image";
    convolution.sockets[0].requiresImage = true;
    convolution.sockets[1].typePolicy = SocketDescriptor::TypePolicy::PreserveInput;
    convolution.sockets[1].typeInputs = {"image"};
    add(nodes, std::move(convolution));

    const auto builtIn = std::ranges::find_if(builtInSubgraphs(), [](const SubgraphDefinition& definition) {
        return definition.id == "builtin.lenia";
    });
    REQUIRE(builtIn != builtInSubgraphs().end());
    const auto errors = validateSubgraph(*builtIn, nodes);
    INFO(nlohmann::json(errors).dump());
    REQUIRE(errors.empty());
    const auto descriptor = describeSubgraph(*builtIn);
    REQUIRE(descriptor.stateful);
    REQUIRE(descriptor.sockets.size() == 7);
    REQUIRE(descriptor.sockets[1].key == "kernelScale");
    REQUIRE(descriptor.sockets[2].key == "growthCenter");
    REQUIRE(descriptor.sockets[3].key == "growthWidth");
    REQUIRE(descriptor.sockets[4].key == "timeStep");
    REQUIRE(descriptor.sockets[6].key == "state");
    const auto kernel = std::ranges::find(builtIn->body.nodes(), "convolution", &NodeRecord::type);
    REQUIRE(kernel != builtIn->body.nodes().end());
    REQUIRE(kernel->parameters["kernelSize"] == 15.0F);
    REQUIRE(kernel->parameters["kernel"].size() == 225);
    REQUIRE(kernel->parameters["preset"] == "Annular Ring");
    REQUIRE(std::ranges::any_of(builtIn->body.nodes(), [](const NodeRecord& node) {
        return node.type == "math" && node.parameters.value("operation", -1.0F) ==
            static_cast<float>(MathOperation::Exp);
    }));
}

TEST_CASE("built-in discrete reaction uses Vector Math for its seed distance") {
    const auto& body = builtInSubgraphs().front().body;
    const auto distance = std::ranges::find_if(body.nodes(), [](const NodeRecord& node) {
        return node.type == "vector_math" &&
            node.parameters.value("operation", -1.0F) ==
                static_cast<float>(VectorMathOperation::Distance);
    });
    REQUIRE(distance != body.nodes().end());
    REQUIRE(std::ranges::any_of(body.links(), [&](const LinkRecord& link) {
        return link.toNode == distance->id && link.toSocket == "a" &&
               link.fromSocket == "coordinates";
    }));
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
    REQUIRE(json["formatVersion"] == 4);
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

TEST_CASE("simulation convolution exposes only its single-pass form") {
    auto nodes = registry();
    NodeDescriptor convolutionDescriptor{"convolution", 1, "Convolution", "Filter",
        {{"image", "Image", ValueType::ColorImage, SocketDirection::Input},
         {"image", "Image", ValueType::ColorImage, SocketDirection::Output}},
        {{"iterations", "Iterations", 1.0F, 1.0F, 32.0F}}};
    convolutionDescriptor.lowerable = true;
    add(nodes, std::move(convolutionDescriptor));
    auto definition = builtInSubgraphs().front();
    const auto convolution = definition.body.addNode("convolution");
    const auto* node = definition.body.findNode(convolution);
    REQUIRE(node != nullptr);

    NodeDescriptor descriptor;
    const auto* resolved = resolveSubgraphBodyDescriptor(definition, *node, nodes, descriptor);
    REQUIRE(resolved != nullptr);
    REQUIRE(std::ranges::find(resolved->parameters, "iterations",
                              &ParameterDescriptor::key) == resolved->parameters.end());
    REQUIRE(std::ranges::none_of(resolved->sockets, [](const SocketDescriptor& socket) {
        return socket.direction == SocketDirection::Input && socket.key == "iterations";
    }));

    definition.body.findNode(convolution)->parameters["iterations"] = 2.0F;
    const auto errors = validateSubgraph(definition, nodes);
    REQUIRE(std::ranges::any_of(errors, [](const std::string& error) {
        return error.find("Convolution") != std::string::npos &&
               error.find("one iteration") != std::string::npos;
    }));
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
    definition.body.addNode("dither");
    for (const auto* type : {"float", "math", "threshold", "select",
                             "coordinates", "laplacian", "dither"}) {
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
    REQUIRE(channelDescriptor->sockets.front().contract == SocketContract::VectorFieldOnly);
    REQUIRE(channelDescriptor->sockets[1].contract == SocketContract::ScalarFieldOnly);
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
        return error.find("cannot connect vector_field") != std::string::npos;
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
    REQUIRE(serialized["formatVersion"] == 4);
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
        return link.fromNode == previous->id && link.fromSocket == "value";
    });
    REQUIRE(outgoing != definition.body.links().end());
    outgoing->fromSocket = "a";
    graph.subgraphs().push_back(std::move(definition));

    auto restored = deserializeProject(serializeProject(graph), nodes);
    const auto& loaded = restored.subgraphs().front();
    const auto loadedPrevious = std::ranges::find(loaded.body.nodes(),
        std::string("simulation_previous_state"), &NodeRecord::type);
    REQUIRE(loadedPrevious != loaded.body.nodes().end());
    REQUIRE(loadedPrevious->needsAttention);
    REQUIRE_FALSE(validateSubgraph(loaded, nodes).empty());

    // A stale editable definition that is not instantiated must not disable the
    // otherwise independent root graph.
    REQUIRE(restored.compile(nodes).valid);

    const auto instance = restored.addNode("subgraph");
    restored.findNode(instance)->subgraphId = loaded.id;
    REQUIRE_FALSE(restored.compile(nodes).valid);
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

TEST_CASE("removing the last consumer prunes the orphaned subgraph") {
    auto nodes = registry();
    Graph graph;
    auto definition = builtInSubgraphs().front();
    definition.id = "project.gray_scott.orphaned";
    definition.immutable = false;
    graph.subgraphs().push_back(definition);
    const auto instance = graph.addNode("subgraph");
    graph.findNode(instance)->subgraphId = definition.id;

    REQUIRE(graph.subgraphs().size() == 1);
    REQUIRE(graph.removeNode(instance));
    REQUIRE(graph.subgraphs().empty());
}

TEST_CASE("removing one consumer keeps a shared subgraph alive") {
    auto nodes = registry();
    Graph graph;
    auto definition = builtInSubgraphs().front();
    definition.id = "project.gray_scott.shared";
    definition.immutable = false;
    graph.subgraphs().push_back(definition);
    const auto first = graph.addNode("subgraph");
    const auto second = graph.addNode("subgraph");
    graph.findNode(first)->subgraphId = definition.id;
    graph.findNode(second)->subgraphId = definition.id;

    REQUIRE(graph.removeNode(first));
    REQUIRE(graph.subgraphs().size() == 1);
    REQUIRE(resolveSubgraph(graph, definition.id) != nullptr);
}

TEST_CASE("copy then delete then paste restores the pruned subgraph") {
    auto nodes = registry();
    Graph graph;
    auto definition = builtInSubgraphs().front();
    definition.id = "project.gray_scott.copied";
    definition.immutable = false;
    graph.subgraphs().push_back(definition);
    const auto instance = graph.addNode("subgraph");
    graph.findNode(instance)->subgraphId = definition.id;

    // Copy captures the full subgraph definition before the node is deleted.
    const auto copied = serializeProject(graph)["subgraphs"];

    // Delete the node, which orphans and prunes the definition.
    REQUIRE(graph.removeNode(instance));
    REQUIRE(graph.subgraphs().empty());

    // Paste re-creates the definition exactly like the clipboard path does.
    for (const auto& definitionJson : copied) {
        const nlohmann::json document{{"formatVersion", 2},
            {"project", {{"width", 1024}, {"height", 1024}, {"targetFps", 60}}},
            {"subgraphs", nlohmann::json::array({definitionJson})},
            {"nodes", nlohmann::json::array()}, {"links", nlohmann::json::array()},
            {"activeOutput", 0}};
        graph.subgraphs().push_back(
            deserializeProject(document, nodes).subgraphs().front());
    }
    const auto pasted = graph.addNode("subgraph");
    graph.findNode(pasted)->subgraphId = definition.id;

    REQUIRE(resolveSubgraph(graph, definition.id) != nullptr);
    REQUIRE(graph.compile(nodes).valid);
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

TEST_CASE("auto-layout snaps a single-consumer chain behind its consumer through a merge") {
    GraphBody body;
    const auto s1 = body.addNode("float", {0, 0});
    const auto s2 = body.addNode("float", {0, 1000});
    const auto relay = body.addNode("math", {500, 0});
    const auto merge = body.addNode("math", {900, 500});
    body.addLink(s1, "value", relay, "a");
    body.addLink(relay, "result", merge, "a");
    body.addLink(s2, "value", merge, "b");
    layoutGraph(body);
    const auto* relayNode = body.findNode(relay);
    const auto* mergeNode = body.findNode(merge);
    REQUIRE(relayNode->position.y == mergeNode->position.y);
}

TEST_CASE("auto-layout leaves a source feeding a merge on its own lane") {
    GraphBody body;
    const auto s1 = body.addNode("float", {0, 0});
    const auto s2 = body.addNode("float", {0, 1000});
    const auto m = body.addNode("math", {500, 0});
    body.addLink(s1, "value", m, "a");
    body.addLink(s2, "value", m, "b");
    layoutGraph(body);
    const float first = body.findNode(s1)->position.y;
    const float second = body.findNode(s2)->position.y;
    REQUIRE(first != second);
    REQUIRE(body.findNode(m)->position.y == (first + second) / 2.0F);
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
