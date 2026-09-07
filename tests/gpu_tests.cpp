#include "reaction/gpu/gpu_runtime.hpp"
#include "reaction/gpu/shader_ir.hpp"
#include "reaction/core/persistence.hpp"

#include <GLFW/glfw3.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <png.h>


#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace reaction {
namespace {

class HiddenContext {
public:
    HiddenContext() {
        glfwSetErrorCallback([](int, const char*) {});
        if (glfwInit() == GLFW_FALSE) SKIP("No desktop display is available for OpenGL integration tests");
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(32, 32, "GPU test", nullptr, nullptr);
        if (!window) { glfwTerminate(); SKIP("OpenGL 4.3 context is unavailable"); }
        glfwMakeContextCurrent(window);
        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
            glfwDestroyWindow(window); window = nullptr; glfwTerminate();
            SKIP("OpenGL loader initialization failed");
        }
    }
    ~HiddenContext() { if (window) glfwDestroyWindow(window); glfwTerminate(); }
    GLFWwindow* window = nullptr;
};

std::vector<float> readImage(ImageHandle image) {
    std::vector<float> values(static_cast<std::size_t>(image.width * image.height * 4));
    glBindTexture(GL_TEXTURE_2D, image.texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, values.data());
    return values;
}

NodeId addDiscreteReaction(Graph& graph) {
    const auto id = graph.addNode("subgraph");
    graph.findNode(id)->subgraphId = "builtin.reaction_diffusion.discrete";
    graph.findNode(id)->parameters = {{"feed", .055F}, {"kill", .062F}, {"diffA", 1.0F},
        {"diffB", .5F}, {"structureScale", 1.0F}, {"dt", 1.0F},
        {"iterations", 8.0F}, {"autoReset", 0.0F}};
    return id;
}

double median(std::vector<double> values) {
    std::ranges::sort(values);
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) * .5;
}

class CapturingLoweringContext final : public ShaderLoweringContext {
public:
    explicit CapturingLoweringContext(ShaderValueType target) : target_(target) {}

    ShaderValue input(std::string_view socket, std::string_view parameterKey,
                      float fallback) override {
        return inputAt(socket, "uv", parameterKey, fallback);
    }
    ShaderValue inputAt(std::string_view socket, std::string_view,
                        std::string_view parameterKey, float fallback) override {
        trace.push_back("input:" + std::string(socket));
        if (const auto found = inputs.find(std::string(socket)); found != inputs.end())
            return found->second;
        return parameter(parameterKey, fallback);
    }
    ShaderValue parameter(std::string_view key, float) override {
        trace.push_back("parameter:" + std::string(key));
        return {ShaderValueType::Scalar, "p_" + std::string(key)};
    }
    std::string helper(std::string_view name, std::string source) override {
        trace.push_back("helper:" + std::string(name));
        helpers.emplace_back(std::move(source));
        return std::string(name);
    }
    ShaderValue emit(std::string expression, std::string_view socket = {}) override {
        trace.push_back("emit:" + std::string(socket));
        emitted = {target_, std::move(expression)};
        return emitted;
    }
    ShaderValueType valueType() const override { return target_; }

    std::unordered_map<std::string, ShaderValue> inputs;
    std::vector<std::string> trace;
    std::vector<std::string> helpers;
    ShaderValue emitted;

private:
    ShaderValueType target_;
};

struct LoweredChain {
    std::vector<std::string> trace;
    std::vector<std::string> expressions;
};

LoweredChain lowerMathThresholdSelect(const NodeRegistry& registry, ShaderValueType type) {
    LoweredChain result;
    auto math = registry.create("math");
    math->setParameters({{"operation", 0.0F}});
    CapturingLoweringContext mathContext(type);
    mathContext.inputs = {{"a", {type, "source"}},
                          {"b", {ShaderValueType::Scalar, "amount"}}};
    REQUIRE(math->lowerShader(mathContext));
    result.trace.insert(result.trace.end(), mathContext.trace.begin(), mathContext.trace.end());
    result.expressions.push_back(mathContext.emitted.name);

    auto threshold = registry.create("threshold");
    CapturingLoweringContext thresholdContext(type);
    thresholdContext.inputs = {{"value", {type, mathContext.emitted.name}}};
    REQUIRE(threshold->lowerShader(thresholdContext));
    result.trace.insert(result.trace.end(), thresholdContext.trace.begin(), thresholdContext.trace.end());
    result.expressions.push_back(thresholdContext.emitted.name);

    auto select = registry.create("select");
    CapturingLoweringContext selectContext(type);
    selectContext.inputs = {{"condition", {type, thresholdContext.emitted.name}},
                            {"ifTrue", {type, mathContext.emitted.name}},
                            {"ifFalse", {ShaderValueType::Scalar, "fallback"}}};
    REQUIRE(select->lowerShader(selectContext));
    result.trace.insert(result.trace.end(), selectContext.trace.begin(), selectContext.trace.end());
    result.expressions.push_back(selectContext.emitted.name);
    return result;
}

} // namespace

TEST_CASE("unused legacy subgraphs do not invalidate the text test root graph") {
    NodeRegistry registry;
    registerBuiltInNodes(registry);
    const auto projectPath = std::filesystem::path(__FILE__).parent_path().parent_path() /
                             "text_test";
    const auto graph = loadProject(projectPath, registry);
    const auto result = graph.compile(registry);
    INFO(nlohmann::json(result.errors).dump());
    REQUIRE(result.valid);
}

TEST_CASE("text test evaluates a non-black output and materializes node previews") {
    HiddenContext context;
    NodeRegistry registry;
    registerBuiltInNodes(registry);
    const auto projectPath = std::filesystem::path(__FILE__).parent_path().parent_path() /
                             "text_test";
    auto graph = loadProject(projectPath, registry);
    graph.settings = {64, 96, 60};

    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0.0, 1.0 / 60.0, true));
    REQUIRE(runtime.outputImage());

    const auto pixels = readImage(runtime.outputImage());
    REQUIRE(std::ranges::any_of(pixels, [](float value) {
        return std::isfinite(value) && std::abs(value) > 1.0e-4F;
    }));

    const auto previewCount = std::ranges::count_if(runtime.values(), [](const auto& entry) {
        return std::ranges::any_of(entry.second, [](const Value& value) {
            const auto* image = std::get_if<ImageHandle>(&value);
            return image && static_cast<bool>(*image);
        });
    });
    REQUIRE(previewCount > 0);
}

TEST_CASE("simulation shader prunes interfaces and lowers one vec2 state neighborhood") {
    const auto& definition = builtInSubgraphs().front();
    const auto initialization = generateSimulationShader(definition, true);
    const auto update = generateSimulationShader(definition, false);

    REQUIRE(initialization.find("in_seed") != std::string::npos);
    REQUIRE(initialization.find("in_feedMultiplier") == std::string::npos);
    REQUIRE(initialization.find("param_feed") == std::string::npos);
    REQUIRE(update.find("in_seed") == std::string::npos);
    REQUIRE(update.find("in_feedMultiplier") != std::string::npos);
    REQUIRE(update.find("in_structureScale") != std::string::npos);
    REQUIRE(update.find("param_iterations") == std::string::npos);
    REQUIRE(update.find("float lap_") == std::string::npos);
    REQUIRE(update.find("float laplacianKernel_") == std::string::npos);
    REQUIRE(update.find("vec2 laplacianKernel_") != std::string::npos);
    const auto helperAt = update.find("At(vec2 q,float r)");
    REQUIRE(helperAt != std::string::npos);
    REQUIRE(update.find("At(vec2 q,float r)", helperAt + 1) == std::string::npos);
    REQUIRE(update.find(".x") != std::string::npos);
    REQUIRE(update.find(".y") != std::string::npos);

    std::size_t neighborhoodSamples = 0;
    for (std::size_t at = update.find("sampleState0(q+pixel*"); at != std::string::npos;
         at = update.find("sampleState0(q+pixel*", at + 1)) ++neighborhoodSamples;
    REQUIRE(neighborhoodSamples == 8);
}

TEST_CASE("generic cellular automata lowers its scalar Moore-neighborhood rule") {
    const auto definition = std::ranges::find_if(builtInSubgraphs(), [](const SubgraphDefinition& item) {
        return item.id == "builtin.cellular_automata.generic";
    });
    REQUIRE(definition != builtInSubgraphs().end());
    const auto initialization = generateSimulationShader(*definition, true);
    const auto update = generateSimulationShader(*definition, false);
    REQUIRE(initialization.find("in_initialState") != std::string::npos);
    REQUIRE(update.find("in_birthMask") != std::string::npos);
    REQUIRE(update.find("in_survivalMask") != std::string::npos);
    REQUIRE(update.find("uint bits=uint") != std::string::npos);
    REQUIRE(update.find("convolutionKernel") != std::string::npos);
    REQUIRE(update.find("vec2(p") != std::string::npos);

    HiddenContext context;
    GpuRuntime gpu;
    REQUIRE_NOTHROW(gpu.compileCompute(update, "Generic Cellular Automata / update"));
}

TEST_CASE("flood fill lowers its four and eight-connected offset neighborhoods") {
    const auto definition = std::ranges::find_if(builtInSubgraphs(), [](const SubgraphDefinition& item) {
        return item.id == "builtin.flood_fill";
    });
    REQUIRE(definition != builtInSubgraphs().end());
    const auto initialization = generateSimulationShader(*definition, true);
    const auto update = generateSimulationShader(*definition, false);
    REQUIRE(initialization.find("in_seedMask") != std::string::npos);
    REQUIRE(initialization.find("in_passableMask") != std::string::npos);
    REQUIRE(update.find("in_connectivity") != std::string::npos);
    REQUIRE(std::ranges::count(definition->body.nodes(), "state_input_sample_offset",
                               &NodeRecord::type) == 8);

    HiddenContext context;
    GpuRuntime gpu;
    REQUIRE_NOTHROW(gpu.compileCompute(update, "Flood Fill / update"));
}

TEST_CASE("simulation shader uses the central widest-edge coercion") {
    NodeRegistry registry;
    registerBuiltInNodes(registry);
    SubgraphDefinition definition;
    definition.id = "test.simulation.widening";
    definition.name = "Simulation Widening";
    definition.execution = SubgraphExecution::Simulation;
    definition.interface = {{"image", "Image", SubgraphInterfaceKind::Output,
                             ValueType::ScalarField}};
    auto addNode = [&](std::string type, nlohmann::json parameters = nlohmann::json::object()) {
        const auto id = definition.body.addNode(std::move(type));
        definition.body.findNode(id)->parameters = std::move(parameters);
        return id;
    };
    const auto constant = addNode("float", {{"value", 0.5F}});
    const auto initial = addNode("simulation_initial_state");
    const auto previous = addNode("simulation_previous_state");
    const auto channels = addNode("simulation_channel");
    const auto mix = addNode("mix");
    const auto separate = addNode("separate_vector");
    const auto next = addNode("simulation_next_state");
    const auto output = addNode("subgraph_output", {{"key", "image"}});
    definition.body.addLink(constant, "value", initial, "a");
    definition.body.addLink(constant, "value", initial, "b");
    definition.body.addLink(previous, "state", channels, "state");
    definition.body.addLink(channels, "a", mix, "a");
    definition.body.addLink(previous, "state", mix, "b");
    definition.body.addLink(mix, "result", separate, "value");
    definition.body.addLink(separate, "x", next, "a");
    definition.body.addLink(channels, "b", next, "b");
    definition.body.addLink(next, "a", output, "value");

    const auto errors = validateSubgraph(definition, registry);
    INFO(nlohmann::json(errors).dump());
    REQUIRE(errors.empty());
    const auto shader = generateSimulationShader(definition, false);
    REQUIRE(shader.find("vec2(v_" + std::to_string(channels) + "_a)") != std::string::npos);
 }

TEST_CASE("shader planner specializes and fuses a linear image chain") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph;
    const auto source = graph.addNode("perlin");
    const auto add = graph.addNode("math");
    const auto absolute = graph.addNode("math");
    graph.findNode(add)->parameters = {{"operation", 0.0F}, {"b", .25F}};
    graph.findNode(absolute)->parameters = {{"operation", 7.0F}};
    graph.addLink(source, "image", add, "a");
    graph.addLink(add, "result", absolute, "a");
    const auto compiled = graph.compile(registry);
    REQUIRE(compiled.valid);

    const auto regions = planShaderRegions(graph, registry, compiled, 16, 1024);
    REQUIRE(regions.size() == 1);
    REQUIRE(regions.front().nodes == std::vector<NodeId>{source, add, absolute});
    const auto generated = generateComputeShader(regions.front(), {absolute});
    const auto originalSpecialization = generated.specializationKey;
    REQUIRE(generated.outputs.size() == 1);
    REQUIRE(generated.inputs.size() == 10);
    REQUIRE(generated.source.find("operation") == std::string::npos);
    REQUIRE(generated.source.find("hasA") == std::string::npos);
    REQUIRE(generated.source.find("// node " + std::to_string(add)) != std::string::npos);
    REQUIRE(generated.source.find("// node " + std::to_string(absolute)) != std::string::npos);
    REQUIRE(generated.annotations.size() == 4);

    const auto originalSource = generated.source;
    graph.findNode(add)->parameters["b"] = .75F;
    const auto numericEdit = generateComputeShader(
        planShaderRegions(graph, registry, graph.compile(registry), 16, 1024).front(),
        {absolute});
    REQUIRE(numericEdit.source == originalSource);
    REQUIRE(numericEdit.specializationKey == originalSpecialization);
    graph.findNode(add)->parameters["operation"] = 2.0F;
    const auto operationEdit = generateComputeShader(
        planShaderRegions(graph, registry, graph.compile(registry), 16, 1024).front(),
        {absolute});
    REQUIRE(operationEdit.source != originalSource);
    REQUIRE(operationEdit.specializationKey != originalSpecialization);
}

TEST_CASE("shared scalar node lowering follows one contract") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto scalar = lowerMathThresholdSelect(registry, ShaderValueType::Scalar);
    REQUIRE(scalar.expressions[0].find('+') != std::string::npos);
    REQUIRE(scalar.expressions[1].find("step(") != std::string::npos);
    REQUIRE(scalar.expressions[2].find('?') != std::string::npos);
}

TEST_CASE("simulation information nodes lower to per-dispatch uniforms") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    SubgraphDefinition definition;
    definition.id = "test.simulation.info";
    definition.name = "Simulation Info";
    definition.execution = SubgraphExecution::Simulation;
    definition.stateSlots = {{"state", "State", ValueType::ScalarField}};
    const auto addNode = [&](std::string type) {
        return definition.body.addNode(std::move(type));
    };
    const auto initial = addNode("simulation_initial_state");
    const auto iteration = addNode("simulation_iteration_info");
    const auto step = addNode("simulation_step_info");
    const auto add = addNode("math");
    const auto next = addNode("simulation_next_state");
    definition.body.findNode(initial)->parameters = {{"slot", 0.0F}};
    definition.body.findNode(next)->parameters = {{"slot", 0.0F}};
    definition.body.findNode(add)->parameters = {{"operation", 0.0F}};
    definition.body.addLink(iteration, "normalizedIteration", add, "a");
    definition.body.addLink(step, "deltaTime", add, "b");
    definition.body.addLink(add, "result", next, "value");

    NodeDescriptor iterationDescriptor;
    const auto* iterationResolved = resolveSubgraphBodyDescriptor(
        definition, *definition.body.findNode(iteration), registry, iterationDescriptor);
    REQUIRE(iterationResolved != nullptr);
    REQUIRE(iterationResolved->displayName == "Simulation Iteration Info");
    REQUIRE(iterationResolved->sockets.size() == 5);

    const auto update = generateSimulationShader(definition, false);
    REQUIRE(update.find("uniform float simulationIterationIndex;") != std::string::npos);
    REQUIRE(update.find("uniform float simulationIterationCount;") != std::string::npos);
    REQUIRE(update.find("uniform float simulationDeltaTime;") != std::string::npos);
    REQUIRE(update.find("uniform float simulationStep;") != std::string::npos);
    REQUIRE(update.find("max(simulationIterationCount-1.0,1.0)") != std::string::npos);
}

TEST_CASE("central widening emits canonical channels and keeps constants uniform-backed") {
    REQUIRE(convertShaderValue({ShaderValueType::Scalar, "s"}, ShaderValueType::Vec2) ==
            "vec2(s)");
    REQUIRE(convertShaderValue({ShaderValueType::Scalar, "s"}, ShaderValueType::Vec4) ==
            "vec4(s,s,s,1.0)");
    REQUIRE(convertShaderValue({ShaderValueType::Vec2, "v"}, ShaderValueType::Vec4) ==
            "vec4(v,0.0,1.0)");
    REQUIRE_THROWS_AS(convertShaderValue({ShaderValueType::Vec4, "c"},
                                         ShaderValueType::Scalar), std::invalid_argument);
    REQUIRE_THROWS_AS(convertShaderValue({ShaderValueType::Vec4, "c"},
                                         ShaderValueType::Vec2), std::invalid_argument);

    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph;
    const auto scalar = graph.addNode("float");
    const auto coordinates = graph.addNode("coordinates");
    const auto mix = graph.addNode("mix");
    graph.addLink(scalar, "value", mix, "a");
    graph.addLink(coordinates, "coordinates", mix, "b");
    const auto compiled = graph.compile(registry);
    INFO(nlohmann::json(compiled.errors).dump());
    REQUIRE(compiled.valid);
    REQUIRE(compiled.socketType(mix, "result") == ValueType::VectorField);
    const auto region = lowerShaderRegion(graph, registry, compiled, {mix});
    const auto shader = generateComputeShader(region, {mix});
    REQUIRE(std::ranges::count(shader.inputs, ShaderInputKind::Float,
                               &ShaderInputRequirement::kind) >= 1);
    REQUIRE(std::ranges::count(shader.inputs, ShaderInputKind::Field,
                               &ShaderInputRequirement::kind) == 1);
    REQUIRE(shader.source.find("vec2(inputFloat_") != std::string::npos);
}

TEST_CASE("general shader planner fuses threshold and select and preserves multi-output sockets") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    for (const auto* type : {"float", "math", "threshold", "select", "coordinates", "laplacian",
                             "invert", "color_ramp", "mix", "perlin"}) {
        INFO(type);
        REQUIRE(registry.descriptor(type)->lowerable);
    }
    Graph graph;
    const auto source = graph.addNode("image");
    const auto red = graph.addNode("color_r");
    const auto threshold = graph.addNode("threshold");
    const auto select = graph.addNode("select");
    graph.addLink(source, "image", red, "color");
    graph.addLink(red, "value", threshold, "value");
    graph.addLink(threshold, "result", select, "condition");
    graph.addLink(source, "image", select, "ifTrue");
    const auto compiled = graph.compile(registry);
    INFO(nlohmann::json(compiled.errors).dump());
    REQUIRE(compiled.valid);
    const auto regions = planShaderRegions(graph, registry, compiled, 16, 1024);
    REQUIRE(regions.size() == 1);
    REQUIRE(regions.front().nodes == std::vector<NodeId>{red, threshold, select});
    const auto generated = generateComputeShader(regions.front(), {select});
    REQUIRE(generated.source.find("step(") != std::string::npos);
    REQUIRE(generated.source.find(".r") != std::string::npos);

    Graph coordinatesGraph;
    const auto coordinates = coordinatesGraph.addNode("coordinates");
    const auto coordinateRegions = planShaderRegions(
        coordinatesGraph, registry, coordinatesGraph.compile(registry), 16, 1024);
    REQUIRE(coordinateRegions.size() == 1);
    const auto coordinateShader = generateComputeShader(coordinateRegions.front(), {coordinates});
    REQUIRE(coordinateShader.outputs.size() == 1);
    REQUIRE(coordinateShader.outputs[0].socket == "coordinates");
}

TEST_CASE("Laplacian neighborhood inputs form region boundaries") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph;
    const auto source = graph.addNode("perlin");
    const auto laplacian = graph.addNode("laplacian");
    const auto threshold = graph.addNode("threshold");
    graph.addLink(source, "image", laplacian, "value");
    graph.addLink(laplacian, "result", threshold, "value");
    const auto compiled = graph.compile(registry);
    REQUIRE(compiled.valid);
    const auto regions = planShaderRegions(graph, registry, compiled, 16, 1024);
    REQUIRE(regions.size() == 2);
    REQUIRE(regions[0].nodes == std::vector<NodeId>{source});
    REQUIRE(regions[1].nodes == std::vector<NodeId>{laplacian, threshold});
    const auto generated = generateComputeShader(regions[1], {threshold});
    REQUIRE(generated.source.find("laplacianKernel_") != std::string::npos);
    REQUIRE(std::ranges::count_if(generated.inputs, [](const auto& input) {
        return input.kind == ShaderInputKind::Field;
    }) == 1);
}

TEST_CASE("convolution fuses with clamp sampling and specializes") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    REQUIRE(registry.descriptor("convolution")->lowerable);

    Graph graph;
    const auto source = graph.addNode("perlin");
    const auto convolution = graph.addNode("convolution");
    const auto threshold = graph.addNode("threshold");
    graph.addLink(source, "image", convolution, "image");
    graph.addLink(convolution, "image", threshold, "value");
    auto compiled = graph.compile(registry);
    REQUIRE(compiled.valid);
    auto regions = planShaderRegions(graph, registry, compiled, 16, 1024);
    REQUIRE(regions.size() == 2);
    REQUIRE(regions[0].nodes == std::vector<NodeId>{source});
    REQUIRE(regions[1].nodes == std::vector<NodeId>{convolution, threshold});

    const auto generated = generateComputeShader(regions[1], {threshold});
    REQUIRE(generated.source.find("convolutionKernel_") != std::string::npos);
    REQUIRE(generated.source.find("texelFetch(") != std::string::npos);
    REQUIRE(generated.source.find("fract(") == std::string::npos);
    REQUIRE(generated.source.find("operation") == std::string::npos);
    REQUIRE(generated.source.find("normalize") == std::string::npos);

    const auto originalSource = generated.source;
    const auto originalKey = generated.specializationKey;
    graph.findNode(convolution)->parameters["normalize"] = 1.0F;
    compiled = graph.compile(registry);
    regions = planShaderRegions(graph, registry, compiled, 16, 1024);
    const auto normalized = generateComputeShader(regions[1], {threshold});
    REQUIRE(normalized.specializationKey != originalKey);
    REQUIRE(normalized.source == originalSource);
}

TEST_CASE("texture samplers lower arbitrary-coordinate source reads") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* textureSample = registry.descriptor("texture_sample");
    const auto* offsetSample = registry.descriptor("state_input_sample_offset");
    REQUIRE(textureSample != nullptr);
    REQUIRE(offsetSample != nullptr);
    REQUIRE(textureSample->lowerable);
    REQUIRE(offsetSample->lowerable);
    REQUIRE(textureSample->sockets[0].requiresImage);
    REQUIRE(offsetSample->sockets[0].requiresImage);
    REQUIRE(offsetSample->neighborhoodSocket == "source");
    REQUIRE(offsetSample->sockets[1].contract == SocketContract::VectorNumeric);
    auto variantNode = registry.create("state_input_sample_offset");
    variantNode->setParameters({{"sampling", 0.0F}, {"addressMode", 0.0F}});
    const auto nearestClamp = variantNode->shaderVariantKey(variantNode->parameters());
    variantNode->setParameters({{"sampling", 1.0F}, {"addressMode", 3.0F}});
    REQUIRE(variantNode->shaderVariantKey(variantNode->parameters()) != nearestClamp);

    Graph graph;
    const auto source = graph.addNode("perlin");
    const auto sample = graph.addNode("state_input_sample_offset");
    const auto threshold = graph.addNode("threshold");
    graph.findNode(sample)->parameters = {{"sampling", 0.0F}, {"addressMode", 2.0F}};
    graph.addLink(source, "image", sample, "source");
    graph.addLink(sample, "sampled", threshold, "value");
    const auto compiled = graph.compile(registry);
    REQUIRE(compiled.valid);
    const auto regions = planShaderRegions(graph, registry, compiled, 16, 1024);
    REQUIRE(regions.size() == 2);
    REQUIRE(regions[0].nodes == std::vector<NodeId>{source});
    REQUIRE(regions[1].nodes == std::vector<NodeId>{sample, threshold});
    const auto generated = generateComputeShader(regions[1], {threshold});
    REQUIRE(generated.source.find("texelFetch(inputImage_") != std::string::npos);
    REQUIRE(generated.source.find("pixelSize") != std::string::npos);
    REQUIRE(generated.source.find("floor((uv+") != std::string::npos);

    Graph textureGraph;
    const auto textureSource = textureGraph.addNode("perlin");
    const auto texture = textureGraph.addNode("texture_sample");
    textureGraph.addLink(textureSource, "image", texture, "source");
    const auto textureCompiled = textureGraph.compile(registry);
    REQUIRE(textureCompiled.valid);
    const auto textureRegions = planShaderRegions(textureGraph, registry, textureCompiled, 16, 1024);
    REQUIRE(textureRegions.size() == 2);
    const auto textureGenerated = generateComputeShader(textureRegions[1], {texture});
    REQUIRE(textureGenerated.source.find("clamp(uv,vec2(0.0),vec2(1.0))") != std::string::npos);
}

TEST_CASE("convolution iterations above one stay a materialized boundary") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph;
    const auto source = graph.addNode("perlin");
    const auto convolution = graph.addNode("convolution");
    graph.findNode(convolution)->parameters = {{"iterations", 3.0F}};
    const auto threshold = graph.addNode("threshold");
    graph.addLink(source, "image", convolution, "image");
    graph.addLink(convolution, "image", threshold, "value");
    const auto compiled = graph.compile(registry);
    REQUIRE(compiled.valid);
    const auto regions = planShaderRegions(graph, registry, compiled, 16, 1024);
    const auto contains = [&](NodeId id) {
        return std::ranges::any_of(regions, [&](const auto& region) {
            return std::ranges::find(region.nodes, id) != region.nodes.end();
        });
    };
    REQUIRE_FALSE(contains(convolution));
    REQUIRE(contains(source));
    REQUIRE(contains(threshold));
}

TEST_CASE("boundary lowering specializes constants and fields without flexible uniforms") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph;
    const auto source = graph.addNode("math");
    const auto threshold = graph.addNode("threshold");
    graph.addLink(source, "result", threshold, "value");
    const auto compiled = graph.compile(registry);
    REQUIRE(compiled.valid);

    const auto constant = lowerShaderRegion(graph, registry, compiled, {threshold},
        [](NodeId, std::string_view) { return ShaderInputKind::Float; });
    REQUIRE(constant.inputs.front().kind == ShaderInputKind::Float);
    REQUIRE_FALSE(constant.candidateOutputs.front().value.field);
    const auto constantShader = generateComputeShader(constant, {threshold});
    REQUIRE(constantShader.source.find("sampler2D") == std::string::npos);
    REQUIRE(constantShader.source.find("inputHasImage") == std::string::npos);

    const auto field = lowerShaderRegion(graph, registry, compiled, {threshold},
        [](NodeId, std::string_view) { return ShaderInputKind::Field; });
    REQUIRE(field.inputs.front().kind == ShaderInputKind::Field);
    REQUIRE(field.candidateOutputs.front().value.field);
    const auto fieldShader = generateComputeShader(field, {threshold});
    REQUIRE(fieldShader.source.find("sampler2D") != std::string::npos);
    REQUIRE(fieldShader.source != constantShader.source);
}

TEST_CASE("constant-only lowering stays semantic and fusion off plans solo regions") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph constants;
    const auto value = constants.addNode("float");
    const auto math = constants.addNode("math");
    constants.addLink(value, "value", math, "a");
    const auto compiled = constants.compile(registry);
    REQUIRE(compiled.valid);

    const auto fused = planShaderRegions(constants, registry, compiled, 16, 1024, true);
    REQUIRE(fused.size() == 1);
    REQUIRE(fused.front().nodes == std::vector<NodeId>{value, math});
    REQUIRE_FALSE(fused.front().candidateOutputs.front().value.field);
    REQUIRE(fused.front().candidateOutputs.front().value.type == ShaderValueType::Scalar);

    const auto solo = planShaderRegions(constants, registry, compiled, 16, 1024, false);
    REQUIRE(solo.size() == 2);
    REQUIRE(solo[0].nodes.size() == 1);
    REQUIRE(solo[1].nodes.size() == 1);
}

TEST_CASE("every fully lowerable built-in executes as a solo generated region") {
    HiddenContext window;
    NodeRegistry registry; registerBuiltInNodes(registry);
    const std::array<const char*, 10> independent = {
        "float", "math", "mix", "threshold", "select", "compare",
        "color_ramp", "perlin", "coordinates", "bit_test"};
    for (const auto* type : independent) {
        INFO(type);
        Graph graph; graph.settings = {4, 4, 60};
        const auto node = graph.addNode(type);
        GpuRuntime gpu;
        GraphRuntime runtime(graph, registry, gpu);
        runtime.setFusionEnabled(false);
        REQUIRE(runtime.evaluate(0.0, 0.0, false));
        REQUIRE(runtime.fusionInfo(node).has_value());
        REQUIRE(runtime.fusionInfo(node)->nodeCount == 1);
        REQUIRE(runtime.fusionInfo(node)->mode == GeneratedExecutionMode::Generated);
        if (std::string_view(type) == "float" || std::string_view(type) == "math" ||
            std::string_view(type) == "mix" || std::string_view(type) == "threshold" ||
            std::string_view(type) == "select" || std::string_view(type) == "compare" ||
            std::string_view(type) == "bit_test")
            REQUIRE(std::holds_alternative<float>(runtime.values().at(node).front()));
    }

    for (const auto* type : {"invert", "laplacian", "convolution"}) {
        INFO(type);
        Graph graph; graph.settings = {4, 4, 60};
        const auto source = graph.addNode("perlin");
        const auto node = graph.addNode(type);
        graph.addLink(source, "image", node,
                      std::string_view(type) == "laplacian" ? "value" : "image");
        GpuRuntime gpu;
        GraphRuntime runtime(graph, registry, gpu);
        runtime.setFusionEnabled(false);
        REQUIRE(runtime.evaluate(0.0, 0.0, false));
        REQUIRE(runtime.fusionInfo(node)->nodeCount == 1);
        REQUIRE(runtime.fusionInfo(node)->mode == GeneratedExecutionMode::Generated);
        REQUIRE(std::holds_alternative<ImageHandle>(runtime.values().at(node).front()));
    }
}

TEST_CASE("requiresImage reports the contributing node for constant inputs") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph;
    const auto constant = graph.addNode("math");
    const auto convolution = graph.addNode("convolution");
    graph.addLink(constant, "result", convolution, "image");
    const auto compiled = graph.compile(registry);
    REQUIRE_FALSE(compiled.valid);
    REQUIRE(std::ranges::any_of(compiled.errors, [](const auto& error) {
        return error.find("semantically inadmissible") != std::string::npos;
    }));
}

TEST_CASE("native convolution escape hatch still rejects a constant source") {
    HiddenContext window;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {4, 4, 60};
    const auto constant = graph.addNode("math");
    const auto convolution = graph.addNode("convolution");
    graph.findNode(convolution)->parameters = {{"iterations", 2.0F}};
    graph.addLink(constant, "result", convolution, "image");
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0.0, 0.0, false));
    REQUIRE(runtime.fusionInfo(convolution).has_value());
    REQUIRE(runtime.fusionInfo(convolution)->compileFailed);
    REQUIRE(std::holds_alternative<std::monostate>(
        runtime.values().at(convolution).front()));
}

TEST_CASE("shader planner keeps branch and join boundaries materialized") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph;
    const auto source = graph.addNode("reaction_diffusion");
    const auto branch = graph.addNode("math");
    const auto left = graph.addNode("math");
    const auto right = graph.addNode("math");
    const auto join = graph.addNode("math");
    graph.addLink(source, "image", branch, "a");
    graph.addLink(branch, "result", left, "a");
    graph.addLink(branch, "result", right, "a");
    graph.addLink(left, "result", join, "a");
    graph.addLink(right, "result", join, "b");
    const auto compiled = graph.compile(registry);
    REQUIRE(compiled.valid);
    const auto regions = planShaderRegions(graph, registry, compiled, 16, 1024);
    REQUIRE(regions.size() == 4);
    REQUIRE(std::ranges::all_of(regions, [](const auto& region) {
        return region.nodes.size() == 1;
    }));
}

TEST_CASE("shader requirements deduplicate bindings and split at resource limits") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph deduplicated;
    const auto source = deduplicated.addNode("image");
    const auto math = deduplicated.addNode("math");
    deduplicated.addLink(source, "image", math, "a");
    deduplicated.addLink(source, "image", math, "b");
    auto compiled = deduplicated.compile(registry);
    const auto one = planShaderRegions(deduplicated, registry, compiled, 16, 1024);
    REQUIRE(one.size() == 1);
    REQUIRE(std::ranges::count_if(one.front().inputs, [](const auto& input) {
        return input.kind == ShaderInputKind::Field;
    }) == 1);

    Graph limited;
    const auto a = limited.addNode("image");
    const auto b = limited.addNode("image");
    const auto c = limited.addNode("image");
    const auto first = limited.addNode("math");
    const auto second = limited.addNode("math");
    const auto third = limited.addNode("math");
    limited.addLink(a, "image", first, "a");
    limited.addLink(first, "result", second, "a");
    limited.addLink(b, "image", second, "b");
    limited.addLink(second, "result", third, "a");
    limited.addLink(c, "image", third, "b");
    compiled = limited.compile(registry);
    const auto split = planShaderRegions(limited, registry, compiled, 2, 1024);
    REQUIRE(split.size() == 2);
    REQUIRE(split[0].nodes == std::vector<NodeId>{first, second});
    REQUIRE(split[1].nodes == std::vector<NodeId>{third});

    Graph uniformLimited;
    const auto image = uniformLimited.addNode("image");
    const auto addOne = uniformLimited.addNode("math");
    const auto addTwo = uniformLimited.addNode("math");
    uniformLimited.addLink(image, "image", addOne, "a");
    uniformLimited.addLink(addOne, "result", addTwo, "a");
    compiled = uniformLimited.compile(registry);
    const auto uniformSplit = planShaderRegions(
        uniformLimited, registry, compiled, 16, 1);
    REQUIRE(uniformSplit.size() == 2);
    REQUIRE(uniformSplit[0].nodes == std::vector<NodeId>{addOne});
    REQUIRE(uniformSplit[1].nodes == std::vector<NodeId>{addTwo});
}

TEST_CASE("image input exposes a PNG-backed image output") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("image");
    REQUIRE(descriptor != nullptr);
    REQUIRE(descriptor->displayName == "Image");
    REQUIRE(descriptor->category == "Input");
    REQUIRE(descriptor->sockets.size() == 1);
    REQUIRE(descriptor->sockets[0].direction == SocketDirection::Output);
    REQUIRE(descriptor->sockets[0].contract == exactContract(ValueType::ColorImage));
}

TEST_CASE("image input loads PNG pixels with the graph image orientation") {
    HiddenContext window;
    GpuRuntime gpu;
    NodeRegistry registry; registerBuiltInNodes(registry);
    auto node = registry.create("image");

    const auto path = std::filesystem::temp_directory_path() / "reaction-image-node-test.png";
    const std::array<png_byte, 8> pixels = {
        255, 0, 0, 255, // top: red
        0, 0, 255, 128  // bottom: blue
    };
    png_image png{};
    png.version = PNG_IMAGE_VERSION;
    png.width = 1;
    png.height = 2;
    png.format = PNG_FORMAT_RGBA;
    REQUIRE(png_image_write_to_file(&png, path.string().c_str(), 0, pixels.data(), 4, nullptr));

    node->setParameters({{"path", path.string()}});
    EvaluationContext context{1, 2, 0.0, 0.0, 0, false, &gpu};
    std::array<Value, 1> outputs;
    node->evaluate(context, {}, outputs);
    const auto image = std::get<ImageHandle>(outputs[0]);
    const auto values = readImage(image);
    REQUIRE(values[0] == Catch::Approx(0.0F).margin(.002F));
    REQUIRE(values[2] == Catch::Approx(1.0F).margin(.002F));
    REQUIRE(values[3] == Catch::Approx(128.0F / 255.0F).margin(.002F));
    REQUIRE(values[4] == Catch::Approx(1.0F).margin(.002F));
    REQUIRE(values[6] == Catch::Approx(0.0F).margin(.002F));
    std::filesystem::remove(path);
}

TEST_CASE("empty to field boundary change re-specializes exactly once") {
    HiddenContext window;
    GpuRuntime gpu;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {2, 2, 60};
    const auto image = graph.addNode("image");
    const auto threshold = graph.addNode("threshold");
    graph.addLink(image, "image", threshold, "value");
    GraphRuntime runtime(graph, registry, gpu);

    REQUIRE(runtime.evaluate(0.0, 0.0, false));
    const auto emptyCount = runtime.fusionInfo(threshold)->specializationCount;

    const auto path = std::filesystem::temp_directory_path() /
                      "reaction-boundary-specialization-test.png";
    const std::array<png_byte, 4> pixels = {255, 255, 255, 255};
    png_image png{};
    png.version = PNG_IMAGE_VERSION;
    png.width = 1;
    png.height = 1;
    png.format = PNG_FORMAT_RGBA;
    REQUIRE(png_image_write_to_file(&png, path.string().c_str(), 0,
                                    pixels.data(), 4, nullptr));
    graph.findNode(image)->parameters = {{"path", path.string()}};

    REQUIRE(runtime.evaluate(0.0, 0.0, false));
    REQUIRE(runtime.fusionInfo(threshold)->specializationCount == emptyCount + 1);
    REQUIRE(std::holds_alternative<ImageHandle>(runtime.values().at(threshold).front()));
    REQUIRE(runtime.evaluate(0.0, 0.0, false));
    REQUIRE(runtime.fusionInfo(threshold)->specializationCount == emptyCount + 1);
    std::filesystem::remove(path);
}

TEST_CASE("reaction defaults use the sustained classic Gray-Scott region") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("reaction_diffusion");
    REQUIRE(descriptor != nullptr);
    const auto defaultFor = [&](std::string_view key) {
        return std::ranges::find(descriptor->parameters, key, &ParameterDescriptor::key)->defaultValue;
    };
    REQUIRE(defaultFor("feed") == Catch::Approx(0.055F));
    REQUIRE(defaultFor("kill") == Catch::Approx(0.062F));
    REQUIRE(defaultFor("diffA") == Catch::Approx(1.0F));
    REQUIRE(defaultFor("diffB") == Catch::Approx(0.5F));
    REQUIRE(defaultFor("structureScale") == Catch::Approx(1.0F));
    REQUIRE(defaultFor("dt") == Catch::Approx(1.0F));
    REQUIRE(defaultFor("iterations") == Catch::Approx(8.0F));
    REQUIRE(defaultFor("autoReset") == Catch::Approx(0.0F));
}

TEST_CASE("procedural node descriptors expose safe controls and vector sockets") {
    NodeRegistry registry; registerBuiltInNodes(registry);

    const auto* sample = registry.descriptor("texture_sample");
    REQUIRE(sample != nullptr);
    REQUIRE(sample->sockets.size() == 4);
    REQUIRE(sample->sockets[1].key == "coordinates");
    REQUIRE(sample->sockets[1].contract == SocketContract::VectorNumeric);

    for (const auto type : {"texture_sample", "repeat_fold", "worley_noise", "gradient", "wave"}) {
        const auto* descriptor = registry.descriptor(type);
        REQUIRE(descriptor != nullptr);
        const auto parameter = std::ranges::find_if(descriptor->parameters, [](const auto& item) {
            return item.control == ParameterDescriptor::Control::Enum;
        });
        REQUIRE(parameter != descriptor->parameters.end());
        REQUIRE_FALSE(parameter->enumOptions.empty());
        REQUIRE(parameter->maximum == Catch::Approx(
            static_cast<float>(parameter->enumOptions.size() - 1)));
    }

    const auto* transform = registry.descriptor("transform_2d");
    REQUIRE(transform != nullptr);
    REQUIRE(std::ranges::find(transform->parameters, "rotation",
                              &ParameterDescriptor::key) != transform->parameters.end());

    const auto* fold = registry.descriptor("repeat_fold");
    REQUIRE(fold != nullptr);
    for (const auto key : {"value", "period", "offset", "foldCenter"})
        REQUIRE(std::ranges::find(fold->parameters, key,
                                  &ParameterDescriptor::key) != fold->parameters.end());

    REQUIRE(registry.contains("vector"));
    REQUIRE(registry.contains("resolution"));
    REQUIRE(registry.contains("combine_vector"));
    REQUIRE(registry.contains("separate_vector"));
}

TEST_CASE("Resolution exposes render dimensions as semantic constants") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("resolution");
    REQUIRE(descriptor != nullptr);
    REQUIRE(descriptor->sockets.size() == 3);
    REQUIRE(descriptor->sockets[0].key == "resolution");
    REQUIRE(descriptor->sockets[0].contract == exactContract(ValueType::Vec2));
    REQUIRE(descriptor->sockets[1].key == "pixelSize");
    REQUIRE(descriptor->sockets[1].contract == exactContract(ValueType::Vec2));
    REQUIRE(descriptor->sockets[2].key == "aspectRatio");
    REQUIRE(descriptor->sockets[2].contract == exactContract(ValueType::Float));

    auto node = registry.create("resolution");
    std::array<Value, 3> values;
    EvaluationContext context;
    context.width = 800;
    context.height = 500;
    node->evaluate(context, {}, values);
    REQUIRE(std::get<Vec2>(values[0]).x == Catch::Approx(800.0F));
    REQUIRE(std::get<Vec2>(values[0]).y == Catch::Approx(500.0F));
    REQUIRE(std::get<Vec2>(values[1]).x == Catch::Approx(1.0F / 800.0F));
    REQUIRE(std::get<Vec2>(values[1]).y == Catch::Approx(1.0F / 500.0F));
    REQUIRE(std::get<float>(values[2]) == Catch::Approx(1.6F));
}

TEST_CASE("Table maps promoted indices through editable addressed values") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("table");
    REQUIRE(descriptor != nullptr);
    REQUIRE(descriptor->lowerable);
    REQUIRE(descriptor->sockets[0].contract == SocketContract::Numeric);
    REQUIRE(descriptor->sockets[1].contract == SocketContract::Numeric);

    auto node = registry.create("table");
    node->setParameters({{"values", {0.0F, 10.0F, 20.0F}}, {"sampling", 1.0F},
                         {"address", 1.0F}, {"indexUnits", 0.0F}});
    CapturingLoweringContext lowering(ShaderValueType::Scalar);
    lowering.inputs = {{"index", {ShaderValueType::Scalar, "index"}}};
    REQUIRE(node->lowerShader(lowering));
    REQUIRE(std::ranges::find(lowering.trace, "input:index") != lowering.trace.end());
    REQUIRE(lowering.emitted.name == "tableLookup(index)");
    REQUIRE(lowering.helpers.size() == 1);
    REQUIRE(lowering.helpers.front().find("float values[3]=float[3](0.0,10.0,20.0)") !=
            std::string::npos);
    REQUIRE(lowering.helpers.front().find("int m=value%3") != std::string::npos);
    REQUIRE(lowering.helpers.front().find("floor(position)") != std::string::npos);
    const auto linearVariant = node->shaderVariantKey(node->parameters());

    node->setParameters({{"values", {1.0F, 2.0F}}, {"sampling", 0.0F},
                         {"address", 2.0F}, {"indexUnits", 1.0F}});
    CapturingLoweringContext normalized(ShaderValueType::Scalar);
    normalized.inputs = {{"index", {ShaderValueType::Scalar, "index"}}};
    REQUIRE(node->lowerShader(normalized));
    REQUIRE(normalized.helpers.front().find("index*float(2-1)") != std::string::npos);
    REQUIRE(normalized.helpers.front().find("round(position)") != std::string::npos);
    REQUIRE(node->shaderVariantKey(node->parameters()) != linearVariant);
}

TEST_CASE("Bit Test safely queries Float-backed masks for constants and fields") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("bit_test");
    REQUIRE(descriptor != nullptr);
    REQUIRE(descriptor->lowerable);
    REQUIRE(descriptor->sockets.size() == 3);
    REQUIRE(descriptor->sockets[0].key == "mask");
    REQUIRE(descriptor->sockets[0].contract == exactContract(ValueType::Float));
    REQUIRE(descriptor->sockets[1].key == "bit");
    REQUIRE(descriptor->sockets[1].contract == SocketContract::Numeric);
    REQUIRE(descriptor->sockets[2].contract == SocketContract::Numeric);

    auto node = registry.create("bit_test");
    CapturingLoweringContext lowering(ShaderValueType::Scalar);
    lowering.inputs = {{"mask", {ShaderValueType::Scalar, "mask"}},
                       {"bit", {ShaderValueType::Scalar, "bit"}}};
    REQUIRE(node->lowerShader(lowering));
    REQUIRE(lowering.emitted.name == "integerMaskTest(mask,bit)");
    REQUIRE(lowering.helpers.size() == 1);
    REQUIRE(lowering.helpers.front().find("round(value)") != std::string::npos);
    REQUIRE(lowering.helpers.front().find("index<0.0||index>23.0") != std::string::npos);
    REQUIRE(lowering.helpers.front().find("bits>>uint(index)") != std::string::npos);

    HiddenContext context;
    Graph graph; graph.settings = {4, 1, 60};
    const auto coordinates = graph.addNode("coordinates");
    const auto separate = graph.addNode("separate_vector");
    const auto scale = graph.addNode("math");
    const auto floor = graph.addNode("math");
    const auto bitTest = graph.addNode("bit_test");
    graph.findNode(scale)->parameters = {{"operation", 2.0F}, {"b", 8.0F}};
    graph.findNode(floor)->parameters = {{"operation", 12.0F}};
    graph.findNode(bitTest)->parameters = {{"mask", 10.0F}};
    graph.addLink(coordinates, "coordinates", separate, "value");
    graph.addLink(separate, "x", scale, "a");
    graph.addLink(scale, "result", floor, "a");
    graph.addLink(floor, "result", bitTest, "bit");
    REQUIRE(graph.compile(registry).valid);

    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto fused = readImage(std::get<ImageHandle>(runtime.values().at(bitTest).front()));
    REQUIRE(fused[0] == Catch::Approx(1.0F));
    REQUIRE(fused[4] == Catch::Approx(1.0F));
    REQUIRE(fused[8] == Catch::Approx(0.0F));
    REQUIRE(fused[12] == Catch::Approx(0.0F));

    runtime.setFusionEnabled(false);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto solo = readImage(std::get<ImageHandle>(runtime.values().at(bitTest).front()));
    REQUIRE(solo == fused);
}

TEST_CASE("Deterministic Hash exposes typed outputs and fixed integer lowering") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("hash");
    REQUIRE(descriptor != nullptr);
    REQUIRE(descriptor->lowerable);
    REQUIRE(descriptor->sockets.size() == 5);
    REQUIRE(descriptor->sockets[0].contract == SocketContract::VectorNumeric);
    REQUIRE(descriptor->sockets[1].contract == exactContract(ValueType::Float));
    REQUIRE(descriptor->sockets[2].contract == exactContract(ValueType::Float));
    REQUIRE(descriptor->sockets[3].contract == SocketContract::Numeric);
    REQUIRE(descriptor->sockets[4].contract == SocketContract::VectorNumeric);

    auto node = registry.create("hash");
    node->setParameters({{"inputMode", 1.0F}});
    CapturingLoweringContext lowering(ShaderValueType::Scalar);
    lowering.inputs = {{"position", {ShaderValueType::Vec2, "position"}},
                       {"seed", {ShaderValueType::Scalar, "seed"}},
                       {"salt", {ShaderValueType::Scalar, "salt"}}};
    REQUIRE(node->lowerShader(lowering));
    REQUIRE(std::ranges::find(lowering.trace, "helper:deterministicHash") != lowering.trace.end());
    REQUIRE(lowering.emitted.name.find("vec2(") != std::string::npos);
    REQUIRE(lowering.emitted.name.find("floor(position)") != std::string::npos);

    node->setParameters({{"inputMode", 0.0F}});
    CapturingLoweringContext rawLowering(ShaderValueType::Scalar);
    rawLowering.inputs = {{"position", {ShaderValueType::Vec2, "position"}},
                          {"seed", {ShaderValueType::Scalar, "seed"}},
                          {"salt", {ShaderValueType::Scalar, "salt"}}};
    REQUIRE(node->lowerShader(rawLowering));
    REQUIRE(rawLowering.emitted.name.find("floatBitsToUint(position.x)") != std::string::npos);
}

TEST_CASE("Hash promotes a vector field while disconnected hashes stay constant") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph;
    const auto coordinates = graph.addNode("coordinates");
    const auto fieldHash = graph.addNode("hash");
    const auto constantHash = graph.addNode("hash");
    const auto output = graph.addNode("output");
    graph.addLink(coordinates, "coordinates", fieldHash, "position");
    graph.addLink(fieldHash, "scalar", output, "image");
    const auto compiled = graph.compile(registry);
    REQUIRE(compiled.valid);
    const auto regions = planShaderRegions(graph, registry, compiled, 16, 1024);
    const auto fieldRegion = std::ranges::find(regions, fieldHash, &ShaderRegion::id);
    REQUIRE(fieldRegion != regions.end());
    const auto fieldOutput = std::ranges::find_if(fieldRegion->candidateOutputs, [](const auto& output) {
        return output.socket == "scalar";
    });
    REQUIRE(fieldOutput != fieldRegion->candidateOutputs.end());
    REQUIRE(fieldOutput->value.field);

    const auto constantRegion = std::ranges::find(regions, constantHash, &ShaderRegion::id);
    REQUIRE(constantRegion != regions.end());
    const auto scalarOutput = std::ranges::find_if(constantRegion->candidateOutputs, [](const auto& output) {
        return output.socket == "scalar";
    });
    REQUIRE(scalarOutput != constantRegion->candidateOutputs.end());
    REQUIRE_FALSE(scalarOutput->requiresImage);
}

TEST_CASE("reaction multiplier inputs accept both scalar and image values") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("reaction_diffusion");
    REQUIRE(descriptor != nullptr);
    const auto typeFor = [&](std::string_view key) {
        return std::ranges::find(descriptor->sockets, key, &SocketDescriptor::key)->contract;
    };
    REQUIRE(typeFor("feedMultiplier") == SocketContract::Numeric);
    REQUIRE(typeFor("killMultiplier") == SocketContract::Numeric);

    Graph graph;
    const auto feed = graph.addNode("float");
    const auto kill = graph.addNode("float");
    const auto reaction = graph.addNode("reaction_diffusion");
    graph.addLink(feed, "value", reaction, "feedMultiplier");
    graph.addLink(kill, "value", reaction, "killMultiplier");
    REQUIRE(graph.compile(registry).valid);
}

TEST_CASE("float preview accepts and passes through float values") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("float_preview");
    REQUIRE(descriptor != nullptr);
    REQUIRE(descriptor->displayName == "Float Preview");
    REQUIRE(descriptor->sockets.size() == 2);
    REQUIRE(descriptor->sockets[0].contract == exactContract(ValueType::Float));
    REQUIRE(descriptor->sockets[1].contract == exactContract(ValueType::Float));

    Graph graph;
    const auto source = graph.addNode("float");
    const auto preview = graph.addNode("float_preview");
    graph.addLink(source, "value", preview, "value");
    REQUIRE(graph.compile(registry).valid);
}

TEST_CASE("image invert exposes AnyField input and output") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("invert");
    REQUIRE(descriptor != nullptr);
    REQUIRE(descriptor->displayName == "Image Invert");
    REQUIRE(descriptor->sockets.size() == 2);
    REQUIRE(descriptor->sockets[0].contract == SocketContract::AnyField);
    REQUIRE(descriptor->sockets[1].contract == SocketContract::AnyField);

    Graph graph;
    const auto source = graph.addNode("perlin");
    const auto invert = graph.addNode("invert");
    graph.addLink(source, "image", invert, "image");
    REQUIRE(graph.compile(registry).valid);
}

TEST_CASE("convolution exposes bounded iteration control") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("convolution");
    REQUIRE(descriptor != nullptr);
    const auto iteration = std::ranges::find(
        descriptor->parameters, "iterations", &ParameterDescriptor::key);
    REQUIRE(iteration != descriptor->parameters.end());
    REQUIRE(iteration->defaultValue == 1.0F);
    REQUIRE(iteration->minimum == 1.0F);
    REQUIRE(iteration->maximum == 32.0F);
}

TEST_CASE("spatial simulation operators are ordinary registered nodes") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* coordinates = registry.descriptor("coordinates");
    REQUIRE(coordinates != nullptr);
    REQUIRE(coordinates->displayName == "Canvas Coordinates");
    REQUIRE(coordinates->category == "Input");
    REQUIRE(coordinates->sockets.size() == 1);
    REQUIRE(coordinates->sockets[0].key == "coordinates");
    REQUIRE(coordinates->sockets[0].contract == SocketContract::VectorFieldOnly);
    REQUIRE(coordinates->parameters.size() == 1);
    REQUIRE(coordinates->parameters[0].key == "pixels");

    const auto* laplacian = registry.descriptor("laplacian");
    REQUIRE(laplacian != nullptr);
    REQUIRE(laplacian->displayName == "Laplacian");
    REQUIRE(laplacian->category == "Filter");
    REQUIRE(laplacian->sockets.size() == 3);
    REQUIRE(laplacian->sockets[0].key == "value");
    REQUIRE(laplacian->sockets[0].contract == SocketContract::AnyField);
    REQUIRE(laplacian->sockets[1].key == "scale");
    REQUIRE(laplacian->sockets[1].contract == exactContract(ValueType::Float));
    REQUIRE(laplacian->sockets[1].optional);
    REQUIRE(laplacian->sockets[2].key == "result");
    REQUIRE(laplacian->sockets[2].contract == SocketContract::AnyField);
    REQUIRE(laplacian->parameters.size() == 1);
    REQUIRE(laplacian->parameters[0].defaultValue == 1.0F);
    REQUIRE(laplacian->parameters[0].minimum == .25F);
    REQUIRE(laplacian->parameters[0].maximum == 8.0F);

    Graph graph;
    const auto uv = graph.addNode("coordinates");
    const auto filter = graph.addNode("laplacian");
    graph.addLink(uv, "coordinates", filter, "value");
    REQUIRE(graph.compile(registry).valid);
}

TEST_CASE("Select is an ordinary registered node with exact nonzero semantics") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("select");
    REQUIRE(descriptor != nullptr);
    REQUIRE(descriptor->displayName == "Select");
    REQUIRE(descriptor->category == "Logic");
    REQUIRE(descriptor->sockets.size() == 4);

    auto node = registry.create("select");
    CapturingLoweringContext lowering(ShaderValueType::Scalar);
    lowering.inputs = {{"condition", {ShaderValueType::Scalar, "condition"}},
                       {"ifTrue", {ShaderValueType::Scalar, "yes"}},
                       {"ifFalse", {ShaderValueType::Scalar, "no"}}};
    REQUIRE(node->lowerShader(lowering));
    REQUIRE(lowering.emitted.name == "((condition)!=0.0?yes:no)");
}

TEST_CASE("canvas coordinates and Laplacian execute through the normal graph runtime") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {4, 4, 60};
    const auto coordinates = graph.addNode("coordinates");
    const auto laplacian = graph.addNode("laplacian");
    graph.addLink(coordinates, "coordinates", laplacian, "value");
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));

    const auto& coordinateValues = runtime.values().at(coordinates);
    REQUIRE(coordinateValues.size() == 1);
    const auto coordinatesValue = readImage(std::get<ImageHandle>(coordinateValues[0]));
    REQUIRE(coordinatesValue[0] == Catch::Approx(.125F).margin(.001F));
    REQUIRE(coordinatesValue[1] == Catch::Approx(.125F).margin(.001F));
    REQUIRE(coordinatesValue[3 * 4] == Catch::Approx(.875F).margin(.001F));
    REQUIRE(coordinatesValue[4 * 4] == Catch::Approx(.125F).margin(.001F));
    REQUIRE(coordinatesValue[1] == Catch::Approx(.125F).margin(.001F));
    REQUIRE(coordinatesValue[4 * 4 + 1] == Catch::Approx(.375F).margin(.001F));

    const auto result = readImage(std::get<ImageHandle>(runtime.values().at(laplacian)[0]));
    REQUIRE(result[0] == Catch::Approx(.3F).margin(.003F));
    REQUIRE(result[(1 * 4 + 1) * 4] == Catch::Approx(0.0F).margin(.003F));
}

TEST_CASE("Perlin advances by evaluated frame rather than wall time") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {32, 24, 60};
    const auto perlin = graph.addNode("perlin");
    const auto output = graph.addNode("output");
    graph.addLink(perlin, "image", output, "image"); graph.activeOutput = output;
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(1.25, 1.0 / 60.0, true));
    const auto firstHandle = runtime.outputImage();
    const auto first = readImage(firstHandle);
    REQUIRE(std::ranges::all_of(first, [](float value) { return std::isfinite(value) && value >= 0.0F && value <= 1.0F; }));
    REQUIRE(runtime.evaluate(25.0, 1.0 / 60.0, true));
    REQUIRE(runtime.outputImage().texture == firstHandle.texture);
    REQUIRE(readImage(runtime.outputImage()) != first);
    runtime.reset();
    REQUIRE(runtime.evaluate(100.0, 1.0 / 60.0, true));
    REQUIRE(readImage(runtime.outputImage()) == first);
}

TEST_CASE("scalar Math safe division is emitted by its only implementation") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    auto math = registry.create("math");
    math->setParameters({{"operation", 3}, {"a", 1.0}, {"b", 0.0}});
    CapturingLoweringContext lowering(ShaderValueType::Scalar);
    REQUIRE(math->lowerShader(lowering));
    REQUIRE(lowering.emitted.name.find("1e-6") != std::string::npos);
}

TEST_CASE("fused Math chain matches solo lowering and materializes previews on demand") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {16, 16, 60};
    const auto coordinates = graph.addNode("coordinates");
    const auto separate = graph.addNode("separate_vector");
    const auto add = graph.addNode("math");
    const auto multiply = graph.addNode("math");
    graph.findNode(add)->parameters = {{"operation", 0.0F}, {"b", .125F}};
    graph.findNode(multiply)->parameters = {{"operation", 2.0F}, {"b", .75F}};
    graph.addLink(coordinates, "coordinates", separate, "value");
    graph.addLink(separate, "x", add, "a");
    graph.addLink(add, "result", multiply, "a");
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));
    REQUIRE(runtime.fusionInfo(add).has_value());
    REQUIRE(runtime.fusionInfo(add)->interior);
    REQUIRE(runtime.fusionInfo(multiply)->nodeCount == 4);
    REQUIRE_FALSE(runtime.values().contains(add));
    const auto generated = readImage(std::get<ImageHandle>(runtime.values().at(multiply).front()));

    runtime.setFusionEnabled(false);
    REQUIRE(runtime.evaluate(0, 0, false));
    REQUIRE(runtime.values().contains(add));
    const auto solo = readImage(std::get<ImageHandle>(runtime.values().at(multiply).front()));
    REQUIRE(generated == solo);

    runtime.setFusionEnabled(true);
    runtime.setIntermediatePreview(add);
    REQUIRE(runtime.evaluate(0, 0, false));
    REQUIRE(runtime.values().contains(add));
    REQUIRE(runtime.fusionInfo(add)->materialized);
    runtime.setIntermediatePreview(std::nullopt);
    REQUIRE(runtime.evaluate(0, 0, false));
    REQUIRE_FALSE(runtime.values().contains(add));
}

TEST_CASE("every generated Math operation compiles and fused pixels agree with solo lowering") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    for (int operation = 0; operation < static_cast<int>(kMathOperationNames.size()); ++operation) {
        Graph graph; graph.settings = {8, 8, 60};
        const auto coordinates = graph.addNode("coordinates");
        const auto separate = graph.addNode("separate_vector");
        const auto math = graph.addNode("math");
        graph.findNode(math)->parameters = {{"operation", static_cast<float>(operation)},
            {"b", .4F}, {"c", .8F}, {"inMin", .1F}, {"inMax", .9F},
            {"outMin", -.25F}, {"outMax", 1.25F}};
        graph.addLink(coordinates, "coordinates", separate, "value");
        graph.addLink(separate, "x", math, "a");
        GpuRuntime gpu;
        GraphRuntime runtime(graph, registry, gpu);
        REQUIRE(runtime.evaluate(0, 0, false));
        INFO("operation=" << operation);
        REQUIRE(runtime.generatedShaders().size() == 1);
        REQUIRE(runtime.generatedShaders().front().mode == GeneratedExecutionMode::Generated);
        const auto generated = readImage(std::get<ImageHandle>(runtime.values().at(math).front()));
        runtime.setFusionEnabled(false);
        REQUIRE(runtime.evaluate(0, 0, false));
        const auto solo = readImage(std::get<ImageHandle>(runtime.values().at(math).front()));
        REQUIRE(generated.size() == solo.size());
        for (std::size_t index = 0; index < generated.size(); ++index)
            REQUIRE(generated[index] == Catch::Approx(solo[index]).margin(.002F));
    }
}

TEST_CASE("generated safe power preserves a negative scalar base") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {8, 8, 60};
    const auto coordinates = graph.addNode("coordinates");
    const auto separate = graph.addNode("separate_vector");
    const auto math = graph.addNode("math");
    graph.findNode(math)->parameters = {{"operation", 4.0F}, {"a", -.5F}};
    graph.addLink(coordinates, "coordinates", separate, "value");
    graph.addLink(separate, "x", math, "b");
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto pixels = readImage(std::get<ImageHandle>(runtime.values().at(math).front()));
    REQUIRE(std::ranges::all_of(pixels, [](float value) { return std::isfinite(value); }));
    REQUIRE(pixels.front() < 0.0F);
}

TEST_CASE("non-throwing compute compilation returns structured diagnostics") {
    HiddenContext context;
    GpuRuntime gpu;
    const auto result = gpu.tryCompileCompute("#version 430\nthis is not GLSL\n", "invalid test");
    REQUIRE_FALSE(result);
    REQUIRE_FALSE(result.log.empty());
    REQUIRE(result.reportedLine > 0);
}

TEST_CASE("Mix exposes blend modes and applies the selected mode to scalar inputs") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("mix");
    REQUIRE(descriptor != nullptr);
    const auto mode = std::ranges::find(descriptor->parameters, "mode", &ParameterDescriptor::key);
    REQUIRE(mode != descriptor->parameters.end());
    REQUIRE(mode->minimum == 0.0F);
    REQUIRE(mode->maximum == 9.0F);

    Graph graph; graph.settings = {16, 16, 60};
    const auto mix = graph.addNode("mix");
    graph.findNode(mix)->parameters = {{"mode", 2.0F}, {"a", .25F}, {"b", .8F}, {"factor", 1.0F}};
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto& value = runtime.values().at(mix).front();
    REQUIRE(std::holds_alternative<float>(value));
    REQUIRE(std::get<float>(value) == Catch::Approx(.2F).margin(0.005));
}

TEST_CASE("Mix image path compiles and produces a GPU image") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {16, 16, 60};
    const auto source = graph.addNode("perlin");
    const auto mix = graph.addNode("mix");
    graph.findNode(mix)->parameters = {{"mode", 4}, {"b", .75F}, {"factor", .5F}};
    graph.addLink(source, "image", mix, "a");
    GpuRuntime gpu; GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto& value = runtime.values().at(mix).front();
    REQUIRE(std::holds_alternative<ImageHandle>(value));
    REQUIRE(std::get<ImageHandle>(value).texture != 0);
}

TEST_CASE("threshold produces a binary scalar result") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {16, 16, 60};
    const auto threshold = graph.addNode("threshold");
    graph.findNode(threshold)->parameters = {{"value", 0.6F}, {"threshold", 0.5F}};
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto& high = runtime.values().at(threshold).front();
    REQUIRE(std::holds_alternative<float>(high));
    REQUIRE(std::get<float>(high) == 1.0F);

    graph.findNode(threshold)->parameters["value"] = 0.4F;
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto& low = runtime.values().at(threshold).front();
    REQUIRE(std::holds_alternative<float>(low));
    REQUIRE(std::get<float>(low) == 0.0F);
}

TEST_CASE("convolution identity preserves its source image") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    REQUIRE(registry.descriptor("convolution") != nullptr);
    Graph graph; graph.settings = {32, 24, 60};
    const auto source = graph.addNode("perlin");
    const auto convolution = graph.addNode("convolution");
    graph.findNode(convolution)->parameters["iterations"] = 4;
    const auto output = graph.addNode("output");
    graph.addLink(source, "image", convolution, "image");
    graph.addLink(convolution, "image", output, "image"); graph.activeOutput = output;
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto sourceImage = std::get<ImageHandle>(runtime.values().at(source).front());
    REQUIRE(readImage(runtime.outputImage()) == readImage(sourceImage));
}

TEST_CASE("erosion and dilation bound the source image") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {24, 20, 60};
    const auto source = graph.addNode("perlin");
    const auto erosion = graph.addNode("convolution");
    const auto dilation = graph.addNode("convolution");
    const std::vector<float> footprint(9, 1.0F);
    graph.findNode(erosion)->parameters = {
        {"kernelSize", 3}, {"kernel", footprint}, {"operation", 1}, {"iterations", 2}};
    graph.findNode(dilation)->parameters = {
        {"kernelSize", 3}, {"kernel", footprint}, {"operation", 2}, {"iterations", 2}};
    graph.addLink(source, "image", erosion, "image");
    graph.addLink(source, "image", dilation, "image");
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));

    const auto sourcePixels = readImage(std::get<ImageHandle>(runtime.values().at(source).front()));
    const auto erosionPixels = readImage(std::get<ImageHandle>(runtime.values().at(erosion).front()));
    const auto dilationPixels = readImage(std::get<ImageHandle>(runtime.values().at(dilation).front()));
    REQUIRE(sourcePixels.size() == erosionPixels.size());
    REQUIRE(sourcePixels.size() == dilationPixels.size());
    for (std::size_t index = 0; index < sourcePixels.size(); ++index) {
        REQUIRE(erosionPixels[index] <= sourcePixels[index] + 0.0001F);
        REQUIRE(dilationPixels[index] + 0.0001F >= sourcePixels[index]);
    }
}

TEST_CASE("reaction diffusion evolves, resets, and reallocates on resize") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {32, 32, 60};
    const auto reaction = graph.addNode("reaction_diffusion");
    const auto output = graph.addNode("output");
    graph.findNode(reaction)->parameters["iterations"] = 8;
    graph.addLink(reaction, "image", output, "image"); graph.activeOutput = output;
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 1.0 / 60.0, false));
    const auto initial = readImage(runtime.outputImage());
    for (int frame = 0; frame < 30; ++frame) REQUIRE(runtime.evaluate(frame / 60.0, 1.0 / 60.0, true));
    const auto evolved = readImage(runtime.outputImage());
    REQUIRE(evolved != initial);
    REQUIRE(*std::ranges::max_element(evolved) > 0.1F);
    runtime.reset();
    REQUIRE(runtime.evaluate(0, 0, false));
    REQUIRE(readImage(runtime.outputImage()) == initial);
    graph.settings = {40, 24, 60};
    REQUIRE(runtime.evaluate(0, 0, false));
    REQUIRE(runtime.outputImage().width == 40);
    REQUIRE(runtime.outputImage().height == 24);
}

TEST_CASE("disconnected reaction multipliers are equivalent to solid white") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {24, 24, 60};
    const auto white = graph.addNode("color_ramp");
    graph.findNode(white)->parameters["value"] = 1.0F;
    const auto unmapped = graph.addNode("reaction_diffusion");
    const auto mapped = graph.addNode("reaction_diffusion");
    graph.findNode(unmapped)->parameters["iterations"] = 3;
    graph.findNode(mapped)->parameters["iterations"] = 3;
    graph.addLink(white, "image", mapped, "feedMultiplier");
    graph.addLink(white, "image", mapped, "killMultiplier");
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    for (int frame = 0; frame < 5; ++frame) {
        REQUIRE(runtime.evaluate(frame / 60.0, 1.0 / 60.0, true));
    }
    const auto& values = runtime.values();
    const auto unmappedImage = std::get<ImageHandle>(values.at(unmapped).front());
    const auto mappedImage = std::get<ImageHandle>(values.at(mapped).front());
    REQUIRE(readImage(unmappedImage) == readImage(mappedImage));
}

TEST_CASE("discrete reaction evolves, pauses, resets, resizes, and exposes A and B") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {32, 32, 60};
    const auto reaction = addDiscreteReaction(graph);
    const auto exposeA = graph.addNode("output");
    const auto exposeB = graph.addNode("output");
    graph.addLink(reaction, "a", exposeA, "image");
    graph.addLink(reaction, "b", exposeB, "image");
    GpuRuntime gpu; GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto initialValues = runtime.values().at(reaction);
    REQUIRE(initialValues.size() == 3);
    const auto initial = readImage(std::get<ImageHandle>(initialValues[0]));
    const auto initialA = readImage(std::get<ImageHandle>(initialValues[1]));
    const auto initialB = readImage(std::get<ImageHandle>(initialValues[2]));
    const auto cachedTexture = std::get<ImageHandle>(initialValues[0]).texture;
    REQUIRE(initial == initialB);
    REQUIRE(initialA != initialB);
    for (int frame = 0; frame < 20; ++frame) REQUIRE(runtime.evaluate(frame / 60.0, 1.0 / 60.0, true));
    const auto evolved = readImage(std::get<ImageHandle>(runtime.values().at(reaction)[0]));
    REQUIRE(evolved != initial);
    REQUIRE(runtime.evaluate(1, 0, false));
    REQUIRE(std::get<ImageHandle>(runtime.values().at(reaction)[0]).texture == cachedTexture);
    REQUIRE(readImage(std::get<ImageHandle>(runtime.values().at(reaction)[0])) == evolved);
    runtime.resetNode(reaction); REQUIRE(runtime.evaluate(0, 0, false));
    REQUIRE(readImage(std::get<ImageHandle>(runtime.values().at(reaction)[0])) == initial);
    graph.settings = {40, 24, 60}; REQUIRE(runtime.evaluate(0, 0, false));
    REQUIRE(std::get<ImageHandle>(runtime.values().at(reaction)[0]).width == 40);
}

TEST_CASE("simulation subgraphs do not materialize unconnected secondary outputs") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {24, 24, 60};
    const auto reaction = addDiscreteReaction(graph);
    GpuRuntime gpu; GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto& outputs = runtime.values().at(reaction);
    REQUIRE(std::holds_alternative<ImageHandle>(outputs[0]));
    REQUIRE(std::holds_alternative<std::monostate>(outputs[1]));
    REQUIRE(std::holds_alternative<std::monostate>(outputs[2]));
}

TEST_CASE("unused subgraph edits preserve live simulation state") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {24, 24, 60};
    auto definition = builtInSubgraphs().front();
    definition.id = "project.unused_edit";
    definition.immutable = false;
    graph.subgraphs().push_back(std::move(definition));
    const auto reaction = graph.addNode("subgraph");
    graph.findNode(reaction)->subgraphId = "project.unused_edit";
    graph.findNode(reaction)->parameters = {{"feed", .055F}, {"kill", .062F},
        {"diffA", 1.0F}, {"diffB", .5F}, {"structureScale", 1.0F},
        {"dt", 1.0F}, {"iterations", 8.0F}, {"autoReset", 0.0F}};
    GpuRuntime gpu; GraphRuntime runtime(graph, registry, gpu);
    for (int frame = 0; frame < 8; ++frame)
        REQUIRE(runtime.evaluate(frame / 60.0, 1.0 / 60.0, true));
    const auto evolved = readImage(std::get<ImageHandle>(runtime.values().at(reaction)[0]));

    const auto unused = graph.findSubgraph("project.unused_edit")->body.addNode("float");
    graph.findSubgraph("project.unused_edit")->body.findNode(unused)->parameters["value"] = .25F;
    runtime.rebuild();
    REQUIRE(runtime.evaluate(1.0, 0.0, false));
    REQUIRE(readImage(std::get<ImageHandle>(runtime.values().at(reaction)[0])) == evolved);
}

TEST_CASE("discrete reaction accepts scalar and image multiplier inputs") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {24, 24, 60};
    const auto scalar = graph.addNode("float");
    graph.findNode(scalar)->parameters["value"] = 1.0F;
    const auto white = graph.addNode("color_ramp");
    graph.findNode(white)->parameters["value"] = 1.0F;
    const auto baseline = addDiscreteReaction(graph);
    const auto scalarMapped = addDiscreteReaction(graph);
    const auto imageMapped = addDiscreteReaction(graph);
    graph.findNode(baseline)->parameters["iterations"] = 3;
    graph.findNode(scalarMapped)->parameters["iterations"] = 3;
    graph.findNode(imageMapped)->parameters["iterations"] = 3;
    graph.addLink(scalar, "value", scalarMapped, "feedMultiplier");
    graph.addLink(scalar, "value", scalarMapped, "killMultiplier");
    graph.addLink(white, "image", imageMapped, "feedMultiplier");
    graph.addLink(white, "image", imageMapped, "killMultiplier");
    GpuRuntime gpu; GraphRuntime runtime(graph, registry, gpu);
    for (int frame = 0; frame < 5; ++frame) REQUIRE(runtime.evaluate(frame / 60.0, 1.0 / 60.0, true));
    const auto expected = readImage(std::get<ImageHandle>(runtime.values().at(baseline)[0]));
    REQUIRE(readImage(std::get<ImageHandle>(runtime.values().at(scalarMapped)[0])) == expected);
    REQUIRE(readImage(std::get<ImageHandle>(runtime.values().at(imageMapped)[0])) == expected);
}

TEST_CASE("structural subgraph edits change the fused simulation") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {24, 24, 60};

    auto definition = builtInSubgraphs().front();
    definition.id = "project.edited_discrete";
    definition.immutable = false;
    const auto nextState = std::ranges::find(
        definition.body.nodes(), std::string("simulation_next_state"), &NodeRecord::type);
    REQUIRE(nextState != definition.body.nodes().end());
    const NodeId nextStateId = nextState->id;
    const auto zero = definition.body.addNode("float", {1700, 820});
    definition.body.findNode(zero)->parameters["value"] = 0.0F;
    const auto select = definition.body.addNode("select", {1880, 820});
    definition.body.findNode(select)->parameters = {
        {"condition", -0.25F}, {"ifFalse", 1.0F}};
    definition.body.addLink(zero, "value", select, "ifTrue");
    definition.body.addLink(select, "result", nextStateId, "b");
    graph.subgraphs().push_back(std::move(definition));

    const auto reaction = graph.addNode("subgraph");
    graph.findNode(reaction)->subgraphId = "project.edited_discrete";
    graph.findNode(reaction)->parameters = {{"feed", .055F}, {"kill", .062F},
        {"diffA", 1.0F}, {"diffB", .5F}, {"structureScale", 1.0F},
        {"dt", 1.0F}, {"iterations", 1.0F}, {"autoReset", 0.0F}};
    const auto exposeB = graph.addNode("output");
    graph.addLink(reaction, "b", exposeB, "image");

    GpuRuntime gpu; GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 1.0 / 60.0, true));
    const auto b = readImage(std::get<ImageHandle>(runtime.values().at(reaction)[2]));
    for (std::size_t pixel = 0; pixel < b.size(); pixel += 4)
        REQUIRE(b[pixel] == Catch::Approx(0.0F).margin(.0001F));
}

TEST_CASE("discrete reaction default B agrees with the monolithic implementation") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {32, 32, 60};
    const auto monolithic = graph.addNode("reaction_diffusion");
    const auto discrete = addDiscreteReaction(graph);
    graph.findNode(monolithic)->parameters["iterations"] = 8;
    GpuRuntime gpu; GraphRuntime runtime(graph, registry, gpu);
    for (int frame = 0; frame < 12; ++frame) REQUIRE(runtime.evaluate(frame / 60.0, 1.0 / 60.0, true));
    const auto expected = readImage(std::get<ImageHandle>(runtime.values().at(monolithic)[0]));
    const auto actual = readImage(std::get<ImageHandle>(runtime.values().at(discrete)[0]));
    REQUIRE(expected.size() == actual.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
        REQUIRE(actual[index] == Catch::Approx(expected[index]).margin(.003F));
}

TEST_CASE("discrete reaction stays within twice monolithic GPU time") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph monolithicGraph; monolithicGraph.settings = {512, 512, 60};
    Graph discreteGraph; discreteGraph.settings = {512, 512, 60};
    const auto monolithic = monolithicGraph.addNode("reaction_diffusion");
    const auto discrete = addDiscreteReaction(discreteGraph);
    monolithicGraph.findNode(monolithic)->parameters["iterations"] = 8;
    GpuRuntime gpu;
    GraphRuntime monolithicRuntime(monolithicGraph, registry, gpu);
    GraphRuntime discreteRuntime(discreteGraph, registry, gpu);
    for (int frame = 0; frame < 20; ++frame) {
        REQUIRE(monolithicRuntime.evaluate(frame / 60.0, 1.0 / 60.0, true));
        REQUIRE(discreteRuntime.evaluate(frame / 60.0, 1.0 / 60.0, true));
    }
    std::vector<double> monolithicTimes, discreteTimes;
    for (int frame = 0; frame < 50; ++frame) {
        if (frame % 2 == 0) {
            REQUIRE(monolithicRuntime.evaluate((frame + 20) / 60.0, 1.0 / 60.0, true));
            REQUIRE(discreteRuntime.evaluate((frame + 20) / 60.0, 1.0 / 60.0, true));
        } else {
            REQUIRE(discreteRuntime.evaluate((frame + 20) / 60.0, 1.0 / 60.0, true));
            REQUIRE(monolithicRuntime.evaluate((frame + 20) / 60.0, 1.0 / 60.0, true));
        }
        monolithicTimes.push_back(monolithicRuntime.gpuMilliseconds().at(monolithic));
        discreteTimes.push_back(discreteRuntime.gpuMilliseconds().at(discrete));
    }
    const double baseline = median(monolithicTimes);
    INFO("monolithic median=" << baseline << " ms, discrete median=" << median(discreteTimes) << " ms");
    REQUIRE(baseline > 0.0);
    REQUIRE(median(discreteTimes) <= baseline * 2.0);
}

TEST_CASE("color ramp factor fed by a field generator agrees when fused and solo") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {8, 1, 60};
    const auto gradient = graph.addNode("gradient");
    const auto ramp = graph.addNode("color_ramp");
    graph.addLink(gradient, "value", ramp, "value");
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto fused = readImage(std::get<ImageHandle>(runtime.values().at(ramp).front()));
    REQUIRE(runtime.generatedShaders().size() == 1);
    REQUIRE(runtime.generatedShaders().front().mode == GeneratedExecutionMode::Generated);
    const auto generatedInfo = runtime.generatedShaders().front();
    const auto& inputs = generatedInfo.shader.inputs;
    REQUIRE(inputs.size() == 9);
    REQUIRE(inputs.front().kind == ShaderInputKind::Field);
    REQUIRE(runtime.fusionInfo(ramp)->mode == GeneratedExecutionMode::Generated);
    // A fixed image fed through fusion would flatten the ramp to its end color;
    // the field must instead drive the interpolation.
    REQUIRE_FALSE(fused[0] == fused[7 * 4]);
    REQUIRE(fused[0] == Catch::Approx(0.015F).margin(.002F));
    REQUIRE(fused[7 * 4] == Catch::Approx(1.0F).margin(.002F));
    runtime.setFusionEnabled(false);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto solo = readImage(std::get<ImageHandle>(runtime.values().at(ramp).front()));
    REQUIRE(fused.size() == solo.size());
    for (std::size_t index = 0; index < fused.size(); ++index)
        REQUIRE(fused[index] == Catch::Approx(solo[index]).margin(.002F));
}

TEST_CASE("text_test fusion matches solo lowering", "[.][text-test]") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto source = std::filesystem::current_path() / "text_test";
    if (!std::filesystem::exists(source)) {
        WARN("text_test project not found in " << std::filesystem::current_path());
        return;
    }

    GpuRuntime gpuA; GpuRuntime gpuB;
    auto graphA = loadProject(source, registry);
    auto graphB = loadProject(source, registry);
    if (graphA.nodes().empty()) { FAIL("text_test failed to load"); return; }
    GraphRuntime runtimeA(graphA, registry, gpuA);
    runtimeA.setFusionEnabled(true);
    GraphRuntime runtimeB(graphB, registry, gpuB);
    runtimeB.setFusionEnabled(false);

    double time = 0.0;
    for (int frame = 0; frame < 5; ++frame) {
        REQUIRE(runtimeA.evaluate(time, 1.0 / 31.0, false));
        REQUIRE(runtimeB.evaluate(time, 1.0 / 31.0, false));
        time += 1.0 / 31.0;
    }

    const auto outA = runtimeA.outputImage();
    const auto outB = runtimeB.outputImage();
    REQUIRE(outA.texture != 0);
    REQUIRE(outB.texture != 0);
    INFO("sizeA=" << outA.width << "x" << outA.height
         << " sizeB=" << outB.width << "x" << outB.height);
    const auto pixelsA = readImage(outA);
    const auto pixelsB = readImage(outB);
    REQUIRE(pixelsA.size() == pixelsB.size());

    std::size_t differing = 0;
    float maxError = 0.0F;
    for (std::size_t index = 0; index < pixelsA.size(); ++index) {
        const float error = std::fabs(pixelsA[index] - pixelsB[index]);
        maxError = std::max(maxError, error);
        if (error > 1e-4F) ++differing;
    }
    // Fused execution must reproduce solo lowered regions exactly, including
    // VectorNumeric field producers such as transform coordinates and Worley positions.
    REQUIRE(differing == 0);
}

TEST_CASE("compare lowers all modes to scalar truth comparisons") {
    NodeRegistry registry;
    registerBuiltInNodes(registry);
    const std::array<const char*, 12> shaders = {
        "float(sa<sb)", "float(sa<=sb)", "float(sa>sb)", "float(sa>=sb)",
        "float(abs(sa-sb)<=p_epsilon)", "float(abs(sa-sb)>p_epsilon)",
        "float(sb<=sa&&sa<=sc)", "float(sb<sa&&sa<sc)",
        "float((abs(sa)>p_epsilon)&&(abs(sb)>p_epsilon))",
        "float((abs(sa)>p_epsilon)||(abs(sb)>p_epsilon))",
        "float(((abs(sa)>p_epsilon))!=((abs(sb)>p_epsilon)))",
        "float(!((abs(sa)>p_epsilon)))"};
    for (int mode = 0; mode < 12; ++mode) {
        auto compare = registry.create("compare");
        compare->setParameters({{"mode", static_cast<float>(mode)},
                                {"epsilon", 1e-4F}});
        CapturingLoweringContext lowering(ShaderValueType::Scalar);
        lowering.inputs = {{"a", {ShaderValueType::Scalar, "sa"}},
                           {"b", {ShaderValueType::Scalar, "sb"}},
                           {"c", {ShaderValueType::Scalar, "sc"}}};
        REQUIRE(compare->lowerShader(lowering));
        REQUIRE(lowering.emitted.name == shaders[static_cast<std::size_t>(mode)]);
    }
}

} // namespace reaction
