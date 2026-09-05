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
    std::string helper(std::string_view name, std::string) override {
        trace.push_back("helper:" + std::string(name));
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
    REQUIRE(update.find("param_structureScale") != std::string::npos);
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
    for (std::size_t at = update.find("sampleState(q+pixel*"); at != std::string::npos;
         at = update.find("sampleState(q+pixel*", at + 1)) ++neighborhoodSamples;
    REQUIRE(neighborhoodSamples == 8);
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

TEST_CASE("shared node lowering follows the same contract for Scalar and Vec4 planners") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto scalar = lowerMathThresholdSelect(registry, ShaderValueType::Scalar);
    const auto vector = lowerMathThresholdSelect(registry, ShaderValueType::Vec4);
    REQUIRE(scalar.trace == vector.trace);
    REQUIRE(scalar.expressions[0].find('+') != std::string::npos);
    REQUIRE(vector.expressions[0].find('+') != std::string::npos);
    REQUIRE(scalar.expressions[1].find("step(") != std::string::npos);
    REQUIRE(vector.expressions[1].find("step(") != std::string::npos);
    REQUIRE(scalar.expressions[2].find('?') != std::string::npos);
    REQUIRE(vector.expressions[2].find(".r") != std::string::npos);
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
    const auto threshold = graph.addNode("threshold");
    const auto select = graph.addNode("select");
    graph.addLink(source, "image", threshold, "value");
    graph.addLink(threshold, "result", select, "condition");
    graph.addLink(source, "image", select, "ifTrue");
    const auto regions = planShaderRegions(graph, registry, graph.compile(registry), 16, 1024);
    REQUIRE(regions.size() == 1);
    REQUIRE(regions.front().nodes == std::vector<NodeId>{threshold, select});
    const auto generated = generateComputeShader(regions.front(), {select});
    REQUIRE(generated.source.find("step(") != std::string::npos);
    REQUIRE(generated.source.find(".r") != std::string::npos);

    Graph coordinatesGraph;
    const auto coordinates = coordinatesGraph.addNode("coordinates");
    const auto coordinateRegions = planShaderRegions(
        coordinatesGraph, registry, coordinatesGraph.compile(registry), 16, 1024);
    REQUIRE(coordinateRegions.size() == 1);
    const auto coordinateShader = generateComputeShader(coordinateRegions.front(), {coordinates});
    REQUIRE(coordinateShader.outputs.size() == 2);
    REQUIRE(coordinateShader.outputs[0].socket == "x");
    REQUIRE(coordinateShader.outputs[1].socket == "y");
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
        return input.kind == ShaderInputKind::Image;
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
    REQUIRE(normalized.source != originalSource);
    REQUIRE(normalized.specializationKey != originalKey);
    REQUIRE(normalized.source.find("weightSum") != std::string::npos);
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

TEST_CASE("shader planner keeps branch and join boundaries materialized") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph;
    const auto source = graph.addNode("image");
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
        return input.kind == ShaderInputKind::Image;
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
    REQUIRE(descriptor->sockets[0].type == ValueType::Image2D);
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
    REQUIRE(sample->sockets[1].type == ValueType::AnyVector);

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
    REQUIRE(registry.contains("combine_vector"));
    REQUIRE(registry.contains("separate_vector"));
}

TEST_CASE("reaction multiplier inputs accept both scalar and image values") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("reaction_diffusion");
    REQUIRE(descriptor != nullptr);
    const auto typeFor = [&](std::string_view key) {
        return std::ranges::find(descriptor->sockets, key, &SocketDescriptor::key)->type;
    };
    REQUIRE(typeFor("feedMultiplier") == ValueType::AnyNumeric);
    REQUIRE(typeFor("killMultiplier") == ValueType::AnyNumeric);

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
    REQUIRE(descriptor->sockets[0].type == ValueType::Float);
    REQUIRE(descriptor->sockets[1].type == ValueType::Float);

    Graph graph;
    const auto source = graph.addNode("float");
    const auto preview = graph.addNode("float_preview");
    graph.addLink(source, "value", preview, "value");
    REQUIRE(graph.compile(registry).valid);
}

TEST_CASE("image invert exposes image-only input and output") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    const auto* descriptor = registry.descriptor("invert");
    REQUIRE(descriptor != nullptr);
    REQUIRE(descriptor->displayName == "Image Invert");
    REQUIRE(descriptor->sockets.size() == 2);
    REQUIRE(descriptor->sockets[0].type == ValueType::Image2D);
    REQUIRE(descriptor->sockets[1].type == ValueType::Image2D);

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
    REQUIRE(coordinates->sockets.size() == 2);
    REQUIRE(coordinates->sockets[0].key == "x");
    REQUIRE(coordinates->sockets[0].type == ValueType::Image2D);
    REQUIRE(coordinates->sockets[1].key == "y");
    REQUIRE(coordinates->sockets[1].type == ValueType::Image2D);

    const auto* laplacian = registry.descriptor("laplacian");
    REQUIRE(laplacian != nullptr);
    REQUIRE(laplacian->displayName == "Laplacian");
    REQUIRE(laplacian->category == "Filter");
    REQUIRE(laplacian->sockets.size() == 3);
    REQUIRE(laplacian->sockets[0].key == "value");
    REQUIRE(laplacian->sockets[0].type == ValueType::AnyVector);
    REQUIRE(laplacian->sockets[1].key == "scale");
    REQUIRE(laplacian->sockets[1].type == ValueType::Float);
    REQUIRE(laplacian->sockets[1].optional);
    REQUIRE(laplacian->sockets[2].key == "result");
    REQUIRE(laplacian->sockets[2].type == ValueType::AnyVector);
    REQUIRE(laplacian->parameters.size() == 1);
    REQUIRE(laplacian->parameters[0].defaultValue == 1.0F);
    REQUIRE(laplacian->parameters[0].minimum == .25F);
    REQUIRE(laplacian->parameters[0].maximum == 8.0F);

    Graph graph;
    const auto uv = graph.addNode("coordinates");
    const auto filter = graph.addNode("laplacian");
    graph.addLink(uv, "x", filter, "value");
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
    EvaluationContext context{};
    std::array<Value, 3> inputs{Value{-0.25F}, Value{2.0F}, Value{3.0F}};
    std::array<Value, 1> outputs;
    node->evaluate(context, inputs, outputs);
    REQUIRE(std::get<float>(outputs[0]) == 2.0F);
    inputs[0] = 0.0F;
    node->evaluate(context, inputs, outputs);
    REQUIRE(std::get<float>(outputs[0]) == 3.0F);
}

TEST_CASE("canvas coordinates and Laplacian execute through the normal graph runtime") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {4, 4, 60};
    const auto coordinates = graph.addNode("coordinates");
    const auto laplacian = graph.addNode("laplacian");
    graph.addLink(coordinates, "x", laplacian, "value");
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));

    const auto& coordinateValues = runtime.values().at(coordinates);
    REQUIRE(coordinateValues.size() == 2);
    const auto x = readImage(std::get<ImageHandle>(coordinateValues[0]));
    const auto y = readImage(std::get<ImageHandle>(coordinateValues[1]));
    REQUIRE(x[0] == Catch::Approx(.125F).margin(.001F));
    REQUIRE(x[3 * 4] == Catch::Approx(.875F).margin(.001F));
    REQUIRE(x[4 * 4] == Catch::Approx(.125F).margin(.001F));
    REQUIRE(y[0] == Catch::Approx(.125F).margin(.001F));
    REQUIRE(y[4 * 4] == Catch::Approx(.375F).margin(.001F));

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

TEST_CASE("scalar Math protects division and remains on CPU") {
    NodeRegistry registry; registerBuiltInNodes(registry);
    auto math = registry.create("math");
    EvaluationContext context;
    std::array<Value, 1> outputs;
    math->setParameters({{"operation", 3}, {"a", 1.0}, {"b", 0.0}});
    math->evaluate(context, {}, outputs);
    REQUIRE(std::holds_alternative<float>(outputs.front()));
    REQUIRE(std::isfinite(std::get<float>(outputs.front())));
    REQUIRE(std::get<float>(outputs.front()) == 1'000'000.0F);
    math->setParameters({{"operation", 3}, {"a", 1.0}, {"b", -1.0e-12F}});
    math->evaluate(context, {}, outputs);
    REQUIRE(std::get<float>(outputs.front()) == -1'000'000.0F);
}

TEST_CASE("generated Math chain matches legacy execution and materializes previews on demand") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {16, 16, 60};
    const auto coordinates = graph.addNode("coordinates");
    const auto add = graph.addNode("math");
    const auto multiply = graph.addNode("math");
    graph.findNode(add)->parameters = {{"operation", 0.0F}, {"b", .125F}};
    graph.findNode(multiply)->parameters = {{"operation", 2.0F}, {"b", .75F}};
    graph.addLink(coordinates, "x", add, "a");
    graph.addLink(add, "result", multiply, "a");
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));
    REQUIRE(runtime.fusionInfo(add).has_value());
    REQUIRE(runtime.fusionInfo(add)->interior);
    REQUIRE(runtime.fusionInfo(multiply)->nodeCount == 3);
    REQUIRE_FALSE(runtime.values().contains(add));
    const auto generated = readImage(std::get<ImageHandle>(runtime.values().at(multiply).front()));

    runtime.setFusionEnabled(false);
    REQUIRE(runtime.evaluate(0, 0, false));
    REQUIRE(runtime.values().contains(add));
    const auto legacy = readImage(std::get<ImageHandle>(runtime.values().at(multiply).front()));
    REQUIRE(generated == legacy);

    runtime.setFusionEnabled(true);
    runtime.setIntermediatePreview(add);
    REQUIRE(runtime.evaluate(0, 0, false));
    REQUIRE(runtime.values().contains(add));
    REQUIRE(runtime.fusionInfo(add)->materialized);
    runtime.setIntermediatePreview(std::nullopt);
    REQUIRE(runtime.evaluate(0, 0, false));
    REQUIRE_FALSE(runtime.values().contains(add));
}

TEST_CASE("every generated Math operation compiles and agrees with legacy pixels") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    for (int operation = 0; operation < static_cast<int>(kMathOperationNames.size()); ++operation) {
        Graph graph; graph.settings = {8, 8, 60};
        const auto coordinates = graph.addNode("coordinates");
        const auto math = graph.addNode("math");
        graph.findNode(math)->parameters = {{"operation", static_cast<float>(operation)},
            {"b", .4F}, {"c", .8F}, {"inMin", .1F}, {"inMax", .9F},
            {"outMin", -.25F}, {"outMax", 1.25F}};
        graph.addLink(coordinates, "x", math, "a");
        GpuRuntime gpu;
        GraphRuntime runtime(graph, registry, gpu);
        REQUIRE(runtime.evaluate(0, 0, false));
        INFO("operation=" << operation);
        REQUIRE(runtime.generatedShaders().size() == 1);
        REQUIRE(runtime.generatedShaders().front().mode == GeneratedExecutionMode::Generated);
        const auto generated = readImage(std::get<ImageHandle>(runtime.values().at(math).front()));
        runtime.setFusionEnabled(false);
        REQUIRE(runtime.evaluate(0, 0, false));
        const auto legacy = readImage(std::get<ImageHandle>(runtime.values().at(math).front()));
        REQUIRE(generated.size() == legacy.size());
        for (std::size_t index = 0; index < generated.size(); ++index)
            REQUIRE(generated[index] == Catch::Approx(legacy[index]).margin(.002F));
    }
}

TEST_CASE("generated safe power preserves a negative scalar base") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {8, 8, 60};
    const auto coordinates = graph.addNode("coordinates");
    const auto math = graph.addNode("math");
    graph.findNode(math)->parameters = {{"operation", 4.0F}, {"a", -.5F}};
    graph.addLink(coordinates, "x", math, "b");
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
    REQUIRE(std::get<float>(value) == Catch::Approx(.2F));
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

TEST_CASE("color ramp factor fed by a field generator fuses and agrees with legacy") {
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
    REQUIRE(inputs.front().kind == ShaderInputKind::Flexible);
    REQUIRE(runtime.fusionInfo(ramp)->mode == GeneratedExecutionMode::Generated);
    // A fixed image fed through fusion would flatten the ramp to its end color;
    // the field must instead drive the interpolation.
    REQUIRE_FALSE(fused[0] == fused[7 * 4]);
    REQUIRE(fused[0] == Catch::Approx(0.015F).margin(.002F));
    REQUIRE(fused[7 * 4] == Catch::Approx(1.0F).margin(.002F));
    runtime.setFusionEnabled(false);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto legacy = readImage(std::get<ImageHandle>(runtime.values().at(ramp).front()));
    REQUIRE(fused.size() == legacy.size());
    for (std::size_t index = 0; index < fused.size(); ++index)
        REQUIRE(fused[index] == Catch::Approx(legacy[index]).margin(.002F));
}

TEST_CASE("text_test fusion matches legacy", "[.][text-test]") {
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
    // Fused execution must reproduce the native (fusion-off) result exactly room
    // for room. The mix node here is fed by AnyVector field producers
    // (transform_2d coordinates, worley featurePosition); without FlexibleVector
    // lowering these fused inputs regress to constant fallbacks.
    REQUIRE(differing == 0);
}

} // namespace reaction
