#include "reaction/gpu/gpu_runtime.hpp"

#include <GLFW/glfw3.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
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

} // namespace

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

TEST_CASE("Perlin compute is stable, bounded, and reuses its texture") {
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
    REQUIRE(runtime.evaluate(1.25, 1.0 / 60.0, true));
    REQUIRE(runtime.outputImage().texture == firstHandle.texture);
    REQUIRE(readImage(runtime.outputImage()) == first);
}

TEST_CASE("scalar Math protects division and remains on CPU") {
    HiddenContext context;
    NodeRegistry registry; registerBuiltInNodes(registry);
    Graph graph; graph.settings = {16, 16, 60};
    const auto math = graph.addNode("math");
    auto* record = graph.findNode(math);
    record->parameters = {{"operation", 3}, {"a", 1.0}, {"b", 0.0}};
    GpuRuntime gpu;
    GraphRuntime runtime(graph, registry, gpu);
    REQUIRE(runtime.evaluate(0, 0, false));
    const auto& value = runtime.values().at(math).front();
    REQUIRE(std::holds_alternative<float>(value));
    REQUIRE(std::isfinite(std::get<float>(value)));
    REQUIRE(std::get<float>(value) == 1'000'000.0F);
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

} // namespace reaction
