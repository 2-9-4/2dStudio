#pragma once

#include "reaction/gpu/gpu_runtime.hpp"

#include <GLFW/glfw3.h>
#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace reaction::test_support {

class HiddenContext {
public:
    HiddenContext() {
        glfwSetErrorCallback([](int, const char*) {});
        if (glfwInit() == GLFW_FALSE)
            SKIP("No desktop display is available for OpenGL integration tests");
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(32, 32, "GPU test", nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            SKIP("OpenGL 4.3 context is unavailable");
        }
        glfwMakeContextCurrent(window);
        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
            glfwDestroyWindow(window);
            window = nullptr;
            glfwTerminate();
            SKIP("OpenGL loader initialization failed");
        }
    }

    ~HiddenContext() {
        if (window) glfwDestroyWindow(window);
        glfwTerminate();
    }

    HiddenContext(const HiddenContext&) = delete;
    HiddenContext& operator=(const HiddenContext&) = delete;

    GLFWwindow* window = nullptr;
};

inline std::vector<float> readImage(ImageHandle image) {
    std::vector<float> values(
        static_cast<std::size_t>(image.width * image.height * 4));
    glBindTexture(GL_TEXTURE_2D, image.texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, values.data());
    return values;
}

// Small integration harness for node conformance tests. It owns the GL context,
// registry, graph, GPU runtime, and graph runtime in the required lifetime order
// so tests can focus on graph semantics rather than repeated setup.
class RuntimeHarness {
public:
    explicit RuntimeHarness(int width = 4, int height = 4, int targetFps = 60)
        : context_(), gpu_() {
        registerBuiltInNodes(registry);
        graph.settings = {width, height, targetFps};
    }

    NodeId add(std::string type, nlohmann::json parameters = nlohmann::json::object()) {
        const auto id = graph.addNode(std::move(type));
        graph.findNode(id)->parameters = std::move(parameters);
        return id;
    }

    LinkId connect(NodeId from, std::string fromSocket,
                   NodeId to, std::string toSocket) {
        return graph.addLink(from, std::move(fromSocket), to, std::move(toSocket));
    }

    GraphRuntime& build(bool fusionEnabled = true) {
        runtime = std::make_unique<GraphRuntime>(graph, registry, gpu_);
        runtime->setFusionEnabled(fusionEnabled);
        return *runtime;
    }

    GraphRuntime& evaluated(double time = 0.0, double deltaTime = 0.0,
                            bool playing = false) {
        auto& current = runtime ? *runtime : build();
        if (!current.evaluate(time, deltaTime, playing))
            throw std::runtime_error("GraphRuntime evaluation failed");
        return current;
    }

    [[nodiscard]] const Value& value(NodeId node, std::size_t output = 0) const {
        if (!runtime) throw std::logic_error("RuntimeHarness has not been built");
        const auto found = runtime->values().find(node);
        if (found == runtime->values().end() || output >= found->second.size())
            throw std::out_of_range("Node output was not materialized");
        return found->second[output];
    }

    [[nodiscard]] ImageHandle image(NodeId node, std::size_t output = 0) const {
        return std::get<ImageHandle>(value(node, output));
    }

    [[nodiscard]] std::vector<float> pixels(
        NodeId node, std::size_t output = 0) const {
        return readImage(image(node, output));
    }

    [[nodiscard]] ValueType semanticType(
        NodeId node, std::string_view socket,
        SocketDirection direction = SocketDirection::Output) const {
        const auto compiled = graph.compile(registry);
        const auto type = compiled.socketType(node, socket, direction);
        if (!type) throw std::out_of_range("Socket did not resolve to a concrete type");
        return *type;
    }

    [[nodiscard]] const NodeDescriptor& descriptor(std::string_view type) const {
        const auto* result = registry.descriptor(type);
        if (!result) throw std::out_of_range("Node descriptor is not registered");
        return *result;
    }

    [[nodiscard]] std::pair<std::vector<float>, std::vector<float>>
    fusedAndSoloPixels(NodeId node, std::size_t output = 0,
                       double time = 0.0, double deltaTime = 0.0,
                       bool playing = false) {
        auto& current = runtime ? *runtime : build(true);
        current.setFusionEnabled(true);
        if (!current.evaluate(time, deltaTime, playing))
            throw std::runtime_error("Fused GraphRuntime evaluation failed");
        auto fused = pixels(node, output);

        current.setFusionEnabled(false);
        if (!current.evaluate(time, deltaTime, playing))
            throw std::runtime_error("Solo GraphRuntime evaluation failed");
        auto solo = pixels(node, output);
        return {std::move(fused), std::move(solo)};
    }

    NodeRegistry registry;
    Graph graph;
    std::unique_ptr<GraphRuntime> runtime;

private:
    // Declaration order matters: OpenGL must exist before GpuRuntime and must
    // outlive it.
    HiddenContext context_;
    GpuRuntime gpu_;
};

} // namespace reaction::test_support
