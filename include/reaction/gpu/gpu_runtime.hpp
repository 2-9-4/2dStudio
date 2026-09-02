#pragma once

#include "reaction/core/graph.hpp"

#include <glad/glad.h>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace reaction {

class GpuRuntime {
public:
    GpuRuntime();
    ~GpuRuntime();
    GpuRuntime(const GpuRuntime&) = delete;
    GpuRuntime& operator=(const GpuRuntime&) = delete;

    [[nodiscard]] GLuint compileCompute(std::string_view source) const;
    [[nodiscard]] GLuint compileComputeCached(std::string_view source);
    [[nodiscard]] GLuint createTexture(int width, int height, GLenum format = GL_RGBA16F) const;
    void ensureTexture(GLuint& texture, int& currentWidth, int& currentHeight,
                       int width, int height, GLenum format = GL_RGBA16F) const;
    void dispatch(GLuint program, int width, int height) const;
    void drawFullscreen(GLuint texture, GLuint vertexArray,
                        int framebufferWidth, int framebufferHeight) const;

private:
    GLuint previewProgram_ = 0;
    std::unordered_map<std::string, GLuint> computeCache_;
};

void registerBuiltInNodes(NodeRegistry& registry);
[[nodiscard]] std::unique_ptr<NodeInstance> createSubgraphInstance(
    const SubgraphDefinition& definition);

class GraphRuntime {
public:
    GraphRuntime(Graph& graph, const NodeRegistry& registry, GpuRuntime& gpu);
    ~GraphRuntime();

    void rebuild();
    bool evaluate(double time, double deltaTime, bool playing);
    void reset();
    void resetNode(NodeId id);
    void clear();
    [[nodiscard]] ImageHandle outputImage() const;
    [[nodiscard]] const CompileResult& compileResult() const { return compiled_; }
    [[nodiscard]] const std::unordered_map<NodeId, double>& gpuMilliseconds() const { return timings_; }
    [[nodiscard]] const std::unordered_map<NodeId, std::vector<Value>>& values() const { return values_; }

private:
    Graph& graph_;
    const NodeRegistry& registry_;
    GpuRuntime& gpu_;
    CompileResult compiled_;
    std::unordered_map<NodeId, std::unique_ptr<NodeInstance>> instances_;
    std::unordered_map<NodeId, std::vector<Value>> values_;
    std::unordered_map<NodeId, double> timings_;
    std::unordered_map<NodeId, GLuint> timerQueries_;
    std::unordered_map<NodeId, nlohmann::json> previousParameters_;
    std::unordered_set<NodeId> pendingResets_;
    bool needsReset_ = true;
    bool forceDirty_ = true;
    int lastWidth_ = 0;
    int lastHeight_ = 0;
};

} // namespace reaction
