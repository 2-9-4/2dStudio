#pragma once

#include "reaction/core/graph.hpp"
#include "reaction/gpu/shader_ir.hpp"

#include <glad/glad.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace reaction {

struct ComputeCompileResult {
    GLuint program = 0;
    std::string log;
    std::size_t reportedLine = 0;
    explicit operator bool() const noexcept { return program != 0; }
};

class GpuRuntime {
public:
    GpuRuntime();
    ~GpuRuntime();
    GpuRuntime(const GpuRuntime&) = delete;
    GpuRuntime& operator=(const GpuRuntime&) = delete;

    [[nodiscard]] GLuint compileCompute(std::string_view source,
                                        std::string_view label = "compute shader") const;
    [[nodiscard]] GLuint compileComputeCached(std::string_view source,
                                              std::string_view label = "cached compute shader");
    [[nodiscard]] ComputeCompileResult tryCompileCompute(
        std::string_view source, std::string_view label = "generated compute shader") const;
    [[nodiscard]] int maximumComputeTextureInputs() const;
    [[nodiscard]] int maximumComputeUniformComponents() const;
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

enum class GeneratedExecutionMode { Generated, LegacyFallback, ForcedLegacy };

struct GeneratedShaderInfo {
    std::uint64_t id = 0;
    NodeId tail = 0;
    std::vector<NodeId> nodes;
    GeneratedShader shader;
    GeneratedExecutionMode mode = GeneratedExecutionMode::Generated;
    std::string diagnostic;
    std::size_t errorLine = 0;
    double gpuMilliseconds = 0.0;
    // True only when this region was dispatched (or fell back) in the latest evaluation.
    // Otherwise its prior outputs are being reused.
    bool evaluated = false;
};

struct NodeFusionInfo {
    std::uint64_t region = 0;
    NodeId tail = 0;
    std::size_t nodeCount = 0;
    bool interior = false;
    bool materialized = false;
    bool compileFailed = false;
    GeneratedExecutionMode mode = GeneratedExecutionMode::Generated;
};

void registerBuiltInNodes(NodeRegistry& registry);
[[nodiscard]] std::unique_ptr<NodeInstance> createSubgraphInstance(
    const SubgraphDefinition& definition, const NodeRegistry& registry);
// Exposed for source-level tests and shader inspection. Simulation graphs have
// feedback and neighborhood-sampling semantics, so they use a dedicated planner
// even though individual arithmetic expressions share the normal shader lowering.
[[nodiscard]] std::string generateSimulationShader(
    const SubgraphDefinition& definition, bool initialization);

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
    [[nodiscard]] bool wasEvaluated(NodeId id) const { return evaluated_.contains(id); }
    [[nodiscard]] const std::unordered_map<NodeId, std::vector<Value>>& values() const { return values_; }
    [[nodiscard]] std::vector<GeneratedShaderInfo> generatedShaders() const;
    [[nodiscard]] std::optional<NodeFusionInfo> fusionInfo(NodeId id) const;
    void setFusionEnabled(bool enabled);
    [[nodiscard]] bool fusionEnabled() const;
    void setIntermediatePreview(std::optional<NodeId> id);
    [[nodiscard]] std::optional<NodeId> intermediatePreview() const;

private:
    struct FusionState;
    void rebuildFusion();
    [[nodiscard]] std::string fusionSignature() const;
    Graph& graph_;
    const NodeRegistry& registry_;
    GpuRuntime& gpu_;
    CompileResult compiled_;
    std::unordered_map<NodeId, std::unique_ptr<NodeInstance>> instances_;
    std::unordered_map<NodeId, std::vector<Value>> values_;
    std::unordered_map<NodeId, double> timings_;
    std::unordered_set<NodeId> evaluated_;
    std::unordered_map<NodeId, GLuint> timerQueries_;
    std::unordered_map<NodeId, nlohmann::json> previousParameters_;
    std::unordered_map<NodeId, std::string> subgraphSignatures_;
    std::unordered_set<NodeId> pendingResets_;
    bool needsReset_ = true;
    bool forceDirty_ = true;
    std::uint64_t frame_ = 0;
    int lastWidth_ = 0;
    int lastHeight_ = 0;
    std::unique_ptr<FusionState> fusion_;
};

} // namespace reaction
