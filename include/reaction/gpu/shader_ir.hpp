#pragma once

#include "reaction/core/graph.hpp"
#include "reaction/core/math.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace reaction {

enum class ShaderValueType { Scalar, Vec2, Vec4 };
enum class ShaderInputKind { Image, Scalar, Flexible, FlexibleVector };

struct ShaderValue {
    ShaderValueType type = ShaderValueType::Vec4;
    std::string name;
};

struct ShaderInputRequirement {
    ShaderInputKind kind = ShaderInputKind::Scalar;
    std::string key;
    std::string uniformName;
    int binding = -1;
    NodeId sourceNode = 0;
    std::string sourceSocket;
    NodeId parameterNode = 0;
    std::string parameterKey;
    float fallback = 0.0F;
    ShaderValue value;
    // Flexible inputs carry a sampler plus a broadcast scalar and select which one
    // is live at dispatch time with a has-image flag.
    std::string scalarUniformName;
    std::string hasImageUniformName;
    // FlexibleVector inputs carry a sampler plus a vec2 constant and select which
    // one is live at dispatch time with a has-image flag.
    std::string vectorUniformName;
};

struct ShaderInstruction {
    ShaderValue result;
    std::string expression;
    NodeId contributor = 0;
    std::string contributorLabel;
};

struct ShaderHelper {
    std::string name;
    std::string source;
};

struct ShaderOutputRequirement {
    NodeId node = 0;
    std::string socket;
    ShaderValue value;
    int binding = -1;
    std::string imageName;
};

struct ShaderSourceAnnotation {
    NodeId contributor = 0;
    std::string label;
    std::size_t firstLine = 0;
    std::size_t lastLine = 0;
};

struct ShaderRegion {
    std::uint64_t id = 0;
    std::vector<NodeId> nodes;
    std::vector<ShaderInputRequirement> inputs;
    std::vector<ShaderHelper> helpers;
    std::vector<ShaderInstruction> instructions;
    std::vector<ShaderOutputRequirement> candidateOutputs;
    std::string specializationKey;
};

struct GeneratedShader {
    std::string source;
    std::vector<ShaderInputRequirement> inputs;
    std::vector<ShaderOutputRequirement> outputs;
    std::vector<ShaderSourceAnnotation> annotations;
    std::string specializationKey;
};

// A node lowerer asks only for values it actually uses. The region builder turns
// those requests into deduplicated samplers, uniforms, and SSA instructions.
class ShaderLoweringContext {
public:
    virtual ~ShaderLoweringContext() = default;
    [[nodiscard]] virtual ShaderValue input(std::string_view socket,
                                            std::string_view parameterKey,
                                            float fallback) = 0;
    [[nodiscard]] virtual ShaderValue inputAt(std::string_view socket,
                                              std::string_view uvExpression,
                                              std::string_view parameterKey,
                                              float fallback) = 0;
    // Samples an image input at an integer-pixel offset clamped to the texture
    // bounds, reproducing texelFetch semantics inside generated regions. Region
    // builders override this with an exact texelFetch; contexts that only see
    // subgraph expressions keep the default inputAt routing.
    [[nodiscard]] virtual ShaderValue inputTexel(std::string_view socket,
                                                 std::string_view pixelExpression,
                                                 std::string_view parameterKey,
                                                 float fallback) {
        return inputAt(socket, pixelExpression, parameterKey, fallback);
    }
    [[nodiscard]] virtual ShaderValue parameter(std::string_view key,
                                                float fallback) = 0;
    // Registers a helper local to the current node and returns its collision-free
    // name. Occurrences of `name` in source are rewritten to that returned name.
    [[nodiscard]] virtual std::string helper(std::string_view name,
                                             std::string source) = 0;
    [[nodiscard]] virtual ShaderValue emit(std::string expression,
                                           std::string_view socket = {}) = 0;
    [[nodiscard]] virtual ShaderValueType valueType() const = 0;
};

[[nodiscard]] std::string mathGlslExpression(
    MathOperation operation, const std::vector<ShaderValue>& operands,
    const std::vector<ShaderValue>& remapParameters, ShaderValueType type);

[[nodiscard]] std::vector<ShaderRegion> planShaderRegions(
    const Graph& graph, const NodeRegistry& registry, const CompileResult& compiled,
    int maximumImageInputs, int maximumScalarInputs);

[[nodiscard]] GeneratedShader generateComputeShader(
    const ShaderRegion& region, const std::vector<NodeId>& materializedNodes);

} // namespace reaction
