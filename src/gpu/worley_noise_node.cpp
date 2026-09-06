#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "procedural_node_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

namespace reaction {
namespace {

using node_support::ParameterNode;
using node_support::parameter;
using procedural::NumericInput;
using procedural::VectorInput;

constexpr std::string_view kWorleyNoisePrefix = R"GLSL(#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba16f, binding=0) writeonly uniform image2D f1Image;
layout(rgba16f, binding=1) writeonly uniform image2D f2Image;
layout(rgba16f, binding=2) writeonly uniform image2D f2MinusF1Image;
layout(rgba16f, binding=3) writeonly uniform image2D cellIdImage;
layout(rgba16f, binding=4) writeonly uniform image2D featurePositionImage;
layout(binding=0) uniform sampler2D coordinatesImage;
layout(binding=1) uniform sampler2D frequencyImage;
layout(binding=2) uniform sampler2D offsetImage;
layout(binding=3) uniform sampler2D jitterImage;
uniform int hasCoordinates, hasFrequency, hasOffset, hasJitter, distanceMetric;
uniform vec2 coordinatesConstant, offsetConstant;
uniform float frequencyConstant, jitterConstant;
uniform uint seed;
const float epsilon = 1e-6;
)GLSL";

std::string worleyNoiseShader() {
    return std::string(kWorleyNoisePrefix) + std::string(procedural::kGlslBroadcastHelpers) + R"GLSL(
float safeDenominator(float value) {
    return abs(value) < epsilon ? (value < 0.0 ? -epsilon : epsilon) : value;
}
uint hashValue(ivec2 cell, uint stream) {
    uint h = uint(cell.x) * 0x8da6b343u + uint(cell.y) * 0xd8163841u + seed * 0xcb1ab31fu + stream * 0x165667b1u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}
float hashUnit(ivec2 cell, uint stream) {
    // Dropping the low eight bits gives a representable [0, 1) float.
    return float(hashValue(cell, stream) >> 8u) * (1.0 / 16777216.0);
}
float metricDistance(vec2 delta) {
    if (distanceMetric == 1) return abs(delta.x) + abs(delta.y);
    if (distanceMetric == 2) return max(abs(delta.x), abs(delta.y));
    return length(delta);
}
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(f1Image);
    if (any(greaterThanEqual(pixel, size))) return;
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
    vec2 coordinates = hasCoordinates > 0 ? texture(coordinatesImage, uv).rg :
                       (hasCoordinates < 0 ? uv : coordinatesConstant);
    float frequency = proceduralScalar(frequencyImage, hasFrequency, frequencyConstant, uv);
    vec2 offset = proceduralVector(offsetImage, hasOffset, offsetConstant, uv);
    float jitter = clamp(proceduralScalar(jitterImage, hasJitter, jitterConstant, uv), 0.0, 1.0);
    vec2 point = coordinates * frequency + offset;
    ivec2 cell = ivec2(floor(point));
    float f1 = 1e30, f2 = 1e30, nearestId = 0.0;
    vec2 nearestFeature = vec2(0.0);
    // A fixed 5x5 neighborhood is sufficient for a feature anywhere in each
    // adjacent jittered cell and keeps noise independent of output resolution.
    for (int y = -2; y <= 2; ++y) for (int x = -2; x <= 2; ++x) {
        ivec2 candidateCell = cell + ivec2(x, y);
        vec2 randomPoint = vec2(hashUnit(candidateCell, 0u), hashUnit(candidateCell, 1u));
        vec2 feature = vec2(candidateCell) + mix(vec2(0.5), randomPoint, jitter);
        float distance = metricDistance(point - feature);
        if (distance < f1) {
            f2 = f1; f1 = distance; nearestFeature = feature;
            nearestId = hashUnit(candidateCell, 2u);
        } else if (distance < f2) {
            f2 = distance;
        }
    }
    vec2 featurePosition = (nearestFeature - offset) / safeDenominator(frequency);
    imageStore(f1Image, pixel, vec4(f1, f1, f1, 1.0));
    imageStore(f2Image, pixel, vec4(f2, f2, f2, 1.0));
    imageStore(f2MinusF1Image, pixel, vec4(f2 - f1, f2 - f1, f2 - f1, 1.0));
    imageStore(cellIdImage, pixel, vec4(nearestId, nearestId, nearestId, 1.0));
    imageStore(featurePositionImage, pixel, vec4(featurePosition, 0.0, 1.0));
})GLSL";
}

float safeFrequency(float value) {
    return std::abs(value) < procedural::kEpsilon
        ? std::copysign(procedural::kEpsilon, value == 0.0F ? 1.0F : value) : value;
}

std::uint32_t hashValue(int x, int y, std::uint32_t seed, std::uint32_t stream) {
    std::uint32_t value = static_cast<std::uint32_t>(x) * 0x8da6b343U +
                          static_cast<std::uint32_t>(y) * 0xd8163841U +
                          seed * 0xcb1ab31fU + stream * 0x165667b1U;
    value = (value ^ (value >> 16U)) * 0x7feb352dU;
    value = (value ^ (value >> 15U)) * 0x846ca68bU;
    return value ^ (value >> 16U);
}

float hashUnit(int x, int y, std::uint32_t seed, std::uint32_t stream) {
    return static_cast<float>(hashValue(x, y, seed, stream) >> 8U) / 16777216.0F;
}

float distanceForMetric(Vec2 delta, int metric) {
    if (metric == 1) return std::abs(delta.x) + std::abs(delta.y);
    if (metric == 2) return std::max(std::abs(delta.x), std::abs(delta.y));
    return std::sqrt(delta.x * delta.x + delta.y * delta.y);
}

struct WorleyResult {
    float f1{};
    float f2{};
    float cellId{};
    Vec2 featurePosition{};
};

WorleyResult calculateWorley(Vec2 coordinates, float frequency, Vec2 offset, float jitter,
                             std::uint32_t seed, int metric) {
    const float clampedJitter = std::clamp(jitter, 0.0F, 1.0F);
    const Vec2 point{coordinates.x * frequency + offset.x, coordinates.y * frequency + offset.y};
    const int cellX = static_cast<int>(std::floor(point.x));
    const int cellY = static_cast<int>(std::floor(point.y));
    WorleyResult result{1.0e30F, 1.0e30F, 0.0F, {}};
    for (int y = -2; y <= 2; ++y) for (int x = -2; x <= 2; ++x) {
        const int candidateX = cellX + x;
        const int candidateY = cellY + y;
        const Vec2 random{hashUnit(candidateX, candidateY, seed, 0),
                          hashUnit(candidateX, candidateY, seed, 1)};
        const Vec2 feature{static_cast<float>(candidateX) + 0.5F + (random.x - 0.5F) * clampedJitter,
                           static_cast<float>(candidateY) + 0.5F + (random.y - 0.5F) * clampedJitter};
        const float distance = distanceForMetric({point.x - feature.x, point.y - feature.y}, metric);
        if (distance < result.f1) {
            result.f2 = result.f1;
            result.f1 = distance;
            result.cellId = hashUnit(candidateX, candidateY, seed, 2);
            result.featurePosition = {(feature.x - offset.x) / safeFrequency(frequency),
                                      (feature.y - offset.y) / safeFrequency(frequency)};
        } else if (distance < result.f2) {
            result.f2 = distance;
        }
    }
    return result;
}

class WorleyNoiseNode final : public ParameterNode {
public:
    ~WorleyNoiseNode() override {
        if (textures_[0]) glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
        if (program_) glDeleteProgram(program_);
    }

    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"worley_noise", 1, "Worley / Voronoi Noise", "Generator",
            {{"coordinates", "Coordinates", SocketContract::VectorNumeric, SocketDirection::Input, true},
             {"frequency", "Frequency", SocketContract::Numeric, SocketDirection::Input, true},
             {"offset", "Offset", SocketContract::VectorNumeric, SocketDirection::Input, true},
             {"jitter", "Jitter", SocketContract::Numeric, SocketDirection::Input, true},
             {"seed", "Seed", ValueType::Float, SocketDirection::Input, true},
             {"f1", "F1", SocketContract::Numeric, SocketDirection::Output},
             {"f2", "F2", SocketContract::Numeric, SocketDirection::Output},
             {"f2MinusF1", "F2 - F1", SocketContract::Numeric, SocketDirection::Output},
             {"cellId", "Cell ID", SocketContract::Numeric, SocketDirection::Output},
             {"featurePosition", "Feature Position", SocketContract::VectorNumeric, SocketDirection::Output}},
            {{"frequency", "Frequency", 5.0F, -100.0F, 100.0F},
             {"offsetX", "Offset X", 0.0F, -100.0F, 100.0F},
             {"offsetY", "Offset Y", 0.0F, -100.0F, 100.0F},
             {"jitter", "Jitter", 1.0F, 0.0F, 1.0F},
             {"seed", "Seed", 0.0F, -100000.0F, 100000.0F},
             {"distanceMetric", "Distance Metric", 0.0F, 0.0F, 2.0F, ParameterDescriptor::Control::Enum,
              {"Euclidean", "Manhattan", "Chebyshev"}}}};
        result.sockets[0].fieldDefault = true;
        const std::vector<std::string> inputs{"coordinates", "frequency", "offset", "jitter"};
        for (std::size_t index = 5; index < 9; ++index) {
            result.sockets[index].typePolicy = SocketDescriptor::TypePolicy::NumericPromotion;
            result.sockets[index].typeInputs = inputs;
        }
        result.sockets[9].typePolicy = SocketDescriptor::TypePolicy::VectorPromotion;
        result.sockets[9].typeInputs = inputs;
        return result;
    }

    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        const VectorInput coordinates = procedural::coordinateInput(inputs, 0);
        const NumericInput frequency = procedural::numericInput(inputs, 1, parameter(parameters_, "frequency", 5.0F));
        const VectorInput offset = procedural::vectorInput(inputs, 2,
            {parameter(parameters_, "offsetX", 0.0F), parameter(parameters_, "offsetY", 0.0F)});
        const NumericInput jitter = procedural::numericInput(inputs, 3, parameter(parameters_, "jitter", 1.0F));
        const float seedValue = node_support::floatAt(inputs, 4, parameter(parameters_, "seed", 0.0F));
        const auto seed = static_cast<std::uint32_t>(static_cast<std::int32_t>(seedValue));
        const int metric = std::clamp(static_cast<int>(parameter(parameters_, "distanceMetric", 0.0F)), 0, 2);
        const bool fieldOutput = coordinates.isField() || frequency.isField() ||
                                 offset.isField() || jitter.isField();
        if (!fieldOutput) {
            const auto result = calculateWorley(coordinates.constant, frequency.constant, offset.constant,
                                                jitter.constant, seed, metric);
            outputs[0] = result.f1;
            outputs[1] = result.f2;
            outputs[2] = result.f2 - result.f1;
            outputs[3] = result.cellId;
            outputs[4] = result.featurePosition;
            return;
        }

        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        for (std::size_t index = 0; index < textures_.size(); ++index)
            gpu.ensureTexture(textures_[index], widths_[index], heights_[index],
                              context.width, context.height, GL_RGBA16F);
        if (!program_) program_ = gpu.compileCompute(worleyNoiseShader(), "Worley / Voronoi Noise / compute");
        glUseProgram(program_);
        for (int index = 0; index < 5; ++index)
            glBindImageTexture(index, textures_[static_cast<std::size_t>(index)], 0, GL_FALSE, 0,
                               GL_WRITE_ONLY, GL_RGBA16F);
        procedural::bindVectorInput(program_, 0, "coordinatesImage", "hasCoordinates", "coordinatesConstant", coordinates);
        procedural::bindNumericInput(program_, 1, "frequencyImage", "hasFrequency", "frequencyConstant", frequency);
        procedural::bindVectorInput(program_, 2, "offsetImage", "hasOffset", "offsetConstant", offset);
        procedural::bindNumericInput(program_, 3, "jitterImage", "hasJitter", "jitterConstant", jitter);
        glUniform1ui(glGetUniformLocation(program_, "seed"), seed);
        node_support::uniform(program_, "distanceMetric", metric);
        gpu.dispatch(program_, context.width, context.height);
        for (std::size_t index = 0; index < textures_.size(); ++index)
            outputs[index] = ImageHandle{textures_[index], context.width, context.height,
                index == 4 ? ValueType::VectorField : ValueType::ScalarField};
    }

private:
    std::array<GLuint, 5> textures_{};
    std::array<int, 5> widths_{};
    std::array<int, 5> heights_{};
    GLuint program_ = 0;
};

} // namespace

void registerWorleyNoiseNode(NodeRegistry& registry) {
    registry.add(WorleyNoiseNode::describe(), [] { return std::make_unique<WorleyNoiseNode>(); });
}

} // namespace reaction
