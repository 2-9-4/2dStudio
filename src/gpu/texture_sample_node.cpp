#include "nodes_internal.hpp"
#include "procedural_node_support.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

namespace reaction {
namespace {

using node_support::TextureNode;
using node_support::imageAt;
using node_support::parameter;
using node_support::uniform;

constexpr std::string_view kTextureSampleShader = R"GLSL(#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba16f, binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D sourceImage;
layout(binding=1) uniform sampler2D uImage;
layout(binding=2) uniform sampler2D vImage;
layout(binding=3) uniform sampler2D borderImage;
uniform int hasSource, hasU, hasV, useDefaultU, useDefaultV, hasBorder, sampling, addressMode;
uniform float uConstant, vConstant;
uniform vec4 borderConstant;

vec2 mirrorRepeat(vec2 value) {
    vec2 m = value - 2.0 * floor(value / 2.0);
    return 1.0 - abs(m - 1.0);
}

vec4 fetchClamped(ivec2 point, ivec2 size) {
    return texelFetch(sourceImage, clamp(point, ivec2(0), size - ivec2(1)), 0);
}

vec4 sampleNearest(vec2 coordinate, ivec2 size) {
    return fetchClamped(ivec2(floor(coordinate * vec2(size))), size);
}

vec4 sampleLinear(vec2 coordinate, ivec2 size) {
    vec2 position = coordinate * vec2(size) - 0.5;
    ivec2 base = ivec2(floor(position));
    vec2 fraction = fract(position);
    vec4 a = fetchClamped(base, size);
    vec4 b = fetchClamped(base + ivec2(1, 0), size);
    vec4 c = fetchClamped(base + ivec2(0, 1), size);
    vec4 d = fetchClamped(base + ivec2(1, 1), size);
    return mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 outputSize = imageSize(outputImage);
    if (any(greaterThanEqual(pixel, outputSize))) return;
    vec2 outputUv = (vec2(pixel) + 0.5) / vec2(outputSize);
    if (hasSource == 0) { imageStore(outputImage, pixel, vec4(0.0)); return; }

    // Unconnected coordinate sockets default to the canvas coordinate fields.
    vec2 coordinate = vec2(hasU != 0 ? texture(uImage, outputUv).r :
                           (useDefaultU != 0 ? outputUv.x : uConstant),
                           hasV != 0 ? texture(vImage, outputUv).r :
                           (useDefaultV != 0 ? outputUv.y : vConstant));
    bool outside = any(lessThan(coordinate, vec2(0.0))) ||
                   any(greaterThan(coordinate, vec2(1.0)));
    if (addressMode == 3 && outside) {
        imageStore(outputImage, pixel,
                   hasBorder != 0 ? texture(borderImage, outputUv) : borderConstant);
        return;
    }
    if (addressMode == 0) coordinate = clamp(coordinate, 0.0, 1.0);
    else if (addressMode == 1) coordinate = fract(coordinate);
    else if (addressMode == 2) coordinate = mirrorRepeat(coordinate);

    ivec2 sourceSize = textureSize(sourceImage, 0);
    vec4 value = sampling == 0 ? sampleNearest(coordinate, sourceSize)
                               : sampleLinear(coordinate, sourceSize);
    imageStore(outputImage, pixel, value);
})GLSL";

struct BorderInput {
    ImageHandle image{};
    Vec2 vector{};
    float scalar = 0.0F;
    enum class Kind { Scalar, Vector, Image } kind = Kind::Scalar;
};

BorderInput borderInput(std::span<const Value> values, std::size_t index) {
    BorderInput result;
    if (index >= values.size()) return result;
    if (const auto* image = std::get_if<ImageHandle>(&values[index])) {
        result.image = *image;
        result.kind = BorderInput::Kind::Image;
    } else if (const auto* vector = std::get_if<Vec2>(&values[index])) {
        result.vector = *vector;
        result.kind = BorderInput::Kind::Vector;
    } else if (const auto* scalar = std::get_if<float>(&values[index])) {
        result.scalar = *scalar;
    }
    return result;
}

class TextureSampleNode final : public TextureNode {
public:
    static NodeDescriptor describe() {
        return NodeDescriptor{"texture_sample", 1, "Texture Sample", "Coordinates",
            {{"source", "Source", ValueType::Image2D, SocketDirection::Input},
             {"u", "U", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"v", "V", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"border", "Border Value", ValueType::AnyVector, SocketDirection::Input, true},
             {"sampled", "Sampled", ValueType::Image2D, SocketDirection::Output}},
            {{"sampling", "Sampling", 1.0F, 0.0F, 1.0F, ParameterDescriptor::Control::Enum,
              {"Nearest", "Linear"}},
             {"addressMode", "Address Mode", 0.0F, 0.0F, 3.0F, ParameterDescriptor::Control::Enum,
              {"Clamp", "Repeat", "Mirror", "Border"}}}};
    }

    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kTextureSampleShader, "Texture Sample / compute");

        const auto source = imageAt(inputs, 0);
        const auto u = procedural::numericInput(inputs, 1, 0.5F);
        const auto v = procedural::numericInput(inputs, 2, 0.5F);
        const auto border = borderInput(inputs, 3);
        glUseProgram(program_);
        procedural::bindColorInput(program_, 0, "sourceImage", "hasSource", source);
        procedural::bindNumericInput(program_, 1, "uImage", "hasU", "uConstant", u);
        procedural::bindNumericInput(program_, 2, "vImage", "hasV", "vConstant", v);
        const bool useDefaultU = inputs.size() <= 1 || std::holds_alternative<std::monostate>(inputs[1]);
        const bool useDefaultV = inputs.size() <= 2 || std::holds_alternative<std::monostate>(inputs[2]);
        uniform(program_, "useDefaultU", useDefaultU ? 1 : 0);
        uniform(program_, "useDefaultV", useDefaultV ? 1 : 0);
        procedural::bindColorInput(program_, 3, "borderImage", "hasBorder", border.image);
        const Vec2 borderValue = border.kind == BorderInput::Kind::Vector ? border.vector : Vec2{border.scalar, border.scalar};
        glUniform4f(glGetUniformLocation(program_, "borderConstant"), borderValue.x, borderValue.y,
                    border.kind == BorderInput::Kind::Scalar ? border.scalar : 0.0F, 0.0F);
        uniform(program_, "sampling", std::clamp(static_cast<int>(parameter(parameters_, "sampling", 1.0F)), 0, 1));
        uniform(program_, "addressMode", std::clamp(static_cast<int>(parameter(parameters_, "addressMode", 0.0F)), 0, 3));
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

template <typename T>
void addNode(NodeRegistry& registry) {
    const auto descriptor = T::describe();
    registry.add(descriptor, [] { return std::make_unique<T>(); });
}

} // namespace

void registerTextureSampleNode(NodeRegistry& registry) {
    addNode<TextureSampleNode>(registry);
}

} // namespace reaction
