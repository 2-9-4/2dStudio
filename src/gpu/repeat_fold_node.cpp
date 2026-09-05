#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "procedural_node_support.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

namespace reaction {
namespace {

using node_support::TextureNode;
using node_support::parameter;
using procedural::NumericInput;
using procedural::bindNumericInput;
using procedural::fract;
using procedural::numericInput;
using procedural::positivePeriod;

enum class RepeatFoldMode { Repeat = 0, MirrorRepeat = 1, FoldPositive = 2, FoldNegative = 3 };

RepeatFoldMode modeFrom(const nlohmann::json& parameters) {
    return static_cast<RepeatFoldMode>(std::clamp(
        static_cast<int>(parameter(parameters, "mode", 0.0F)), 0, 3));
}

float apply(RepeatFoldMode mode, float value, float period, float offset, float foldCenter) {
    switch (mode) {
    case RepeatFoldMode::Repeat:
        return offset + fract((value - offset) / positivePeriod(period)) * period;
    case RepeatFoldMode::MirrorRepeat: {
        const float m = (value - offset) / positivePeriod(period);
        const float modulo = m - 2.0F * std::floor(m / 2.0F);
        return offset + (1.0F - std::abs(modulo - 1.0F)) * period;
    }
    case RepeatFoldMode::FoldPositive: return foldCenter + std::abs(value - foldCenter);
    case RepeatFoldMode::FoldNegative: return foldCenter - std::abs(value - foldCenter);
    }
    return value;
}

constexpr std::string_view kRepeatFoldShader = R"GLSL(#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba16f, binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D valueImage;
layout(binding=1) uniform sampler2D periodImage;
layout(binding=2) uniform sampler2D offsetImage;
layout(binding=3) uniform sampler2D foldCenterImage;
uniform int hasValue, hasPeriod, hasOffset, hasFoldCenter, mode;
uniform float valueConstant, periodConstant, offsetConstant, foldCenterConstant;
const float epsilon = 1e-6;
float periodSafeguard(float value) {
    return abs(value) < epsilon ? (value < 0.0 ? -epsilon : epsilon) : value;
}
float repeatFold(float value, float period, float offset, float foldCenter) {
    if (mode == 0) return offset + fract((value - offset) / periodSafeguard(period)) * period;
    if (mode == 1) {
        float t = (value - offset) / periodSafeguard(period);
        float m = t - 2.0 * floor(t / 2.0);
        return offset + (1.0 - abs(m - 1.0)) * period;
    }
    if (mode == 2) return foldCenter + abs(value - foldCenter);
    return foldCenter - abs(value - foldCenter);
}
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(outputImage);
    if (any(greaterThanEqual(pixel, size))) return;
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
    float value = hasValue != 0 ? texture(valueImage, uv).r : valueConstant;
    float period = hasPeriod != 0 ? texture(periodImage, uv).r : periodConstant;
    float offset = hasOffset != 0 ? texture(offsetImage, uv).r : offsetConstant;
    float foldCenter = hasFoldCenter != 0 ? texture(foldCenterImage, uv).r : foldCenterConstant;
    float result = repeatFold(value, period, offset, foldCenter);
    imageStore(outputImage, pixel, vec4(result, result, result, 1.0));
})GLSL";

class RepeatFoldNode final : public TextureNode {
public:
    static NodeDescriptor describe() {
        return {"repeat_fold", 1, "Repeat / Fold", "Math",
            {{"value", "Value", ValueType::AnyNumeric, SocketDirection::Input},
             {"period", "Period", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"offset", "Offset", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"foldCenter", "Fold Center", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"value", "Value", ValueType::AnyNumeric, SocketDirection::Output}},
            {{"mode", "Mode", 0.0F, 0.0F, 3.0F, ParameterDescriptor::Control::Enum,
              {"Repeat", "Mirror Repeat", "Fold Positive", "Fold Negative"}}}};
    }

    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        const auto mode = modeFrom(parameters_);
        const NumericInput value = numericInput(inputs, 0, 0.0F);
        const NumericInput period = numericInput(inputs, 1, 1.0F);
        const NumericInput offset = numericInput(inputs, 2, 0.0F);
        const NumericInput foldCenter = numericInput(inputs, 3, 0.0F);
        const bool outputIsField = value.isField() || foldCenter.isField() ||
            ((mode == RepeatFoldMode::Repeat || mode == RepeatFoldMode::MirrorRepeat) &&
             (period.isField() || offset.isField()));
        if (!outputIsField) {
            outputs[0] = apply(mode, value.constant, period.constant, offset.constant, foldCenter.constant);
            return;
        }

        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kRepeatFoldShader, "Repeat / Fold / compute");
        glUseProgram(program_);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        bindNumericInput(program_, 0, "valueImage", "hasValue", "valueConstant", value);
        bindNumericInput(program_, 1, "periodImage", "hasPeriod", "periodConstant", period);
        bindNumericInput(program_, 2, "offsetImage", "hasOffset", "offsetConstant", offset);
        bindNumericInput(program_, 3, "foldCenterImage", "hasFoldCenter", "foldCenterConstant", foldCenter);
        node_support::uniform(program_, "mode", static_cast<int>(mode));
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

} // namespace

void registerRepeatFoldNode(NodeRegistry& registry) {
    registry.add(RepeatFoldNode::describe(), [] { return std::make_unique<RepeatFoldNode>(); });
}

} // namespace reaction
