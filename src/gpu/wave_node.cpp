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
using procedural::kTwoPi;
using procedural::numericInput;

enum class Waveform { Sine = 0, Cosine = 1, Triangle = 2, Saw = 3, ReverseSaw = 4, Square = 5 };

Waveform waveformFrom(const nlohmann::json& parameters) {
    return static_cast<Waveform>(std::clamp(
        static_cast<int>(parameter(parameters, "waveform", 0.0F)), 0, 5));
}

float waveform(Waveform shape, float z, float dutyCycle) {
    const float q = fract(z);
    switch (shape) {
    case Waveform::Sine: return std::sin(kTwoPi * z);
    case Waveform::Cosine: return std::cos(kTwoPi * z);
    case Waveform::Triangle: return 1.0F - 4.0F * std::abs(q - 0.5F);
    case Waveform::Saw: return 2.0F * q - 1.0F;
    case Waveform::ReverseSaw: return 1.0F - 2.0F * q;
    case Waveform::Square: return q < std::clamp(dutyCycle, 0.0F, 1.0F) ? 1.0F : -1.0F;
    }
    return 0.0F;
}

constexpr std::string_view kWaveShader = R"GLSL(#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba16f, binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D phaseImage;
layout(binding=1) uniform sampler2D frequencyImage;
layout(binding=2) uniform sampler2D phaseOffsetImage;
layout(binding=3) uniform sampler2D amplitudeImage;
layout(binding=4) uniform sampler2D biasImage;
layout(binding=5) uniform sampler2D dutyCycleImage;
uniform int hasPhase, hasFrequency, hasPhaseOffset, hasAmplitude, hasBias, hasDutyCycle;
uniform int waveform, normalize01;
uniform float phaseConstant, frequencyConstant, phaseOffsetConstant, amplitudeConstant, biasConstant, dutyCycleConstant;
const float twoPi = 6.28318530717958647692;
float baseWave(float z, float dutyCycle) {
    float q = fract(z);
    if (waveform == 0) return sin(twoPi * z);
    if (waveform == 1) return cos(twoPi * z);
    if (waveform == 2) return 1.0 - 4.0 * abs(q - 0.5);
    if (waveform == 3) return 2.0 * q - 1.0;
    if (waveform == 4) return 1.0 - 2.0 * q;
    return q < clamp(dutyCycle, 0.0, 1.0) ? 1.0 : -1.0;
}
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(outputImage);
    if (any(greaterThanEqual(pixel, size))) return;
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
    float phase = hasPhase != 0 ? texture(phaseImage, uv).r : phaseConstant;
    float frequency = hasFrequency != 0 ? texture(frequencyImage, uv).r : frequencyConstant;
    float phaseOffset = hasPhaseOffset != 0 ? texture(phaseOffsetImage, uv).r : phaseOffsetConstant;
    float amplitude = hasAmplitude != 0 ? texture(amplitudeImage, uv).r : amplitudeConstant;
    float bias = hasBias != 0 ? texture(biasImage, uv).r : biasConstant;
    float dutyCycle = hasDutyCycle != 0 ? texture(dutyCycleImage, uv).r : dutyCycleConstant;
    float result = baseWave(phase * frequency + phaseOffset, dutyCycle);
    if (normalize01 != 0) result = result * 0.5 + 0.5;
    result = result * amplitude + bias;
    imageStore(outputImage, pixel, vec4(result, result, result, 1.0));
})GLSL";

class WaveNode final : public TextureNode {
public:
    static NodeDescriptor describe() {
        return {"wave", 1, "Wave", "Generator",
            {{"phase", "Phase", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"frequency", "Frequency", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"phaseOffset", "Phase Offset", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"amplitude", "Amplitude", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"bias", "Bias", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"dutyCycle", "Duty Cycle", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"value", "Value", ValueType::AnyNumeric, SocketDirection::Output}},
            {{"waveform", "Waveform", 0.0F, 0.0F, 5.0F, ParameterDescriptor::Control::Enum,
              {"Sine", "Cosine", "Triangle", "Saw", "Reverse Saw", "Square"}},
             {"normalize01", "Normalize 0-1", 0.0F, 0.0F, 1.0F, ParameterDescriptor::Control::Boolean},
             {"phase", "Phase", 0.0F, -100.0F, 100.0F},
             {"frequency", "Frequency", 1.0F, -100.0F, 100.0F},
             {"phaseOffset", "Phase Offset", 0.0F, -100.0F, 100.0F},
             {"amplitude", "Amplitude", 1.0F, -100.0F, 100.0F},
             {"bias", "Bias", 0.0F, -100.0F, 100.0F},
             {"dutyCycle", "Duty Cycle", 0.5F, 0.0F, 1.0F}}};
    }

    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        const NumericInput phase = numericInput(inputs, 0, 0.0F);
        const NumericInput frequency = numericInput(inputs, 1, 1.0F);
        const NumericInput phaseOffset = numericInput(inputs, 2, 0.0F);
        const NumericInput amplitude = numericInput(inputs, 3, 1.0F);
        const NumericInput bias = numericInput(inputs, 4, 0.0F);
        const NumericInput dutyCycle = numericInput(inputs, 5, 0.5F);
        const auto shape = waveformFrom(parameters_);
        const bool normalize = parameter(parameters_, "normalize01", 0.0F) > 0.5F;
        const bool outputIsField = phase.isField() || frequency.isField() || phaseOffset.isField() ||
            amplitude.isField() || bias.isField() || dutyCycle.isField();
        if (!outputIsField) {
            float result = waveform(shape, phase.constant * frequency.constant + phaseOffset.constant,
                                    dutyCycle.constant);
            if (normalize) result = result * 0.5F + 0.5F;
            outputs[0] = result * amplitude.constant + bias.constant;
            return;
        }

        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kWaveShader, "Wave / compute");
        glUseProgram(program_);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        bindNumericInput(program_, 0, "phaseImage", "hasPhase", "phaseConstant", phase);
        bindNumericInput(program_, 1, "frequencyImage", "hasFrequency", "frequencyConstant", frequency);
        bindNumericInput(program_, 2, "phaseOffsetImage", "hasPhaseOffset", "phaseOffsetConstant", phaseOffset);
        bindNumericInput(program_, 3, "amplitudeImage", "hasAmplitude", "amplitudeConstant", amplitude);
        bindNumericInput(program_, 4, "biasImage", "hasBias", "biasConstant", bias);
        bindNumericInput(program_, 5, "dutyCycleImage", "hasDutyCycle", "dutyCycleConstant", dutyCycle);
        node_support::uniform(program_, "waveform", static_cast<int>(shape));
        node_support::uniform(program_, "normalize01", normalize ? 1 : 0);
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

} // namespace

void registerWaveNode(NodeRegistry& registry) {
    registry.add(WaveNode::describe(), [] { return std::make_unique<WaveNode>(); });
}

} // namespace reaction
