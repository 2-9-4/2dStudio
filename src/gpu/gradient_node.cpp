#include "reaction/gpu/gpu_runtime.hpp"
#include "reaction/core/node.hpp"

#include "node_support.hpp"
#include "procedural_node_support.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

namespace reaction {
namespace {

using node_support::TextureNode;
using node_support::parameter;

std::string gradientShader() {
    return std::string{R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;
layout(rgba16f,binding=0)writeonly uniform image2D outputImage;
layout(binding=0)uniform sampler2D coordinatesImage;
layout(binding=1)uniform sampler2D centerImage;
layout(binding=2)uniform sampler2D angleImage;
layout(binding=3)uniform sampler2D scaleImage;
layout(binding=4)uniform sampler2D offsetImage;
uniform int hasCoordinates,hasCenter,hasAngle,hasScale,hasOffset,mode,clampOutput;
uniform vec2 coordinatesConstant,centerConstant;
uniform float angleConstant,scaleConstant,offsetConstant;
)GLSL"} + std::string(procedural::kGlslBroadcastHelpers) + R"GLSL(
void main(){
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy),size=imageSize(outputImage);
    if(any(greaterThanEqual(pixel,size)))return;
    vec2 uv=(vec2(pixel)+0.5)/vec2(size);
    vec2 coordinates=hasCoordinates!=0
        ?proceduralVector(coordinatesImage,hasCoordinates,coordinatesConstant,uv):uv;
    vec2 center=proceduralVector(centerImage,hasCenter,centerConstant,uv);
    float angle=proceduralScalar(angleImage,hasAngle,angleConstant,uv);
    float scale=proceduralScalar(scaleImage,hasScale,scaleConstant,uv);
    float offset=proceduralScalar(offsetImage,hasOffset,offsetConstant,uv);
    vec2 d=coordinates-center;
    float value;
    if(mode==0)value=dot(d,vec2(cos(angle),sin(angle)))*scale+offset;
    else if(mode==1)value=abs(dot(d,vec2(cos(angle),sin(angle))))*scale+offset;
    else if(mode==2)value=length(d)*scale+offset;
    else if(mode==3)value=fract(atan(d.y,d.x)/(2.0*3.141592653589793)*scale+offset);
    else value=(abs(d.x)+abs(d.y))*scale+offset;
    if(clampOutput!=0)value=clamp(value,0.0,1.0);
    imageStore(outputImage,pixel,vec4(value,0.0,0.0,1.0));
})GLSL";
}

float gradientValue(int mode, Vec2 coordinates, Vec2 center, float angle, float scale,
                    float offset, bool clampOutput) {
    const float dx = coordinates.x - center.x;
    const float dy = coordinates.y - center.y;
    float value = 0.0F;
    switch (mode) {
    case 0: value = (dx * std::cos(angle) + dy * std::sin(angle)) * scale + offset; break;
    case 1: value = std::abs(dx * std::cos(angle) + dy * std::sin(angle)) * scale + offset; break;
    case 2: value = std::sqrt(dx * dx + dy * dy) * scale + offset; break;
    case 3: value = procedural::fract(std::atan2(dy, dx) / procedural::kTwoPi * scale + offset); break;
    default: value = (std::abs(dx) + std::abs(dy)) * scale + offset; break;
    }
    return clampOutput ? std::clamp(value, 0.0F, 1.0F) : value;
}

class GradientNode final : public TextureNode {
public:
    static NodeDescriptor describe() {
        return {"gradient", 1, "Gradient", "Generator",
            {{"coordinates", "Coordinates", ValueType::AnyVector, SocketDirection::Input, true},
             {"center", "Center", ValueType::AnyVector, SocketDirection::Input, true},
             {"angle", "Angle", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"scale", "Scale", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"offset", "Offset", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"value", "Value", ValueType::AnyNumeric, SocketDirection::Output}},
            {{"mode", "Mode", 0.0F, 0.0F, 4.0F, ParameterDescriptor::Control::Enum,
              {"Linear", "Reflected Linear", "Radial", "Angular", "Diamond"}},
             {"clamp", "Clamp", 0.0F, 0.0F, 1.0F, ParameterDescriptor::Control::Boolean},
             {"angle", "Angle", 0.0F, -20.0F, 20.0F},
             {"scale", "Scale", 1.0F, -100.0F, 100.0F},
             {"offset", "Offset", 0.0F, -100.0F, 100.0F}}};
    }

    [[nodiscard]] const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        const auto coordinates = procedural::coordinateInput(inputs, 0);
        const auto center = procedural::vectorInput(inputs, 1, {0.5F, 0.5F});
        const auto angle = procedural::numericInput(inputs, 2, parameter(parameters_, "angle", 0.0F));
        const auto scale = procedural::numericInput(inputs, 3, parameter(parameters_, "scale", 1.0F));
        const auto offset = procedural::numericInput(inputs, 4, parameter(parameters_, "offset", 0.0F));
        const int mode = static_cast<int>(parameter(parameters_, "mode", 0.0F));
        const bool clampOutput = parameter(parameters_, "clamp", 0.0F) != 0.0F;
        if (!coordinates.isField() && !center.isField() &&
            !angle.isField() && !scale.isField() && !offset.isField()) {
            outputs[0] = gradientValue(mode, coordinates.constant, center.constant, angle.constant,
                scale.constant, offset.constant, clampOutput);
            return;
        }

        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (program_ == 0) program_ = gpu.compileCompute(gradientShader(), "Gradient / compute");
        glUseProgram(program_);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        procedural::bindVectorInput(program_, 0, "coordinatesImage", "hasCoordinates",
                                    "coordinatesConstant", coordinates);
        procedural::bindVectorInput(program_, 1, "centerImage", "hasCenter", "centerConstant", center);
        procedural::bindNumericInput(program_, 2, "angleImage", "hasAngle", "angleConstant", angle);
        procedural::bindNumericInput(program_, 3, "scaleImage", "hasScale", "scaleConstant", scale);
        procedural::bindNumericInput(program_, 4, "offsetImage", "hasOffset", "offsetConstant", offset);
        node_support::uniform(program_, "mode", std::clamp(mode, 0, 4));
        node_support::uniform(program_, "clampOutput", clampOutput ? 1 : 0);
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

} // namespace

void registerGradientNodes(NodeRegistry& registry) {
    registry.add(GradientNode::describe(), [] { return std::make_unique<GradientNode>(); });
}

} // namespace reaction
