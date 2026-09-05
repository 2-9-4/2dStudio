#include "reaction/gpu/gpu_runtime.hpp"
#include "reaction/core/node.hpp"

#include "node_support.hpp"
#include "procedural_node_support.hpp"

#include <memory>
#include <string>

namespace reaction {
namespace {

using node_support::TextureNode;
using node_support::parameter;
using procedural::NumericInput;
using procedural::VectorInput;

std::string domainWarpShader() {
    return std::string{R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;
layout(rgba16f,binding=0)writeonly uniform image2D outputImage;
layout(binding=0)uniform sampler2D coordinatesImage;
layout(binding=1)uniform sampler2D xOffsetImage;
layout(binding=2)uniform sampler2D yOffsetImage;
layout(binding=3)uniform sampler2D strengthImage;
uniform int hasCoordinates,hasXOffset,hasYOffset,hasStrength;
uniform vec2 coordinatesConstant;
uniform float xOffsetConstant,yOffsetConstant,strengthConstant;
)GLSL"} + std::string(procedural::kGlslBroadcastHelpers) + R"GLSL(
void main(){
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy),size=imageSize(outputImage);
    if(any(greaterThanEqual(pixel,size)))return;
    vec2 uv=(vec2(pixel)+0.5)/vec2(size);
    // An unconnected Coordinates socket means the canvas coordinate field, not
    // a constant vec2.  This is why it is handled separately from broadcast inputs.
    vec2 coordinates=hasCoordinates!=0
        ?proceduralVector(coordinatesImage,hasCoordinates,coordinatesConstant,uv):uv;
    float xOffset=proceduralScalar(xOffsetImage,hasXOffset,xOffsetConstant,uv);
    float yOffset=proceduralScalar(yOffsetImage,hasYOffset,yOffsetConstant,uv);
    float strength=proceduralScalar(strengthImage,hasStrength,strengthConstant,uv);
    vec2 result=coordinates+vec2(xOffset,yOffset)*strength;
    imageStore(outputImage,pixel,vec4(result,0.0,1.0));
})GLSL";
}

class DomainWarpNode final : public TextureNode {
public:
    static NodeDescriptor describe() {
        return {"domain_warp", 1, "Domain Warp", "Coordinate",
            {{"coordinates", "Coordinates", ValueType::AnyVector, SocketDirection::Input, true},
             {"xOffset", "X Offset", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"yOffset", "Y Offset", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"strength", "Strength", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"coordinates", "Coordinates", ValueType::AnyVector, SocketDirection::Output}},
            {{"xOffset", "X Offset", 0.0F, -10.0F, 10.0F},
             {"yOffset", "Y Offset", 0.0F, -10.0F, 10.0F},
             {"strength", "Strength", 1.0F, -10.0F, 10.0F}}};
    }

    [[nodiscard]] const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        const auto coordinates = procedural::vectorInput(inputs, 0);
        const auto xOffset = procedural::numericInput(inputs, 1, parameter(parameters_, "xOffset", 0.0F));
        const auto yOffset = procedural::numericInput(inputs, 2, parameter(parameters_, "yOffset", 0.0F));
        const auto strength = procedural::numericInput(inputs, 3, parameter(parameters_, "strength", 1.0F));

        // The omitted Coordinates input has a field default (canvas UV), while
        // an explicitly supplied Float2/Float has ordinary constant promotion.
        const bool hasConstantCoordinates = inputs.size() > 0 &&
            (std::holds_alternative<Vec2>(inputs[0]) || std::holds_alternative<float>(inputs[0]));
        if (hasConstantCoordinates && !coordinates.isField() && !xOffset.isField() &&
            !yOffset.isField() && !strength.isField()) {
            outputs[0] = Vec2{coordinates.constant.x + xOffset.constant * strength.constant,
                              coordinates.constant.y + yOffset.constant * strength.constant};
            return;
        }

        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(domainWarpShader(), "Domain Warp / compute");
        glUseProgram(program_);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

        procedural::bindVectorInput(program_, 0, "coordinatesImage", "hasCoordinates",
                                    "coordinatesConstant", coordinates);
        procedural::bindNumericInput(program_, 1, "xOffsetImage", "hasXOffset",
                                     "xOffsetConstant", xOffset);
        procedural::bindNumericInput(program_, 2, "yOffsetImage", "hasYOffset",
                                     "yOffsetConstant", yOffset);
        procedural::bindNumericInput(program_, 3, "strengthImage", "hasStrength",
                                     "strengthConstant", strength);
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

} // namespace

void registerDomainWarpNodes(NodeRegistry& registry) {
    registry.add(DomainWarpNode::describe(), [] { return std::make_unique<DomainWarpNode>(); });
}

} // namespace reaction
