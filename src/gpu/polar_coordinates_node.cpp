#include "reaction/gpu/gpu_runtime.hpp"
#include "reaction/core/node.hpp"

#include "node_support.hpp"
#include "procedural_node_support.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <string>

namespace reaction {
namespace {

using node_support::ParameterNode;
using node_support::parameter;

std::string polarCoordinatesShader() {
    return std::string{R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;
layout(rgba16f,binding=0)writeonly uniform image2D radiusImage;
layout(rgba16f,binding=1)writeonly uniform image2D angleImage;
layout(rgba16f,binding=2)writeonly uniform image2D normalizedAngleImage;
layout(rgba16f,binding=3)writeonly uniform image2D coordinatesImage;
layout(binding=0)uniform sampler2D inputCoordinates;
layout(binding=1)uniform sampler2D centerImage;
layout(binding=2)uniform sampler2D radiusScaleImage;
layout(binding=3)uniform sampler2D angleOffsetImage;
layout(binding=4)uniform sampler2D radiusInputImage;
layout(binding=5)uniform sampler2D angleInputImage;
uniform int hasCoordinates,hasCenter,hasRadiusScale,hasAngleOffset,hasRadius,hasAngle,mode;
uniform vec2 coordinatesConstant,centerConstant;
uniform float radiusScaleConstant,angleOffsetConstant,radiusConstant,angleConstant;
)GLSL"} + std::string(procedural::kGlslBroadcastHelpers) + R"GLSL(
void main(){
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy),size=imageSize(radiusImage);
    if(any(greaterThanEqual(pixel,size)))return;
    vec2 uv=(vec2(pixel)+0.5)/vec2(size);
    vec2 center=proceduralVector(centerImage,hasCenter,centerConstant,uv);
    if(mode==0){
        vec2 coordinates=hasCoordinates!=0
            ?proceduralVector(inputCoordinates,hasCoordinates,coordinatesConstant,uv):uv;
        float radiusScale=proceduralScalar(radiusScaleImage,hasRadiusScale,radiusScaleConstant,uv);
        float angleOffset=proceduralScalar(angleOffsetImage,hasAngleOffset,angleOffsetConstant,uv);
        vec2 d=coordinates-center;
        float angle=atan(d.y,d.x)+angleOffset;
        float radius=length(d)*radiusScale;
        imageStore(radiusImage,pixel,vec4(radius,0.0,0.0,1.0));
        imageStore(angleImage,pixel,vec4(angle,0.0,0.0,1.0));
        imageStore(normalizedAngleImage,pixel,vec4(fract(angle/(2.0*3.141592653589793)),0.0,0.0,1.0));
        imageStore(coordinatesImage,pixel,vec4(0.0,0.0,0.0,1.0));
    }else{
        float radius=proceduralScalar(radiusInputImage,hasRadius,radiusConstant,uv);
        float angle=proceduralScalar(angleInputImage,hasAngle,angleConstant,uv);
        vec2 coordinates=center+radius*vec2(cos(angle),sin(angle));
        imageStore(radiusImage,pixel,vec4(0.0,0.0,0.0,1.0));
        imageStore(angleImage,pixel,vec4(0.0,0.0,0.0,1.0));
        imageStore(normalizedAngleImage,pixel,vec4(0.0,0.0,0.0,1.0));
        imageStore(coordinatesImage,pixel,vec4(coordinates,0.0,1.0));
    }
})GLSL";
}

class PolarCoordinatesNode final : public ParameterNode {
public:
    ~PolarCoordinatesNode() override {
        if (textures_[0] != 0) glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
        if (program_ != 0) glDeleteProgram(program_);
    }

    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"polar_coordinates", 1, "Polar Coordinates", "Coordinate",
            {{"coordinates", "Coordinates", SocketContract::VectorNumeric, SocketDirection::Input, true},
             {"center", "Center", SocketContract::VectorNumeric, SocketDirection::Input, true},
             {"radiusScale", "Radius Scale", SocketContract::Numeric, SocketDirection::Input, true},
             {"angleOffset", "Angle Offset", SocketContract::Numeric, SocketDirection::Input, true},
             {"radius", "Radius", SocketContract::Numeric, SocketDirection::Input, true},
             {"angle", "Angle", SocketContract::Numeric, SocketDirection::Input, true},
             {"radius", "Radius", SocketContract::Numeric, SocketDirection::Output},
             {"angle", "Angle", SocketContract::Numeric, SocketDirection::Output},
             {"normalizedAngle", "Normalized Angle", SocketContract::Numeric, SocketDirection::Output},
             {"outputCoordinates", "Coordinates", SocketContract::VectorNumeric, SocketDirection::Output}},
            {{"mode", "Mode", 0.0F, 0.0F, 1.0F, ParameterDescriptor::Control::Enum,
              {"Cartesian to Polar", "Polar to Cartesian"}},
             {"radiusScale", "Radius Scale", 1.0F, -10.0F, 10.0F},
             {"angleOffset", "Angle Offset", 0.0F, -20.0F, 20.0F}}};
        result.sockets[0].fieldDefault = true;
        const std::vector<std::string> inputs{"coordinates", "center", "radiusScale",
                                              "angleOffset"};
        for (std::size_t index = 6; index < 9; ++index) {
            result.sockets[index].typePolicy = SocketDescriptor::TypePolicy::NumericPromotion;
            result.sockets[index].typeInputs = inputs;
        }
        result.sockets[9].typePolicy = SocketDescriptor::TypePolicy::VectorPromotion;
        result.sockets[9].typeInputs = {"center", "radius", "angle"};
        return result;
    }

    [[nodiscard]] const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        const int mode = static_cast<int>(parameter(parameters_, "mode", 0.0F));
        const auto coordinates = procedural::coordinateInput(inputs, 0);
        const auto center = procedural::vectorInput(inputs, 1, {0.5F, 0.5F});
        const auto radiusScale = procedural::numericInput(inputs, 2, parameter(parameters_, "radiusScale", 1.0F));
        const auto angleOffset = procedural::numericInput(inputs, 3, parameter(parameters_, "angleOffset", 0.0F));
        const auto radius = procedural::numericInput(inputs, 4);
        const auto angle = procedural::numericInput(inputs, 5);
        const bool hasConstantCoordinates = inputs.size() > 0 &&
            (std::holds_alternative<Vec2>(inputs[0]) || std::holds_alternative<float>(inputs[0]));

        if (mode == 0 && hasConstantCoordinates && !coordinates.isField() && !center.isField() &&
            !radiusScale.isField() && !angleOffset.isField()) {
            const float dx = coordinates.constant.x - center.constant.x;
            const float dy = coordinates.constant.y - center.constant.y;
            const float rawAngle = std::atan2(dy, dx) + angleOffset.constant;
            outputs[0] = std::sqrt(dx * dx + dy * dy) * radiusScale.constant;
            outputs[1] = rawAngle;
            outputs[2] = procedural::fract(rawAngle / procedural::kTwoPi);
            return;
        }
        if (mode != 0 && !center.isField() && !radius.isField() && !angle.isField()) {
            outputs[3] = Vec2{center.constant.x + radius.constant * std::cos(angle.constant),
                              center.constant.y + radius.constant * std::sin(angle.constant)};
            return;
        }

        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        for (std::size_t index = 0; index < textures_.size(); ++index)
            gpu.ensureTexture(textures_[index], widths_[index], heights_[index], context.width,
                              context.height, GL_RGBA16F);
        if (program_ == 0) program_ = gpu.compileCompute(polarCoordinatesShader(), "Polar Coordinates / compute");
        glUseProgram(program_);
        for (std::size_t index = 0; index < textures_.size(); ++index)
            glBindImageTexture(static_cast<GLuint>(index), textures_[index], 0, GL_FALSE, 0,
                               GL_WRITE_ONLY, GL_RGBA16F);
        procedural::bindVectorInput(program_, 0, "inputCoordinates", "hasCoordinates",
                                    "coordinatesConstant", coordinates);
        procedural::bindVectorInput(program_, 1, "centerImage", "hasCenter", "centerConstant", center);
        procedural::bindNumericInput(program_, 2, "radiusScaleImage", "hasRadiusScale",
                                     "radiusScaleConstant", radiusScale);
        procedural::bindNumericInput(program_, 3, "angleOffsetImage", "hasAngleOffset",
                                     "angleOffsetConstant", angleOffset);
        procedural::bindNumericInput(program_, 4, "radiusInputImage", "hasRadius", "radiusConstant", radius);
        procedural::bindNumericInput(program_, 5, "angleInputImage", "hasAngle", "angleConstant", angle);
        node_support::uniform(program_, "mode", mode == 0 ? 0 : 1);
        gpu.dispatch(program_, context.width, context.height);
        if (mode == 0) {
            outputs[0] = ImageHandle{textures_[0], context.width, context.height, ValueType::ScalarField};
            outputs[1] = ImageHandle{textures_[1], context.width, context.height, ValueType::ScalarField};
            outputs[2] = ImageHandle{textures_[2], context.width, context.height, ValueType::ScalarField};
        } else {
            outputs[3] = ImageHandle{textures_[3], context.width, context.height, ValueType::VectorField};
        }
    }

private:
    std::array<GLuint, 4> textures_{};
    std::array<int, 4> widths_{};
    std::array<int, 4> heights_{};
    GLuint program_ = 0;
};

} // namespace

void registerPolarCoordinatesNodes(NodeRegistry& registry) {
    registry.add(PolarCoordinatesNode::describe(), [] { return std::make_unique<PolarCoordinatesNode>(); });
}

} // namespace reaction
