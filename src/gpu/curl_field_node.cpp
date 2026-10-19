#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "procedural_node_support.hpp"

#include <algorithm>
#include <memory>
#include <string>

namespace reaction {
namespace {

// Generic 2D perpendicular-gradient operator. Works with Perlin, Worley,
// Fractal Noise, imported scalar images, or any scalar field—not a new
// hardcoded noise algorithm coupled to a particular particle simulation.
constexpr std::string_view kCurlShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba16f,binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D scalarImage;
layout(binding=1) uniform sampler2D strengthImage;
uniform int hasStrength,normalizeDirection;
uniform float strengthConstant;
void main(){
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy),size=imageSize(outputImage);
    if(any(greaterThanEqual(pixel,size))) return;
    vec2 uv=(vec2(pixel)+0.5)/vec2(size);
    vec2 texel=1.0/vec2(size);
    float dx=(texture(scalarImage,uv+vec2(texel.x,0.0)).r-
              texture(scalarImage,uv-vec2(texel.x,0.0)).r)*0.5;
    float dy=(texture(scalarImage,uv+vec2(0.0,texel.y)).r-
              texture(scalarImage,uv-vec2(0.0,texel.y)).r)*0.5;
    vec2 tangent=vec2(dy,-dx);
    if(normalizeDirection!=0){
        float magnitude=length(tangent);
        tangent=magnitude>1e-6?tangent/magnitude:vec2(0.0);
    }
    float gain=hasStrength!=0?texture(strengthImage,uv).r:strengthConstant;
    imageStore(outputImage,pixel,vec4(tangent*gain,0.0,1.0));
})GLSL";

class CurlFieldNode final : public node_support::TextureNode {
public:
    static NodeDescriptor describe(){
        return {"curl_field",1,"Curl Field","Generator",
            {{"source","Scalar Field",ValueType::ScalarField,SocketDirection::Input},
             {"strength","Strength",SocketContract::Numeric,SocketDirection::Input,true},
             {"vector","Vector Field",ValueType::VectorField,SocketDirection::Output}},
            {{"strength","Strength",1.0F,-100.0F,100.0F},
             {"normalize","Normalize Direction",0.0F,0.0F,1.0F,
              ParameterDescriptor::Control::Boolean}}};
    }
    const NodeDescriptor& descriptor() const override{
        static const auto value=describe();return value;
    }
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,
                  std::span<Value> outputs) override{
        const ImageHandle source=node_support::imageAt(inputs,0);
        if(!source){outputs[0]={};return;}
        const auto strength=procedural::numericParameterInput(
            inputs,1,parameters_,"strength",1.0F);
        ensure(context);
        auto& gpu=*static_cast<GpuRuntime*>(context.gpu);
        if(!program_)program_=gpu.compileCompute(kCurlShader,"Curl Field / compute");
        glUseProgram(program_);
        glBindImageTexture(0,texture_,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F);
        node_support::bindTexture(0,source.texture);
        node_support::uniform(program_,"scalarImage",0);
        procedural::bindNumericInput(program_,1,"strengthImage","hasStrength",
                                     "strengthConstant",strength);
        node_support::uniform(program_,"normalizeDirection",
            node_support::parameter(parameters_,"normalize",0.0F)>0.5F?1:0);
        gpu.dispatch(program_,context.width,context.height);
        outputs[0]=ImageHandle{texture_,context.width,context.height,ValueType::VectorField};
    }
};

} // namespace

void registerCurlFieldNode(NodeRegistry& registry){
    registry.add(CurlFieldNode::describe(),[]{return std::make_unique<CurlFieldNode>();});
}

} // namespace reaction
