#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "procedural_node_support.hpp"
#include "reaction/core/node_builder.hpp"

#include <algorithm>
#include <memory>

namespace reaction {
namespace {

// A reverse-coordinate map: feed this into any coordinate-aware generator or
// Texture Sample. Each evaluation reads the original coordinates, so there are
// no broken/reconnected grid edges.
constexpr std::string_view kVortex = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D coordinatesImage;
layout(binding=1) uniform sampler2D centerImage;
layout(binding=2) uniform sampler2D twistImage;
layout(binding=3) uniform sampler2D radiusImage;
layout(binding=4) uniform sampler2D separationImage;
layout(binding=5) uniform sampler2D squeezeImage;
layout(binding=6) uniform sampler2D rippleImage;
layout(binding=7) uniform sampler2D periodImage;
layout(binding=8) uniform sampler2D widthImage;
uniform int hasCoordinates,hasCenter,hasTwist,hasRadius,hasSeparation;
uniform int hasSqueeze,hasRipple,hasPeriod,hasWidth;
uniform vec2 coordinatesConstant,centerConstant;
uniform float twistConstant,radiusConstant,separationConstant,squeezeConstant;
uniform float rippleConstant,periodConstant,widthConstant;
uniform int mode,falloff;
float envelope(float r,float radius){
    float u=r/max(radius,1e-4);
    if(falloff==0)return exp(-u*u);
    if(falloff==1)return exp(-u);
    return pow(max(1.0-u,0.0),2.0)*(1.0+2.0*min(u,1.0));
}
vec2 turn(vec2 p,vec2 origin,float angle,float radius){
    vec2 d=p-origin;
    float a=angle*envelope(length(d),radius);
    return origin+mat2(cos(a),sin(a),-sin(a),cos(a))*d;
}
void main(){
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy),size=imageSize(outputImage);
    if(any(greaterThanEqual(pixel,size)))return;
    vec2 uv=(vec2(pixel)+0.5)/vec2(size);
    vec2 p=hasCoordinates>0?texture(coordinatesImage,uv).rg:
           (hasCoordinates<0?uv:coordinatesConstant);
    vec2 center=hasCenter!=0?texture(centerImage,uv).rg:centerConstant;
    float twist=hasTwist!=0?texture(twistImage,uv).r:twistConstant;
    float radius=max(1e-4,hasRadius!=0?texture(radiusImage,uv).r:radiusConstant);
    float separation=hasSeparation!=0?texture(separationImage,uv).r:separationConstant;
    float squeeze=hasSqueeze!=0?texture(squeezeImage,uv).r:squeezeConstant;
    float ripple=hasRipple!=0?texture(rippleImage,uv).r:rippleConstant;
    float period=max(1e-4,hasPeriod!=0?texture(periodImage,uv).r:periodConstant);
    float ringWidth=max(1e-4,hasWidth!=0?texture(widthImage,uv).r:widthConstant);

    if(mode==0) {
        p=turn(p,center,twist,radius);
    } else {
        // Adjacent counter-rotating foci; a genuine dipole, not two overlay images.
        vec2 left=center-vec2(separation*0.5,0.0);
        vec2 right=center+vec2(separation*0.5,0.0);
        p=turn(p,left,twist,radius);
        p=turn(p,right,-twist,radius);
    }
    vec2 d=p-center;
    float r=length(d);
    float ring=exp(-pow((r-radius)/ringWidth,2.0));
    float compression=clamp(1.0-squeeze*envelope(r,radius),0.15,3.0);
    float oscillation=ripple*sin(6.28318530718*(r-radius)/period)*ring;
    p=center+d*compression+(r>1e-6?d/r:vec2(0.0))*oscillation;
    imageStore(outputImage,pixel,vec4(p,0.0,1.0));
})GLSL";

class VortexWarpNode final : public node_support::TextureNode {
public:
    static NodeDescriptor describe() {
        return NodeDescriptorBuilder{"vortex_warp",1,"Vortex / Dipole Warp","Coordinate"}
            .optionalInput("coordinates","Coordinates",SocketContract::VectorNumeric)
            .fieldDefault("coordinates")
            .optionalInput("center","Center",SocketContract::VectorNumeric)
            .output("coordinates","Warped Coordinates",ValueType::VectorField)
            .enumParameter("mode","Foci",1,{"Single","Dipole (counter-rotating)"})
            .enumParameter("falloff","Falloff",0,{"Gaussian","Exponential","Smooth cutoff"})
            .floatParameter("twist","Twist (radians)",5.5F,-30.0F,30.0F)
            .floatParameter("radius","Influence Radius (UV)",0.23F,0.005F,2.0F)
            .floatParameter("separation","Dipole Separation (UV)",0.20F,0.0F,1.0F)
            .floatParameter("squeeze","Radial Squeeze",0.36F,-1.0F,0.8F)
            .floatParameter("ripple","Transition Ripple (UV)",0.025F,-0.3F,0.3F)
            .floatParameter("period","Ripple Wavelength (UV)",0.08F,0.005F,1.0F)
            .floatParameter("ringWidth","Transition Width (UV)",0.10F,0.005F,1.0F)
            .producedField()
            .build();
    }
    const NodeDescriptor& descriptor()const override{
        static const auto d=describe();return d;
    }
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,
                  std::span<Value> outputs)override{
        const auto coords=procedural::coordinateInput(inputs,0);
        const auto center=procedural::vectorInput(inputs,1,{0.5F,0.5F});
        constexpr const char* keys[]={"twist","radius","separation","squeeze",
                                      "ripple","period","ringWidth"};
        constexpr float defaults[]={5.5F,0.23F,0.20F,0.36F,0.025F,0.08F,0.10F};
        const char* textures[]={"twistImage","radiusImage","separationImage","squeezeImage",
                                "rippleImage","periodImage","widthImage"};
        const char* has[]={"hasTwist","hasRadius","hasSeparation","hasSqueeze",
                           "hasRipple","hasPeriod","hasWidth"};
        const char* constants[]={"twistConstant","radiusConstant","separationConstant",
                                 "squeezeConstant","rippleConstant","periodConstant","widthConstant"};
        ensure(context,GL_RGBA32F);
        auto& gpu=*static_cast<GpuRuntime*>(context.gpu);
        if(!program_)program_=gpu.compileCompute(kVortex,"Vortex Warp");
        glUseProgram(program_);
        glBindImageTexture(0,texture_,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA32F);
        procedural::bindVectorInput(program_,0,"coordinatesImage","hasCoordinates","coordinatesConstant",coords);
        procedural::bindVectorInput(program_,1,"centerImage","hasCenter","centerConstant",center);
        for(std::size_t i=0;i<7;i++){
            const auto value=procedural::numericParameterInput(
                inputs,i+2,parameters_,keys[i],defaults[i]);
            procedural::bindNumericInput(program_,static_cast<int>(i+2),
                textures[i],has[i],constants[i],value);
        }
        node_support::uniform(program_,"mode",std::clamp(static_cast<int>(
            node_support::parameter(parameters_,"mode",1)),0,1));
        node_support::uniform(program_,"falloff",std::clamp(static_cast<int>(
            node_support::parameter(parameters_,"falloff",0)),0,2));
        gpu.dispatch(program_,context.width,context.height);
        outputs[0]=ImageHandle{texture_,context.width,context.height,ValueType::VectorField};
    }
};
}
void registerVortexWarpNode(NodeRegistry& registry){
    registry.add(VortexWarpNode::describe(),[]{return std::make_unique<VortexWarpNode>();});
}
} // namespace reaction
