#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "procedural_node_support.hpp"

#include <algorithm>
#include <memory>
#include <string>

namespace reaction {
namespace {

// A general-purpose, warpable 2D noise field. The five modes share the same
// hashing and octave pipeline, rather than introducing a separate kernel for
// each visual style. Coordinates accept any normal Vector Numeric field.
constexpr std::string_view kFractalNoiseShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba16f,binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D coordinatesImage;
layout(binding=1) uniform sampler2D frequencyImage;
uniform int hasCoordinates,hasFrequency,mode,octaves;
uniform vec2 coordinatesConstant,offset;
uniform float frequencyConstant,gain,lacunarity;
uniform uint seed;
)GLSL";

std::string noiseShader() {
    return std::string(kFractalNoiseShader) +
        std::string(procedural::kGlslBroadcastHelpers) + R"GLSL(
uint hashCell(ivec2 p, uint salt){
    uint h = uint(p.x)*0x8da6b343u+uint(p.y)*0xd8163841u+
             seed*0xcb1ab31fu+salt*0x165667b1u;
    h=(h^(h>>16u))*0x7feb352du;
    h=(h^(h>>15u))*0x846ca68bu;
    return h^(h>>16u);
}
float rand01(ivec2 p,uint salt){
    return float(hashCell(p,salt)>>8u)*(1.0/16777216.0);
}
vec2 gradient(ivec2 p) {
    float a=rand01(p,0u)*6.283185307179586;
    return vec2(cos(a),sin(a));
}
float noiseValue(vec2 p,bool valueLattice){
    ivec2 c=ivec2(floor(p));
    vec2 f=fract(p);
    vec2 t=f*f*f*(f*(f*6.0-15.0)+10.0);
    float v00,v10,v01,v11;
    if(valueLattice){
        v00=rand01(c,1u)*2.0-1.0;
        v10=rand01(c+ivec2(1,0),1u)*2.0-1.0;
        v01=rand01(c+ivec2(0,1),1u)*2.0-1.0;
        v11=rand01(c+ivec2(1,1),1u)*2.0-1.0;
    }else{
        v00=dot(gradient(c),f)*1.41421356237;
        v10=dot(gradient(c+ivec2(1,0)),f-vec2(1,0))*1.41421356237;
        v01=dot(gradient(c+ivec2(0,1)),f-vec2(0,1))*1.41421356237;
        v11=dot(gradient(c+ivec2(1,1)),f-vec2(1,1))*1.41421356237;
    }
    return mix(mix(v00,v10,t.x),mix(v01,v11,t.x),t.y);
}
void main(){
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy),size=imageSize(outputImage);
    if(any(greaterThanEqual(pixel,size))) return;
    vec2 uv=(vec2(pixel)+0.5)/vec2(size);
    vec2 xy=proceduralVector(coordinatesImage,hasCoordinates,coordinatesConstant,uv);
    float freq=proceduralScalar(frequencyImage,hasFrequency,frequencyConstant,uv);
    vec2 p=xy*freq+offset;
    float amplitude=1.0,sum=0.0,weight=0.0;
    for(int i=0;i<8;i++){
        if(i>=octaves) break;
        float n=noiseValue(p,mode==4);
        if(mode==1) n=abs(n)*2.0-1.0;            // Billow
        else if(mode==2) {float r=1.0-abs(n);n=(r*r)*2.0-1.0;} // Ridged
        else if(mode==3) n=abs(n)*2.0-1.0;       // Turbulence
        sum+=n*amplitude;
        weight+=amplitude;
        amplitude*=gain;
        p*=lacunarity;
    }
    float value=clamp(0.5+0.5*sum/max(weight,1e-6),0.0,1.0);
    imageStore(outputImage,pixel,vec4(value,0.0,0.0,1.0));
})GLSL";
}

class FractalNoiseNode final : public node_support::TextureNode {
public:
    static NodeDescriptor describe(){
        auto result=NodeDescriptor{"fractal_noise",1,"Fractal Noise","Generator",
            {{"coordinates","Coordinates",SocketContract::VectorNumeric,SocketDirection::Input,true},
             {"frequency","Frequency",SocketContract::Numeric,SocketDirection::Input,true},
             {"image","Scalar Field",ValueType::ScalarField,SocketDirection::Output}},
            {{"mode","Noise Style",0.0F,0.0F,4.0F,ParameterDescriptor::Control::Enum,
              {"Gradient fBm","Billow","Ridged","Turbulence","Value fBm"}},
             {"frequency","Frequency",5.0F,-100.0F,100.0F},
             {"seed","Seed",0.0F,-100000.0F,100000.0F},
             {"octaves","Octaves",4.0F,1.0F,8.0F,ParameterDescriptor::Control::Integer},
             {"gain","Amplitude Gain",0.5F,0.0F,1.0F},
             {"lacunarity","Lacunarity",2.0F,1.0F,4.0F},
             {"offsetX","Offset X",0.0F,-100.0F,100.0F},
             {"offsetY","Offset Y",0.0F,-100.0F,100.0F}}};
        result.sockets[0].fieldDefault=true;
        result.producedField=true;
        return result;
    }
    const NodeDescriptor& descriptor() const override{
        static const auto value=describe();return value;
    }
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,
                  std::span<Value> outputs) override{
        const auto coordinates=procedural::coordinateInput(inputs,0);
        const auto frequency=procedural::numericParameterInput(
            inputs,1,parameters_,"frequency",5.0F);
        ensure(context);
        auto& gpu=*static_cast<GpuRuntime*>(context.gpu);
        if(!program_)program_=gpu.compileCompute(noiseShader(),"Fractal Noise / compute");
        glUseProgram(program_);
        glBindImageTexture(0,texture_,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F);
        procedural::bindVectorInput(program_,0,"coordinatesImage","hasCoordinates",
                                    "coordinatesConstant",coordinates);
        procedural::bindNumericInput(program_,1,"frequencyImage","hasFrequency",
                                     "frequencyConstant",frequency);
        node_support::uniform(program_,"mode",std::clamp(
            static_cast<int>(node_support::parameter(parameters_,"mode",0.0F)),0,4));
        node_support::uniform(program_,"octaves",std::clamp(
            static_cast<int>(node_support::parameter(parameters_,"octaves",4.0F)),1,8));
        node_support::uniform(program_,"gain",node_support::parameter(parameters_,"gain",0.5F));
        node_support::uniform(program_,"lacunarity",
                              node_support::parameter(parameters_,"lacunarity",2.0F));
        glUniform1ui(glGetUniformLocation(program_,"seed"),static_cast<unsigned int>(
            static_cast<int>(node_support::parameter(parameters_,"seed",0.0F))));
        glUniform2f(glGetUniformLocation(program_,"offset"),
            node_support::parameter(parameters_,"offsetX",0.0F),
            node_support::parameter(parameters_,"offsetY",0.0F));
        gpu.dispatch(program_,context.width,context.height);
        outputs[0]=ImageHandle{texture_,context.width,context.height,ValueType::ScalarField};
    }
};

} // namespace

void registerFractalNoiseNodes(NodeRegistry& registry){
    registry.add(FractalNoiseNode::describe(),[]{return std::make_unique<FractalNoiseNode>();});
}

} // namespace reaction
