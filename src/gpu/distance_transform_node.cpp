#include "nodes_internal.hpp"
#include "node_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>

namespace reaction {
namespace {

// Jump flooding is deliberately a native materialization boundary: every pass
// samples a previous seed texture, which cannot be represented by one fused
// per-pixel expression.  Seed positions are in pixel-centre coordinates and
// B is a validity bit.
constexpr std::string_view kInit = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba16f,binding=0) writeonly uniform image2D seeds;
layout(binding=0) uniform sampler2D mask;
uniform float threshold; uniform int foreground;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy), s=imageSize(seeds);if(any(greaterThanEqual(p,s)))return;
 float v=texelFetch(mask,p,0).r; bool hit=foreground!=0?v>=threshold:v<threshold;
 imageStore(seeds,p,hit?vec4(vec2(p)+.5,1.,0.):vec4(0.));})GLSL";

constexpr std::string_view kJump = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(binding=0) uniform sampler2D previousSeeds;
layout(rgba16f,binding=0) writeonly uniform image2D nextSeeds;
uniform int stepPixels;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(nextSeeds);if(any(greaterThanEqual(p,s)))return;
 vec2 point=vec2(p)+.5,best=vec2(0.);float bestD=1e30;bool valid=false;
 for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x){ivec2 q=clamp(p+ivec2(x,y)*stepPixels,ivec2(0),s-1);vec4 c=texelFetch(previousSeeds,q,0);if(c.b<.5)continue;float d=dot(point-c.xy,point-c.xy);if(d<bestD){bestD=d;best=c.xy;valid=true;}}
 imageStore(nextSeeds,p,valid?vec4(best,1.,0.):vec4(0.));})GLSL";

constexpr std::string_view kDistance = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba16f,binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D seeds;
uniform int metric, pixels; uniform float maximumDistance;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outputImage);if(any(greaterThanEqual(p,s)))return;vec4 seed=texelFetch(seeds,p,0);float d=maximumDistance;
 if(seed.b>.5){vec2 delta=(vec2(p)+.5-seed.xy);if(metric==1)d=abs(delta.x)+abs(delta.y);else if(metric==2)d=max(abs(delta.x),abs(delta.y));else d=length(delta);if(pixels==0)d/=float(max(s.x,s.y));d=min(d,maximumDistance);}
 imageStore(outputImage,p,vec4(d,d,d,1.));})GLSL";

constexpr std::string_view kSigned = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba16f,binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D foregroundSeeds;
layout(binding=1) uniform sampler2D backgroundSeeds;
layout(binding=2) uniform sampler2D mask;
uniform float threshold; uniform int metric, pixels;
float distanceTo(vec4 seed,vec2 point,ivec2 size){if(seed.b<.5)return 0.;vec2 d=point-seed.xy;float r=metric==1?abs(d.x)+abs(d.y):(metric==2?max(abs(d.x),abs(d.y)):length(d));return pixels!=0?r:r/float(max(size.x,size.y));}
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outputImage);if(any(greaterThanEqual(p,s)))return;vec2 point=vec2(p)+.5;bool inside=texelFetch(mask,p,0).r>=threshold;float d=inside?-distanceTo(texelFetch(backgroundSeeds,p,0),point,s):distanceTo(texelFetch(foregroundSeeds,p,0),point,s);imageStore(outputImage,p,vec4(d,d,d,1.));})GLSL";

class DistanceTransformNode : public node_support::ParameterNode {
public:
    ~DistanceTransformNode() override { for (const auto program : programs_) if (program) glDeleteProgram(program); if (seeds_[0]) glDeleteTextures(4, seeds_.data()); if (output_) glDeleteTextures(1,&output_); }
    static NodeDescriptor describe() {
        auto d=NodeDescriptor{"distance_transform",1,"Distance Transform","Filter",
            {{"mask","Mask",ValueType::ScalarField,SocketDirection::Input},{"threshold","Threshold",ValueType::Float,SocketDirection::Input,true},{"distance","Distance",ValueType::ScalarField,SocketDirection::Output}},
            {{"threshold","Threshold",.5F,0.F,1.F},{"target","Distance To",0.F,0.F,1.F,ParameterDescriptor::Control::Enum,{"Foreground","Background"}},{"metric","Metric",0.F,0.F,2.F,ParameterDescriptor::Control::Enum,{"Euclidean","Manhattan","Chebyshev"}},{"pixels","Units",0.F,0.F,1.F,ParameterDescriptor::Control::Enum,{"Normalized Canvas","Pixels"}},{"maximumDistance","Maximum Distance",65504.F,0.F,65504.F}}}; return d;
    }
    const NodeDescriptor& descriptor() const override { static const auto d=describe();return d; }
    void evaluate(EvaluationContext& c,std::span<const Value> in,std::span<Value> out) override { run(c,node_support::imageAt(in,0),node_support::floatAt(in,1,node_support::parameter(parameters_,"threshold",.5F)),false,out); }
protected:
    void run(EvaluationContext& c,ImageHandle mask,float threshold,bool signedOutput,std::span<Value> out) {
        if(!mask){out[0]={};return;} auto& gpu=*static_cast<GpuRuntime*>(c.gpu); ensure(gpu,c.width,c.height);
        const int metric=std::clamp(static_cast<int>(node_support::parameter(parameters_,"metric",0)),0,2), pixels=std::clamp(static_cast<int>(node_support::parameter(parameters_,"pixels",0)),0,1);
        const int target=std::clamp(static_cast<int>(node_support::parameter(parameters_,"target",0)),0,1);
        const float cap=node_support::parameter(parameters_,"maximumDistance",pixels?65504.F:1.F);
        flood(gpu,mask.texture,threshold,target,seeds_[0],seeds_[1]);
        if(signedOutput){ flood(gpu,mask.texture,threshold,1,seeds_[2],seeds_[3]); glUseProgram(programs_[3]); glBindImageTexture(0,output_,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F); node_support::bindTexture(0,seeds_[0]); node_support::bindTexture(1,seeds_[2]); node_support::bindTexture(2,mask.texture);node_support::uniform(programs_[3],"foregroundSeeds",0);node_support::uniform(programs_[3],"backgroundSeeds",1);node_support::uniform(programs_[3],"mask",2);node_support::uniform(programs_[3],"threshold",threshold);node_support::uniform(programs_[3],"metric",metric);node_support::uniform(programs_[3],"pixels",pixels);gpu.dispatch(programs_[3],c.width,c.height); }
        else {glUseProgram(programs_[2]);glBindImageTexture(0,output_,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F);node_support::bindTexture(0,seeds_[0]);node_support::uniform(programs_[2],"seeds",0);node_support::uniform(programs_[2],"metric",metric);node_support::uniform(programs_[2],"pixels",pixels);node_support::uniform(programs_[2],"maximumDistance",cap);gpu.dispatch(programs_[2],c.width,c.height);}
        out[0]=ImageHandle{output_,c.width,c.height,ValueType::ScalarField};
    }
private:
    void ensure(GpuRuntime& gpu,int w,int h){for(auto& seed:seeds_)gpu.ensureTexture(seed,width_,height_,w,h,GL_RGBA16F);gpu.ensureTexture(output_,outputWidth_,outputHeight_,w,h,GL_RGBA16F);if(!programs_[0]){programs_[0]=gpu.compileCompute(kInit,"Distance Transform / init");programs_[1]=gpu.compileCompute(kJump,"Distance Transform / jump");programs_[2]=gpu.compileCompute(kDistance,"Distance Transform / output");programs_[3]=gpu.compileCompute(kSigned,"SDF Generator / output");}}
    void flood(GpuRuntime& gpu,GLuint mask,float threshold,int foreground,GLuint& first,GLuint& second){glUseProgram(programs_[0]);glBindImageTexture(0,first,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F);node_support::bindTexture(0,mask);node_support::uniform(programs_[0],"mask",0);node_support::uniform(programs_[0],"threshold",threshold);node_support::uniform(programs_[0],"foreground",foreground);gpu.dispatch(programs_[0],width_,height_);GLuint* read=&first,*write=&second;int step=1;for(int largest=std::max(width_,height_);step<largest;step<<=1){} step>>=1;for(;step>0;step>>=1){glUseProgram(programs_[1]);node_support::bindTexture(0,*read);glBindImageTexture(0,*write,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F);node_support::uniform(programs_[1],"previousSeeds",0);node_support::uniform(programs_[1],"stepPixels",step);gpu.dispatch(programs_[1],width_,height_);std::swap(read,write);}if(read!=&first)std::swap(first,second);}
    std::array<GLuint,4> seeds_{};std::array<GLuint,4> programs_{};GLuint output_{};int width_{},height_{},outputWidth_{},outputHeight_{};
};

class SdfGeneratorNode final : public DistanceTransformNode {
public:
    static NodeDescriptor describe(){auto d=NodeDescriptor{"sdf_generator",1,"SDF Generator","Filter",{{"mask","Mask",ValueType::ScalarField,SocketDirection::Input},{"threshold","Threshold",ValueType::Float,SocketDirection::Input,true},{"distance","Signed Distance",ValueType::ScalarField,SocketDirection::Output}},{{"threshold","Threshold",.5F,0.F,1.F},{"metric","Metric",0.F,0.F,2.F,ParameterDescriptor::Control::Enum,{"Euclidean","Manhattan","Chebyshev"}},{"pixels","Units",0.F,0.F,1.F,ParameterDescriptor::Control::Enum,{"Normalized Canvas","Pixels"}}}};return d;}
    const NodeDescriptor& descriptor() const override {static const auto d=describe();return d;}
    void evaluate(EvaluationContext& c,std::span<const Value> in,std::span<Value> out) override {run(c,node_support::imageAt(in,0),node_support::floatAt(in,1,node_support::parameter(parameters_,"threshold",.5F)),true,out);}
};
}
void registerDistanceTransformNodes(NodeRegistry& r){r.add(DistanceTransformNode::describe(),[]{return std::make_unique<DistanceTransformNode>();});r.add(SdfGeneratorNode::describe(),[]{return std::make_unique<SdfGeneratorNode>();});}
}
