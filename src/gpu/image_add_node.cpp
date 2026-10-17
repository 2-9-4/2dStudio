#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "reaction/core/node_builder.hpp"

#include <algorithm>
#include <memory>

namespace reaction {
namespace {
constexpr const char* kImageAdd = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;
layout(binding=0)uniform sampler2D imageA;
layout(binding=1)uniform sampler2D imageB;
layout(rgba16f,binding=0)writeonly uniform image2D resultImage;
uniform float gainA,gainB;
uniform vec3 background;
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(resultImage);
    if(any(greaterThanEqual(p,s)))return;
    vec3 a=texelFetch(imageA,p,0).rgb;
    vec3 b=texelFetch(imageB,p,0).rgb;
    // Add two independently accumulated intensities on a single background.
    imageStore(resultImage,p,vec4(background+gainA*a+gainB*b,1.0));
})GLSL";

class ImageAddNode final : public node_support::TextureNode {
public:
    static NodeDescriptor describe() {
        return NodeDescriptorBuilder{"image_add",1,"Add Image Layers","Compositing"}
            .input("a","Layer A",ValueType::ColorImage)
            .input("b","Layer B",ValueType::ColorImage)
            .output("image","Added Layers",ValueType::ColorImage)
            .floatParameter("gainA","Layer A Gain",1.0F,0.0F,8.0F)
            .floatParameter("gainB","Layer B Gain",1.0F,0.0F,8.0F)
            .floatParameter("backgroundR","Background Red",.05F,0.0F,1.0F)
            .floatParameter("backgroundG","Background Green",.05F,0.0F,1.0F)
            .floatParameter("backgroundB","Background Blue",.05F,0.0F,1.0F)
            .build();
    }
    const NodeDescriptor& descriptor() const override {
        static const auto d=describe();
        return d;
    }
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        if(outputs.empty())return;
        const auto a=node_support::imageAt(inputs,0);
        const auto b=node_support::imageAt(inputs,1);
        if(!a||!b||a.width!=b.width||a.height!=b.height){
            outputs[0]={};
            return;
        }
        ensure(context);
        auto& gpu=*static_cast<GpuRuntime*>(context.gpu);
        if(!program_)program_=gpu.compileCompute(kImageAdd,"Image Add / compute");
        glUseProgram(program_);
        node_support::bindTexture(0,a.texture);
        node_support::bindTexture(1,b.texture);
        node_support::uniform(program_,"imageA",0);
        node_support::uniform(program_,"imageB",1);
        const auto value=[&](std::size_t i,const char* key,float defaultValue) {
            return node_support::floatAt(inputs,i,node_support::parameter(parameters_,key,defaultValue));
        };
        node_support::uniform(program_,"gainA",value(2,"gainA",1.0F));
        node_support::uniform(program_,"gainB",value(3,"gainB",1.0F));
        glUniform3f(glGetUniformLocation(program_,"background"),
                    value(4,"backgroundR",.05F),
                    value(5,"backgroundG",.05F),
                    value(6,"backgroundB",.05F));
        glBindImageTexture(0,texture_,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F);
        gpu.dispatch(program_,context.width,context.height);
        outputs[0]=ImageHandle{texture_,context.width,context.height,ValueType::ColorImage};
    }
};
} // namespace
void registerImageAddNode(NodeRegistry& registry){
    registry.add(ImageAddNode::describe(),[]{return std::make_unique<ImageAddNode>();});
}
} // namespace reaction
