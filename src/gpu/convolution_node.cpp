#include "nodes_internal.hpp"
#include "node_support.hpp"

#include <algorithm>
#include <array>
#include <memory>

namespace reaction {
namespace {

using node_support::TextureNode;
using node_support::imageAt;
using node_support::parameter;
using node_support::uniform;

constexpr int kMaximumKernelSize = 15;
constexpr std::string_view kConvolutionShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;
layout(rgba16f,binding=0)writeonly uniform image2D outImage;
layout(binding=0)uniform sampler2D source;
uniform int kernelSize, normalize, operation; uniform float kernel[225], bias;
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outImage);
    if(any(greaterThanEqual(p,s)))return;
    int radius=kernelSize/2; vec4 sum=vec4(0); float weightSum=0;
    vec4 morphology=operation==1?vec4(3.402823e38):vec4(-3.402823e38);
    for(int y=-7;y<=7;++y)for(int x=-7;x<=7;++x){
        if(abs(x)>radius||abs(y)>radius)continue;
        float weight=kernel[(y+radius)*kernelSize+(x+radius)];
        // Sampling the opposite offset performs a mathematical convolution.
        ivec2 samplePixel=clamp(p-ivec2(x,y),ivec2(0),s-ivec2(1));
        vec4 sampleValue=texelFetch(source,samplePixel,0);
        if(operation==0){sum+=sampleValue*weight;weightSum+=weight;}
        else if(weight!=0.0){morphology=operation==1?min(morphology,sampleValue):max(morphology,sampleValue);}
    }
    if(operation==0){if(normalize!=0&&abs(weightSum)>1e-6)sum/=weightSum;sum+=vec4(bias);}
    else sum=morphology;
    imageStore(outImage,p,sum);
})GLSL";

class ConvolutionNode final : public TextureNode {
public:
    ~ConvolutionNode() override { if (scratch_ != 0) glDeleteTextures(1, &scratch_); }
    static NodeDescriptor describe() { return {"convolution", 1, "Convolution", "Filter",
        {{"image", "Image", ValueType::Image2D, SocketDirection::Input},
         {"image", "Image", ValueType::Image2D, SocketDirection::Output}},
        {{"iterations", "Iterations", 1.0F, 1.0F, 32.0F}}}; }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    void evaluate(EvaluationContext& context, std::span<const Value> inputs, std::span<Value> outputs) override {
        const auto source = imageAt(inputs, 0);
        if (!source) { outputs[0] = {}; return; }
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        gpu.ensureTexture(scratch_, scratchWidth_, scratchHeight_,
                          context.width, context.height, GL_RGBA16F);
        if (!program_) program_ = gpu.compileCompute(kConvolutionShader);
        const int requestedSize = static_cast<int>(parameter(parameters_, "kernelSize", 3));
        const int size = std::clamp(requestedSize | 1, 3, kMaximumKernelSize);
        std::array<float, kMaximumKernelSize * kMaximumKernelSize> kernel{};
        kernel[static_cast<std::size_t>((size / 2) * size + size / 2)] = 1.0F;
        if (const auto it = parameters_.find("kernel"); it != parameters_.end() && it->is_array()) {
            const auto count = std::min(it->size(), static_cast<std::size_t>(size * size));
            for (std::size_t index = 0; index < count; ++index) {
                if ((*it)[index].is_number()) kernel[index] = (*it)[index].get<float>();
            }
        }
        glUseProgram(program_);
        uniform(program_, "kernelSize", size);
        uniform(program_, "normalize", parameter(parameters_, "normalize", 0) > 0.5F ? 1 : 0);
        uniform(program_, "operation", std::clamp(
            static_cast<int>(parameter(parameters_, "operation", 0)), 0, 2));
        uniform(program_, "bias", parameter(parameters_, "bias", 0));
        glUniform1fv(glGetUniformLocation(program_, "kernel[0]"), size * size, kernel.data());
        const int iterations = std::clamp(
            static_cast<int>(parameter(parameters_, "iterations", 1)), 1, 32);
        GLuint inputTexture = source.texture;
        for (int pass = 0; pass < iterations; ++pass) {
            const GLuint outputTexture = ((iterations - pass) % 2 == 1) ? texture_ : scratch_;
            glBindImageTexture(0, outputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, inputTexture);
            gpu.dispatch(program_, context.width, context.height);
            inputTexture = outputTexture;
        }
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }

private:
    GLuint scratch_ = 0;
    int scratchWidth_ = 0;
    int scratchHeight_ = 0;
};

} // namespace

void registerConvolutionNode(NodeRegistry& registry) {
    registry.add(ConvolutionNode::describe(), [] { return std::make_unique<ConvolutionNode>(); });
}

} // namespace reaction
