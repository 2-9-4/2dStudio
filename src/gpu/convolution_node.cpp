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
uniform int kernelSize, normalize; uniform float kernel[225], bias;
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outImage);
    if(any(greaterThanEqual(p,s)))return;
    int radius=kernelSize/2; vec4 sum=vec4(0); float weightSum=0;
    for(int y=-7;y<=7;++y)for(int x=-7;x<=7;++x){
        if(abs(x)>radius||abs(y)>radius)continue;
        float weight=kernel[(y+radius)*kernelSize+(x+radius)];
        // Sampling the opposite offset performs a mathematical convolution.
        ivec2 samplePixel=clamp(p-ivec2(x,y),ivec2(0),s-ivec2(1));
        sum+=texelFetch(source,samplePixel,0)*weight; weightSum+=weight;
    }
    if(normalize!=0&&abs(weightSum)>1e-6)sum/=weightSum;
    imageStore(outImage,p,sum+vec4(bias));
})GLSL";

class ConvolutionNode final : public TextureNode {
public:
    static NodeDescriptor describe() { return {"convolution", 1, "Convolution", "Filter",
        {{"image", "Image", ValueType::Image2D, SocketDirection::Input},
         {"image", "Image", ValueType::Image2D, SocketDirection::Output}}, {}}; }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    void evaluate(EvaluationContext& context, std::span<const Value> inputs, std::span<Value> outputs) override {
        const auto source = imageAt(inputs, 0);
        if (!source) { outputs[0] = {}; return; }
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
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
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, source.texture);
        uniform(program_, "kernelSize", size);
        uniform(program_, "normalize", parameter(parameters_, "normalize", 0) > 0.5F ? 1 : 0);
        uniform(program_, "bias", parameter(parameters_, "bias", 0));
        glUniform1fv(glGetUniformLocation(program_, "kernel[0]"), size * size, kernel.data());
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

} // namespace

void registerConvolutionNode(NodeRegistry& registry) {
    registry.add(ConvolutionNode::describe(), [] { return std::make_unique<ConvolutionNode>(); });
}

} // namespace reaction
