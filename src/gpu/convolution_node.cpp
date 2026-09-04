#include "nodes_internal.hpp"
#include "node_support.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

namespace reaction {
namespace {

using node_support::TextureNode;
using node_support::imageAt;
using node_support::parameter;
using node_support::uniform;

constexpr int kMaximumKernelSize = 15;

struct ConvolutionSpec {
    int operation = 0;
    int size = 3;
    int radius = 1;
    bool normalize = false;
    std::array<float, kMaximumKernelSize * kMaximumKernelSize> kernel{};
};

ConvolutionSpec convolutionSpec(const nlohmann::json& parameters) {
    ConvolutionSpec spec;
    spec.operation = std::clamp(static_cast<int>(parameter(parameters, "operation", 0)), 0, 2);
    const int requestedSize = static_cast<int>(parameter(parameters, "kernelSize", 3));
    spec.size = std::clamp(requestedSize | 1, 3, kMaximumKernelSize);
    spec.radius = spec.size / 2;
    spec.normalize = parameter(parameters, "normalize", 0) > 0.5F;
    spec.kernel[static_cast<std::size_t>(spec.radius * spec.size + spec.radius)] = 1.0F;
    if (const auto it = parameters.find("kernel"); it != parameters.end() && it->is_array()) {
        const auto count = std::min(it->size(), static_cast<std::size_t>(spec.size * spec.size));
        for (std::size_t index = 0; index < count; ++index)
            if ((*it)[index].is_number()) spec.kernel[index] = (*it)[index].get<float>();
    }
    return spec;
}

bool convolutionFusable(const nlohmann::json& parameters) {
    return std::clamp(static_cast<int>(parameter(parameters, "iterations", 1)), 1, 32) <= 1;
}

std::string convolutionVariantKey(const nlohmann::json& parameters, bool includeIterations) {
    const auto spec = convolutionSpec(parameters);
    std::ostringstream key;
    key << "operation=" << spec.operation << ";normalize=" << (spec.normalize ? 1 : 0)
        << ";kernelSize=" << spec.size << ";kernel=" << std::hex;
    for (int index = 0; index < spec.size * spec.size; ++index)
        key << std::bit_cast<std::uint32_t>(spec.kernel[static_cast<std::size_t>(index)]) << ',';
    key << std::dec;
    if (includeIterations)
        key << ";iterations=" << std::clamp(static_cast<int>(parameter(parameters, "iterations", 1)), 1, 32);
    return key.str();
}

std::string glslFloat(float value) {
    std::ostringstream out;
    out << std::setprecision(9) << value;
    auto text = out.str();
    if (text.find_first_of(".e") == std::string::npos) text += ".0";
    return text;
}

std::string kernelDeclaration(const ConvolutionSpec& spec) {
    std::ostringstream out;
    out << "const float kernel[" << spec.size * spec.size << "]=float[" << spec.size * spec.size << "](";
    for (int index = 0; index < spec.size * spec.size; ++index) {
        if (index) out << ',';
        out << glslFloat(spec.kernel[static_cast<std::size_t>(index)]);
    }
    out << ");\n";
    return out.str();
}

std::string loopOpen(const ConvolutionSpec& spec) {
    return "for(int y=-" + std::to_string(spec.radius) + ";y<=" + std::to_string(spec.radius) +
           ";++y)for(int x=-" + std::to_string(spec.radius) + ";x<=" + std::to_string(spec.radius) +
           ";++x){float w=kernel[(y+" + std::to_string(spec.radius) + ")*" +
           std::to_string(spec.size) + "+(x+" + std::to_string(spec.radius) + ")];";
}

class ConvolutionNode final : public TextureNode {
public:
    ~ConvolutionNode() override { if (scratch_ != 0) glDeleteTextures(1, &scratch_); }
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"convolution", 1, "Convolution", "Filter",
            {{"image", "Image", ValueType::Image2D, SocketDirection::Input},
             {"image", "Image", ValueType::Image2D, SocketDirection::Output}},
            {{"iterations", "Iterations", 1.0F, 1.0F, 32.0F}}};
        result.lowerable = true;
        result.neighborhoodSocket = "image";
        return result;
    }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    std::string shaderVariantKey(const nlohmann::json& parameters) const override {
        return convolutionVariantKey(parameters, true);
    }
    bool supportsRegionFusion(const nlohmann::json& parameters) const override {
        return convolutionFusable(parameters);
    }
    bool lowerShader(ShaderLoweringContext& context) const override {
        if (!convolutionFusable(parameters_)) return false;
        const auto spec = convolutionSpec(parameters_);
        const auto bias = context.parameter("bias", 0.0F);
        const auto tap = context.inputAt(
            "image", "q+pixelSize*vec2(float(-x),float(-y))", "image", 0.0F).name;
        std::ostringstream source;
        source << "vec4 convolutionKernel(vec2 q){\n" << kernelDeclaration(spec);
        if (spec.operation == 0) {
            source << "vec4 sum=vec4(0.0);";
            if (spec.normalize) source << "float weightSum=0.0;";
            source << "\n" << loopOpen(spec) << "sum+=" << tap << "*w;";
            if (spec.normalize) source << "weightSum+=w;";
            source << "}\n";
            if (spec.normalize) source << "sum/=max(weightSum,1e-6);\n";
            source << "return sum+vec4(" << bias.name << ");\n";
        } else {
            const bool erode = spec.operation == 1;
            source << "vec4 v=vec4(" << (erode ? "3.402823e38" : "-3.402823e38") << ");\n"
                   << loopOpen(spec) << "if(w!=0.0)v=" << (erode ? "min" : "max") << "(v,"
                   << tap << ");}\nreturn v;\n";
        }
        source << "}\n";
        const auto helper = context.helper("convolutionKernel", source.str());
        (void)context.emit(helper + "(uv)", "image");
        return true;
    }
    void evaluate(EvaluationContext& context, std::span<const Value> inputs, std::span<Value> outputs) override {
        const auto source = imageAt(inputs, 0);
        if (!source) { outputs[0] = {}; return; }
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        gpu.ensureTexture(scratch_, scratchWidth_, scratchHeight_,
                          context.width, context.height, GL_RGBA16F);
        const auto spec = convolutionSpec(parameters_);
        const auto sourceKey = convolutionVariantKey(parameters_, false);
        if (!program_ || sourceKey_ != sourceKey) {
            if (program_) glDeleteProgram(program_);
            program_ = gpu.compileCompute(computeShaderSource(spec), "Convolution / compute");
            sourceKey_ = sourceKey;
        }
        glUseProgram(program_);
        uniform(program_, "bias", parameter(parameters_, "bias", 0));
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
    static std::string computeShaderSource(const ConvolutionSpec& spec) {
        std::ostringstream source;
        source << "#version 430\n"
            "layout(local_size_x=16,local_size_y=16)in;\n"
            "layout(rgba16f,binding=0)writeonly uniform image2D outImage;\n"
            "layout(binding=0)uniform sampler2D source;\n"
            "uniform float bias;\n"
            << kernelDeclaration(spec)
            << "void main(){\n"
            "ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outImage);\n"
            "if(any(greaterThanEqual(p,s)))return;\n";
        if (spec.operation == 0) {
            source << "vec4 sum=vec4(0.0);";
            if (spec.normalize) source << "float weightSum=0.0;";
            source << "\n" << loopOpen(spec)
                   << "sum+=texelFetch(source,clamp(p-ivec2(x,y),ivec2(0),s-ivec2(1)),0)*w;";
            if (spec.normalize) source << "weightSum+=w;";
            source << "}\n";
            if (spec.normalize) source << "sum/=max(weightSum,1e-6);\n";
            source << "sum+=vec4(bias);imageStore(outImage,p,sum);\n";
        } else {
            const bool erode = spec.operation == 1;
            source << "vec4 v=vec4(" << (erode ? "3.402823e38" : "-3.402823e38") << ");\n"
                   << loopOpen(spec) << "if(w!=0.0)v=" << (erode ? "min" : "max")
                   << "(v,texelFetch(source,clamp(p-ivec2(x,y),ivec2(0),s-ivec2(1)),0));}\n"
                   << "imageStore(outImage,p,v);\n";
        }
        source << "}";
        return source.str();
    }

    GLuint scratch_ = 0;
    int scratchWidth_ = 0;
    int scratchHeight_ = 0;
    std::string sourceKey_;
};

} // namespace

void registerConvolutionNode(NodeRegistry& registry) {
    registry.add(ConvolutionNode::describe(), [] { return std::make_unique<ConvolutionNode>(); });
}

} // namespace reaction
