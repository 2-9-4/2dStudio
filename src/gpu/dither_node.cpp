#include "nodes_internal.hpp"
#include "node_support.hpp"

#include "reaction/core/node_builder.hpp"

#include "reaction/gpu/shader_ir.hpp"

#include <algorithm>
#include <memory>

namespace reaction {
namespace {

using node_support::parameter;

// These values are serialized in project files and are part of the node's
// shader specialization key. Append new modes; do not reorder existing ones.
enum class DitherMode : int {
    Threshold = 0,
    Bayer2x2 = 1,
    Bayer4x4 = 2,
    Bayer8x8 = 3,
    Halftone = 4,
    Noise = 5,
};

DitherMode clampMode(float persistedValue) {
    return static_cast<DitherMode>(
        std::clamp(static_cast<int>(persistedValue), 0,
                   static_cast<int>(DitherMode::Noise)));
}

class DitherNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe() {
        return NodeDescriptorBuilder{"dither", 1, "Dither / Halftone", "Filter"}
            .input("image", "Image", SocketContract::AnyImageValue)
            .output("image", "Image", SocketContract::AnyImageValue)
            .enumParameter("mode", "Mode", 0,
                {"Threshold", "Bayer 2x2", "Bayer 4x4", "Bayer 8x8",
                 "Halftone", "Noise"})
            .integerParameter("levels", "Levels", 2, 2, 16)
            .integerParameter("patternSize", "Pattern Size", 8, 2, 64)
            .typePolicy("image", SocketDescriptor::TypePolicy::PreserveInput, {"image"})
            .lowerable()
            .producedField()
            .build();
    }

    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    std::string shaderVariantKey(const nlohmann::json& parameters) const override {
        return "mode=" + std::to_string(
            static_cast<int>(clampMode(parameter(parameters, "mode", 0.0F))));
    }

    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto mode = clampMode(parameter(parameters_, "mode", 0.0F));
        const auto value = context.input("image", "image", 0.0F);
        const auto helper = context.helper("dither", R"GLSL(
float ditherQuantize(float value,float threshold,float levels){
    float count=max(round(levels),2.0);
    float scaled=clamp(value,0.0,1.0)*(count-1.0);
    float result=floor(scaled)+step(threshold,fract(scaled));
    return clamp(result,0.0,count-1.0)/max(count-1.0,1.0);
}
float ditherThreshold(float value){return step(0.5,clamp(value,0.0,1.0));}
int ditherBayerCell(int x,int y){
    return x==0?(y==0?0:3):(y==0?2:1);
}
int ditherBayerIndex(ivec2 pixel,int bitCount){
    int x=pixel.x;
    int y=pixel.y;
    int result=0;
    for(int bit=0;bit<3;bit++){
        if(bit>=bitCount)break;
        // The least-significant coordinate bits form the most-significant
        // Bayer digits, matching the canonical 4x4 matrix orientation.
        result=result*4+ditherBayerCell(x&1,y&1);
        x>>=1;
        y>>=1;
    }
    return result;
}
float ditherBayer(float value,ivec2 pixel,float levels,int bitCount){
    int side=1<<bitCount;
    float threshold=(float(ditherBayerIndex(pixel,bitCount))+0.5)/float(side*side);
    return ditherQuantize(value,threshold,levels);
}
float ditherHalftone(float value,vec2 pixel,float cellSize){
    float size=max(round(cellSize),2.0);
    vec2 local=fract((pixel+vec2(0.5))/size)-vec2(0.5);
    float radius=sqrt(clamp(value,0.0,1.0)*0.5);
    return step(length(local),radius);
}
uint ditherHash(uvec2 pixel){
    uint value=pixel.x*0x8da6b343u^pixel.y*0xd8163841u;
    value^=value>>16u;
    value*=0x85ebca6bu;
    value^=value>>13u;
    value*=0xc2b2ae35u;
    return value^(value>>16u);
}
float ditherNoise(float value,ivec2 pixel,float levels){
    float threshold=float(ditherHash(uvec2(uint(pixel.x),uint(pixel.y)))>>8u)/16777216.0;
    return ditherQuantize(value,threshold,levels);
}
)GLSL");

        const std::string pixel = "ivec2(floor(uv/pixelSize))";
        const auto pixelVector = "vec2(" + pixel + ")";
        ShaderValue levels;
        ShaderValue patternSize;
        if (mode != DitherMode::Threshold)
            levels = context.scalar("levels", "levels", 2.0F);
        if (mode == DitherMode::Halftone)
            patternSize = context.scalar("patternSize", "patternSize", 8.0F);

        const auto apply = [&](const std::string& channel, const std::string& offset) {
            switch (mode) {
            case DitherMode::Threshold:
                return helper + "Threshold(" + channel + ")";
            case DitherMode::Bayer2x2:
                return helper + "Bayer(" + channel + "," + pixel + "," + levels.name + ",1)";
            case DitherMode::Bayer4x4:
                return helper + "Bayer(" + channel + "," + pixel + "," + levels.name + ",2)";
            case DitherMode::Bayer8x8:
                return helper + "Bayer(" + channel + "," + pixel + "," + levels.name + ",3)";
            case DitherMode::Halftone:
                return helper + "Halftone(" + channel + "," + pixelVector + "+" + offset + "," +
                    patternSize.name + ")";
            case DitherMode::Noise:
                return helper + "Noise(" + channel + "," + pixel + "," + levels.name + ")";
            }
            return channel;
        };

        std::string expression;
        switch (value.type) {
        case ShaderValueType::Scalar:
            expression = apply(value.name, "vec2(0.0)");
            break;
        case ShaderValueType::Vec2:
            expression = "vec2(" + apply(value.name + ".x", "vec2(0.0)") + "," +
                apply(value.name + ".y", "vec2(0.0)") + ")";
            break;
        case ShaderValueType::Vec4:
            // Preserve alpha. For color halftones, slight screen offsets keep
            // the RGB dots from collapsing into one monochrome screen.
            const auto redOffset = "vec2(0.0)";
            const auto greenOffset = mode == DitherMode::Halftone
                ? "vec2(" + patternSize.name + "*0.33," + patternSize.name + "*0.17)"
                : "vec2(0.0)";
            const auto blueOffset = mode == DitherMode::Halftone
                ? "vec2(" + patternSize.name + "*0.67," + patternSize.name + "*0.41)"
                : "vec2(0.0)";
            expression = "vec4(" + apply(value.name + ".r", redOffset) + "," +
                apply(value.name + ".g", greenOffset) + "," +
                apply(value.name + ".b", blueOffset) + "," + value.name + ".a)";
            break;
        }
        (void)context.emitTyped(std::move(expression), value.type, "image");
        return true;
    }
};

} // namespace

void registerDitherNode(NodeRegistry& registry) {
    registry.add(DitherNode::describe(), [] { return std::make_unique<DitherNode>(); });
}

} // namespace reaction
