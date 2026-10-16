#include "nodes_internal.hpp"

#include "reaction/gpu/shader_ir.hpp"
#include "node_support.hpp"

#include "reaction/core/node_builder.hpp"

#include <memory>

namespace reaction {
namespace {

// Float graph values can represent every integer through 2^24 exactly, so the
// mask intentionally exposes bits 0 through 23 rather than pretending to be a
// full 32-bit integer value.
constexpr float kMaximumMask = 16777215.0F;
constexpr float kMaximumBit = 23.0F;

class BitTestNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe() {
        return NodeDescriptorBuilder{"bit_test", 1, "Bit Test / Integer Mask", "Math"}
            .optionalInput("mask", "Mask", ValueType::Float)
            .optionalInput("bit", "Bit", SocketContract::Numeric)
            .output("result", "Result", SocketContract::Numeric)
            .integerParameter("mask", "Mask", 0, 0, static_cast<int>(kMaximumMask))
            .integerParameter("bit", "Bit", 0, 0, static_cast<int>(kMaximumBit))
            .lowerable()
            .build();
    }

    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto mask = context.scalar("mask", "mask", 0.0F);
        const auto bit = context.scalar("bit", "bit", 0.0F);
        const auto helper = context.helper("integerMask", R"GLSL(
float integerMaskSanitize(float value){
    return (isnan(value)||isinf(value)) ? 0.0 : clamp(round(value),0.0,16777215.0);
}
float integerMaskTest(float mask,float bit){
    if(isnan(bit)||isinf(bit))return 0.0;
    float index=round(bit);
    if(index<0.0||index>23.0)return 0.0;
    uint bits=uint(integerMaskSanitize(mask));
    return float((bits>>uint(index))&1u);
}
)GLSL");
        (void)context.emitTyped(helper + "Test(" + mask.name + "," + bit.name + ")",
                                ShaderValueType::Scalar, "result");
        return true;
    }
};

} // namespace

void registerBitTestNode(NodeRegistry& registry) {
    registry.add(BitTestNode::describe(), [] { return std::make_unique<BitTestNode>(); });
}

} // namespace reaction
