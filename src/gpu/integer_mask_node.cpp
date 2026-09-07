#include "nodes_internal.hpp"

#include "reaction/gpu/shader_ir.hpp"
#include "node_support.hpp"

#include <memory>

namespace reaction {
namespace {

class IntegerMaskNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"integer_mask", 1, "Integer Mask", "Math",
            {{"value", "Value", ValueType::Float, SocketDirection::Input, true},
             {"value", "Value", ValueType::Float, SocketDirection::Output}},
            {{"value", "Value", 0.0F, 0.0F, 16777215.0F,
              ParameterDescriptor::Control::Integer}}};
        result.lowerable = true;
        return result;
    }

    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    bool lowerShader(ShaderLoweringContext& context) const override {
        (void)context.emitTyped(context.scalar("value", "value", 0.0F).name,
                                ShaderValueType::Scalar, "value");
        return true;
    }
};

} // namespace

void registerIntegerMaskNode(NodeRegistry& registry) {
    registry.add(IntegerMaskNode::describe(), [] { return std::make_unique<IntegerMaskNode>(); });
}

} // namespace reaction
