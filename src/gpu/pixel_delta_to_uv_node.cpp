#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "reaction/core/node_builder.hpp"
#include "reaction/gpu/shader_ir.hpp"
#include <memory>

namespace reaction {
namespace {
class PixelDeltaToUvNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe() {
        return NodeDescriptorBuilder{"pixel_delta_to_uv",1,"Pixel Delta to UV","Coordinates"}
            .input("delta","Pixel Delta",SocketContract::VectorNumeric)
            .output("uvDelta","UV Delta",SocketContract::VectorNumeric)
            .typePolicy("uvDelta",SocketDescriptor::TypePolicy::VectorPromotion,{"delta"})
            .fieldIf("uvDelta",{"delta"})
            .lowerable()
            .build();
    }
    const NodeDescriptor& descriptor() const override {
        static const auto descriptor=describe();return descriptor;
    }
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto delta=context.vector("delta","delta",0.0F);
        (void)context.emitTyped("("+delta.name+")*pixelSize",ShaderValueType::Vec2,"uvDelta");
        return true;
    }
};
}
void registerPixelDeltaToUvNode(NodeRegistry& registry) {
    registry.add(PixelDeltaToUvNode::describe(),
                 [] { return std::make_unique<PixelDeltaToUvNode>(); });
}
}
