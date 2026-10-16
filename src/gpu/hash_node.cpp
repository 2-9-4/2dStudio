#include "nodes_internal.hpp"
#include "node_support.hpp"

#include "reaction/core/node_builder.hpp"

#include "reaction/gpu/shader_ir.hpp"

#include <algorithm>

namespace reaction {
namespace {

using node_support::parameter;

enum class HashPositionMode : int { CanvasSpace, PixelSpace };

HashPositionMode hashPositionMode(float persistedValue) {
    return static_cast<HashPositionMode>(std::clamp(static_cast<int>(persistedValue), 0, 1));
}

class HashNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe() {
        return NodeDescriptorBuilder{"hash", 1, "Deterministic Hash", "Math"}
            .optionalInput("position", "Position", SocketContract::VectorNumeric)
            .optionalInput("seed", "Seed", ValueType::Float)
            .optionalInput("salt", "Salt", ValueType::Float)
            .output("scalar", "Scalar", SocketContract::Numeric)
            .output("vector", "Vector", SocketContract::VectorNumeric)
            .enumParameter("inputMode", "Input Handling", 0,
                           {"Canvas Space", "Pixel Space"})
            .floatParameter("seed", "Seed", 0.0F, -1000000.0F, 1000000.0F)
            .floatParameter("salt", "Salt", 0.0F, -1000000.0F, 1000000.0F)
            .typePolicy("scalar", SocketDescriptor::TypePolicy::NumericPromotion,
                        {"position"})
            .typePolicy("vector", SocketDescriptor::TypePolicy::VectorPromotion,
                        {"position"})
            .lowerable()
            .build();
    }

    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    std::string shaderVariantKey(const nlohmann::json& parameters) const override {
        return "inputMode=" + std::to_string(static_cast<int>(
            hashPositionMode(parameter(parameters, "inputMode", 0.0F))));
    }

    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto position = context.vector("position", "position", 0.0F);
        const auto seed = context.scalar("seed", "seed", 0.0F);
        const auto salt = context.scalar("salt", "salt", 0.0F);
        const auto mode = hashPositionMode(parameter(parameters_, "inputMode", 0.0F));
        const auto positionExpression = mode == HashPositionMode::CanvasSpace
            ? "uvec2(floatBitsToUint(" + position.name + ".x),floatBitsToUint(" +
                position.name + ".y))"
            : "uvec2(ivec2(floor(" + position.name + ")))";
        const auto seedExpression = "uint(int(floor(" + seed.name + ")))";
        const auto saltExpression = "uint(int(floor(" + salt.name + ")))";

        // All operations are uint arithmetic, whose overflow is defined modulo 2^32 in GLSL.
        // Using 24 high-quality bits avoids float rounding up to 1.0.
        const auto helper = context.helper("deterministicHash", R"GLSL(
uint deterministicHash_mix(uint value){
    value+=0x9e3779b9u;
    value=(value^(value>>16u))*0x85ebca6bu;
    value=(value^(value>>13u))*0xc2b2ae35u;
    return value^(value>>16u);
}
uint deterministicHash_value(uvec2 position,uint seed,uint salt,uint stream){
    uint value=position.x*0x8da6b343u;
    value^=position.y*0xd8163841u;
    value^=seed*0xcb1ab31fu;
    value^=salt*0x165667b1u;
    value^=stream*0x27d4eb2du;
    return deterministicHash_mix(value);
}
float deterministicHash_unit(uvec2 position,uint seed,uint salt,uint stream){
    return float(deterministicHash_value(position,seed,salt,stream)>>8u)*(1.0/16777216.0);
}
)GLSL");
        const auto args = positionExpression + "," + seedExpression + "," + saltExpression;
        (void)context.emitTyped(helper + "_unit(" + args + ",0u)", ShaderValueType::Scalar,
                                "scalar");
        (void)context.emitTyped("vec2(" + helper + "_unit(" + args + ",1u)," +
                                    helper + "_unit(" + args + ",2u))",
                                ShaderValueType::Vec2, "vector");
        return true;
    }
};

} // namespace

void registerHashNode(NodeRegistry& registry) {
    registry.add(HashNode::describe(), [] { return std::make_unique<HashNode>(); });
}

} // namespace reaction
