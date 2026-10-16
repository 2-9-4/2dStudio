#include "node_support.hpp"

#include "reaction/core/node_builder.hpp"
#include "procedural_node_support.hpp"

#include "reaction/gpu/shader_ir.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <vector>

namespace reaction {
namespace {

using node_support::parameter;

// Persisted as the zero-based label index of the Operation enum. Append new
// modes; never reorder. These values are serialized in project files and the
// generated-shader specialization key.
enum class CompareMode : int {
    LessThan = 0,
    LessEqual = 1,
    GreaterThan = 2,
    GreaterEqual = 3,
    ApproximatelyEqual = 4,
    NotApproximatelyEqual = 5,
    BetweenInclusive = 6,
    BetweenExclusive = 7,
    And = 8,
    Or = 9,
    Xor = 10,
    Not = 11,
};

CompareMode clampMode(float persistedValue) {
    return static_cast<CompareMode>(
        std::clamp(static_cast<int>(persistedValue), 0, static_cast<int>(CompareMode::Not)));
}

class CompareNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe() {
        return NodeDescriptorBuilder{"compare", 1, "Compare", "Math"}
            .optionalInput("a", "A", SocketContract::Numeric)
            .optionalInput("b", "B", SocketContract::Numeric)
            .optionalInput("c", "C", SocketContract::Numeric)
            .output("result", "Result", SocketContract::Numeric)
            .enumParameter("mode", "Operation", 0,
                {"Less Than", "Less Than Equal", "Greater Than", "Greater Than Equal",
                 "Approximately Equal", "Not Approximately Equal", "Between Inclusive",
                 "Between Exclusive", "AND", "OR", "XOR", "NOT"})
            .floatParameter("a", "A", 0.0F, -10.0F, 10.0F)
            .floatParameter("b", "B", 0.0F, -10.0F, 10.0F)
            .floatParameter("c", "C", 0.0F, -10.0F, 10.0F)
            .floatParameter("epsilon", "Epsilon", 1.0e-4F, 0.0F, 1.0F)
            .lowerable()
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
        const auto reduce = [&](const ShaderValue& value) {
            return convertShaderValue(value, ShaderValueType::Scalar);
        };
        const auto truthy = [&](const ShaderValue& value) {
            return "(abs(" + reduce(value) + ")>" + context.parameter("epsilon", 1.0e-4F).name + ")";
        };
        std::string expression;
        switch (mode) {
        case CompareMode::LessThan:
        case CompareMode::LessEqual:
        case CompareMode::GreaterThan:
        case CompareMode::GreaterEqual:
        case CompareMode::ApproximatelyEqual:
        case CompareMode::NotApproximatelyEqual: {
            const auto a = reduce(context.input("a", "a", 0.0F));
            const auto b = reduce(context.input("b", "b", 0.0F));
            const auto eps = context.parameter("epsilon", 1.0e-4F).name;
            switch (mode) {
            case CompareMode::LessThan: expression = "float(" + a + "<" + b + ")"; break;
            case CompareMode::LessEqual: expression = "float(" + a + "<=" + b + ")"; break;
            case CompareMode::GreaterThan: expression = "float(" + a + ">" + b + ")"; break;
            case CompareMode::GreaterEqual: expression = "float(" + a + ">=" + b + ")"; break;
            case CompareMode::ApproximatelyEqual:
                expression = "float(abs(" + a + "-" + b + ")<=" + eps + ")"; break;
            case CompareMode::NotApproximatelyEqual:
                expression = "float(abs(" + a + "-" + b + ")>" + eps + ")"; break;
            default: break;
            }
            break;
        }
        case CompareMode::BetweenInclusive:
        case CompareMode::BetweenExclusive: {
            const auto a = reduce(context.input("a", "a", 0.0F));
            const auto b = reduce(context.input("b", "b", 0.0F));
            const auto c = reduce(context.input("c", "c", 0.0F));
            expression = mode == CompareMode::BetweenInclusive
                ? "float(" + b + "<=" + a + "&&" + a + "<=" + c + ")"
                : "float(" + b + "<" + a + "&&" + a + "<" + c + ")";
            break;
        }
        case CompareMode::And:
            expression = "float(" + truthy(context.input("a", "a", 0.0F)) + "&&" +
                         truthy(context.input("b", "b", 0.0F)) + ")";
            break;
        case CompareMode::Or:
            expression = "float(" + truthy(context.input("a", "a", 0.0F)) + "||" +
                         truthy(context.input("b", "b", 0.0F)) + ")";
            break;
        case CompareMode::Xor:
            expression = "float((" + truthy(context.input("a", "a", 0.0F)) + ")!=" +
                         "(" + truthy(context.input("b", "b", 0.0F)) + "))";
            break;
        case CompareMode::Not:
            expression = "float(!(" + truthy(context.input("a", "a", 0.0F)) + "))";
            break;
        }
        (void)context.emitTyped(std::move(expression), ShaderValueType::Scalar, "result");
        return true;
    }
};

} // namespace

void registerCompareNode(NodeRegistry& registry) {
    registry.add(CompareNode::describe(), [] { return std::make_unique<CompareNode>(); });
}

} // namespace reaction
