#include "node_support.hpp"
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

using node_support::TextureNode;
using node_support::parameter;
using node_support::uniform;
using procedural::NumericInput;
using procedural::bindNumericInput;
using procedural::kGlslBroadcastHelpers;
using procedural::numericParameterInput;

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

constexpr std::string_view kCompareShader = R"GLSL(#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba16f,binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D imageA;
layout(binding=1) uniform sampler2D imageB;
layout(binding=2) uniform sampler2D imageC;
uniform int hasA,hasB,hasC,mode; uniform float scalarA,scalarB,scalarC,epsilon;
float compareTrue(float x){return x>0.0?1.0:0.0;}
float compare(float a,float b,float c){
  if(mode==0)return compareTrue(b-a);
  if(mode==1)return a<=b?1.0:0.0;
  if(mode==2)return compareTrue(a-b);
  if(mode==3)return a>=b?1.0:0.0;
  if(mode==4)return compareTrue(epsilon-abs(a-b));
  if(mode==5)return compareTrue(abs(a-b)-epsilon);
  if(mode==6)return b<=a&&a<=c?1.0:0.0;
  if(mode==7)return b<a&&a<c?1.0:0.0;
  if(mode==8)return abs(a)>epsilon&&abs(b)>epsilon?1.0:0.0;
  if(mode==9)return abs(a)>epsilon||abs(b)>epsilon?1.0:0.0;
  if(mode==10)return (abs(a)>epsilon)!=(abs(b)>epsilon)?1.0:0.0;
  return abs(a)<=epsilon?1.0:0.0;
}
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outputImage);if(any(greaterThanEqual(p,s)))return;vec2 uv=(vec2(p)+.5)/vec2(s);
  float a=hasA!=0?texture(imageA,uv).r:scalarA,b=hasB!=0?texture(imageB,uv).r:scalarB,c=hasC!=0?texture(imageC,uv).r:scalarC;
  imageStore(outputImage,p,vec4(vec3(compare(a,b,c)),1.0));})GLSL";

CompareMode clampMode(float persistedValue) {
    return static_cast<CompareMode>(
        std::clamp(static_cast<int>(persistedValue), 0, static_cast<int>(CompareMode::Not)));
}

// A truthy assertion for logic modes: abs(value) > epsilon. Approximately Equal
// uses the same tolerance so "==~ b" and "not(abs(b-a) <= epsilon)" agree.
float toBool(float value, float epsilon) {
    return std::abs(value) > epsilon ? 1.0F : 0.0F;
}

float applyCompare(CompareMode mode, float a, float b, float c, float epsilon) {
    switch (mode) {
    case CompareMode::LessThan: return a < b ? 1.0F : 0.0F;
    case CompareMode::LessEqual: return a <= b ? 1.0F : 0.0F;
    case CompareMode::GreaterThan: return a > b ? 1.0F : 0.0F;
    case CompareMode::GreaterEqual: return a >= b ? 1.0F : 0.0F;
    case CompareMode::ApproximatelyEqual: return std::abs(a - b) <= epsilon ? 1.0F : 0.0F;
    case CompareMode::NotApproximatelyEqual: return std::abs(a - b) > epsilon ? 1.0F : 0.0F;
    case CompareMode::BetweenInclusive: return (a >= b && a <= c) ? 1.0F : 0.0F;
    case CompareMode::BetweenExclusive: return (a > b && a < c) ? 1.0F : 0.0F;
    case CompareMode::And: return toBool(a, epsilon) * toBool(b, epsilon);
    case CompareMode::Or: {
        const float ta = toBool(a, epsilon), tb = toBool(b, epsilon);
        return ta + tb - ta * tb;
    }
    case CompareMode::Xor: return toBool(a, epsilon) != toBool(b, epsilon) ? 1.0F : 0.0F;
    case CompareMode::Not: return toBool(a, epsilon) == 0.0F ? 1.0F : 0.0F;
    }
    return 0.0F;
}

class CompareNode final : public TextureNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"compare", 1, "Compare", "Math",
            {{"a", "A", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"b", "B", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"c", "C", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"result", "Result", ValueType::AnyNumeric, SocketDirection::Output}},
            {{"mode", "Operation", 0.0F, 0.0F, 11.0F, ParameterDescriptor::Control::Enum,
              {"Less Than", "Less Than Equal", "Greater Than", "Greater Than Equal",
               "Approximately Equal", "Not Approximately Equal", "Between Inclusive",
               "Between Exclusive", "AND", "OR", "XOR", "NOT"}},
             {"a", "A", 0.0F, -10.0F, 10.0F},
             {"b", "B", 0.0F, -10.0F, 10.0F},
             {"c", "C", 0.0F, -10.0F, 10.0F},
             {"epsilon", "Epsilon", 1.0e-4F, 0.0F, 1.0F}}};
        result.lowerable = true;
        return result;
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
        const auto type = context.valueType();
        const auto reduce = [&](const ShaderValue& value) {
            return value.type == ShaderValueType::Scalar ? value.name : "(" + value.name + ").r";
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
        const ShaderValue result = context.emit(std::move(expression));
        if (type == ShaderValueType::Vec4)
            (void)context.emit(
                "vec4(vec3(" + result.name + "),1.0)", "result");
        else if (result.type != type)
            (void)context.emit(result.name, "result");
        return true;
    }
    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        const auto epsilon = parameter(parameters_, "epsilon", 1.0e-4F);
        const auto mode = clampMode(parameter(parameters_, "mode", 0.0F));
        const NumericInput a = numericParameterInput(inputs, 0, parameters_, "a", 0.0F);
        const NumericInput b = numericParameterInput(inputs, 1, parameters_, "b", 0.0F);
        const NumericInput c = numericParameterInput(inputs, 2, parameters_, "c", 0.0F);
        if (!a.isField() && !b.isField() && !c.isField()) {
            outputs[0] = applyCompare(mode, a.constant, b.constant, c.constant, epsilon);
            return;
        }
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kCompareShader, "Compare / compute");
        glUseProgram(program_);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        bindNumericInput(program_, 0, "imageA", "hasA", "scalarA", a);
        bindNumericInput(program_, 1, "imageB", "hasB", "scalarB", b);
        bindNumericInput(program_, 2, "imageC", "hasC", "scalarC", c);
        uniform(program_, "mode", static_cast<int>(mode));
        uniform(program_, "epsilon", epsilon);
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

} // namespace

void registerCompareNode(NodeRegistry& registry) {
    registry.add(CompareNode::describe(), [] { return std::make_unique<CompareNode>(); });
}

} // namespace reaction