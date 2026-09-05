#include "reaction/gpu/gpu_runtime.hpp"
#include "reaction/core/math.hpp"
#include "reaction/core/vector_math.hpp"
#include "reaction/gpu/shader_ir.hpp"
#include "nodes_internal.hpp"
#include "node_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace reaction {
namespace {

using node_support::ParameterNode;
using node_support::TextureNode;
using node_support::bindTexture;
using node_support::floatAt;
using node_support::imageAt;
using node_support::parameter;
using node_support::uniform;

std::string shaderTypeName(ShaderValueType type) {
    switch (type) {
    case ShaderValueType::Scalar: return "float";
    case ShaderValueType::Vec2: return "vec2";
    case ShaderValueType::Vec4: return "vec4";
    }
    return "float";
}

class PerlinNode final : public ParameterNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"perlin", 1, "Perlin Noise", "Generator",
            {{"image", "Image", ValueType::AnyNumeric, SocketDirection::Output}},
            {{"seed", "Seed", 1, 0, 10000}, {"scale", "Scale", 6, 0.05F, 100},
             {"octaves", "Octaves", 4, 1, 8}, {"persistence", "Persistence", 0.5F, 0, 1},
             {"lacunarity", "Lacunarity", 2, 1, 4}, {"speed", "Speed", 0.08F, -2, 2},
             {"offsetX", "Offset X", 0, -100, 100}, {"offsetY", "Offset Y", 0, -100, 100}}};
        result.timeDependent = true;
        result.lowerable = true;
        result.producedField = true;
        return result;
    }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto seed = context.parameter("seed", 1.0F);
        const auto scale = context.parameter("scale", 6.0F);
        const auto octaves = context.parameter("octaves", 4.0F);
        const auto persistence = context.parameter("persistence", 0.5F);
        const auto lacunarity = context.parameter("lacunarity", 2.0F);
        const auto speed = context.parameter("speed", 0.08F);
        const auto offsetX = context.parameter("offsetX", 0.0F);
        const auto offsetY = context.parameter("offsetY", 0.0F);
        const auto time = context.parameter("__time", 0.0F);
        std::string source =
            "uint perlinNoise_hash(uvec3 p,int seed){uint h=p.x*374761393u+p.y*668265263u+p.z*1442695041u+uint(seed)*2246822519u;h=(h^(h>>13u))*1274126177u;return h^(h>>16u);}\n"
            "vec3 perlinNoise_gradient(ivec3 p,int seed){uint h=perlinNoise_hash(uvec3(p),seed);float z=float(h&65535u)/32767.5-1.0;float a=float(h>>16u)*(6.28318530718/65535.0),r=sqrt(max(1.0-z*z,0.0));return vec3(r*cos(a),r*sin(a),z);}\n"
            "float perlinNoise_value(vec3 p,int seed){ivec3 c=ivec3(floor(p));vec3 f=fract(p),u=f*f*f*(f*(f*6.0-15.0)+10.0);float n000=dot(perlinNoise_gradient(c+ivec3(0,0,0),seed),f-vec3(0,0,0)),n100=dot(perlinNoise_gradient(c+ivec3(1,0,0),seed),f-vec3(1,0,0)),n010=dot(perlinNoise_gradient(c+ivec3(0,1,0),seed),f-vec3(0,1,0)),n110=dot(perlinNoise_gradient(c+ivec3(1,1,0),seed),f-vec3(1,1,0)),n001=dot(perlinNoise_gradient(c+ivec3(0,0,1),seed),f-vec3(0,0,1)),n101=dot(perlinNoise_gradient(c+ivec3(1,0,1),seed),f-vec3(1,0,1)),n011=dot(perlinNoise_gradient(c+ivec3(0,1,1),seed),f-vec3(0,1,1)),n111=dot(perlinNoise_gradient(c+ivec3(1,1,1),seed),f-vec3(1,1,1));return mix(mix(mix(n000,n100,u.x),mix(n010,n110,u.x),u.y),mix(mix(n001,n101,u.x),mix(n011,n111,u.x),u.y),u.z);}\n"
            "float perlinNoise_fbm(vec3 p,int seed,float octaves,float persistence,float lacunarity){float sum=0.0,amplitude=1.0,norm=0.0;for(int i=0;i<8;i++){if(float(i)>=octaves)break;sum+=perlinNoise_value(p,seed)*amplitude;norm+=amplitude;amplitude*=persistence;p*=lacunarity;}return clamp(0.5+0.85*sum/max(norm,0.0001),0.0,1.0);}\n";
        const auto helper = context.helper("perlinNoise", std::move(source));
        const auto expression = helper + "_fbm(vec3(uv*" + scale.name + "+vec2(" +
            offsetX.name + "," + offsetY.name + ")," + time.name + "*" + speed.name +
            "),int(" + seed.name + ")," + octaves.name + "," + persistence.name + "," +
            lacunarity.name + ")";
        (void)context.emitTyped(expression, ShaderValueType::Scalar, "image");
        return true;
    }
};

class MathNode final : public ParameterNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"math", 1, "Math", "Math",
            {{"a", "A", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"b", "B", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"c", "C", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"result", "Result", ValueType::AnyNumeric, SocketDirection::Output}},
            {{"operation", "Operation", 0, 0, 30, ParameterDescriptor::Control::Enum,
              {"Add", "Subtract", "Multiply", "Divide", "Power", "Minimum",
               "Maximum", "Absolute", "Sine", "Cosine", "Clamp", "Remap",
               "Floor", "Ceil", "Round", "Fraction", "Square Root", "Exp",
               "Natural Log", "Log2", "Sign", "Tangent", "Arc Sine", "Arc Cosine",
               "Arc Tangent", "Modulo", "Arc Tangent 2", "Step", "Hypotenuse",
               "Smoothstep", "Multiply Accumulate"}},
             {"a", "A", 0, -10, 10}, {"b", "B", 0, -10, 10},
             {"c", "C", 1, -10, 10}, {"inMin", "Input Min", 0, -10, 10}, {"inMax", "Input Max", 1, -10, 10},
             {"outMin", "Output Min", 0, -10, 10}, {"outMax", "Output Max", 1, -10, 10}}};
        result.lowerable = true;
        return result;
    }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    std::string shaderVariantKey(const nlohmann::json& parameters) const override {
        return "operation=" + std::to_string(
            static_cast<int>(mathOperation(parameter(parameters, "operation", 0))));
    }
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto operation = mathOperation(parameter(parameters_, "operation", 0));
        static constexpr std::array<const char*, 3> sockets{"a", "b", "c"};
        static constexpr std::array<float, 3> defaults{0.0F, 0.0F, 1.0F};
        std::vector<ShaderValue> operands;
        for (int index = 0; index < mathOperationOperandCount(operation); ++index)
            operands.push_back(context.input(sockets[static_cast<std::size_t>(index)],
                sockets[static_cast<std::size_t>(index)], defaults[static_cast<std::size_t>(index)]));
        std::vector<ShaderValue> remap;
        if (operation == MathOperation::Remap) {
            remap.push_back(context.parameter("inMin", 0.0F));
            remap.push_back(context.parameter("inMax", 1.0F));
            remap.push_back(context.parameter("outMin", 0.0F));
            remap.push_back(context.parameter("outMax", 1.0F));
        }
        const auto type = promotedShaderType(operands);
        (void)context.emitTyped(mathGlslExpression(operation, operands, remap, type), type);
        return true;
    }
};

class VectorMathNode final : public ParameterNode {
public:
    static NodeDescriptor describe() { return vectorMathDescriptor(VectorMathOperation::Add); }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    std::string shaderVariantKey(const nlohmann::json& parameters) const override {
        return "operation=" + std::to_string(static_cast<int>(
            vectorMathOperation(parameter(parameters, "operation", 0.0F))));
    }
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto operation = vectorMathOperation(parameter(parameters_, "operation", 0.0F));
        const auto a = context.vector("a", "a", 0.0F);
        const auto b = [&] { return context.vector("b", "b", 0.0F); };
        const auto scalar = [&] { return context.scalar("scalar", "scalar", 0.0F).name; };
        const auto min = [&] { return context.scalar("min", "min", 0.0F).name; };
        const auto max = [&] { return context.scalar("max", "max", 1.0F).name; };
        const auto safeNormalize = [&](const std::string& value) {
            return "(" + value + "/max(length(" + value + "),1e-8))";
        };
        std::string expression;
        ShaderValueType type = ShaderValueType::Vec2;
        switch (operation) {
        case VectorMathOperation::Add: expression = a.name + "+" + b().name; break;
        case VectorMathOperation::Subtract: expression = a.name + "-" + b().name; break;
        case VectorMathOperation::MultiplyComponents: expression = a.name + "*" + b().name; break;
        case VectorMathOperation::DivideComponents: { const auto divisor = b(); expression = a.name + "/(mix(vec2(1.0),sign(" + divisor.name + "),step(vec2(1e-6),abs(" + divisor.name + ")))*max(abs(" + divisor.name + "),vec2(1e-6)))"; break; }
        case VectorMathOperation::Scale: expression = a.name + "*" + scalar(); break;
        case VectorMathOperation::Negate: expression = "-" + a.name; break;
        case VectorMathOperation::Absolute: expression = "abs(" + a.name + ")"; break;
        case VectorMathOperation::Minimum: expression = "min(" + a.name + "," + b().name + ")"; break;
        case VectorMathOperation::Maximum: expression = "max(" + a.name + "," + b().name + ")"; break;
        case VectorMathOperation::Clamp: expression = "clamp(" + a.name + ",vec2(" + min() + "),vec2(" + max() + "))"; break;
        case VectorMathOperation::Normalize: expression = "(" + a.name + "*(step(1e-8,length(" + a.name + "))/max(length(" + a.name + "),1e-8)))"; break;
        case VectorMathOperation::SetLength: expression = safeNormalize(a.name) + "*" + scalar() + "*step(1e-8,length(" + a.name + "))"; break;
        case VectorMathOperation::ClampLength: expression = safeNormalize(a.name) + "*clamp(length(" + a.name + ")," + min() + "," + max() + ")*step(1e-8,length(" + a.name + "))"; break;
        case VectorMathOperation::LimitLength: expression = safeNormalize(a.name) + "*min(length(" + a.name + ")," + scalar() + ")*step(1e-8,length(" + a.name + "))"; break;
        case VectorMathOperation::Rotate: { const auto angle = scalar(); expression = "mat2(cos(" + angle + "),sin(" + angle + "),-sin(" + angle + "),cos(" + angle + "))*" + a.name; break; }
        case VectorMathOperation::PerpendicularClockwise: expression = "vec2(-" + a.name + ".y," + a.name + ".x)"; break;
        case VectorMathOperation::PerpendicularCounterClockwise: expression = "vec2(" + a.name + ".y,-" + a.name + ".x)"; break;
        case VectorMathOperation::Reflect: { const auto normal = b(); expression = a.name + "-2.0*dot(" + a.name + "," + normal.name + "/max(length(" + normal.name + "),1e-8))*" + normal.name + "/max(length(" + normal.name + "),1e-8)*step(1e-8,length(" + normal.name + "))"; break; }
        case VectorMathOperation::Project: { const auto axis = b(); expression = axis.name + "*dot(" + a.name + "," + axis.name + ")/max(dot(" + axis.name + "," + axis.name + "),1e-8)"; break; }
        case VectorMathOperation::Reject: { const auto axis = b(); expression = a.name + "-" + axis.name + "*dot(" + a.name + "," + axis.name + ")/max(dot(" + axis.name + "," + axis.name + "),1e-8)"; break; }
        case VectorMathOperation::Mix: expression = "mix(" + a.name + "," + b().name + "," + scalar() + ")"; break;
        case VectorMathOperation::Length: expression = "length(" + a.name + ")"; type = ShaderValueType::Scalar; break;
        case VectorMathOperation::LengthSquared: expression = "dot(" + a.name + "," + a.name + ")"; type = ShaderValueType::Scalar; break;
        case VectorMathOperation::Distance: expression = "distance(" + a.name + "," + b().name + ")"; type = ShaderValueType::Scalar; break;
        case VectorMathOperation::DistanceSquared: { const auto d = "(" + a.name + "-" + b().name + ")"; expression = "dot(" + d + "," + d + ")"; type = ShaderValueType::Scalar; break; }
        case VectorMathOperation::Dot: expression = "dot(" + a.name + "," + b().name + ")"; type = ShaderValueType::Scalar; break;
        case VectorMathOperation::Cross: expression = a.name + ".x*" + b().name + ".y-" + a.name + ".y*" + b().name + ".x"; type = ShaderValueType::Scalar; break;
        case VectorMathOperation::Angle: expression = "atan(" + a.name + ".y," + a.name + ".x)"; type = ShaderValueType::Scalar; break;
        case VectorMathOperation::AngleBetween: { const auto other = b(); expression = "(step(1e-8,length(" + a.name + "))*step(1e-8,length(" + other.name + "))*atan(" + a.name + ".x*" + other.name + ".y-" + a.name + ".y*" + other.name + ".x,dot(" + a.name + "," + other.name + ")))"; type = ShaderValueType::Scalar; break; }
        case VectorMathOperation::MinimumComponent: expression = "min(" + a.name + ".x," + a.name + ".y)"; type = ShaderValueType::Scalar; break;
        case VectorMathOperation::MaximumComponent: expression = "max(" + a.name + ".x," + a.name + ".y)"; type = ShaderValueType::Scalar; break;
        case VectorMathOperation::SumComponents: expression = a.name + ".x+" + a.name + ".y"; type = ShaderValueType::Scalar; break;
        case VectorMathOperation::ProductComponents: expression = a.name + ".x*" + a.name + ".y"; type = ShaderValueType::Scalar; break;
        }
        (void)context.emitTyped(std::move(expression), type, "result");
        return true;
    }
};

class MixNode final : public ParameterNode {
public:
    static NodeDescriptor describe() { auto result = NodeDescriptor{"mix",1,"Mix","Color",
        {{"a","A",ValueType::AnyNumeric,SocketDirection::Input,true},{"b","B",ValueType::AnyNumeric,SocketDirection::Input,true},
         {"factor","Factor",ValueType::AnyNumeric,SocketDirection::Input,true},{"result","Result",ValueType::AnyNumeric,SocketDirection::Output}},
        {{"mode","Mode",0,0,9,ParameterDescriptor::Control::Enum,
          {"Mix", "Add", "Multiply", "Screen", "Overlay", "Difference", "Darken",
           "Lighten", "Color Dodge", "Color Burn"}},
         {"a","A",0,0,1},{"b","B",1,0,1},{"factor","Factor",0.5F,0,1}}}; result.lowerable=true; return result; }
    const NodeDescriptor& descriptor() const override { static const auto value=describe();return value; }
    std::string shaderVariantKey(const nlohmann::json& parameters) const override {
        return "mode=" + std::to_string(std::clamp(
            static_cast<int>(parameter(parameters, "mode", 0)), 0, 9));
    }
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto a = context.input("a", "a", 0.0F);
        const auto b = context.input("b", "b", 1.0F);
        const auto factorValue = context.scalar("factor", "factor", 0.5F);
        const auto type = promotedShaderType({a, b});
        const auto av = convertShaderValue(a, type), bv = convertShaderValue(b, type);
        const auto factor = factorValue.name;
        const int mode = std::clamp(static_cast<int>(parameter(parameters_, "mode", 0)), 0, 9);
        std::string blend = bv;
        switch (mode) {
        case 1: blend = "(" + av + "+" + bv + ")"; break;
        case 2: blend = "(" + av + "*" + bv + ")"; break;
        case 3: blend = "(" + av + "+" + bv + "-" + av + "*" + bv + ")"; break;
        case 4: blend = "mix(2.0*" + av + "*" + bv + ",1.0-2.0*(1.0-" + av + ")*(1.0-" + bv + "),step(" + shaderTypeName(type) + "(0.5)," + av + "))"; break;
        case 5: blend = "abs(" + av + "-" + bv + ")"; break;
        case 6: blend = "min(" + av + "," + bv + ")"; break;
        case 7: blend = "max(" + av + "," + bv + ")"; break;
        case 8: blend = "(" + av + "/max(" + shaderTypeName(type) + "(1e-6),1.0-" + bv + "))"; break;
        case 9: blend = "(1.0-(1.0-" + av + ")/max(" + shaderTypeName(type) + "(1e-6)," + bv + "))"; break;
        default: break;
        }
        const auto f = "clamp(" + factor + ",0.0,1.0)";
        if (type == ShaderValueType::Vec4)
            (void)context.emitTyped("vec4(mix((" + av + ").rgb,(" + blend + ").rgb," + f +
                               "),mix((" + av + ").a,(" + bv + ").a," + f + "))",
                               ShaderValueType::Vec4);
        else
            (void)context.emitTyped("mix(" + av + "," + blend + "," + f + ")", type);
        return true;
    }
};

class ThresholdNode final : public ParameterNode {
public:
    static NodeDescriptor describe() { auto result = NodeDescriptor{"threshold", 1, "Threshold", "Math",
        {{"value", "Value", ValueType::AnyNumeric, SocketDirection::Input, true},
         {"result", "Result", ValueType::AnyNumeric, SocketDirection::Output}},
        {{"value", "Value", 0.5F, 0, 1}, {"threshold", "Threshold", 0.5F, 0, 1}}}; result.lowerable=true; return result; }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto value = context.input("value", "value", 0.5F);
        const auto cutoff = context.parameter("threshold", 0.5F);
        (void)context.emitTyped("step(" + convertShaderValue(cutoff, value.type) + "," +
                               value.name + ")", value.type);
        return true;
    }
};

class SelectNode final : public ParameterNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"select", 1, "Select", "Logic",
                {{"condition", "Condition", ValueType::AnyNumeric, SocketDirection::Input, true},
                 {"ifTrue", "If True", ValueType::AnyNumeric, SocketDirection::Input, true},
                 {"ifFalse", "If False", ValueType::AnyNumeric, SocketDirection::Input, true},
                 {"result", "Result", ValueType::AnyNumeric, SocketDirection::Output}},
                {{"condition", "Condition", 0.0F, -10.0F, 10.0F},
                 {"ifTrue", "If True", 1.0F, -10.0F, 10.0F},
                 {"ifFalse", "If False", 0.0F, -10.0F, 10.0F}}};
        result.lowerable = true;
        return result;
    }
    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto condition = context.input("condition", "condition", 0.0F);
        const auto yes = context.input("ifTrue", "ifTrue", 1.0F);
        const auto no = context.input("ifFalse", "ifFalse", 0.0F);
        const auto type = promotedShaderType({yes, no});
        const auto scalarCondition = convertShaderValue(condition, ShaderValueType::Scalar);
        const auto expression = "((" + scalarCondition + ")!=0.0?" +
            convertShaderValue(yes, type) + ":" + convertShaderValue(no, type) + ")";
        (void)context.emitTyped(std::move(expression), type);
        return true;
    }
};

class CoordinatesNode final : public ParameterNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"coordinates", 1, "Canvas Coordinates", "Input",
                {{"coordinates", "Coordinates", ValueType::AnyVector, SocketDirection::Output}},
                {{"pixels", "Pixels", 0.0F, 0.0F, 1.0F,
                  ParameterDescriptor::Control::Boolean}}};
        result.lowerable = true;
        result.producedField = true;
        return result;
    }
    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }
    std::string shaderVariantKey(const nlohmann::json& parameters) const override {
        return "pixels=" + std::to_string(parameter(parameters, "pixels", 0.0F) > 0.5F);
    }
    bool lowerShader(ShaderLoweringContext& context) const override {
        const bool pixels = parameter(parameters_, "pixels", 0.0F) > 0.5F;
        (void)context.emitTyped(pixels ? "uv/pixelSize" : "uv", ShaderValueType::Vec2,
                                "coordinates");
        return true;
    }
};

class LaplacianNode final : public ParameterNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"laplacian", 1, "Laplacian", "Filter",
                {{"value", "Value", ValueType::AnyVector, SocketDirection::Input},
                 {"scale", "Scale", ValueType::Float, SocketDirection::Input, true},
                 {"result", "Result", ValueType::AnyVector, SocketDirection::Output}},
                {{"scale", "Scale", 1.0F, .25F, 8.0F}}};
        result.lowerable = true;
        result.neighborhoodSocket = "value";
        result.sockets[0].requiresImage = true;
        return result;
    }
    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto center = context.inputAt("value", "q", "value", 0.0F);
        const auto type = center.type;
        const auto zero = shaderTypeName(type) + "(0.0)";
        const auto sample = [&](std::string delta) {
            return context.inputAt("value", "q+pixel*" + std::move(delta), "value", 0.0F).name;
        };
        const auto scale = context.inputAt("scale", "q", "scale", 1.0F);
        std::string source = shaderTypeName(type) +
            " laplacianKernelAt(vec2 q,float r){vec2 pixel=pixelSize;return -" + center.name +
            "+0.2*(" + sample("vec2(-r,0.0)") + "+" + sample("vec2(r,0.0)") + "+" +
            sample("vec2(0.0,-r)") + "+" + sample("vec2(0.0,r)") + ")+0.05*(" +
            sample("vec2(-r,-r)") + "+" + sample("vec2(r,-r)") + "+" +
            sample("vec2(-r,r)") + "+" + sample("vec2(r,r)") + ");}\n" +
            shaderTypeName(type) + " laplacianKernel(vec2 q){float scale=" + scale.name +
            ";if(abs(scale-1.0)<0.001)return laplacianKernelAt(q,1.0);" + shaderTypeName(type) +
            " v=" + zero + ";for(int i=0;i<3;i++)v+=laplacianKernelAt(q,mix(1.0,scale,float(i)/2.0))/3.0;return v;}\n";
        const auto helper = context.helper("laplacianKernel", std::move(source));
        (void)context.emitTyped(helper + "(uv)", type);
        return true;
    }
};

class InvertNode final : public ParameterNode {
public:
    static NodeDescriptor describe() { auto result = NodeDescriptor{"invert", 1, "Image Invert", "Utility",
        {{"image", "Image", ValueType::Image2D, SocketDirection::Input},
         {"image", "Image", ValueType::Image2D, SocketDirection::Output}}, {}}; result.lowerable=true; return result; }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto value = context.color("image", "image", 0.0F);
        (void)context.emitTyped("vec4(vec3(1.0)-(" + value.name + ").rgb,(" + value.name + ").a)",
                                ShaderValueType::Vec4, "image");
        return true;
    }
};

class ColorRampNode final : public ParameterNode {
public:
    static NodeDescriptor describe(){auto result=NodeDescriptor{"color_ramp",1,"Color Ramp","Color",
        {{"value","Value",ValueType::AnyNumeric,SocketDirection::Input,true},{"image","Image",ValueType::Image2D,SocketDirection::Output}},
        {{"value","Value",0.5F,0,1},{"low","Low threshold",.02F,0,1},{"high","High threshold",.4F,0,1},
         {"r0","Start R",.015F,0,1},{"g0","Start G",.01F,0,1},{"b0","Start B",.04F,0,1},
         {"r1","End R",1,0,1},{"g1","End G",.35F,0,1},{"b1","End B",.08F,0,1}}};result.lowerable=true;return result;}
    const NodeDescriptor& descriptor()const override{static const auto value=describe();return value;}
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto value = context.scalar("value", "value", 0.5F);
        const auto low = context.parameter("low", 0.02F), high = context.parameter("high", 0.4F);
        const auto r0 = context.parameter("r0", .015F), g0 = context.parameter("g0", .01F), b0 = context.parameter("b0", .04F);
        const auto r1 = context.parameter("r1", 1.0F), g1 = context.parameter("g1", .35F), b1 = context.parameter("b1", .08F);
        const auto scalar = value.name;
        const auto t = "clamp((" + scalar + "-" + low.name + ")/max(" + high.name + "-" + low.name + ",1e-6),0.0,1.0)";
        (void)context.emitTyped("mix(vec4(" + r0.name + "," + g0.name + "," + b0.name + ",1.0),vec4(" +
                           r1.name + "," + g1.name + "," + b1.name + ",1.0)," + t + ")",
                           ShaderValueType::Vec4, "image");
        return true;
    }
};

constexpr std::string_view kReactionInit = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(rg16f,binding=0)writeonly uniform image2D stateOut;layout(binding=0)uniform sampler2D seedImage;uniform int hasSeed;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(stateOut);if(any(greaterThanEqual(p,s)))return;vec2 uv=(vec2(p)+.5)/vec2(s);float seed=hasSeed!=0?texture(seedImage,uv).r:step(length(uv-vec2(.5)),.075);imageStore(stateOut,p,vec4(1.0-seed*.5,seed,0,1));})GLSL";
constexpr std::string_view kReactionStep = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(rg16f,binding=0)writeonly uniform image2D stateOut;layout(binding=0)uniform sampler2D stateIn;
layout(binding=1)uniform sampler2D feedImage;layout(binding=2)uniform sampler2D killImage;uniform int hasFeed,hasKill;uniform float feed,kill,diffA,diffB,dt,structureScale;
vec2 sampleState(vec2 uv){return texture(stateIn,fract(uv)).rg;}
vec2 laplacian(vec2 uv,vec2 o){vec2 q=sampleState(uv),lap=-q;
lap+=.2*(sampleState(uv+vec2(-o.x,0))+sampleState(uv+vec2(o.x,0))+sampleState(uv+vec2(0,-o.y))+sampleState(uv+vec2(0,o.y)));
lap+=.05*(sampleState(uv+vec2(-o.x,-o.y))+sampleState(uv+vec2(o.x,-o.y))+sampleState(uv+vec2(-o.x,o.y))+sampleState(uv+vec2(o.x,o.y)));return lap;}
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(stateOut);if(any(greaterThanEqual(p,s)))return;
vec2 uv=(vec2(p)+.5)/vec2(s),pixelSize=1.0/vec2(s),q=sampleState(uv),lap;
if(abs(structureScale-1.0)<.001)lap=laplacian(uv,pixelSize);else{lap=vec2(0);for(int i=0;i<3;i++){float radius=mix(1.0,structureScale,float(i)/2.0);lap+=laplacian(uv,pixelSize*radius)/3.0;}}
float feedMask=1.0,killMask=1.0;
if(hasFeed!=0)feedMask=clamp(texture(feedImage,uv).r,0,1);if(hasKill!=0)killMask=clamp(texture(killImage,uv).r,0,1);
float f=feed*feedMask,k=kill*killMask,reaction=q.x*q.y*q.y;
float a=q.x+(diffA*lap.x-reaction+f*(1-q.x))*dt,b=q.y+(diffB*lap.y+reaction-(k+f)*q.y)*dt;imageStore(stateOut,p,vec4(clamp(a,0,1),clamp(b,0,1),0,1));})GLSL";
constexpr std::string_view kReactionOutput = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(rgba16f,binding=0)writeonly uniform image2D outImage;layout(binding=0)uniform sampler2D stateIn;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outImage);if(any(greaterThanEqual(p,s)))return;float b=texelFetch(stateIn,p,0).g;imageStore(outImage,p,vec4(b,b,b,1));})GLSL";

constexpr std::string_view kReactionCollapse = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(binding=0)uniform sampler2D stateIn;
layout(std430,binding=0)buffer CollapseState{uint activityFlag;};uniform float threshold;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=textureSize(stateIn,0);if(any(greaterThanEqual(p,s)))return;if(texelFetch(stateIn,p,0).g>threshold)atomicOr(activityFlag,1u);})GLSL";

class ReactionNode final : public TextureNode {
public:
    ~ReactionNode()override{if(state_[0])glDeleteTextures(2,state_.data());if(initProgram_)glDeleteProgram(initProgram_);if(outputProgram_)glDeleteProgram(outputProgram_);if(collapseProgram_)glDeleteProgram(collapseProgram_);if(collapseBuffer_)glDeleteBuffers(1,&collapseBuffer_);}
    static NodeDescriptor describe(){auto result=NodeDescriptor{"reaction_diffusion",1,"Reaction Diffusion (Monolithic)","Simulation",
        {{"feedMultiplier","Feed Multiplier",ValueType::AnyNumeric,SocketDirection::Input,true},{"killMultiplier","Kill Multiplier",ValueType::AnyNumeric,SocketDirection::Input,true},
         {"seed","Seed",ValueType::Image2D,SocketDirection::Input,true},{"image","Image",ValueType::Image2D,SocketDirection::Output}},
        {{"feed","Feed",.055F,0,.1F},{"kill","Kill",.062F,0,.1F},{"diffA","Diffusion A",1,0,2},{"diffB","Diffusion B",.5F,0,2},
         {"structureScale","Structure Scale",1,.25F,8},
         {"dt","Timestep",1,0.01F,2},{"iterations","Iterations",8,1,64,ParameterDescriptor::Control::Integer},
         {"autoReset","Auto Reset",0,0,1,ParameterDescriptor::Control::Boolean}}};result.timeDependent=true;result.stateful=true;return result;}
    const NodeDescriptor& descriptor()const override{static const auto value=describe();return value;}
    void reset(EvaluationContext&)override{resetPending_=true;collapseCheckCounter_=0;}
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,std::span<Value> outputs)override{
        auto& gpu=*static_cast<GpuRuntime*>(context.gpu);ensure(context);
        if(stateWidth_!=context.width||stateHeight_!=context.height){if(state_[0])glDeleteTextures(2,state_.data());state_[0]=gpu.createTexture(context.width,context.height,GL_RG16F);state_[1]=gpu.createTexture(context.width,context.height,GL_RG16F);stateWidth_=context.width;stateHeight_=context.height;resetPending_=true;}
        if (!initProgram_) initProgram_ = gpu.compileCompute(kReactionInit,"Reaction Diffusion (Monolithic) / initialize");
        if (!program_) program_ = gpu.compileCompute(kReactionStep,"Reaction Diffusion (Monolithic) / update");
        if (!outputProgram_) outputProgram_ = gpu.compileCompute(kReactionOutput,"Reaction Diffusion (Monolithic) / output");
        if (!collapseProgram_) collapseProgram_ = gpu.compileCompute(kReactionCollapse,"Reaction Diffusion (Monolithic) / collapse check");
        if (!collapseBuffer_) {glGenBuffers(1,&collapseBuffer_);glBindBuffer(GL_SHADER_STORAGE_BUFFER,collapseBuffer_);const GLuint zero=0;glBufferData(GL_SHADER_STORAGE_BUFFER,sizeof(zero),&zero,GL_DYNAMIC_READ);}
        const auto initialize=[&]{const auto seed=imageAt(inputs,2);glUseProgram(initProgram_);glBindImageTexture(0,state_[0],0,GL_FALSE,0,GL_WRITE_ONLY,GL_RG16F);if(seed)bindTexture(0,seed.texture);uniform(initProgram_,"hasSeed",seed?1:0);gpu.dispatch(initProgram_,context.width,context.height);index_=0;resetPending_=false;};
        if(resetPending_)initialize();
        if(context.playing){const auto feedImage=imageAt(inputs,0),killImage=imageAt(inputs,1);const auto feedMultiplier=floatAt(inputs,0,1),killMultiplier=floatAt(inputs,1,1);const int iterations=std::clamp(static_cast<int>(parameter(parameters_,"iterations",8)),1,64);
            for(int i=0;i<iterations;++i){const int next=1-index_;glUseProgram(program_);bindTexture(0,state_[index_]);if(feedImage)bindTexture(1,feedImage.texture);if(killImage)bindTexture(2,killImage.texture);
                glBindImageTexture(0,state_[next],0,GL_FALSE,0,GL_WRITE_ONLY,GL_RG16F);uniform(program_,"hasFeed",feedImage?1:0);uniform(program_,"hasKill",killImage?1:0);
                uniform(program_,"feed",parameter(parameters_,"feed",.055F)*feedMultiplier);uniform(program_,"kill",parameter(parameters_,"kill",.062F)*killMultiplier);uniform(program_,"diffA",parameter(parameters_,"diffA",1));uniform(program_,"diffB",parameter(parameters_,"diffB",.5F));uniform(program_,"structureScale",parameter(parameters_,"structureScale",1));uniform(program_,"dt",parameter(parameters_,"dt",1));
                gpu.dispatch(program_,context.width,context.height);index_=next;}}
        const bool autoReset=parameter(parameters_,"autoReset",0)>.5F;
        if(autoReset&&++collapseCheckCounter_>=8){collapseCheckCounter_=0;const GLuint zero=0;glBindBuffer(GL_SHADER_STORAGE_BUFFER,collapseBuffer_);glBufferSubData(GL_SHADER_STORAGE_BUFFER,0,sizeof(zero),&zero);glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,collapseBuffer_);glUseProgram(collapseProgram_);bindTexture(0,state_[index_]);uniform(collapseProgram_,"threshold",.001F);gpu.dispatch(collapseProgram_,context.width,context.height);GLuint active=0;glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,sizeof(active),&active);if(active==0)initialize();}
        glUseProgram(outputProgram_);bindTexture(0,state_[index_]);glBindImageTexture(0,texture_,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F);gpu.dispatch(outputProgram_,context.width,context.height);outputs[0]=ImageHandle{texture_,context.width,context.height};
    }
private:std::array<GLuint,2> state_{};GLuint initProgram_=0,outputProgram_=0,collapseProgram_=0,collapseBuffer_=0;int stateWidth_=0,stateHeight_=0,index_=0,collapseCheckCounter_=0;bool resetPending_=true;
};

class OutputNode final : public ParameterNode {
public:
    static NodeDescriptor describe(){return {"output",1,"Output","Output",{{"image","Image",ValueType::Image2D,SocketDirection::Input},{"image","Image",ValueType::Image2D,SocketDirection::Output}},{}};}
    const NodeDescriptor& descriptor()const override{static const auto value=describe();return value;}
    void evaluate(EvaluationContext&,std::span<const Value> inputs,std::span<Value> outputs)override{
        // A generated scalar/vector producer normally materializes before this
        // image-only boundary. Keep an unexpected constant from escaping as an
        // invalid image value if that boundary cannot be generated.
        outputs[0] = !inputs.empty() && std::holds_alternative<ImageHandle>(inputs[0])
            ? inputs[0] : Value{};
    }
};

template <typename T> void addNode(NodeRegistry& registry) {
    registry.add(T::describe(), [] { return std::make_unique<T>(); });
}

} // namespace

void registerBuiltInNodes(NodeRegistry& registry) {
    registerInputNodes(registry);
    registerTextureSampleNode(registry);
    registerTransform2DNode(registry);
    registerDomainWarpNodes(registry);
    registerGradientNodes(registry);
    registerPolarCoordinatesNodes(registry);
    registerRepeatFoldNode(registry);
    registerWaveNode(registry);
    registerWorleyNoiseNode(registry);
    registerCompareNode(registry);
    registerHashNode(registry);
    registerTableNode(registry);
    addNode<PerlinNode>(registry); addNode<CoordinatesNode>(registry);
    addNode<MathNode>(registry); addNode<VectorMathNode>(registry); addNode<MixNode>(registry); addNode<ThresholdNode>(registry); addNode<SelectNode>(registry); addNode<InvertNode>(registry); addNode<ColorRampNode>(registry);
    registerConvolutionNode(registry); addNode<LaplacianNode>(registry);
    addNode<ReactionNode>(registry); addNode<OutputNode>(registry);
}

} // namespace reaction
