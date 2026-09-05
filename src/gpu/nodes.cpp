#include "reaction/gpu/gpu_runtime.hpp"
#include "reaction/core/math.hpp"
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

std::string converted(const ShaderValue& value, ShaderValueType type) {
    return value.type == type ? value.name : shaderTypeName(type) + "(" + value.name + ")";
}

constexpr std::string_view kPerlinShader = R"GLSL(#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba16f, binding=0) writeonly uniform image2D outputImage;
uniform float scale, persistence, lacunarity, timeValue;
uniform int octaves, seed;
uniform vec2 offset;
uint hash(uvec3 p) {
    uint h = p.x * 374761393u + p.y * 668265263u + p.z * 1442695041u + uint(seed) * 2246822519u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}
vec3 gradient(ivec3 p) {
    uint h=hash(uvec3(p));float z=float(h&65535u)/32767.5-1.0;
    float a=float(h>>16u)*(6.28318530718/65535.0),r=sqrt(max(1.0-z*z,0.0));
    return vec3(r*cos(a),r*sin(a),z);
}
float perlin(vec3 p) {
    ivec3 cell=ivec3(floor(p));vec3 f=fract(p),u=f*f*f*(f*(f*6.0-15.0)+10.0);
    float n000=dot(gradient(cell+ivec3(0,0,0)),f-vec3(0,0,0));
    float n100=dot(gradient(cell+ivec3(1,0,0)),f-vec3(1,0,0));
    float n010=dot(gradient(cell+ivec3(0,1,0)),f-vec3(0,1,0));
    float n110=dot(gradient(cell+ivec3(1,1,0)),f-vec3(1,1,0));
    float n001=dot(gradient(cell+ivec3(0,0,1)),f-vec3(0,0,1));
    float n101=dot(gradient(cell+ivec3(1,0,1)),f-vec3(1,0,1));
    float n011=dot(gradient(cell+ivec3(0,1,1)),f-vec3(0,1,1));
    float n111=dot(gradient(cell+ivec3(1,1,1)),f-vec3(1,1,1));
    return mix(mix(mix(n000,n100,u.x),mix(n010,n110,u.x),u.y),
               mix(mix(n001,n101,u.x),mix(n011,n111,u.x),u.y),u.z);
}
void main() {
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy); ivec2 size=imageSize(outputImage);
    if(any(greaterThanEqual(pixel,size))) return;
    vec3 p=vec3((vec2(pixel)+0.5)/vec2(size)*scale+offset,timeValue);
    float sum=0.0, amplitude=1.0, norm=0.0;
    for(int i=0;i<octaves;i++){sum+=perlin(p)*amplitude;norm+=amplitude;amplitude*=persistence;p*=lacunarity;}
    float value=clamp(0.5+0.85*sum/max(norm,0.0001),0.0,1.0);
    imageStore(outputImage,pixel,vec4(value,value,value,1.0));
})GLSL";

class PerlinNode final : public TextureNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"perlin", 1, "Perlin Noise", "Generator",
            {{"image", "Image", ValueType::Image2D, SocketDirection::Output}},
            {{"seed", "Seed", 1, 0, 10000}, {"scale", "Scale", 6, 0.05F, 100},
             {"octaves", "Octaves", 4, 1, 8}, {"persistence", "Persistence", 0.5F, 0, 1},
             {"lacunarity", "Lacunarity", 2, 1, 4}, {"speed", "Speed", 0.08F, -2, 2},
             {"offsetX", "Offset X", 0, -100, 100}, {"offsetY", "Offset Y", 0, -100, 100}}};
        result.timeDependent = true;
        result.lowerable = true;
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
        (void)context.emit("vec4(vec3(" + expression + "),1.0)", "image");
        return true;
    }
    void evaluate(EvaluationContext& context, std::span<const Value>, std::span<Value> outputs) override {
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kPerlinShader, "Perlin Noise / compute");
        glUseProgram(program_);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        uniform(program_, "seed", static_cast<int>(parameter(parameters_, "seed", 1)));
        uniform(program_, "scale", parameter(parameters_, "scale", 6));
        uniform(program_, "octaves", static_cast<int>(parameter(parameters_, "octaves", 4)));
        uniform(program_, "persistence", parameter(parameters_, "persistence", 0.5F));
        uniform(program_, "lacunarity", parameter(parameters_, "lacunarity", 2));
        // Frame-derived animation is deterministic even when rendering or encoding stalls.
        // The 60 FPS baseline preserves the established meaning of existing Speed values.
        constexpr float baselineFps = 60.0F;
        uniform(program_, "timeValue", static_cast<float>(context.frame) /
                                       baselineFps * parameter(parameters_, "speed", 0.08F));
        glUniform2f(glGetUniformLocation(program_, "offset"), parameter(parameters_, "offsetX", 0), parameter(parameters_, "offsetY", 0));
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

constexpr std::string_view kMathShader = R"GLSL(#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba16f,binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D imageA; layout(binding=1) uniform sampler2D imageB; layout(binding=2) uniform sampler2D imageC;
uniform int hasA,hasB,hasC,operation; uniform vec4 scalarA,scalarB,scalarC; uniform vec4 remapRange;
vec4 safePow(vec4 a,vec4 b){return (step(0.0,a)*2.0-1.0)*pow(max(abs(a),vec4(1e-6)),b);}
vec4 applyOp(vec4 a,vec4 b,vec4 c){
 if(operation==0)return a+b;if(operation==1)return a-b;if(operation==2)return a*b;
 if(operation==3)return a/((step(0.0,b)*2.0-1.0)*max(abs(b),vec4(1e-6)));
 if(operation==4)return safePow(a,b);if(operation==5)return min(a,b);if(operation==6)return max(a,b);
 if(operation==7)return abs(a);if(operation==8)return sin(a);if(operation==9)return cos(a);
 if(operation==10)return clamp(a,b,c);
 if(operation==11)return mix(vec4(remapRange.z),vec4(remapRange.w),clamp((a-vec4(remapRange.x))/max(remapRange.y-remapRange.x,1e-6),0.0,1.0));
 if(operation==12)return floor(a);if(operation==13)return ceil(a);if(operation==14)return round(a);
 if(operation==15)return fract(a);if(operation==16)return sqrt(max(a,vec4(0.0)));
 if(operation==17)return exp(a);if(operation==18)return log(max(a,vec4(1e-6)));if(operation==19)return log2(max(a,vec4(1e-6)));
 if(operation==20)return sign(a);if(operation==21)return tan(a);
 if(operation==22)return asin(clamp(a,vec4(-1.0),vec4(1.0)));if(operation==23)return acos(clamp(a,vec4(-1.0),vec4(1.0)));
 if(operation==24)return atan(a);if(operation==25)return mod(a,b);if(operation==26)return atan(a,b);
 if(operation==27)return step(a,b);if(operation==28)return sqrt(a*a+b*b);
 if(operation==29){vec4 t=clamp((a-b)/max(c-b,vec4(1e-6)),0.0,1.0);return t*t*(3.0-2.0*t);}
 return a*b+c;
}
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outputImage);if(any(greaterThanEqual(p,s)))return;vec2 uv=(vec2(p)+.5)/vec2(s);
 vec4 a=hasA!=0?texture(imageA,uv):scalarA,b=hasB!=0?texture(imageB,uv):scalarB,c=hasC!=0?texture(imageC,uv):scalarC;
 imageStore(outputImage,p,applyOp(a,b,c));})GLSL";

class MathNode final : public TextureNode {
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
        (void)context.emit(mathGlslExpression(operation, operands, remap, context.valueType()));
        return true;
    }
    void evaluate(EvaluationContext& context, std::span<const Value> inputs, std::span<Value> outputs) override {
        const auto imageA = imageAt(inputs, 0), imageB = imageAt(inputs, 1), imageC = imageAt(inputs, 2);
        const float a = floatAt(inputs, 0, parameter(parameters_, "a", 0));
        const float b = floatAt(inputs, 1, parameter(parameters_, "b", 0));
        const float c = floatAt(inputs, 2, parameter(parameters_, "c", 1));
        const auto operation = mathOperation(parameter(parameters_, "operation", 0));
        if (!imageA && !imageB && !imageC) {
            outputs[0] = applyMathOperation(operation, a, b, c,
                parameter(parameters_, "inMin", 0), parameter(parameters_, "inMax", 1),
                parameter(parameters_, "outMin", 0), parameter(parameters_, "outMax", 1));
            return;
        }
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kMathShader, "Math / compute");
        glUseProgram(program_);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        const std::array images{imageA, imageB, imageC};
        const std::array scalars{a, b, c};
        for (int i = 0; i < 3; ++i) {
            if (images[static_cast<std::size_t>(i)]) bindTexture(i, images[static_cast<std::size_t>(i)].texture);
            const std::string has = std::string("has") + static_cast<char>('A' + i);
            const std::string scalar = std::string("scalar") + static_cast<char>('A' + i);
            uniform(program_, has.c_str(), images[static_cast<std::size_t>(i)] ? 1 : 0);
            glUniform4f(glGetUniformLocation(program_, scalar.c_str()), scalars[static_cast<std::size_t>(i)], scalars[static_cast<std::size_t>(i)], scalars[static_cast<std::size_t>(i)], scalars[static_cast<std::size_t>(i)]);
        }
        uniform(program_, "operation", static_cast<int>(operation));
        glUniform4f(glGetUniformLocation(program_, "remapRange"), parameter(parameters_, "inMin", 0), parameter(parameters_, "inMax", 1), parameter(parameters_, "outMin", 0), parameter(parameters_, "outMax", 1));
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

constexpr std::string_view kMixShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(rgba16f,binding=0)writeonly uniform image2D outImage;
layout(binding=0)uniform sampler2D texA;layout(binding=1)uniform sampler2D texB;layout(binding=2)uniform sampler2D texF;
uniform int hasA,hasB,hasF,mixMode;uniform vec4 valA,valB;uniform float valF;
vec3 blend(vec3 a,vec3 b){
 if(mixMode==1)return a+b;
 if(mixMode==2)return a*b;
 if(mixMode==3)return a+b-a*b;
 if(mixMode==4)return mix(2.0*a*b,1.0-2.0*(1.0-a)*(1.0-b),step(vec3(.5),a));
 if(mixMode==5)return abs(a-b);
 if(mixMode==6)return min(a,b);
 if(mixMode==7)return max(a,b);
 if(mixMode==8)return a/max(vec3(1e-6),1.0-b);
 if(mixMode==9)return 1.0-(1.0-a)/max(vec3(1e-6),b);
 return b;
}
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outImage);if(any(greaterThanEqual(p,s)))return;vec2 uv=(vec2(p)+.5)/vec2(s);
vec4 a=hasA!=0?texture(texA,uv):valA,b=hasB!=0?texture(texB,uv):valB;float f=hasF!=0?texture(texF,uv).r:valF;
imageStore(outImage,p,vec4(mix(a.rgb,blend(a.rgb,b.rgb),clamp(f,0,1)),mix(a.a,b.a,clamp(f,0,1))));})GLSL";

float applyMix(int mode, float a, float b) {
    switch (mode) {
    case 1: return a + b;
    case 2: return a * b;
    case 3: return a + b - a * b;
    case 4: return a < .5F ? 2.0F * a * b : 1.0F - 2.0F * (1.0F - a) * (1.0F - b);
    case 5: return std::abs(a - b);
    case 6: return std::min(a, b);
    case 7: return std::max(a, b);
    case 8: return a / std::max(1.0e-6F, 1.0F - b);
    case 9: return 1.0F - (1.0F - a) / std::max(1.0e-6F, b);
    default: return b;
    }
}

class MixNode final : public TextureNode {
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
        const auto type = context.valueType();
        const auto a = context.input("a", "a", 0.0F);
        const auto b = context.input("b", "b", 1.0F);
        const auto factorValue = context.input("factor", "factor", 0.5F);
        const auto av = converted(a, type), bv = converted(b, type);
        const auto factor = factorValue.type == ShaderValueType::Scalar
            ? factorValue.name : "(" + factorValue.name + ").r";
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
            (void)context.emit("vec4(mix((" + av + ").rgb,(" + blend + ").rgb," + f +
                               "),mix((" + av + ").a,(" + bv + ").a," + f + "))");
        else
            (void)context.emit("mix(" + av + "," + blend + "," + f + ")");
        return true;
    }
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,std::span<Value> outputs) override {
        const auto ia=imageAt(inputs,0),ib=imageAt(inputs,1),iff=imageAt(inputs,2);
        const float a=floatAt(inputs,0,parameter(parameters_,"a",0)),b=floatAt(inputs,1,parameter(parameters_,"b",1)),f=floatAt(inputs,2,parameter(parameters_,"factor",.5F));
        const int mode=std::clamp(static_cast<int>(parameter(parameters_,"mode",0)),0,9);
        if(!ia&&!ib&&!iff){outputs[0]=std::lerp(a,applyMix(mode,a,b),std::clamp(f,0.0F,1.0F));return;}
        ensure(context);auto& gpu=*static_cast<GpuRuntime*>(context.gpu);if(!program_)program_=gpu.compileCompute(kMixShader,"Mix / compute");glUseProgram(program_);
        glBindImageTexture(0,texture_,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F);const std::array imgs{ia,ib,iff};
        for(int i=0;i<3;++i)if(imgs[static_cast<std::size_t>(i)])bindTexture(i,imgs[static_cast<std::size_t>(i)].texture);
        uniform(program_,"hasA",ia?1:0);uniform(program_,"hasB",ib?1:0);uniform(program_,"hasF",iff?1:0);
        glUniform4f(glGetUniformLocation(program_,"valA"),a,a,a,a);glUniform4f(glGetUniformLocation(program_,"valB"),b,b,b,b);uniform(program_,"valF",f);uniform(program_,"mixMode",mode);
        gpu.dispatch(program_,context.width,context.height);outputs[0]=ImageHandle{texture_,context.width,context.height};
    }
};

constexpr std::string_view kThresholdShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(rgba16f,binding=0)writeonly uniform image2D outImage;
layout(binding=0)uniform sampler2D source;uniform float threshold;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outImage);if(any(greaterThanEqual(p,s)))return;
vec2 uv=(vec2(p)+.5)/vec2(s);float value=texture(source,uv).r;float result=value>=threshold?1.0:0.0;imageStore(outImage,p,vec4(result,result,result,1));})GLSL";

class ThresholdNode final : public TextureNode {
public:
    static NodeDescriptor describe() { auto result = NodeDescriptor{"threshold", 1, "Threshold", "Math",
        {{"value", "Value", ValueType::AnyNumeric, SocketDirection::Input, true},
         {"result", "Result", ValueType::AnyNumeric, SocketDirection::Output}},
        {{"value", "Value", 0.5F, 0, 1}, {"threshold", "Threshold", 0.5F, 0, 1}}}; result.lowerable=true; return result; }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto value = context.input("value", "value", 0.5F);
        const auto cutoff = context.parameter("threshold", 0.5F);
        const auto type = context.valueType();
        if (type == ShaderValueType::Vec4) {
            const auto scalarValue = value.type == ShaderValueType::Scalar
                ? value.name : "(" + value.name + ").r";
            (void)context.emit("vec4(vec3(step(" + cutoff.name + "," + scalarValue + ")),1.0)");
        } else {
            (void)context.emit("step(" + cutoff.name + "," + value.name + ")");
        }
        return true;
    }
    void evaluate(EvaluationContext& context, std::span<const Value> inputs, std::span<Value> outputs) override {
        const auto source = imageAt(inputs, 0);
        const float value = floatAt(inputs, 0, parameter(parameters_, "value", 0.5F));
        const float cutoff = parameter(parameters_, "threshold", 0.5F);
        if (!source) { outputs[0] = value >= cutoff ? 1.0F : 0.0F; return; }
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kThresholdShader, "Threshold / compute");
        glUseProgram(program_);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        bindTexture(0, source.texture);
        uniform(program_, "threshold", cutoff);
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

constexpr std::string_view kSelectShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;
layout(rgba16f,binding=0)writeonly uniform image2D outImage;
layout(binding=0)uniform sampler2D conditionImage;
layout(binding=1)uniform sampler2D trueImage;
layout(binding=2)uniform sampler2D falseImage;
uniform int hasCondition,hasTrue,hasFalse;
uniform float scalarCondition;
uniform vec4 scalarTrue,scalarFalse;
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outImage);
    if(any(greaterThanEqual(p,s)))return;
    vec2 uv=(vec2(p)+.5)/vec2(s);
    float condition=hasCondition!=0?texture(conditionImage,uv).r:scalarCondition;
    vec4 yes=hasTrue!=0?texture(trueImage,uv):scalarTrue;
    vec4 no=hasFalse!=0?texture(falseImage,uv):scalarFalse;
    imageStore(outImage,p,condition!=0.0?yes:no);
})GLSL";

class SelectNode final : public TextureNode {
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
        const auto type = context.valueType();
        const auto scalarCondition = condition.type == ShaderValueType::Scalar
            ? condition.name : "(" + condition.name + ").r";
        const auto expression = "((" + scalarCondition + ")!=0.0?" +
            converted(yes, type) + ":" + converted(no, type) + ")";
        (void)context.emit(std::move(expression));
        return true;
    }
    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        const auto conditionImage = imageAt(inputs, 0);
        const auto trueImage = imageAt(inputs, 1);
        const auto falseImage = imageAt(inputs, 2);
        const float condition = floatAt(inputs, 0, parameter(parameters_, "condition", 0.0F));
        const float ifTrue = floatAt(inputs, 1, parameter(parameters_, "ifTrue", 1.0F));
        const float ifFalse = floatAt(inputs, 2, parameter(parameters_, "ifFalse", 0.0F));
        if (!conditionImage && !trueImage && !falseImage) {
            outputs[0] = condition != 0.0F ? ifTrue : ifFalse;
            return;
        }
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kSelectShader, "Select / compute");
        glUseProgram(program_);
        const std::array images{conditionImage, trueImage, falseImage};
        for (int index = 0; index < 3; ++index)
            if (images[static_cast<std::size_t>(index)])
                bindTexture(index, images[static_cast<std::size_t>(index)].texture);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        uniform(program_, "hasCondition", conditionImage ? 1 : 0);
        uniform(program_, "hasTrue", trueImage ? 1 : 0);
        uniform(program_, "hasFalse", falseImage ? 1 : 0);
        uniform(program_, "scalarCondition", condition);
        glUniform4f(glGetUniformLocation(program_, "scalarTrue"), ifTrue, ifTrue, ifTrue, ifTrue);
        glUniform4f(glGetUniformLocation(program_, "scalarFalse"), ifFalse, ifFalse, ifFalse, ifFalse);
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

constexpr std::string_view kCoordinatesShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;
layout(rgba16f,binding=0)writeonly uniform image2D xImage;
layout(rgba16f,binding=1)writeonly uniform image2D yImage;
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(xImage);
    if(any(greaterThanEqual(p,s)))return;
    vec2 uv=(vec2(p)+.5)/vec2(s);
    imageStore(xImage,p,vec4(uv.x,uv.x,uv.x,1));
    imageStore(yImage,p,vec4(uv.y,uv.y,uv.y,1));
})GLSL";

class CoordinatesNode final : public ParameterNode {
public:
    ~CoordinatesNode() override {
        if (textures_[0]) glDeleteTextures(2, textures_.data());
        if (program_) glDeleteProgram(program_);
    }
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"coordinates", 1, "Canvas Coordinates", "Input",
                {{"x", "X", ValueType::Image2D, SocketDirection::Output},
                 {"y", "Y", ValueType::Image2D, SocketDirection::Output}}, {}};
        result.lowerable = true;
        return result;
    }
    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }
    bool lowerShader(ShaderLoweringContext& context) const override {
        if (context.valueType() == ShaderValueType::Scalar) {
            (void)context.emit("uv.x", "x");
            (void)context.emit("uv.y", "y");
        } else {
            (void)context.emit("vec4(uv.x,uv.x,uv.x,1.0)", "x");
            (void)context.emit("vec4(uv.y,uv.y,uv.y,1.0)", "y");
        }
        return true;
    }
    void evaluate(EvaluationContext& context, std::span<const Value>,
                  std::span<Value> outputs) override {
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        for (std::size_t index = 0; index < textures_.size(); ++index) {
            gpu.ensureTexture(textures_[index], widths_[index], heights_[index],
                              context.width, context.height, GL_RGBA16F);
        }
        if (!program_) program_ = gpu.compileCompute(kCoordinatesShader, "Canvas Coordinates / compute");
        glUseProgram(program_);
        glBindImageTexture(0, textures_[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        glBindImageTexture(1, textures_[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{textures_[0], context.width, context.height};
        outputs[1] = ImageHandle{textures_[1], context.width, context.height};
    }

private:
    std::array<GLuint, 2> textures_{};
    std::array<int, 2> widths_{};
    std::array<int, 2> heights_{};
    GLuint program_ = 0;
};

constexpr std::string_view kLaplacianShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;
layout(rgba16f,binding=0)writeonly uniform image2D outImage;
layout(binding=0)uniform sampler2D source;
uniform float scale;
vec4 sampleValue(vec2 uv){return texture(source,fract(uv));}
vec4 laplacianAt(vec2 uv,float radius){
    vec2 pixel=1.0/vec2(imageSize(outImage));
    vec2 o=pixel*radius;
    vec4 result=-sampleValue(uv);
    result+=.2*(sampleValue(uv+vec2(-o.x,0))+sampleValue(uv+vec2(o.x,0))+
                 sampleValue(uv+vec2(0,-o.y))+sampleValue(uv+vec2(0,o.y)));
    result+=.05*(sampleValue(uv+vec2(-o.x,-o.y))+sampleValue(uv+vec2(o.x,-o.y))+
                  sampleValue(uv+vec2(-o.x,o.y))+sampleValue(uv+vec2(o.x,o.y)));
    return result;
}
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outImage);
    if(any(greaterThanEqual(p,s)))return;
    vec2 uv=(vec2(p)+.5)/vec2(s);
    vec4 result;
    if(abs(scale-1.0)<.001)result=laplacianAt(uv,1.0);
    else{result=vec4(0);for(int i=0;i<3;i++)result+=laplacianAt(uv,mix(1.0,scale,float(i)/2.0))/3.0;}
    imageStore(outImage,p,result);
})GLSL";

class LaplacianNode final : public TextureNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"laplacian", 1, "Laplacian", "Filter",
                {{"value", "Value", ValueType::AnyVector, SocketDirection::Input},
                 {"scale", "Scale", ValueType::Float, SocketDirection::Input, true},
                 {"result", "Result", ValueType::AnyVector, SocketDirection::Output}},
                {{"scale", "Scale", 1.0F, .25F, 8.0F}}};
        result.lowerable = true;
        result.neighborhoodSocket = "value";
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
        (void)context.emit(helper + "(uv)");
        return true;
    }
    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        const auto source = imageAt(inputs, 0);
        if (!source) {
            if (!inputs.empty() && std::holds_alternative<float>(inputs[0]))
                outputs[0] = 0.0F;
            else if (!inputs.empty() && std::holds_alternative<Vec2>(inputs[0]))
                outputs[0] = Vec2{};
            else outputs[0] = {};
            return;
        }
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kLaplacianShader, "Laplacian / compute");
        glUseProgram(program_);
        bindTexture(0, source.texture);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        uniform(program_, "scale", floatAt(inputs, 1, parameter(parameters_, "scale", 1.0F)));
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

constexpr std::string_view kInvertShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(rgba16f,binding=0)writeonly uniform image2D outImage;
layout(binding=0)uniform sampler2D source;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outImage);if(any(greaterThanEqual(p,s)))return;vec4 value=texelFetch(source,p,0);imageStore(outImage,p,vec4(vec3(1.0)-value.rgb,value.a));})GLSL";

class InvertNode final : public TextureNode {
public:
    static NodeDescriptor describe() { auto result = NodeDescriptor{"invert", 1, "Image Invert", "Utility",
        {{"image", "Image", ValueType::Image2D, SocketDirection::Input},
         {"image", "Image", ValueType::Image2D, SocketDirection::Output}}, {}}; result.lowerable=true; return result; }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto value = context.input("image", "image", 0.0F);
        (void)context.emit("vec4(vec3(1.0)-(" + value.name + ").rgb,(" + value.name + ").a)", "image");
        return true;
    }
    void evaluate(EvaluationContext& context, std::span<const Value> inputs, std::span<Value> outputs) override {
        const auto source = imageAt(inputs, 0);
        if (!source) { outputs[0] = {}; return; }
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kInvertShader, "Image Invert / compute");
        glUseProgram(program_);
        bindTexture(0, source.texture);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

constexpr std::string_view kRampShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(rgba16f,binding=0)writeonly uniform image2D outImage;
layout(binding=0)uniform sampler2D source;uniform int hasSource;uniform float scalarValue,low,high;uniform vec4 colorA,colorB;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outImage);if(any(greaterThanEqual(p,s)))return;vec2 uv=(vec2(p)+.5)/vec2(s);float v=hasSource!=0?texture(source,uv).r:scalarValue;float t=clamp((v-low)/max(high-low,1e-6),0,1);imageStore(outImage,p,mix(colorA,colorB,t));})GLSL";

class ColorRampNode final : public TextureNode {
public:
    static NodeDescriptor describe(){auto result=NodeDescriptor{"color_ramp",1,"Color Ramp","Color",
        {{"value","Value",ValueType::AnyNumeric,SocketDirection::Input,true},{"image","Image",ValueType::Image2D,SocketDirection::Output}},
        {{"value","Value",0.5F,0,1},{"low","Low threshold",.02F,0,1},{"high","High threshold",.4F,0,1},
         {"r0","Start R",.015F,0,1},{"g0","Start G",.01F,0,1},{"b0","Start B",.04F,0,1},
         {"r1","End R",1,0,1},{"g1","End G",.35F,0,1},{"b1","End B",.08F,0,1}}};result.lowerable=true;return result;}
    const NodeDescriptor& descriptor()const override{static const auto value=describe();return value;}
    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto value = context.input("value", "value", 0.5F);
        const auto low = context.parameter("low", 0.02F), high = context.parameter("high", 0.4F);
        const auto r0 = context.parameter("r0", .015F), g0 = context.parameter("g0", .01F), b0 = context.parameter("b0", .04F);
        const auto r1 = context.parameter("r1", 1.0F), g1 = context.parameter("g1", .35F), b1 = context.parameter("b1", .08F);
        const auto scalar = value.type == ShaderValueType::Scalar ? value.name : "(" + value.name + ").r";
        const auto t = "clamp((" + scalar + "-" + low.name + ")/max(" + high.name + "-" + low.name + ",1e-6),0.0,1.0)";
        (void)context.emit("mix(vec4(" + r0.name + "," + g0.name + "," + b0.name + ",1.0),vec4(" +
                           r1.name + "," + g1.name + "," + b1.name + ",1.0)," + t + ")", "image");
        return true;
    }
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,std::span<Value> outputs)override{
        ensure(context);auto& gpu=*static_cast<GpuRuntime*>(context.gpu);if(!program_)program_=gpu.compileCompute(kRampShader,"Color Ramp / compute");const auto source=imageAt(inputs,0);
        glUseProgram(program_);glBindImageTexture(0,texture_,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F);if(source)bindTexture(0,source.texture);
        uniform(program_,"hasSource",source?1:0);uniform(program_,"scalarValue",floatAt(inputs,0,parameter(parameters_,"value",.5F)));
        uniform(program_,"low",parameter(parameters_,"low",.02F));uniform(program_,"high",parameter(parameters_,"high",.4F));
        glUniform4f(glGetUniformLocation(program_,"colorA"),parameter(parameters_,"r0",.015F),parameter(parameters_,"g0",.01F),parameter(parameters_,"b0",.04F),1);
        glUniform4f(glGetUniformLocation(program_,"colorB"),parameter(parameters_,"r1",1),parameter(parameters_,"g1",.35F),parameter(parameters_,"b1",.08F),1);
        gpu.dispatch(program_,context.width,context.height);outputs[0]=ImageHandle{texture_,context.width,context.height};
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
    void evaluate(EvaluationContext&,std::span<const Value> inputs,std::span<Value> outputs)override{outputs[0]=inputs.empty()?Value{}:inputs[0];}
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
    addNode<PerlinNode>(registry); addNode<CoordinatesNode>(registry);
    addNode<MathNode>(registry); addNode<MixNode>(registry); addNode<ThresholdNode>(registry); addNode<SelectNode>(registry); addNode<InvertNode>(registry); addNode<ColorRampNode>(registry);
    registerConvolutionNode(registry); addNode<LaplacianNode>(registry);
    addNode<ReactionNode>(registry); addNode<OutputNode>(registry);
}

} // namespace reaction
