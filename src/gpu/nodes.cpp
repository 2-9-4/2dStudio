#include "reaction/gpu/gpu_runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace reaction {
namespace {

float parameter(const nlohmann::json& values, const char* key, float fallback) {
    return values.contains(key) && values[key].is_number()
        ? values[key].get<float>() : fallback;
}

ImageHandle imageAt(std::span<const Value> values, std::size_t index) {
    if (index >= values.size()) return {};
    if (const auto* image = std::get_if<ImageHandle>(&values[index])) return *image;
    return {};
}

float floatAt(std::span<const Value> values, std::size_t index, float fallback = 0.0F) {
    if (index >= values.size()) return fallback;
    if (const auto* number = std::get_if<float>(&values[index])) return *number;
    return fallback;
}

void uniform(GLuint program, const char* name, float value) {
    glUniform1f(glGetUniformLocation(program, name), value);
}

void uniform(GLuint program, const char* name, int value) {
    glUniform1i(glGetUniformLocation(program, name), value);
}

void bindTexture(int unit, GLuint texture) {
    glActiveTexture(static_cast<GLenum>(GL_TEXTURE0 + unit));
    glBindTexture(GL_TEXTURE_2D, texture);
}

class ParameterNode : public NodeInstance {
public:
    [[nodiscard]] nlohmann::json parameters() const override { return parameters_; }
    void setParameters(const nlohmann::json& values) override { parameters_ = values; }
protected:
    nlohmann::json parameters_ = nlohmann::json::object();
};

class FloatNode final : public ParameterNode {
public:
    static NodeDescriptor describe() {
        return {"float", 1, "Float", "Input",
                {{"value", "Value", ValueType::Float, SocketDirection::Output}},
                {{"value", "Value", 0.5F, -10.0F, 10.0F}}};
    }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    void evaluate(EvaluationContext&, std::span<const Value>, std::span<Value> outputs) override {
        outputs[0] = parameter(parameters_, "value", 0.5F);
    }
};

class TimeNode final : public ParameterNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"time", 1, "Time", "Input",
            {{"time", "Time", ValueType::Float, SocketDirection::Output},
             {"delta", "Delta", ValueType::Float, SocketDirection::Output}},
            {{"speed", "Speed", 1.0F, -4.0F, 4.0F}}};
        result.timeDependent = true;
        return result;
    }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    void evaluate(EvaluationContext& context, std::span<const Value>, std::span<Value> outputs) override {
        const float speed = parameter(parameters_, "speed", 1.0F);
        outputs[0] = static_cast<float>(context.time) * speed;
        outputs[1] = static_cast<float>(context.deltaTime) * speed;
    }
};

class TextureNode : public ParameterNode {
public:
    ~TextureNode() override {
        if (texture_ != 0) glDeleteTextures(1, &texture_);
        if (program_ != 0) glDeleteProgram(program_);
    }
protected:
    void ensure(EvaluationContext& context, GLenum format = GL_RGBA16F) {
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        gpu.ensureTexture(texture_, width_, height_, context.width, context.height, format);
    }
    GLuint texture_ = 0;
    GLuint program_ = 0;
    int width_ = 0;
    int height_ = 0;
};

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
        return result;
    }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    void evaluate(EvaluationContext& context, std::span<const Value>, std::span<Value> outputs) override {
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kPerlinShader);
        glUseProgram(program_);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        uniform(program_, "seed", static_cast<int>(parameter(parameters_, "seed", 1)));
        uniform(program_, "scale", parameter(parameters_, "scale", 6));
        uniform(program_, "octaves", static_cast<int>(parameter(parameters_, "octaves", 4)));
        uniform(program_, "persistence", parameter(parameters_, "persistence", 0.5F));
        uniform(program_, "lacunarity", parameter(parameters_, "lacunarity", 2));
        uniform(program_, "timeValue", static_cast<float>(context.time) * parameter(parameters_, "speed", 0.08F));
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
vec4 safePow(vec4 a,vec4 b){return sign(a)*pow(max(abs(a),vec4(1e-6)),b);}
vec4 applyOp(vec4 a,vec4 b,vec4 c){
 if(operation==0)return a+b;if(operation==1)return a-b;if(operation==2)return a*b;
 if(operation==3)return a/(max(abs(b),vec4(1e-6))*sign(b+vec4(1e-12)));
 if(operation==4)return safePow(a,b);if(operation==5)return min(a,b);if(operation==6)return max(a,b);
 if(operation==7)return abs(a);if(operation==8)return sin(a);if(operation==9)return cos(a);
 if(operation==10)return clamp(a,b,c);
 return mix(vec4(remapRange.z),vec4(remapRange.w),clamp((a-vec4(remapRange.x))/max(remapRange.y-remapRange.x,1e-6),0.0,1.0));
}
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outputImage);if(any(greaterThanEqual(p,s)))return;vec2 uv=(vec2(p)+.5)/vec2(s);
 vec4 a=hasA!=0?texture(imageA,uv):scalarA,b=hasB!=0?texture(imageB,uv):scalarB,c=hasC!=0?texture(imageC,uv):scalarC;
 imageStore(outputImage,p,applyOp(a,b,c));})GLSL";

float applyMath(int op, float a, float b, float c, const nlohmann::json& values) {
    switch (op) {
    case 0: return a + b; case 1: return a - b; case 2: return a * b;
    case 3: return a / (std::abs(b) < 1.0e-6F ? std::copysign(1.0e-6F, b == 0 ? 1.0F : b) : b);
    case 4: return std::copysign(std::pow(std::max(std::abs(a), 1.0e-6F), b), a);
    case 5: return std::min(a, b); case 6: return std::max(a, b); case 7: return std::abs(a);
    case 8: return std::sin(a); case 9: return std::cos(a); case 10: return std::clamp(a, b, c);
    default: {
        const auto inMin = parameter(values, "inMin", 0), inMax = parameter(values, "inMax", 1);
        const auto t = std::clamp((a - inMin) / std::max(inMax - inMin, 1.0e-6F), 0.0F, 1.0F);
        return std::lerp(parameter(values, "outMin", 0), parameter(values, "outMax", 1), t);
    }}
}

class MathNode final : public TextureNode {
public:
    static NodeDescriptor describe() {
        return {"math", 1, "Math", "Math",
            {{"a", "A", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"b", "B", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"c", "C", ValueType::AnyNumeric, SocketDirection::Input, true},
             {"result", "Result", ValueType::AnyNumeric, SocketDirection::Output}},
            {{"operation", "Operation", 0, 0, 11}, {"a", "A", 0, -10, 10}, {"b", "B", 0, -10, 10},
             {"c", "C", 1, -10, 10}, {"inMin", "Input Min", 0, -10, 10}, {"inMax", "Input Max", 1, -10, 10},
             {"outMin", "Output Min", 0, -10, 10}, {"outMax", "Output Max", 1, -10, 10}}};
    }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    void evaluate(EvaluationContext& context, std::span<const Value> inputs, std::span<Value> outputs) override {
        const auto imageA = imageAt(inputs, 0), imageB = imageAt(inputs, 1), imageC = imageAt(inputs, 2);
        const float a = floatAt(inputs, 0, parameter(parameters_, "a", 0));
        const float b = floatAt(inputs, 1, parameter(parameters_, "b", 0));
        const float c = floatAt(inputs, 2, parameter(parameters_, "c", 1));
        const int operation = static_cast<int>(parameter(parameters_, "operation", 0));
        if (!imageA && !imageB && !imageC) { outputs[0] = applyMath(operation, a, b, c, parameters_); return; }
        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kMathShader);
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
        uniform(program_, "operation", operation);
        glUniform4f(glGetUniformLocation(program_, "remapRange"), parameter(parameters_, "inMin", 0), parameter(parameters_, "inMax", 1), parameter(parameters_, "outMin", 0), parameter(parameters_, "outMax", 1));
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }
};

constexpr std::string_view kMixShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(rgba16f,binding=0)writeonly uniform image2D outImage;
layout(binding=0)uniform sampler2D texA;layout(binding=1)uniform sampler2D texB;layout(binding=2)uniform sampler2D texF;
uniform int hasA,hasB,hasF;uniform vec4 valA,valB;uniform float valF;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outImage);if(any(greaterThanEqual(p,s)))return;vec2 uv=(vec2(p)+.5)/vec2(s);
vec4 a=hasA!=0?texture(texA,uv):valA,b=hasB!=0?texture(texB,uv):valB;float f=hasF!=0?texture(texF,uv).r:valF;imageStore(outImage,p,mix(a,b,clamp(f,0,1)));})GLSL";

class MixNode final : public TextureNode {
public:
    static NodeDescriptor describe() { return {"mix",1,"Mix","Color",
        {{"a","A",ValueType::AnyNumeric,SocketDirection::Input,true},{"b","B",ValueType::AnyNumeric,SocketDirection::Input,true},
         {"factor","Factor",ValueType::AnyNumeric,SocketDirection::Input,true},{"result","Result",ValueType::AnyNumeric,SocketDirection::Output}},
        {{"a","A",0,0,1},{"b","B",1,0,1},{"factor","Factor",0.5F,0,1}}}; }
    const NodeDescriptor& descriptor() const override { static const auto value=describe();return value; }
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,std::span<Value> outputs) override {
        const auto ia=imageAt(inputs,0),ib=imageAt(inputs,1),iff=imageAt(inputs,2);
        const float a=floatAt(inputs,0,parameter(parameters_,"a",0)),b=floatAt(inputs,1,parameter(parameters_,"b",1)),f=floatAt(inputs,2,parameter(parameters_,"factor",.5F));
        if(!ia&&!ib&&!iff){outputs[0]=std::lerp(a,b,std::clamp(f,0.0F,1.0F));return;}
        ensure(context);auto& gpu=*static_cast<GpuRuntime*>(context.gpu);if(!program_)program_=gpu.compileCompute(kMixShader);glUseProgram(program_);
        glBindImageTexture(0,texture_,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F);const std::array imgs{ia,ib,iff};
        for(int i=0;i<3;++i)if(imgs[static_cast<std::size_t>(i)])bindTexture(i,imgs[static_cast<std::size_t>(i)].texture);
        uniform(program_,"hasA",ia?1:0);uniform(program_,"hasB",ib?1:0);uniform(program_,"hasF",iff?1:0);
        glUniform4f(glGetUniformLocation(program_,"valA"),a,a,a,a);glUniform4f(glGetUniformLocation(program_,"valB"),b,b,b,b);uniform(program_,"valF",f);
        gpu.dispatch(program_,context.width,context.height);outputs[0]=ImageHandle{texture_,context.width,context.height};
    }
};

constexpr std::string_view kRampShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(rgba16f,binding=0)writeonly uniform image2D outImage;
layout(binding=0)uniform sampler2D source;uniform int hasSource;uniform float scalarValue,low,high;uniform vec4 colorA,colorB;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(outImage);if(any(greaterThanEqual(p,s)))return;vec2 uv=(vec2(p)+.5)/vec2(s);float v=hasSource!=0?texture(source,uv).r:scalarValue;float t=clamp((v-low)/max(high-low,1e-6),0,1);imageStore(outImage,p,mix(colorA,colorB,t));})GLSL";

class ColorRampNode final : public TextureNode {
public:
    static NodeDescriptor describe(){return {"color_ramp",1,"Color Ramp","Color",
        {{"value","Value",ValueType::AnyNumeric,SocketDirection::Input,true},{"image","Image",ValueType::Image2D,SocketDirection::Output}},
        {{"value","Value",0.5F,0,1},{"low","Low",.02F,0,1},{"high","High",.4F,0,1},
         {"r0","Start R",.015F,0,1},{"g0","Start G",.01F,0,1},{"b0","Start B",.04F,0,1},
         {"r1","End R",1,0,1},{"g1","End G",.35F,0,1},{"b1","End B",.08F,0,1}}};}
    const NodeDescriptor& descriptor()const override{static const auto value=describe();return value;}
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,std::span<Value> outputs)override{
        ensure(context);auto& gpu=*static_cast<GpuRuntime*>(context.gpu);if(!program_)program_=gpu.compileCompute(kRampShader);const auto source=imageAt(inputs,0);
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
    static NodeDescriptor describe(){auto result=NodeDescriptor{"reaction_diffusion",1,"Reaction Diffusion","Simulation",
        {{"feedMultiplier","Feed Multiplier",ValueType::Image2D,SocketDirection::Input,true},{"killMultiplier","Kill Multiplier",ValueType::Image2D,SocketDirection::Input,true},
         {"seed","Seed",ValueType::Image2D,SocketDirection::Input,true},{"image","Image",ValueType::Image2D,SocketDirection::Output}},
        {{"feed","Feed",.055F,0,.1F},{"kill","Kill",.062F,0,.1F},{"diffA","Diffusion A",1,0,2},{"diffB","Diffusion B",.5F,0,2},
         {"structureScale","Structure Scale",1,.25F,8},
         {"dt","Timestep",1,0.01F,2},{"iterations","Iterations",8,1,64},
         {"autoReset","Auto Reset",0,0,1}}};result.timeDependent=true;result.stateful=true;return result;}
    const NodeDescriptor& descriptor()const override{static const auto value=describe();return value;}
    void reset(EvaluationContext&)override{resetPending_=true;collapseCheckCounter_=0;}
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,std::span<Value> outputs)override{
        auto& gpu=*static_cast<GpuRuntime*>(context.gpu);ensure(context);
        if(stateWidth_!=context.width||stateHeight_!=context.height){if(state_[0])glDeleteTextures(2,state_.data());state_[0]=gpu.createTexture(context.width,context.height,GL_RG16F);state_[1]=gpu.createTexture(context.width,context.height,GL_RG16F);stateWidth_=context.width;stateHeight_=context.height;resetPending_=true;}
        if (!initProgram_) initProgram_ = gpu.compileCompute(kReactionInit);
        if (!program_) program_ = gpu.compileCompute(kReactionStep);
        if (!outputProgram_) outputProgram_ = gpu.compileCompute(kReactionOutput);
        if (!collapseProgram_) collapseProgram_ = gpu.compileCompute(kReactionCollapse);
        if (!collapseBuffer_) {glGenBuffers(1,&collapseBuffer_);glBindBuffer(GL_SHADER_STORAGE_BUFFER,collapseBuffer_);const GLuint zero=0;glBufferData(GL_SHADER_STORAGE_BUFFER,sizeof(zero),&zero,GL_DYNAMIC_READ);}
        const auto initialize=[&]{const auto seed=imageAt(inputs,2);glUseProgram(initProgram_);glBindImageTexture(0,state_[0],0,GL_FALSE,0,GL_WRITE_ONLY,GL_RG16F);if(seed)bindTexture(0,seed.texture);uniform(initProgram_,"hasSeed",seed?1:0);gpu.dispatch(initProgram_,context.width,context.height);index_=0;resetPending_=false;};
        if(resetPending_)initialize();
        if(context.playing){const auto feedImage=imageAt(inputs,0),killImage=imageAt(inputs,1);const int iterations=std::clamp(static_cast<int>(parameter(parameters_,"iterations",8)),1,64);
            for(int i=0;i<iterations;++i){const int next=1-index_;glUseProgram(program_);bindTexture(0,state_[index_]);if(feedImage)bindTexture(1,feedImage.texture);if(killImage)bindTexture(2,killImage.texture);
                glBindImageTexture(0,state_[next],0,GL_FALSE,0,GL_WRITE_ONLY,GL_RG16F);uniform(program_,"hasFeed",feedImage?1:0);uniform(program_,"hasKill",killImage?1:0);
                uniform(program_,"feed",parameter(parameters_,"feed",.055F));uniform(program_,"kill",parameter(parameters_,"kill",.062F));uniform(program_,"diffA",parameter(parameters_,"diffA",1));uniform(program_,"diffB",parameter(parameters_,"diffB",.5F));uniform(program_,"structureScale",parameter(parameters_,"structureScale",1));uniform(program_,"dt",parameter(parameters_,"dt",1));
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
    addNode<FloatNode>(registry); addNode<TimeNode>(registry); addNode<PerlinNode>(registry);
    addNode<MathNode>(registry); addNode<MixNode>(registry); addNode<ColorRampNode>(registry);
    addNode<ReactionNode>(registry); addNode<OutputNode>(registry);
}

} // namespace reaction
