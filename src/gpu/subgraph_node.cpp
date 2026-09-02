#include "reaction/gpu/gpu_runtime.hpp"
#include "node_support.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace reaction {
namespace {

using node_support::bindTexture;
using node_support::parameter;
using node_support::uniform;

std::string identifier(std::string value) {
    for (char& c : value) if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
    return value;
}

class KernelCompiler {
public:
    explicit KernelCompiler(const SubgraphDefinition& definition) : definition_(definition) {
        for (const auto& item : definition.kernel) nodes_.emplace(item.key, &item);
    }

    std::string shader(bool initialization) {
        const auto* endpoint = endpointFor(initialization ? "initial" : "next");
        if (!endpoint) throw std::runtime_error("Simulation subgraph is missing a state endpoint");
        helpers_.str({}); helpers_.clear(); generatedHelpers_.clear();
        statements_.str({}); statements_.clear(); mainValues_.clear();
        const auto expression = emit(endpoint->key, "uv");
        std::ostringstream source;
        source << "#version 430\nlayout(local_size_x=16,local_size_y=16)in;\n"
               << "layout(rg16f,binding=0)writeonly uniform image2D stateOut;\n"
               << "layout(binding=0)uniform sampler2D stateIn;\n";
        declareInterface(source, 1);
        source << "vec2 sampleState(vec2 q){return texture(stateIn,fract(q)).rg;}\n";
        source << helpers_.str();
        source << "void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(stateOut);"
               << "if(any(greaterThanEqual(p,s)))return;vec2 uv=(vec2(p)+0.5)/vec2(s);\n"
               << statements_.str()
               << "imageStore(stateOut,p,vec4(" << expression << ",0.0,1.0));}";
        return source.str();
    }

private:
    const SubgraphKernelNode* endpointFor(std::string_view role) const {
        const auto it = std::ranges::find_if(definition_.kernel, [&](const auto& item) {
            return item.properties.value("role", "") == role;
        });
        return it == definition_.kernel.end() ? nullptr : &*it;
    }

    void declareInterface(std::ostringstream& out, int firstBinding) const {
        int binding = firstBinding;
        for (const auto& item : definition_.interface) {
            const auto name = identifier(item.key);
            if (item.kind == SubgraphInterfaceKind::Input) {
                out << "layout(binding=" << binding++ << ")uniform sampler2D in_" << name << ";\n"
                    << "uniform int mode_" << name << ";uniform float value_" << name << ";\n";
            } else if (item.kind == SubgraphInterfaceKind::Slider) {
                out << "uniform float param_" << name << ";\n";
            }
        }
    }

    std::string emit(const std::string& key, const std::string& uv) {
        if (uv != "uv") return emitInline(key, uv);
        if (const auto found = mainValues_.find(key); found != mainValues_.end()) return found->second;
        const auto value = "v_" + identifier(key);
        const auto expression = emitInline(key, uv);
        statements_ << typeOf(key) << " " << value << "=" << expression << ";\n";
        mainValues_.emplace(key, value);
        return value;
    }

    std::string emitInline(const std::string& key, const std::string& uv) {
        const auto found = nodes_.find(key);
        if (found == nodes_.end()) throw std::runtime_error("Unknown kernel input: " + key);
        const auto& node = *found->second;
        const auto input = [&](std::size_t index) { return emit(node.inputs.at(index), uv); };
        const auto binary = [&](const char* op) {
            if (node.inputs.size() == 1 && node.properties.contains("value"))
                return "(" + input(0) + op + std::to_string(node.properties.at("value").get<float>()) + ")";
            return "(" + input(0) + op + input(1) + ")";
        };
        if (node.operation == "uv") return uv;
        if (node.operation == "constant") return std::to_string(node.properties.value("value", 0.0F));
        if (node.operation == "constant2") return "vec2(" + std::to_string(node.properties.value("x", 0.0F)) + "," + std::to_string(node.properties.value("y", 0.0F)) + ")";
        if (node.operation == "previous_state") return "sampleState(" + uv + ")";
        if (node.operation == "interface") {
            const auto interfaceKey = node.properties.at("key").get<std::string>();
            const auto* item = interfaceItem(interfaceKey);
            const auto name = identifier(interfaceKey);
            if (item && item->kind == SubgraphInterfaceKind::Slider) return "param_" + name;
            const auto fallback = std::to_string(node.properties.value("default", item ? item->defaultValue : 0.0F));
            return "(mode_" + name + "==2?texture(in_" + name + "," + uv + ").r:(mode_" + name + "==1?value_" + name + ":" + fallback + "))";
        }
        if (node.operation == "connected") {
            return "(mode_" + identifier(node.properties.at("key").get<std::string>()) + "!=0?1.0:0.0)";
        }
        if (node.operation == "add") return binary("+");
        if (node.operation == "multiply") return binary("*");
        if (node.operation == "subtract") {
            if (node.inputs.size() == 1 && node.properties.contains("from"))
                return "(" + std::to_string(node.properties.at("from").get<float>()) + "-" + input(0) + ")";
            return binary("-");
        }
        if (node.operation == "divide") return binary("/");
        if (node.operation == "min") return "min(" + input(0) + "," + input(1) + ")";
        if (node.operation == "max") return "max(" + input(0) + "," + input(1) + ")";
        if (node.operation == "abs") return "abs(" + input(0) + ")";
        if (node.operation == "sin") return "sin(" + input(0) + ")";
        if (node.operation == "cos") return "cos(" + input(0) + ")";
        if (node.operation == "pow") return "pow(max(abs(" + input(0) + "),1e-6)," + input(1) + ")";
        if (node.operation == "clamp01") return "clamp(" + input(0) + ",0.0,1.0)";
        if (node.operation == "clamp") return "clamp(" + input(0) + "," + input(1) + "," + input(2) + ")";
        if (node.operation == "remap") return "mix(" + input(3) + "," + input(4) + ",clamp((" +
            input(0) + "-" + input(1) + ")/max(" + input(2) + "-" + input(1) + ",1e-6),0.0,1.0))";
        if (node.operation == "pack2") return "vec2(" + input(0) + "," + input(1) + ")";
        if (node.operation == "swizzle") return "(" + input(0) + ")." + (node.properties.value("channel", 0) == 0 ? "x" : "y");
        if (node.operation == "length") {
            const auto value = node.properties.value("subtract", false)
                ? "(" + input(0) + "-" + input(1) + ")" : input(0);
            return "length(" + value + ")";
        }
        if (node.operation == "step") {
            return node.properties.value("reverse", false)
                ? "step(" + input(0) + "," + input(1) + ")"
                : "step(" + input(1) + "," + input(0) + ")";
        }
        if (node.operation == "select") return "((" + input(0) + ")!=0.0?" + input(1) + ":" + input(2) + ")";
        if (node.operation == "laplacian") return emitLaplacian(node, uv);
        if (node.operation == "output") return input(0);
        throw std::runtime_error("Unsupported kernel operation: " + node.operation);
    }

    std::string typeOf(const std::string& key) {
        const auto found = nodes_.find(key);
        if (found == nodes_.end()) return "float";
        const auto& node = *found->second;
        if (node.operation == "uv" || node.operation == "constant2" ||
            node.operation == "previous_state" || node.operation == "laplacian" ||
            node.operation == "pack2") return "vec2";
        if (node.operation == "swizzle" || node.operation == "length" ||
            node.operation == "constant" || node.operation == "interface" ||
            node.operation == "connected" || node.operation == "step") return "float";
        if (node.operation == "select" && node.inputs.size() > 1) return typeOf(node.inputs[1]);
        if (node.operation == "output" && !node.inputs.empty()) return typeOf(node.inputs[0]);
        for (const auto& input : node.inputs) if (typeOf(input) == "vec2") return "vec2";
        return "float";
    }

    std::string emitLaplacian(const SubgraphKernelNode& node, const std::string& uv) {
        const auto functionName = "lap_" + identifier(node.key);
        if (generatedHelpers_.insert(functionName).second) {
            const auto sample = [&](const std::string& delta) {
                return emit(node.inputs.at(0), "fract(q+pixel*" + delta + ")");
            };
            const auto scaleKey = node.properties.value("scale", std::string("structureScale"));
            helpers_ << "vec2 " << functionName << "At(vec2 q,float r){vec2 pixel=1.0/vec2(textureSize(stateIn,0));return -"
                     << emit(node.inputs.at(0), "q") << "+0.2*(" << sample("vec2(-r,0.0)") << "+" << sample("vec2(r,0.0)")
                     << "+" << sample("vec2(0.0,-r)") << "+" << sample("vec2(0.0,r)") << ")+0.05*("
                     << sample("vec2(-r,-r)") << "+" << sample("vec2(r,-r)") << "+" << sample("vec2(-r,r)")
                     << "+" << sample("vec2(r,r)") << ");}\n"
                     << "vec2 " << functionName << "(vec2 q){float scale=param_" << identifier(scaleKey)
                     << ";if(abs(scale-1.0)<0.001)return " << functionName << "At(q,1.0);vec2 v=vec2(0.0);"
                     << "for(int i=0;i<3;i++)v+=" << functionName << "At(q,mix(1.0,scale,float(i)/2.0))/3.0;return v;}\n";
        }
        return functionName + "(" + uv + ")";
    }

    const SubgraphInterfaceItem* interfaceItem(std::string_view key) const {
        const auto it = std::ranges::find(definition_.interface, key, &SubgraphInterfaceItem::key);
        return it == definition_.interface.end() ? nullptr : &*it;
    }

    const SubgraphDefinition& definition_;
    std::unordered_map<std::string, const SubgraphKernelNode*> nodes_;
    std::ostringstream helpers_;
    std::ostringstream statements_;
    std::unordered_set<std::string> generatedHelpers_;
    std::unordered_map<std::string, std::string> mainValues_;
};

constexpr std::string_view kCollapseShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(binding=0)uniform sampler2D stateIn;
layout(std430,binding=0)buffer CollapseState{uint activityFlag;};uniform float threshold;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=textureSize(stateIn,0);if(any(greaterThanEqual(p,s)))return;if(texelFetch(stateIn,p,0).g>threshold)atomicOr(activityFlag,1u);})GLSL";

constexpr std::string_view kOutputShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;
layout(rgba16f,binding=0)writeonly uniform image2D out0;
layout(rgba16f,binding=1)writeonly uniform image2D out1;
layout(rgba16f,binding=2)writeonly uniform image2D out2;
layout(binding=0)uniform sampler2D stateIn;uniform ivec3 channels;
float channelValue(vec2 state,int channel){return channel==0?state.x:state.y;}
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(out0);if(any(greaterThanEqual(p,s)))return;
vec2 q=texelFetch(stateIn,p,0).rg;float a=channelValue(q,channels.x),b=channelValue(q,channels.y),c=channelValue(q,channels.z);
imageStore(out0,p,vec4(a,a,a,1));imageStore(out1,p,vec4(b,b,b,1));imageStore(out2,p,vec4(c,c,c,1));})GLSL";

class SimulationSubgraphNode final : public node_support::ParameterNode {
public:
    explicit SimulationSubgraphNode(SubgraphDefinition definition)
        : definition_(std::move(definition)), descriptor_(describeSubgraph(definition_)) {
        const auto next = std::ranges::find_if(definition_.kernel, [](const auto& item) {
            return item.properties.is_object() &&
                   item.properties.value("role", std::string{}) == "next";
        });
        for (const auto& port : definition_.interface) {
            if (port.kind != SubgraphInterfaceKind::Output) continue;
            const auto endpoint = std::ranges::find_if(definition_.kernel, [&](const auto& item) {
                return item.operation == "output" && item.properties.is_object() &&
                       item.properties.value("key", std::string{}) == port.key;
            });
            const bool second = next != definition_.kernel.end() && endpoint != definition_.kernel.end() &&
                !endpoint->inputs.empty() && next->inputs.size() > 1 && endpoint->inputs.front() == next->inputs[1];
            outputChannels_.push_back(second ? 1 : 0);
        }
    }
    ~SimulationSubgraphNode() override {
        if (state_[0]) glDeleteTextures(2, state_.data());
        if (output_[0]) glDeleteTextures(3, output_.data());
        for (const auto program : {outputProgram_, collapseProgram_}) if (program) glDeleteProgram(program);
        if (collapseBuffer_) glDeleteBuffers(1, &collapseBuffer_);
    }
    const NodeDescriptor& descriptor() const override { return descriptor_; }
    void reset(EvaluationContext&) override { resetPending_ = true; collapseCheckCounter_ = 0; }

    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        ensureResources(gpu, context.width, context.height);
        if (!initProgram_ || !stepProgram_) {
            KernelCompiler compiler(definition_);
            initProgram_ = gpu.compileComputeCached(compiler.shader(true), definition_.name + " / initialize");
            stepProgram_ = gpu.compileComputeCached(compiler.shader(false), definition_.name + " / update");
            outputProgram_ = gpu.compileCompute(kOutputShader, definition_.name + " / output");
            collapseProgram_ = gpu.compileCompute(kCollapseShader, definition_.name + " / collapse check");
        }
        const auto initialize = [&] {
            glUseProgram(initProgram_);
            glBindImageTexture(0, state_[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F);
            bindInterface(initProgram_, inputs, 1);
            gpu.dispatch(initProgram_, context.width, context.height);
            index_ = 0; resetPending_ = false;
        };
        if (resetPending_) initialize();
        if (context.playing) {
            const int iterations = std::clamp(static_cast<int>(parameter(parameters_, "iterations", 8)), 1, 64);
            for (int iteration = 0; iteration < iterations; ++iteration) {
                const int next = 1 - index_;
                glUseProgram(stepProgram_);
                bindTexture(0, state_[index_]);
                glBindImageTexture(0, state_[next], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F);
                bindInterface(stepProgram_, inputs, 1);
                gpu.dispatch(stepProgram_, context.width, context.height);
                index_ = next;
            }
        }
        if (parameter(parameters_, "autoReset", 0) > .5F && ++collapseCheckCounter_ >= 8) {
            collapseCheckCounter_ = 0; const GLuint zero = 0;
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, collapseBuffer_);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(zero), &zero);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, collapseBuffer_);
            glUseProgram(collapseProgram_); bindTexture(0, state_[index_]);
            uniform(collapseProgram_, "threshold", .001F);
            gpu.dispatch(collapseProgram_, context.width, context.height);
            GLuint active = 0; glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(active), &active);
            if (active == 0) initialize();
        }
        glUseProgram(outputProgram_); bindTexture(0, state_[index_]);
        for (int output = 0; output < 3; ++output)
            glBindImageTexture(output, output_[static_cast<std::size_t>(output)], 0, GL_FALSE, 0,
                               GL_WRITE_ONLY, GL_RGBA16F);
        const std::array<int, 3> channel = {
            outputChannels_.size() > 0 ? outputChannels_[0] : 0,
            outputChannels_.size() > 1 ? outputChannels_[1] : 0,
            outputChannels_.size() > 2 ? outputChannels_[2] : 0};
        glUniform3i(glGetUniformLocation(outputProgram_, "channels"), channel[0], channel[1], channel[2]);
        gpu.dispatch(outputProgram_, context.width, context.height);
        for (std::size_t output = 0; output < outputs.size() && output < output_.size(); ++output)
            outputs[output] = ImageHandle{output_[output], context.width, context.height};
    }

private:
    void ensureResources(GpuRuntime& gpu, int width, int height) {
        if (width_ == width && height_ == height && state_[0]) return;
        if (state_[0]) glDeleteTextures(2, state_.data());
        if (output_[0]) glDeleteTextures(3, output_.data());
        for (auto& texture : state_) texture = gpu.createTexture(width, height, GL_RG16F);
        for (auto& texture : output_) texture = gpu.createTexture(width, height, GL_RGBA16F);
        width_ = width; height_ = height; resetPending_ = true;
        if (!collapseBuffer_) {
            glGenBuffers(1, &collapseBuffer_); glBindBuffer(GL_SHADER_STORAGE_BUFFER, collapseBuffer_);
            const GLuint zero = 0; glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero), &zero, GL_DYNAMIC_READ);
        }
    }

    void bindInterface(GLuint program, std::span<const Value> inputs, int firstTexture) {
        std::size_t inputIndex = 0; int textureUnit = firstTexture;
        for (const auto& item : definition_.interface) {
            const auto name = identifier(item.key);
            if (item.kind == SubgraphInterfaceKind::Input) {
                const Value* value = inputIndex < inputs.size() ? &inputs[inputIndex] : nullptr;
                int mode = 0; float scalar = item.defaultValue;
                if (value) {
                    if (const auto* number = std::get_if<float>(value)) { mode = 1; scalar = *number; }
                    else if (const auto* image = std::get_if<ImageHandle>(value); image && *image) {
                        mode = 2; bindTexture(textureUnit, image->texture);
                        uniform(program, ("in_" + name).c_str(), textureUnit);
                    }
                }
                uniform(program, ("mode_" + name).c_str(), mode);
                uniform(program, ("value_" + name).c_str(), scalar);
                ++textureUnit; ++inputIndex;
            } else if (item.kind == SubgraphInterfaceKind::Slider) {
                uniform(program, ("param_" + name).c_str(),
                        parameter(parameters_, item.key.c_str(), item.defaultValue));
            }
        }
    }

    SubgraphDefinition definition_;
    NodeDescriptor descriptor_;
    std::vector<int> outputChannels_;
    std::array<GLuint, 2> state_{};
    std::array<GLuint, 3> output_{};
    GLuint initProgram_ = 0, stepProgram_ = 0, outputProgram_ = 0, collapseProgram_ = 0;
    GLuint collapseBuffer_ = 0;
    int width_ = 0, height_ = 0, index_ = 0, collapseCheckCounter_ = 0;
    bool resetPending_ = true;
};

} // namespace

std::unique_ptr<NodeInstance> createSubgraphInstance(const SubgraphDefinition& definition) {
    if (definition.execution != SubgraphExecution::Simulation)
        throw std::runtime_error("Pipeline subgraphs are not executable in this milestone");
    const auto errors = validateSubgraph(definition);
    if (!errors.empty()) throw std::runtime_error(errors.front());
    return std::make_unique<SimulationSubgraphNode>(definition);
}

} // namespace reaction
