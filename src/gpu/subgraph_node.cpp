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

class SimulationGraphCompiler {
public:
    explicit SimulationGraphCompiler(const SubgraphDefinition& definition) : definition_(definition) {
        for (const auto& item : definition.body.nodes()) nodes_.emplace(item.id, &item);
    }

    std::string shader(bool initialization) {
        const auto* endpoint = endpointFor(initialization ? "initial" : "next");
        if (!endpoint) throw std::runtime_error("Simulation subgraph is missing a state endpoint");
        helpers_.str({}); helpers_.clear(); generatedHelpers_.clear();
        statements_.str({}); statements_.clear(); mainValues_.clear();
        const auto expression = "vec2(" + emitInput(*endpoint, "a", "uv") + "," +
                                emitInput(*endpoint, "b", "uv") + ")";
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
    const NodeRecord* endpointFor(std::string_view role) const {
        const std::string type = role == "initial" ? "simulation_initial_state"
                                                    : "simulation_next_state";
        const auto& nodes = definition_.body.nodes();
        const auto it = std::ranges::find(nodes, type, &NodeRecord::type);
        return it == nodes.end() ? nullptr : &*it;
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

    const LinkRecord* inputLink(NodeId node, std::string_view socket) const {
        const auto& links = definition_.body.links();
        const auto found = std::ranges::find_if(links, [&](const auto& link) {
            return link.toNode == node && link.toSocket == socket;
        });
        return found == links.end() ? nullptr : &*found;
    }

    std::string emitInput(const NodeRecord& node, std::string_view socket,
                          const std::string& uv, float fallback = 0.0F) {
        const auto* link = inputLink(node.id, socket);
        return link ? emit(link->fromNode, link->fromSocket, uv) : std::to_string(fallback);
    }

    std::string emit(NodeId id, std::string_view socket, const std::string& uv) {
        if (uv != "uv") return emitInline(id, socket, uv);
        const auto cacheKey = std::to_string(id) + "_" + std::string(socket);
        if (const auto found = mainValues_.find(cacheKey); found != mainValues_.end())
            return found->second;
        const auto value = "v_" + cacheKey;
        const auto expression = emitInline(id, socket, uv);
        statements_ << "float " << value << "=" << expression << ";\n";
        mainValues_.emplace(cacheKey, value);
        return value;
    }

    std::string emitInline(NodeId id, std::string_view socket, const std::string& uv) {
        const auto found = nodes_.find(id);
        if (found == nodes_.end())
            throw std::runtime_error("Unknown simulation node: " + std::to_string(id));
        const auto& node = *found->second;
        const auto parameterValue = [&](const char* key, float fallback) {
            return node.parameters.value(key, fallback);
        };
        if (node.type == "float") return std::to_string(parameterValue("value", 0.5F));
        if (node.type == "coordinates") return socket == "y" ? "(" + uv + ").y" : "(" + uv + ").x";
        if (node.type == "simulation_previous_state")
            return "sampleState(" + uv + ")." + (socket == "b" ? "y" : "x");
        if (node.type == "subgraph_input") {
            const auto interfaceKey = node.parameters.at("key").get<std::string>();
            const auto* item = interfaceItem(interfaceKey);
            const auto name = identifier(interfaceKey);
            if (socket == "connected") return "(mode_" + name + "!=0?1.0:0.0)";
            if (item && item->kind == SubgraphInterfaceKind::Slider) return "param_" + name;
            const auto fallback = std::to_string(parameterValue(
                "default", item ? item->defaultValue : 0.0F));
            return "(mode_" + name + "==2?texture(in_" + name + "," + uv + ").r:(mode_" + name + "==1?value_" + name + ":" + fallback + "))";
        }
        if (node.type == "math") {
            const auto a = emitInput(node, "a", uv, parameterValue("a", 0.0F));
            const auto b = emitInput(node, "b", uv, parameterValue("b", 0.0F));
            const auto c = emitInput(node, "c", uv, parameterValue("c", 1.0F));
            switch (std::clamp(static_cast<int>(parameterValue("operation", 0.0F)), 0, 11)) {
            case 0: return "(" + a + "+" + b + ")";
            case 1: return "(" + a + "-" + b + ")";
            case 2: return "(" + a + "*" + b + ")";
            case 3: return "(" + a + "/(max(abs(" + b + "),1e-6)*sign(" + b + "+1e-12)))";
            case 4: return "(sign(" + a + ")*pow(max(abs(" + a + "),1e-6)," + b + "))";
            case 5: return "min(" + a + "," + b + ")";
            case 6: return "max(" + a + "," + b + ")";
            case 7: return "abs(" + a + ")";
            case 8: return "sin(" + a + ")";
            case 9: return "cos(" + a + ")";
            case 10: return "clamp(" + a + "," + b + "," + c + ")";
            default: {
                const auto inMin = std::to_string(parameterValue("inMin", 0.0F));
                const auto inMax = std::to_string(parameterValue("inMax", 1.0F));
                const auto outMin = std::to_string(parameterValue("outMin", 0.0F));
                const auto outMax = std::to_string(parameterValue("outMax", 1.0F));
                return "mix(" + outMin + "," + outMax + ",clamp((" + a + "-" +
                       inMin + ")/max(" + inMax + "-" + inMin + ",1e-6),0.0,1.0))";
            }}
        }
        if (node.type == "threshold") {
            const auto value = emitInput(node, "value", uv, parameterValue("value", 0.5F));
            return "step(" + std::to_string(parameterValue("threshold", 0.5F)) + "," + value + ")";
        }
        if (node.type == "select") {
            const auto condition = emitInput(node, "condition", uv,
                                             parameterValue("condition", 0.0F));
            const auto ifTrue = emitInput(node, "ifTrue", uv,
                                          parameterValue("ifTrue", 1.0F));
            const auto ifFalse = emitInput(node, "ifFalse", uv,
                                           parameterValue("ifFalse", 0.0F));
            return "((" + condition + ")!=0.0?" + ifTrue + ":" + ifFalse + ")";
        }
        if (node.type == "laplacian") return emitLaplacian(node, uv);
        if (node.type == "simulation_next_state")
            return emitInput(node, socket == "b" ? "b" : "a", uv);
        if (node.type == "subgraph_output") return emitInput(node, "value", uv);
        throw std::runtime_error("Unsupported simulation node: " + node.type);
    }

    std::string emitLaplacian(const NodeRecord& node, const std::string& uv) {
        const auto functionName = "lap_" + std::to_string(node.id);
        if (generatedHelpers_.insert(functionName).second) {
            const auto sample = [&](const std::string& delta) {
                const auto* link = inputLink(node.id, "value");
                if (!link) return std::string("0.0");
                return emit(link->fromNode, link->fromSocket, "fract(q+pixel*" + delta + ")");
            };
            const auto* valueLink = inputLink(node.id, "value");
            const auto center = valueLink ? emit(valueLink->fromNode, valueLink->fromSocket, "q")
                                          : std::string("0.0");
            const auto scale = emitInput(node, "scale", "q", 1.0F);
            helpers_ << "float " << functionName << "At(vec2 q,float r){vec2 pixel=1.0/vec2(textureSize(stateIn,0));return -"
                     << center << "+0.2*(" << sample("vec2(-r,0.0)") << "+" << sample("vec2(r,0.0)")
                     << "+" << sample("vec2(0.0,-r)") << "+" << sample("vec2(0.0,r)") << ")+0.05*("
                     << sample("vec2(-r,-r)") << "+" << sample("vec2(r,-r)") << "+" << sample("vec2(-r,r)")
                     << "+" << sample("vec2(r,r)") << ");}\n"
                     << "float " << functionName << "(vec2 q){float scale=" << scale
                     << ";if(abs(scale-1.0)<0.001)return " << functionName << "At(q,1.0);float v=0.0;"
                     << "for(int i=0;i<3;i++)v+=" << functionName << "At(q,mix(1.0,scale,float(i)/2.0))/3.0;return v;}\n";
        }
        return functionName + "(" + uv + ")";
    }

    const SubgraphInterfaceItem* interfaceItem(std::string_view key) const {
        const auto it = std::ranges::find(definition_.interface, key, &SubgraphInterfaceItem::key);
        return it == definition_.interface.end() ? nullptr : &*it;
    }

    const SubgraphDefinition& definition_;
    std::unordered_map<NodeId, const NodeRecord*> nodes_;
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
        const auto& bodyNodes = definition_.body.nodes();
        const auto& bodyLinks = definition_.body.links();
        const auto next = std::ranges::find(bodyNodes, std::string("simulation_next_state"),
                                            &NodeRecord::type);
        for (const auto& port : definition_.interface) {
            if (port.kind != SubgraphInterfaceKind::Output) continue;
            const auto endpoint = std::ranges::find_if(bodyNodes, [&](const auto& item) {
                return item.type == "subgraph_output" &&
                       item.parameters.value("key", std::string{}) == port.key;
            });
            const auto mapping = endpoint == bodyNodes.end() ? bodyLinks.end() :
                std::ranges::find_if(bodyLinks, [&](const auto& link) {
                    return link.toNode == endpoint->id && link.toSocket == "value";
                });
            const bool channelB = next != bodyNodes.end() && mapping != bodyLinks.end() &&
                                  mapping->fromNode == next->id && mapping->fromSocket == "b";
            outputChannels_.push_back(channelB ? 1 : 0);
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
            SimulationGraphCompiler compiler(definition_);
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

std::unique_ptr<NodeInstance> createSubgraphInstance(const SubgraphDefinition& definition,
                                                     const NodeRegistry& registry) {
    if (definition.execution != SubgraphExecution::Simulation)
        throw std::runtime_error("Pipeline subgraphs are not executable in this milestone");
    const auto errors = validateSubgraph(definition, registry);
    if (!errors.empty()) throw std::runtime_error(errors.front());
    return std::make_unique<SimulationSubgraphNode>(definition);
}

} // namespace reaction
