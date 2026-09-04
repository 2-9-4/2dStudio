#include "reaction/gpu/gpu_runtime.hpp"
#include "reaction/core/math.hpp"
#include "reaction/gpu/shader_ir.hpp"
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
        statements_.str({}); statements_.clear(); mainValues_.clear(); mainStateValues_.clear();
        usedInterfaces_.clear(); stateLaplacians_.clear(); mainStateLaplacianValues_.clear();
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

    [[nodiscard]] const std::unordered_set<std::string>& usedInterfaces() const {
        return usedInterfaces_;
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
                if (usedInterfaces_.contains(item.key))
                    out << "layout(binding=" << binding << ")uniform sampler2D in_" << name << ";\n"
                        << "uniform int mode_" << name << ";uniform float value_" << name << ";\n";
                ++binding;
            } else if (item.kind == SubgraphInterfaceKind::Slider &&
                       usedInterfaces_.contains(item.key)) {
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
        if (node.type == "simulation_previous_state") {
            std::string value = "sampleState(" + uv + ")";
            if (uv == "uv") {
                auto foundState = mainStateValues_.find(node.id);
                if (foundState == mainStateValues_.end()) {
                    const auto variable = "v_state_" + std::to_string(node.id);
                    statements_ << "vec2 " << variable << "=" << value << ";\n";
                    foundState = mainStateValues_.emplace(node.id, variable).first;
                }
                value = foundState->second;
            }
            return value + "." + (socket == "b" ? "y" : "x");
        }
        if (node.type == "subgraph_input") {
            const auto interfaceKey = node.parameters.at("key").get<std::string>();
            usedInterfaces_.insert(interfaceKey);
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
            const auto operation = mathOperation(parameterValue("operation", 0.0F));
            std::vector<ShaderValue> operands{{ShaderValueType::Scalar, a}};
            if (mathOperationOperandCount(operation) >= 2)
                operands.push_back({ShaderValueType::Scalar, b});
            if (mathOperationOperandCount(operation) >= 3)
                operands.push_back({ShaderValueType::Scalar, c});
            std::vector<ShaderValue> remap;
            if (operation == MathOperation::Remap) {
                for (const auto* key : {"inMin", "inMax", "outMin", "outMax"})
                    remap.push_back({ShaderValueType::Scalar,
                        std::to_string(parameterValue(key,
                            std::string_view(key) == "inMax" || std::string_view(key) == "outMax"
                                ? 1.0F : 0.0F))});
            }
            return mathGlslExpression(operation, operands, remap, ShaderValueType::Scalar);
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
        const auto* valueLink = inputLink(node.id, "value");
        const auto valueNode = valueLink ? nodes_.find(valueLink->fromNode) : nodes_.end();
        if (valueLink && valueNode != nodes_.end() &&
            valueNode->second->type == "simulation_previous_state" &&
            (valueLink->fromSocket == "a" || valueLink->fromSocket == "b")) {
            // A state texture fetch already returns both simulation channels. Reuse
            // one vec2 neighborhood for every direct A/B Laplacian with the same
            // scale instead of issuing an independent neighborhood per channel.
            const auto scale = emitInput(node, "scale", "q", 1.0F);
            auto found = stateLaplacians_.find(scale);
            if (found == stateLaplacians_.end()) {
                const auto functionName = "state_lap_" + std::to_string(node.id);
                helpers_ << "vec2 " << functionName
                         << "At(vec2 q,float r){vec2 pixel=1.0/vec2(textureSize(stateIn,0));return -sampleState(q)+0.2*("
                         << "sampleState(q+pixel*vec2(-r,0.0))+sampleState(q+pixel*vec2(r,0.0))+"
                         << "sampleState(q+pixel*vec2(0.0,-r))+sampleState(q+pixel*vec2(0.0,r)))+0.05*("
                         << "sampleState(q+pixel*vec2(-r,-r))+sampleState(q+pixel*vec2(r,-r))+"
                         << "sampleState(q+pixel*vec2(-r,r))+sampleState(q+pixel*vec2(r,r)));}\n"
                         << "vec2 " << functionName << "(vec2 q){float scale=" << scale
                         << ";if(abs(scale-1.0)<0.001)return " << functionName
                         << "At(q,1.0);vec2 v=vec2(0.0);for(int i=0;i<3;i++)v+="
                         << functionName << "At(q,mix(1.0,scale,float(i)/2.0))/3.0;return v;}\n";
                found = stateLaplacians_.emplace(scale, functionName).first;
            }
            std::string value = found->second + "(" + uv + ")";
            if (uv == "uv") {
                auto mainValue = mainStateLaplacianValues_.find(found->second);
                if (mainValue == mainStateLaplacianValues_.end()) {
                    const auto variable = "v_" + found->second;
                    statements_ << "vec2 " << variable << "=" << value << ";\n";
                    mainValue = mainStateLaplacianValues_.emplace(found->second, variable).first;
                }
                value = mainValue->second;
            }
            return value + "." + (valueLink->fromSocket == "b" ? "y" : "x");
        }

        const auto functionName = "lap_" + std::to_string(node.id);
        if (generatedHelpers_.insert(functionName).second) {
            const auto sample = [&](const std::string& delta) {
                const auto* link = inputLink(node.id, "value");
                if (!link) return std::string("0.0");
                return emit(link->fromNode, link->fromSocket, "fract(q+pixel*" + delta + ")");
            };
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
    std::unordered_map<NodeId, std::string> mainStateValues_;
    std::unordered_set<std::string> usedInterfaces_;
    std::unordered_map<std::string, std::string> stateLaplacians_;
    std::unordered_map<std::string, std::string> mainStateLaplacianValues_;
};

constexpr std::string_view kCollapseShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(binding=0)uniform sampler2D stateIn;
layout(std430,binding=0)buffer CollapseState{uint activityFlag;};uniform float threshold;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=textureSize(stateIn,0);if(any(greaterThanEqual(p,s)))return;if(texelFetch(stateIn,p,0).g>threshold)atomicOr(activityFlag,1u);})GLSL";

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
        requiredOutputs_.assign(outputChannels_.size(), true);
        output_.assign(outputChannels_.size(), 0);
    }
    ~SimulationSubgraphNode() override {
        if (state_[0]) glDeleteTextures(2, state_.data());
        for (const auto texture : output_) if (texture) glDeleteTextures(1, &texture);
        if (collapseBuffer_) glDeleteBuffers(1, &collapseBuffer_);
    }
    const NodeDescriptor& descriptor() const override { return descriptor_; }
    void reset(EvaluationContext&) override { resetPending_ = true; collapseCheckCounter_ = 0; }
    void setOutputRequirements(const std::vector<bool>& required) override {
        std::vector<bool> normalized(outputChannels_.size(), false);
        for (std::size_t index = 0; index < normalized.size() && index < required.size(); ++index)
            normalized[index] = required[index];
        if (normalized == requiredOutputs_) return;
        requiredOutputs_ = std::move(normalized);
        outputProgram_ = 0;
    }

    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        ensureResources(gpu, context.width, context.height);
        if (!initProgram_ || !stepProgram_) {
            SimulationGraphCompiler compiler(definition_);
            const auto initializationSource = compiler.shader(true);
            initializationInterfaces_ = compiler.usedInterfaces();
            const auto updateSource = compiler.shader(false);
            updateInterfaces_ = compiler.usedInterfaces();
            initProgram_ = gpu.compileComputeCached(initializationSource,
                                                     definition_.name + " / initialize");
            stepProgram_ = gpu.compileComputeCached(updateSource,
                                                     definition_.name + " / update");
            collapseProgram_ = gpu.compileComputeCached(kCollapseShader, definition_.name + " / collapse check");
        }
        if (!outputProgram_)
            outputProgram_ = gpu.compileComputeCached(outputShader(), definition_.name + " / output");
        const auto initialize = [&] {
            glUseProgram(initProgram_);
            glBindImageTexture(0, state_[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F);
            bindInterface(initProgram_, inputs, 1, initializationInterfaces_);
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
                bindInterface(stepProgram_, inputs, 1, updateInterfaces_);
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
        for (std::size_t output = 0; output < output_.size(); ++output)
            if (requiredOutputs_[output])
                glBindImageTexture(static_cast<GLuint>(output), output_[output], 0, GL_FALSE, 0,
                                   GL_WRITE_ONLY, GL_RGBA16F);
        gpu.dispatch(outputProgram_, context.width, context.height);
        for (std::size_t output = 0; output < outputs.size() && output < output_.size(); ++output) {
            if (requiredOutputs_[output])
                outputs[output] = ImageHandle{output_[output], context.width, context.height};
            else outputs[output] = Value{};
        }
    }

private:
    std::string outputShader() const {
        std::ostringstream source;
        source << "#version 430\nlayout(local_size_x=16,local_size_y=16)in;\n"
               << "layout(binding=0)uniform sampler2D stateIn;\n";
        for (std::size_t output = 0; output < requiredOutputs_.size(); ++output)
            if (requiredOutputs_[output])
                source << "layout(rgba16f,binding=" << output
                       << ")writeonly uniform image2D out" << output << ";\n";
        source << "void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=textureSize(stateIn,0);"
               << "if(any(greaterThanEqual(p,s)))return;vec2 q=texelFetch(stateIn,p,0).rg;\n";
        for (std::size_t output = 0; output < requiredOutputs_.size(); ++output) {
            if (!requiredOutputs_[output]) continue;
            const char channel = outputChannels_[output] == 1 ? 'y' : 'x';
            source << "imageStore(out" << output << ",p,vec4(q." << channel << ",q."
                   << channel << ",q." << channel << ",1.0));\n";
        }
        source << "}";
        return source.str();
    }

    void ensureResources(GpuRuntime& gpu, int width, int height) {
        const bool resized = width_ != width || height_ != height || !state_[0];
        if (resized) {
            if (state_[0]) glDeleteTextures(2, state_.data());
            for (auto& texture : state_) texture = gpu.createTexture(width, height, GL_RG16F);
            width_ = width; height_ = height; resetPending_ = true;
        }
        for (std::size_t output = 0; output < output_.size(); ++output) {
            if (!requiredOutputs_[output]) {
                if (output_[output]) glDeleteTextures(1, &output_[output]);
                output_[output] = 0;
            } else if (resized || !output_[output]) {
                if (output_[output]) glDeleteTextures(1, &output_[output]);
                output_[output] = gpu.createTexture(width, height, GL_RGBA16F);
            }
        }
        if (!collapseBuffer_) {
            glGenBuffers(1, &collapseBuffer_); glBindBuffer(GL_SHADER_STORAGE_BUFFER, collapseBuffer_);
            const GLuint zero = 0; glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero), &zero, GL_DYNAMIC_READ);
        }
    }

    void bindInterface(GLuint program, std::span<const Value> inputs, int firstTexture,
                       const std::unordered_set<std::string>& used) {
        std::size_t inputIndex = 0; int textureUnit = firstTexture;
        for (const auto& item : definition_.interface) {
            const auto name = identifier(item.key);
            if (item.kind == SubgraphInterfaceKind::Input) {
                if (!used.contains(item.key)) {
                    ++textureUnit; ++inputIndex;
                    continue;
                }
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
            } else if (item.kind == SubgraphInterfaceKind::Slider && used.contains(item.key)) {
                uniform(program, ("param_" + name).c_str(),
                        parameter(parameters_, item.key.c_str(), item.defaultValue));
            }
        }
    }

    SubgraphDefinition definition_;
    NodeDescriptor descriptor_;
    std::vector<int> outputChannels_;
    std::array<GLuint, 2> state_{};
    std::vector<GLuint> output_;
    std::vector<bool> requiredOutputs_;
    std::unordered_set<std::string> initializationInterfaces_;
    std::unordered_set<std::string> updateInterfaces_;
    GLuint initProgram_ = 0, stepProgram_ = 0, outputProgram_ = 0, collapseProgram_ = 0;
    GLuint collapseBuffer_ = 0;
    int width_ = 0, height_ = 0, index_ = 0, collapseCheckCounter_ = 0;
    bool resetPending_ = true;
};

} // namespace

std::string generateSimulationShader(const SubgraphDefinition& definition, bool initialization) {
    return SimulationGraphCompiler(definition).shader(initialization);
}

std::unique_ptr<NodeInstance> createSubgraphInstance(const SubgraphDefinition& definition,
                                                     const NodeRegistry& registry) {
    if (definition.execution != SubgraphExecution::Simulation)
        throw std::runtime_error("Pipeline subgraphs are not executable in this milestone");
    const auto errors = validateSubgraph(definition, registry);
    if (!errors.empty()) throw std::runtime_error(errors.front());
    return std::make_unique<SimulationSubgraphNode>(definition);
}

} // namespace reaction
