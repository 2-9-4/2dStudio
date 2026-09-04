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

class SimulationGraphCompiler final : public ShaderLoweringContext {
public:
    SimulationGraphCompiler(const SubgraphDefinition& definition, const NodeRegistry& registry)
        : definition_(definition), registry_(registry) {
        for (const auto& item : definition.body.nodes()) nodes_.emplace(item.id, &item);
    }

    std::string shader(bool initialization) {
        const auto* endpoint = endpointFor(initialization ? "initial" : "next");
        if (!endpoint) throw std::runtime_error("Simulation subgraph is missing a state endpoint");
        helpers_.clear(); helperNames_.clear(); statements_.str({}); statements_.clear();
        mainValues_.clear(); usedInterfaces_.clear(); frames_.clear();
        const auto a = lowerInput(*endpoint, "a", "uv", 0.0F);
        const auto b = lowerInput(*endpoint, "b", "uv", 0.0F);
        const auto expression = "vec2(" + a.name + "," + b.name + ")";
        std::ostringstream source;
        source << "#version 430\nlayout(local_size_x=16,local_size_y=16)in;\n"
               << "layout(rg16f,binding=0)writeonly uniform image2D stateOut;\n"
               << "layout(binding=0)uniform sampler2D stateIn;\n";
        declareInterface(source, 1);
        source << "vec2 sampleState(vec2 q){return texture(stateIn,fract(q)).rg;}\n";
        source << "vec2 pixelSize;\n";
        for (const auto& helper : helpers_) source << helper.source;
        source << "void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(stateOut);"
               << "if(any(greaterThanEqual(p,s)))return;vec2 uv=(vec2(p)+0.5)/vec2(s);pixelSize=1.0/vec2(s);\n"
               << statements_.str()
               << "imageStore(stateOut,p,vec4(" << expression << ",0.0,1.0));}";
        return source.str();
    }

    [[nodiscard]] const std::unordered_set<std::string>& usedInterfaces() const {
        return usedInterfaces_;
    }

    ShaderValue input(std::string_view socket, std::string_view parameterKey,
                      float fallback) override {
        return inputAt(socket, frames_.back().uv, parameterKey, fallback);
    }

    ShaderValue inputAt(std::string_view socket, std::string_view uvExpression,
                        std::string_view parameterKey, float fallback) override {
        const auto& frame = frames_.back();
        const auto* link = inputLink(frame.node->id, socket);
        if (link) return lower(link->fromNode, link->fromSocket, std::string(uvExpression));
        return parameter(parameterKey, fallback);
    }

    ShaderValue parameter(std::string_view key, float fallback) override {
        const auto& node = *frames_.back().node;
        if (inputLink(node.id, key))
            return input(key, key, fallback);
        const float value = node.parameters.contains(key) && node.parameters[key].is_number()
            ? node.parameters[key].get<float>() : fallback;
        return {ShaderValueType::Scalar, std::to_string(value)};
    }

    std::string helper(std::string_view name, std::string source) override {
        const auto actual = identifier(std::string(name) + "_" +
                                       std::to_string(frames_.back().node->id));
        if (helperNames_.insert(actual).second) {
            for (std::size_t at = 0; (at = source.find(name, at)) != std::string::npos;
                 at += actual.size()) source.replace(at, name.size(), actual);
            helpers_.push_back({actual, std::move(source)});
        }
        return actual;
    }

    ShaderValue emit(std::string expression, std::string_view socket = {}) override {
        auto& frame = frames_.back();
        const auto output = socket.empty() ? defaultOutputSocket(*frame.node) : std::string(socket);
        const auto type = inferredType(frame.node->id, output);
        if (frame.uv != "uv") {
            for (std::size_t at = 0; (at = expression.find("uv", at)) != std::string::npos;) {
                const bool left = at == 0 || (!std::isalnum(static_cast<unsigned char>(expression[at - 1])) &&
                                               expression[at - 1] != '_');
                const auto after = at + 2;
                const bool right = after == expression.size() ||
                    (!std::isalnum(static_cast<unsigned char>(expression[after])) &&
                     expression[after] != '_');
                if (left && right) {
                    const auto replacement = "(" + frame.uv + ")";
                    expression.replace(at, 2, replacement);
                    at += replacement.size();
                } else at += 2;
            }
        }
        const auto value = materialize(*frame.node, output, frame.uv, type, std::move(expression));
        frame.emitted[output] = value;
        return value;
    }

    ShaderValueType valueType() const override { return frames_.back().valueType; }

private:
    struct Frame {
        const NodeRecord* node = nullptr;
        std::string uv;
        ShaderValueType valueType = ShaderValueType::Scalar;
        std::unordered_map<std::string, ShaderValue> emitted;
    };

    static std::string typeName(ShaderValueType type) {
        switch (type) {
        case ShaderValueType::Scalar: return "float";
        case ShaderValueType::Vec2: return "vec2";
        case ShaderValueType::Vec4: return "vec4";
        }
        return "float";
    }

    static std::string valueKey(NodeId id, std::string_view socket) {
        return std::to_string(id) + ":" + std::string(socket);
    }

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

    ShaderValue lowerInput(const NodeRecord& node, std::string_view socket,
                           const std::string& uv, float fallback = 0.0F) {
        const auto* link = inputLink(node.id, socket);
        return link ? lower(link->fromNode, link->fromSocket, uv)
                    : ShaderValue{ShaderValueType::Scalar, std::to_string(fallback)};
    }

    ShaderValue lower(NodeId id, std::string_view socket, const std::string& uv) {
        const auto cacheKey = valueKey(id, socket);
        if (uv == "uv") if (const auto found = mainValues_.find(cacheKey);
            found != mainValues_.end())
            return found->second;
        const auto found = nodes_.find(id);
        if (found == nodes_.end())
            throw std::runtime_error("Unknown simulation node: " + std::to_string(id));
        const auto& node = *found->second;
        if (node.type == "simulation_previous_state") {
            if (socket != "state") throw std::runtime_error("Unknown previous-state socket");
            return materialize(node, socket, uv, ShaderValueType::Vec2,
                               "sampleState(" + uv + ")");
        }
        if (node.type == "simulation_channel") {
            const auto state = lowerInput(node, "state", uv);
            return materialize(node, socket, uv, ShaderValueType::Scalar,
                               state.name + (socket == "b" ? ".y" : ".x"));
        }
        if (node.type == "subgraph_input") {
            const auto interfaceKey = node.parameters.at("key").get<std::string>();
            usedInterfaces_.insert(interfaceKey);
            const auto* item = interfaceItem(interfaceKey);
            const auto name = identifier(interfaceKey);
            std::string expression;
            if (socket == "connected") expression = "(mode_" + name + "!=0?1.0:0.0)";
            else if (item && item->kind == SubgraphInterfaceKind::Slider)
                expression = "param_" + name;
            else {
                const auto fallback = std::to_string(node.parameters.value(
                    "default", item ? item->defaultValue : 0.0F));
                expression = "(mode_" + name + "==2?texture(in_" + name + "," + uv +
                    ").r:(mode_" + name + "==1?value_" + name + ":" + fallback + "))";
            }
            return materialize(node, socket, uv, ShaderValueType::Scalar,
                               std::move(expression));
        }
        if (node.type == "simulation_initial_state" ||
            node.type == "simulation_next_state")
            return lowerInput(node, socket == "b" ? "b" : "a", uv);
        if (node.type == "subgraph_output") return lowerInput(node, "value", uv);

        auto instance = registry_.create(node.type);
        if (!instance) throw std::runtime_error("Unsupported simulation node: " + node.type);
        instance->setParameters(node.parameters);
        frames_.push_back({&node, uv, inferredType(id, socket), {}});
        if (!instance->lowerShader(*this)) {
            frames_.pop_back();
            throw std::runtime_error("Simulation node does not support shader lowering: " + node.type);
        }
        auto completed = std::move(frames_.back());
        frames_.pop_back();
        const auto emitted = completed.emitted.find(std::string(socket));
        if (emitted != completed.emitted.end()) return emitted->second;
        if (completed.emitted.size() == 1) return completed.emitted.begin()->second;
        throw std::runtime_error("Simulation lowerer did not emit socket '" +
                                 std::string(socket) + "'");
    }

    ShaderValue materialize(const NodeRecord& node, std::string_view socket,
                            const std::string& uv, ShaderValueType type,
                            std::string expression) {
        if (uv != "uv") return {type, std::move(expression)};
        const auto key = valueKey(node.id, socket);
        if (const auto found = mainValues_.find(key); found != mainValues_.end())
            return found->second;
        ShaderValue result{type, "v_" + std::to_string(node.id) + "_" + identifier(std::string(socket))};
        statements_ << typeName(type) << " " << result.name << "=" << expression << ";\n";
        mainValues_[key] = result;
        return result;
    }

    std::string defaultOutputSocket(const NodeRecord& node) const {
        NodeDescriptor storage;
        const auto* descriptor = resolveSubgraphBodyDescriptor(definition_, node, registry_, storage);
        if (descriptor) for (const auto& socket : descriptor->sockets)
            if (socket.direction == SocketDirection::Output) return socket.key;
        return "result";
    }

    ShaderValueType inferredType(NodeId id, std::string_view socket) const {
        const auto found = nodes_.find(id);
        if (found == nodes_.end()) return ShaderValueType::Scalar;
        const auto& node = *found->second;
        if (node.type == "simulation_previous_state" && socket == "state")
            return ShaderValueType::Vec2;
        if (node.type == "laplacian") {
            const auto* link = inputLink(id, "value");
            return link ? inferredType(link->fromNode, link->fromSocket)
                        : ShaderValueType::Scalar;
        }
        return ShaderValueType::Scalar;
    }

    const SubgraphInterfaceItem* interfaceItem(std::string_view key) const {
        const auto it = std::ranges::find(definition_.interface, key, &SubgraphInterfaceItem::key);
        return it == definition_.interface.end() ? nullptr : &*it;
    }

    const SubgraphDefinition& definition_;
    const NodeRegistry& registry_;
    std::unordered_map<NodeId, const NodeRecord*> nodes_;
    std::vector<ShaderHelper> helpers_;
    std::ostringstream statements_;
    std::unordered_set<std::string> helperNames_;
    std::unordered_map<std::string, ShaderValue> mainValues_;
    std::unordered_set<std::string> usedInterfaces_;
    std::vector<Frame> frames_;
};

constexpr std::string_view kCollapseShader = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;layout(binding=0)uniform sampler2D stateIn;
layout(std430,binding=0)buffer CollapseState{uint activityFlag;};uniform float threshold;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=textureSize(stateIn,0);if(any(greaterThanEqual(p,s)))return;if(texelFetch(stateIn,p,0).g>threshold)atomicOr(activityFlag,1u);})GLSL";

class SimulationSubgraphNode final : public node_support::ParameterNode {
public:
    SimulationSubgraphNode(SubgraphDefinition definition, const NodeRegistry& registry)
        : definition_(std::move(definition)), descriptor_(describeSubgraph(definition_)),
          registry_(registry) {
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
            SimulationGraphCompiler compiler(definition_, registry_);
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
            } else if (item.kind == SubgraphInterfaceKind::Slider) {
                if (inputIndex < inputs.size() &&
                    std::holds_alternative<float>(inputs[inputIndex])) {
                    if (used.contains(item.key))
                        uniform(program, ("param_" + name).c_str(),
                                std::get<float>(inputs[inputIndex]));
                } else if (used.contains(item.key)) {
                    uniform(program, ("param_" + name).c_str(),
                            parameter(parameters_, item.key.c_str(), item.defaultValue));
                }
                ++inputIndex;
            }
        }
    }

    SubgraphDefinition definition_;
    NodeDescriptor descriptor_;
    const NodeRegistry& registry_;
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
    NodeRegistry registry;
    registerBuiltInNodes(registry);
    return SimulationGraphCompiler(definition, registry).shader(initialization);
}

std::unique_ptr<NodeInstance> createSubgraphInstance(const SubgraphDefinition& definition,
                                                     const NodeRegistry& registry) {
    if (definition.execution != SubgraphExecution::Simulation)
        throw std::runtime_error("Pipeline subgraphs are not executable in this milestone");
    const auto errors = validateSubgraph(definition, registry);
    if (!errors.empty()) throw std::runtime_error(errors.front());
    return std::make_unique<SimulationSubgraphNode>(definition, registry);
}

} // namespace reaction
