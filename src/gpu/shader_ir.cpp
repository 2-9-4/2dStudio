#include "reaction/gpu/shader_ir.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace reaction {
namespace {

std::string commentLabel(std::string value) {
    std::replace(value.begin(), value.end(), '\n', ' ');
    std::replace(value.begin(), value.end(), '\r', ' ');
    return value;
}

std::string identifier(std::string value) {
    for (char& c : value)
        if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
    return value;
}

std::string glslType(ShaderValueType type) {
    switch (type) {
    case ShaderValueType::Scalar: return "float";
    case ShaderValueType::Vec2: return "vec2";
    case ShaderValueType::Vec4: return "vec4";
    }
    return "float";
}

void replaceAll(std::string& value, std::string_view from, std::string_view to) {
    if (from.empty()) return;
    for (std::size_t at = 0; (at = value.find(from, at)) != std::string::npos;
         at += to.size()) value.replace(at, from.size(), to);
}

const LinkRecord* inputLink(const Graph& graph, NodeId node, std::string_view socket) {
    const auto found = std::ranges::find_if(graph.links(), [&](const LinkRecord& link) {
        return link.toNode == node && link.toSocket == socket;
    });
    return found == graph.links().end() ? nullptr : &*found;
}

class RegionBuilder final : public ShaderLoweringContext {
public:
    RegionBuilder(const Graph& graph, const NodeRegistry& registry,
                  const CompileResult& compiled, const std::vector<NodeId>& nodes)
        : graph_(graph), registry_(registry), compiled_(compiled),
          regionNodes_(nodes.begin(), nodes.end()) {}

    ShaderRegion build(const NodeRegistry& registry, const std::vector<NodeId>& nodes) {
        ShaderRegion result;
        region_ = &result;
        result.nodes = nodes;
        result.id = nodes.empty() ? 0 : nodes.back();
        for (const auto id : nodes) {
            const auto* record = graph_.findNode(id);
            if (!record) throw std::runtime_error("Generated shader references a missing node");
            auto instance = registry.create(record->type);
            if (!instance) throw std::runtime_error("Generated shader references an unknown node type");
            instance->setParameters(record->parameters);
            currentNode_ = record;
            NodeDescriptor storage;
            const auto* descriptor = resolveDescriptor(graph_, *record, registry, storage);
            if (!descriptor) throw std::runtime_error("Generated shader has no descriptor");
            currentDescriptor_ = descriptor;
            currentLabel_ = record->label.empty() && descriptor
                ? descriptor->displayName : record->label;
            if (!instance->lowerShader(*this))
                throw std::runtime_error("Node does not support shader lowering: " + record->type);
            for (const auto& socket : descriptor->sockets) {
                if (socket.direction != SocketDirection::Output) continue;
                const auto found = values_.find(valueKey(id, socket.key));
                if (found != values_.end())
                    result.candidateOutputs.push_back(
                        ShaderOutputRequirement{id, socket.key, found->second, -1, {}});
            }
            const auto variant = instance->shaderVariantKey(record->parameters);
            if (!variant.empty())
                result.specializationKey += std::to_string(id) + ":" + variant + ";";
        }
        region_ = nullptr;
        return result;
    }

    ShaderValue input(std::string_view socket, std::string_view parameterKey,
                      float fallback) override {
        return inputAt(socket, "uv", parameterKey, fallback);
    }

    ShaderValue inputAt(std::string_view socket, std::string_view uvExpression,
                        std::string_view parameterKey, float fallback) override {
        const auto* link = inputLink(graph_, currentNode_->id, socket);
        if (link && regionNodes_.contains(link->fromNode)) {
            if (uvExpression != "uv")
                throw std::runtime_error("Neighborhood input crosses an in-region value");
            const auto found = values_.find(valueKey(link->fromNode, link->fromSocket));
            if (found == values_.end())
                throw std::runtime_error("Generated region is not in dependency order");
            return found->second;
        }
        if (link) {
            const bool image = compiled_.inferredOutputs.contains(link->fromNode) &&
                               compiled_.inferredOutputs.at(link->fromNode) == ValueType::Image2D;
            // AnyNumeric/AnyVector upstream values broadcast as either a constant
            // or a field image. A fused region cannot know which at lowering time,
            // so it must be able to read the value both ways; native evaluation
            // supplies a field image for these producers whenever their inputs are
            // fields.
            const bool flexible =
                !image && anySourceOutput(link->fromNode, link->fromSocket,
                                          ValueType::AnyNumeric);
            const bool flexibleVector =
                !image && !flexible &&
                anySourceOutput(link->fromNode, link->fromSocket, ValueType::AnyVector);
            const std::string key =
                std::string(image ? "image:" : flexible ? "flex:" :
                              flexibleVector ? "flexv:" : "scalar:") +
                std::to_string(link->fromNode) + ":" + link->fromSocket;
            const auto required = requireInput(
                key, image ? ShaderInputKind::Image
                           : flexible ? ShaderInputKind::Flexible
                                      : flexibleVector ? ShaderInputKind::FlexibleVector
                                                       : ShaderInputKind::Scalar,
                link->fromNode, link->fromSocket, 0, {}, fallback);
            if (uvExpression != "uv") {
                const auto& input = region_->inputs[requirements_.at(key)];
                if (input.kind == ShaderInputKind::Image)
                    return {ShaderValueType::Vec4, "texture(" + input.uniformName + ",fract(" +
                        std::string(uvExpression) + "))"};
                if (input.kind == ShaderInputKind::Flexible)
                    return {ShaderValueType::Scalar, "(" + input.hasImageUniformName +
                        "!=0?texture(" + input.uniformName + ",fract(" +
                        std::string(uvExpression) + ")).r:" + input.scalarUniformName + ")"};
                if (input.kind == ShaderInputKind::FlexibleVector)
                    return {ShaderValueType::Vec4, "vec4((" + input.hasImageUniformName +
                        "!=0?texture(" + input.uniformName + ",fract(" +
                        std::string(uvExpression) + ")).rg:" + input.vectorUniformName +
                        "),0.0,1.0)"};
            }
            return required;
        }
        const std::string key = "parameter:" + std::to_string(currentNode_->id) + ":" +
                                std::string(parameterKey);
        return requireInput(key, ShaderInputKind::Scalar, 0, {}, currentNode_->id,
                            std::string(parameterKey), fallback);
    }

    ShaderValue inputTexel(std::string_view socket, std::string_view pixelExpression,
                           std::string_view parameterKey, float fallback) override {
        const auto* link = inputLink(graph_, currentNode_->id, socket);
        if (link && regionNodes_.contains(link->fromNode))
            throw std::runtime_error("Neighborhood input crosses an in-region value");
        if (!link) return parameter(parameterKey, fallback);
        const bool image = compiled_.inferredOutputs.contains(link->fromNode) &&
                           compiled_.inferredOutputs.at(link->fromNode) == ValueType::Image2D;
        if (!image) return inputAt(socket, pixelExpression, parameterKey, fallback);
        const std::string key = "image:" + std::to_string(link->fromNode) + ":" +
                                link->fromSocket;
        const auto required = requireInput(
            key, ShaderInputKind::Image, link->fromNode, link->fromSocket, 0, {}, fallback);
        const auto& uniformName = region_->inputs[requirements_.at(key)].uniformName;
        return {ShaderValueType::Vec4, "texelFetch(" + uniformName + ",clamp(ivec2(" +
            std::string(pixelExpression) + "),ivec2(0),textureSize(" + uniformName +
            ",0)-ivec2(1)),0)"};
    }

    ShaderValue parameter(std::string_view key, float fallback) override {
        // A connected slider-style input socket overrides the parameter uniform.
        if (inputLink(graph_, currentNode_->id, key))
            return input(key, key, fallback);
        const std::string requirementKey = "parameter:" + std::to_string(currentNode_->id) +
                                           ":" + std::string(key);
        return requireInput(requirementKey, ShaderInputKind::Scalar, 0, {}, currentNode_->id,
                            std::string(key), fallback);
    }

    std::string helper(std::string_view name, std::string source) override {
        const std::string actual = identifier(std::string(name) + "_" +
                                              std::to_string(currentNode_->id));
        if (helperNames_.insert(actual).second) {
            replaceAll(source, name, actual);
            region_->helpers.push_back({actual, std::move(source)});
        }
        return actual;
    }

    ShaderValue emit(std::string expression, std::string_view socket = {}) override {
        const auto outputSocket = socket.empty() ? defaultOutputSocket() : std::string(socket);
        ShaderValue result{ShaderValueType::Vec4, "v_" + std::to_string(currentNode_->id) +
                           "_" + identifier(outputSocket)};
        region_->instructions.push_back(
            ShaderInstruction{result, std::move(expression), currentNode_->id, currentLabel_});
        values_[valueKey(currentNode_->id, outputSocket)] = result;
        return result;
    }

    ShaderValueType valueType() const override { return ShaderValueType::Vec4; }

private:
    static std::string valueKey(NodeId id, std::string_view socket) {
        return std::to_string(id) + ":" + std::string(socket);
    }

    std::string defaultOutputSocket() const {
        if (currentDescriptor_) for (const auto& socket : currentDescriptor_->sockets)
            if (socket.direction == SocketDirection::Output) return socket.key;
        return "result";
    }

    ShaderValue requireInput(const std::string& key, ShaderInputKind kind,
                             NodeId sourceNode, std::string sourceSocket,
                             NodeId parameterNode, std::string parameterKey,
                             float fallback) {
        if (const auto found = requirements_.find(key); found != requirements_.end())
            return region_->inputs[found->second].value;
        const auto index = region_->inputs.size();
        const std::string suffix = std::to_string(index);
        ShaderValue value{kind == ShaderInputKind::Image ? ShaderValueType::Vec4
                                                        : ShaderValueType::Scalar,
                          kind == ShaderInputKind::Image ? "sample_" + suffix
                                                         : "inputScalar_" + suffix};
        const int binding = kind == ShaderInputKind::Image ||
                            kind == ShaderInputKind::Flexible ||
                            kind == ShaderInputKind::FlexibleVector
            ? static_cast<int>(std::ranges::count_if(region_->inputs, [](const auto& input) {
                  return input.kind == ShaderInputKind::Image ||
                         input.kind == ShaderInputKind::Flexible ||
                         input.kind == ShaderInputKind::FlexibleVector;
              })) : -1;
        ShaderInputRequirement requirement{kind, key,
            kind == ShaderInputKind::Image ? "inputImage_" + suffix : "inputScalar_" + suffix,
            binding, sourceNode, std::move(sourceSocket), parameterNode,
            std::move(parameterKey), fallback, value};
        if (kind == ShaderInputKind::Flexible) {
            requirement.uniformName = "inputImage_" + suffix;
            requirement.value.name = "inputFlex_" + suffix;
            requirement.scalarUniformName = "inputScalar_" + suffix;
            requirement.hasImageUniformName = "inputHasImage_" + suffix;
            value.name = requirement.value.name;
        }
        if (kind == ShaderInputKind::FlexibleVector) {
            requirement.uniformName = "inputImage_" + suffix;
            requirement.value.name = "inputFlexVec_" + suffix;
            requirement.vectorUniformName = "inputVec_" + suffix;
            requirement.hasImageUniformName = "inputHasImage_" + suffix;
            value = {ShaderValueType::Vec4, requirement.value.name};
        }
        requirements_[key] = index;
        region_->inputs.push_back(std::move(requirement));
        return value;
    }

    bool anySourceOutput(NodeId node, std::string_view socket, ValueType type) const {
        const auto* record = graph_.findNode(node);
        if (!record) return false;
        NodeDescriptor storage;
        const auto* descriptor = resolveDescriptor(graph_, *record, registry_, storage);
        if (!descriptor) return false;
        for (const auto& port : descriptor->sockets)
            if (port.direction == SocketDirection::Output && port.key == socket)
                return port.type == type;
        return false;
    }

    const Graph& graph_;
    const NodeRegistry& registry_;
    const CompileResult& compiled_;
    std::unordered_set<NodeId> regionNodes_;
    ShaderRegion* region_ = nullptr;
    const NodeRecord* currentNode_ = nullptr;
    const NodeDescriptor* currentDescriptor_ = nullptr;
    std::string currentLabel_;
    std::unordered_map<std::string, std::size_t> requirements_;
    std::unordered_map<std::string, ShaderValue> values_;
    std::unordered_set<std::string> helperNames_;
};

ShaderRegion lowerRegion(const Graph& graph, const NodeRegistry& registry,
                         const CompileResult& compiled, const std::vector<NodeId>& nodes) {
    RegionBuilder builder(graph, registry, compiled, nodes);
    return builder.build(registry, nodes);
}

int imageInputCount(const ShaderRegion& region) {
    return static_cast<int>(std::ranges::count_if(region.inputs, [](const auto& input) {
        return input.kind == ShaderInputKind::Image ||
               input.kind == ShaderInputKind::Flexible ||
               input.kind == ShaderInputKind::FlexibleVector;
    }));
}

int scalarInputCount(const ShaderRegion& region) {
    return static_cast<int>(std::ranges::count_if(region.inputs, [](const auto& input) {
        return input.kind == ShaderInputKind::Scalar ||
               input.kind == ShaderInputKind::Flexible ||
               input.kind == ShaderInputKind::FlexibleVector;
    }));
}

} // namespace

std::string mathGlslExpression(MathOperation operation,
                               const std::vector<ShaderValue>& operands,
                               const std::vector<ShaderValue>& remap,
                               ShaderValueType type) {
    const auto convert = [&](const ShaderValue& value) {
        if (value.type == type) return value.name;
        return glslType(type) + "(" + value.name + ")";
    };
    const auto a = convert(operands.at(0));
    const auto scalar = [&](std::string_view value) {
        return glslType(type) + "(" + std::string(value) + ")";
    };
    switch (operation) {
    case MathOperation::Add: return "(" + a + "+" + convert(operands.at(1)) + ")";
    case MathOperation::Subtract: return "(" + a + "-" + convert(operands.at(1)) + ")";
    case MathOperation::Multiply: return "(" + a + "*" + convert(operands.at(1)) + ")";
    case MathOperation::Divide: {
        const auto b = convert(operands.at(1));
        return "(" + a + "/((step(0.0," + b + ")*2.0-1.0)*max(abs(" + b + ")," +
               scalar("1e-6") + ")))";
    }
    case MathOperation::Power:
        return "((step(0.0," + a + ")*2.0-1.0)*pow(max(abs(" + a + ")," +
               scalar("1e-6") + ")," + convert(operands.at(1)) + "))";
    case MathOperation::Minimum: return "min(" + a + "," + convert(operands.at(1)) + ")";
    case MathOperation::Maximum: return "max(" + a + "," + convert(operands.at(1)) + ")";
    case MathOperation::Absolute: return "abs(" + a + ")";
    case MathOperation::Sine: return "sin(" + a + ")";
    case MathOperation::Cosine: return "cos(" + a + ")";
    case MathOperation::Clamp:
        return "clamp(" + a + "," + convert(operands.at(1)) + "," +
               convert(operands.at(2)) + ")";
    case MathOperation::Remap:
        return "mix(" + convert(remap.at(2)) + "," + convert(remap.at(3)) +
               ",clamp((" + a + "-" + convert(remap.at(0)) + ")/max(" +
               convert(remap.at(1)) + "-" + convert(remap.at(0)) + "," +
               scalar("1e-6") + ")," + scalar("0.0") + "," + scalar("1.0") + "))";
    case MathOperation::Floor: return "floor(" + a + ")";
    case MathOperation::Ceil: return "ceil(" + a + ")";
    case MathOperation::Round: return "round(" + a + ")";
    case MathOperation::Fraction: return "fract(" + a + ")";
    case MathOperation::SquareRoot: return "sqrt(max(" + a + "," + scalar("0.0") + "))";
    case MathOperation::Exp: return "exp(" + a + ")";
    case MathOperation::NaturalLog: return "log(max(" + a + "," + scalar("1e-6") + "))";
    case MathOperation::Log2: return "log2(max(" + a + "," + scalar("1e-6") + "))";
    case MathOperation::Sign: return "sign(" + a + ")";
    case MathOperation::Tangent: return "tan(" + a + ")";
    case MathOperation::ArcSine:
        return "asin(clamp(" + a + "," + scalar("-1.0") + "," + scalar("1.0") + "))";
    case MathOperation::ArcCosine:
        return "acos(clamp(" + a + "," + scalar("-1.0") + "," + scalar("1.0") + "))";
    case MathOperation::ArcTangent: return "atan(" + a + ")";
    case MathOperation::Modulo: return "mod(" + a + "," + convert(operands.at(1)) + ")";
    case MathOperation::ArcTangent2: return "atan(" + a + "," + convert(operands.at(1)) + ")";
    case MathOperation::Step: return "step(" + a + "," + convert(operands.at(1)) + ")";
    case MathOperation::Hypotenuse: {
        const auto b = convert(operands.at(1));
        return "sqrt((" + a + ")*(" + a + ")+(" + b + ")*(" + b + "))";
    }
    case MathOperation::Smoothstep: {
        const auto b = convert(operands.at(1));
        const auto c = convert(operands.at(2));
        const auto t = "clamp((" + a + "-" + b + ")/max(" + c + "-" + b + "," +
                       scalar("1e-6") + ")," + scalar("0.0") + "," + scalar("1.0") + ")";
        return "(" + t + "*" + t + "*(3.0-2.0*" + t + "))";
    }
    case MathOperation::MultiplyAccumulate:
        return "(" + a + "*" + convert(operands.at(1)) + "+" +
               convert(operands.at(2)) + ")";
    }
    return a;
}

std::vector<ShaderRegion> planShaderRegions(
    const Graph& graph, const NodeRegistry& registry, const CompileResult& compiled,
    int maximumImageInputs, int maximumScalarInputs) {
    std::unordered_set<NodeId> eligible;
    for (const auto id : compiled.order) {
        const auto* node = graph.findNode(id);
        NodeDescriptor storage;
        const auto* descriptor = node ? resolveDescriptor(graph, *node, registry, storage) : nullptr;
        if (!descriptor || !descriptor->lowerable || !compiled.inferredOutputs.contains(id) ||
            compiled.inferredOutputs.at(id) != ValueType::Image2D) continue;
        auto instance = registry.create(node->type);
        if (!instance) continue;
        instance->setParameters(node->parameters);
        if (instance->supportsRegionFusion(node->parameters)) eligible.insert(id);
    }

    std::unordered_map<NodeId, std::vector<NodeId>> eligiblePredecessors;
    std::unordered_map<NodeId, std::vector<NodeId>> consumers;
    for (const auto& link : graph.links()) {
        const auto* target = graph.findNode(link.toNode);
        NodeDescriptor storage;
        const auto* descriptor = target
            ? resolveDescriptor(graph, *target, registry, storage) : nullptr;
        const bool neighborhoodBoundary = descriptor &&
            !descriptor->neighborhoodSocket.empty() &&
            link.toSocket == descriptor->neighborhoodSocket;
        auto& targets = consumers[link.fromNode];
        if (std::ranges::find(targets, link.toNode) == targets.end()) targets.push_back(link.toNode);
        if (!neighborhoodBoundary && eligible.contains(link.fromNode) && eligible.contains(link.toNode)) {
            auto& predecessors = eligiblePredecessors[link.toNode];
            if (std::ranges::find(predecessors, link.fromNode) == predecessors.end())
                predecessors.push_back(link.fromNode);
        }
    }
    const auto continuation = [&](NodeId id) -> NodeId {
        const auto found = consumers.find(id);
        if (found == consumers.end() || found->second.size() != 1) return 0;
        const auto next = found->second.front();
        return eligible.contains(next) && eligiblePredecessors[next].size() == 1 ? next : 0;
    };

    std::unordered_set<NodeId> hasChainPredecessor;
    for (const auto id : eligible) if (const auto next = continuation(id))
        hasChainPredecessor.insert(next);

    std::vector<ShaderRegion> regions;
    std::unordered_set<NodeId> visited;
    for (const auto start : compiled.order) {
        if (!eligible.contains(start) || hasChainPredecessor.contains(start) || visited.contains(start))
            continue;
        std::vector<NodeId> chain;
        for (NodeId id = start; id != 0 && !visited.contains(id); id = continuation(id)) {
            chain.push_back(id);
            visited.insert(id);
        }

        std::size_t first = 0;
        while (first < chain.size()) {
            std::size_t last = first + 1;
            std::vector<NodeId> acceptedNodes(
                chain.begin() + static_cast<std::ptrdiff_t>(first),
                chain.begin() + static_cast<std::ptrdiff_t>(last));
            ShaderRegion accepted = lowerRegion(graph, registry, compiled, acceptedNodes);
            while (last < chain.size()) {
                const std::vector<NodeId> candidateNodes(
                    chain.begin() + static_cast<std::ptrdiff_t>(first),
                    chain.begin() + static_cast<std::ptrdiff_t>(last + 1));
                auto candidate = lowerRegion(graph, registry, compiled, candidateNodes);
                if (imageInputCount(candidate) > maximumImageInputs ||
                    scalarInputCount(candidate) > maximumScalarInputs) break;
                accepted = std::move(candidate);
                ++last;
            }
            regions.push_back(std::move(accepted));
            first = last;
        }
    }
    // Defensive coverage for an eligible node omitted by malformed chain metadata.
    for (const auto id : compiled.order) if (eligible.contains(id) && !visited.contains(id))
        regions.push_back(lowerRegion(graph, registry, compiled, {id}));
    return regions;
}

GeneratedShader generateComputeShader(const ShaderRegion& region,
                                      const std::vector<NodeId>& materializedNodes) {
    GeneratedShader generated;
    generated.inputs = region.inputs;
    generated.specializationKey = region.specializationKey;
    for (const auto id : materializedNodes) {
        for (const auto& candidate : region.candidateOutputs) {
            if (candidate.node != id) continue;
            auto output = candidate;
            output.binding = static_cast<int>(generated.outputs.size());
            output.imageName = "outputImage_" + std::to_string(generated.outputs.size());
            generated.outputs.push_back(std::move(output));
        }
    }
    if (generated.outputs.empty()) throw std::runtime_error("Generated shader has no outputs");

    std::vector<std::string> lines;
    const auto append = [&](std::string line) { lines.push_back(std::move(line)); };
    append("#version 430");
    append("layout(local_size_x=16, local_size_y=16) in;");
    for (const auto& output : generated.outputs)
        append("layout(rgba16f,binding=" + std::to_string(output.binding) +
               ") writeonly uniform image2D " + output.imageName + ";");
    for (const auto& input : generated.inputs) {
        if (input.kind == ShaderInputKind::Image)
            append("layout(binding=" + std::to_string(input.binding) + ") uniform sampler2D " +
                   input.uniformName + ";");
        else if (input.kind == ShaderInputKind::Flexible) {
            append("layout(binding=" + std::to_string(input.binding) + ") uniform sampler2D " +
                   input.uniformName + ";");
            append("uniform float " + input.scalarUniformName + ";");
            append("uniform int " + input.hasImageUniformName + ";");
        }
        else if (input.kind == ShaderInputKind::FlexibleVector) {
            append("layout(binding=" + std::to_string(input.binding) + ") uniform sampler2D " +
                   input.uniformName + ";");
            append("uniform vec2 " + input.vectorUniformName + ";");
            append("uniform int " + input.hasImageUniformName + ";");
        }
        else append("uniform float " + input.uniformName + ";");
    }
    append("vec2 pixelSize;");
    for (const auto& helper : region.helpers) {
        std::istringstream helperSource(helper.source);
        for (std::string line; std::getline(helperSource, line);) append(std::move(line));
    }
    append("void main() {");
    append("    ivec2 p=ivec2(gl_GlobalInvocationID.xy), s=imageSize(" +
           generated.outputs.front().imageName + ");");
    append("    if(any(greaterThanEqual(p,s))) return;");
    append("    vec2 uv=(vec2(p)+0.5)/vec2(s);");
    append("    pixelSize=1.0/vec2(s);");
    for (std::size_t index = 0; index < generated.inputs.size(); ++index) {
        const auto& input = generated.inputs[index];
        if (input.kind == ShaderInputKind::Image)
            append("    vec4 " + input.value.name + "=texture(" + input.uniformName + ",uv);");
        else if (input.kind == ShaderInputKind::Flexible)
            append("    float " + input.value.name + "=(" + input.hasImageUniformName +
                   "!=0?texture(" + input.uniformName + ",uv).r:" +
                   input.scalarUniformName + ");");
        else if (input.kind == ShaderInputKind::FlexibleVector)
            append("    vec4 " + input.value.name + "=vec4((" + input.hasImageUniformName +
                   "!=0?texture(" + input.uniformName + ",uv).rg:" +
                   input.vectorUniformName + "),0.0,1.0);");
    }
    for (const auto& instruction : region.instructions) {
        const auto first = lines.size() + 1;
        append("    // node " + std::to_string(instruction.contributor) + ": " +
               commentLabel(instruction.contributorLabel));
        append("    " + glslType(instruction.result.type) + " " + instruction.result.name +
               "=" + instruction.expression + ";");
        generated.annotations.push_back({instruction.contributor,
            instruction.contributorLabel, first, lines.size()});
    }
    for (const auto& output : generated.outputs) {
        const auto first = lines.size() + 1;
        append("    // materialized output for node " + std::to_string(output.node));
        append("    imageStore(" + output.imageName + ",p," + output.value.name + ");");
        generated.annotations.push_back({output.node, "Materialized output", first, lines.size()});
    }
    append("}");
    std::ostringstream source;
    for (const auto& line : lines) source << line << '\n';
    generated.source = source.str();
    generated.specializationKey += "outputs:";
    for (const auto& output : generated.outputs)
        generated.specializationKey += std::to_string(output.node) + ":" + output.socket + ",";
    return generated;
}

} // namespace reaction
