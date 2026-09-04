#include "reaction/gpu/shader_ir.hpp"

#include <algorithm>
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

const LinkRecord* inputLink(const Graph& graph, NodeId node, std::string_view socket) {
    const auto found = std::ranges::find_if(graph.links(), [&](const LinkRecord& link) {
        return link.toNode == node && link.toSocket == socket;
    });
    return found == graph.links().end() ? nullptr : &*found;
}

class RegionBuilder final : public ShaderLoweringContext {
public:
    RegionBuilder(const Graph& graph, const CompileResult& compiled,
                  const std::vector<NodeId>& nodes)
        : graph_(graph), compiled_(compiled), regionNodes_(nodes.begin(), nodes.end()) {}

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
            currentLabel_ = record->label.empty() && descriptor
                ? descriptor->displayName : record->label;
            if (!instance->lowerShader(*this))
                throw std::runtime_error("Node does not support shader lowering: " + record->type);
            const auto value = values_.at(id);
            result.candidateOutputs.push_back(
                ShaderOutputRequirement{id, "result", value, -1, {}});
            const float operation = record->parameters.contains("operation") &&
                                    record->parameters["operation"].is_number()
                ? record->parameters["operation"].get<float>() : 0.0F;
            result.specializationKey += std::to_string(id) + ":" +
                std::to_string(static_cast<int>(mathOperation(operation))) + ";";
        }
        region_ = nullptr;
        return result;
    }

    ShaderValue input(std::string_view socket, std::string_view parameterKey,
                      float fallback) override {
        const auto* link = inputLink(graph_, currentNode_->id, socket);
        if (link && regionNodes_.contains(link->fromNode)) {
            const auto found = values_.find(link->fromNode);
            if (found == values_.end())
                throw std::runtime_error("Generated region is not in dependency order");
            return found->second;
        }
        if (link) {
            const bool image = compiled_.inferredOutputs.contains(link->fromNode) &&
                               compiled_.inferredOutputs.at(link->fromNode) == ValueType::Image2D;
            const std::string key = std::string(image ? "image:" : "scalar:") +
                std::to_string(link->fromNode) + ":" + link->fromSocket;
            return requireInput(key, image ? ShaderInputKind::Image : ShaderInputKind::Scalar,
                                link->fromNode, link->fromSocket, 0, {}, fallback);
        }
        const std::string key = "parameter:" + std::to_string(currentNode_->id) + ":" +
                                std::string(parameterKey);
        return requireInput(key, ShaderInputKind::Scalar, 0, {}, currentNode_->id,
                            std::string(parameterKey), fallback);
    }

    ShaderValue parameter(std::string_view key, float fallback) override {
        const std::string requirementKey = "parameter:" + std::to_string(currentNode_->id) +
                                           ":" + std::string(key);
        return requireInput(requirementKey, ShaderInputKind::Scalar, 0, {}, currentNode_->id,
                            std::string(key), fallback);
    }

    ShaderValue emit(std::string expression) override {
        ShaderValue result{ShaderValueType::Vec4, "v_" + std::to_string(currentNode_->id)};
        region_->instructions.push_back(
            ShaderInstruction{result, std::move(expression), currentNode_->id, currentLabel_});
        values_[currentNode_->id] = result;
        return result;
    }

    ShaderValueType valueType() const override { return ShaderValueType::Vec4; }

private:
    ShaderValue requireInput(const std::string& key, ShaderInputKind kind,
                             NodeId sourceNode, std::string sourceSocket,
                             NodeId parameterNode, std::string parameterKey,
                             float fallback) {
        if (const auto found = requirements_.find(key); found != requirements_.end())
            return region_->inputs[found->second].value;
        const auto index = region_->inputs.size();
        const std::string uniform = kind == ShaderInputKind::Image
            ? "inputImage_" + std::to_string(index)
            : "inputScalar_" + std::to_string(index);
        ShaderValue value{ShaderValueType::Vec4, kind == ShaderInputKind::Image
            ? "sample_" + std::to_string(index) : "vec4(" + uniform + ")"};
        const int binding = kind == ShaderInputKind::Image
            ? static_cast<int>(std::ranges::count_if(region_->inputs, [](const auto& input) {
                  return input.kind == ShaderInputKind::Image;
              })) : -1;
        requirements_[key] = index;
        region_->inputs.push_back(ShaderInputRequirement{kind, key, uniform, binding,
            sourceNode, std::move(sourceSocket), parameterNode, std::move(parameterKey),
            fallback, value});
        return value;
    }

    const Graph& graph_;
    const CompileResult& compiled_;
    std::unordered_set<NodeId> regionNodes_;
    ShaderRegion* region_ = nullptr;
    const NodeRecord* currentNode_ = nullptr;
    std::string currentLabel_;
    std::unordered_map<std::string, std::size_t> requirements_;
    std::unordered_map<NodeId, ShaderValue> values_;
};

ShaderRegion lowerRegion(const Graph& graph, const NodeRegistry& registry,
                         const CompileResult& compiled, const std::vector<NodeId>& nodes) {
    RegionBuilder builder(graph, compiled, nodes);
    return builder.build(registry, nodes);
}

int imageInputCount(const ShaderRegion& region) {
    return static_cast<int>(std::ranges::count_if(region.inputs, [](const auto& input) {
        return input.kind == ShaderInputKind::Image;
    }));
}

int scalarInputCount(const ShaderRegion& region) {
    return static_cast<int>(std::ranges::count_if(region.inputs, [](const auto& input) {
        return input.kind == ShaderInputKind::Scalar;
    }));
}

} // namespace

std::string mathGlslExpression(MathOperation operation,
                               const std::vector<ShaderValue>& operands,
                               const std::vector<ShaderValue>& remap,
                               ShaderValueType type) {
    const auto& a = operands.at(0).name;
    const auto scalar = [&](std::string_view value) {
        return type == ShaderValueType::Vec4 ? "vec4(" + std::string(value) + ")"
                                             : std::string(value);
    };
    switch (operation) {
    case MathOperation::Add: return "(" + a + "+" + operands.at(1).name + ")";
    case MathOperation::Subtract: return "(" + a + "-" + operands.at(1).name + ")";
    case MathOperation::Multiply: return "(" + a + "*" + operands.at(1).name + ")";
    case MathOperation::Divide: {
        const auto& b = operands.at(1).name;
        return "(" + a + "/((step(0.0," + b + ")*2.0-1.0)*max(abs(" + b + ")," +
               scalar("1e-6") + ")))";
    }
    case MathOperation::Power:
        return "((step(0.0," + a + ")*2.0-1.0)*pow(max(abs(" + a + ")," +
               scalar("1e-6") + ")," + operands.at(1).name + "))";
    case MathOperation::Minimum: return "min(" + a + "," + operands.at(1).name + ")";
    case MathOperation::Maximum: return "max(" + a + "," + operands.at(1).name + ")";
    case MathOperation::Absolute: return "abs(" + a + ")";
    case MathOperation::Sine: return "sin(" + a + ")";
    case MathOperation::Cosine: return "cos(" + a + ")";
    case MathOperation::Clamp:
        return "clamp(" + a + "," + operands.at(1).name + "," + operands.at(2).name + ")";
    case MathOperation::Remap:
        return "mix(" + remap.at(2).name + "," + remap.at(3).name + ",clamp((" + a + "-" +
               remap.at(0).name + ")/max(" + remap.at(1).name + "-" + remap.at(0).name +
               ",1e-6),0.0,1.0))";
    }
    return a;
}

std::vector<ShaderRegion> planMathShaderRegions(
    const Graph& graph, const NodeRegistry& registry, const CompileResult& compiled,
    int maximumImageInputs, int maximumScalarInputs) {
    std::unordered_set<NodeId> eligible;
    for (const auto id : compiled.order) {
        const auto* node = graph.findNode(id);
        if (node && node->type == "math" && compiled.inferredOutputs.contains(id) &&
            compiled.inferredOutputs.at(id) == ValueType::Image2D) eligible.insert(id);
    }

    std::unordered_map<NodeId, std::vector<NodeId>> eligiblePredecessors;
    std::unordered_map<NodeId, std::vector<NodeId>> consumers;
    for (const auto& link : graph.links()) {
        consumers[link.fromNode].push_back(link.toNode);
        if (eligible.contains(link.fromNode) && eligible.contains(link.toNode))
            eligiblePredecessors[link.toNode].push_back(link.fromNode);
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
        const auto found = std::ranges::find(region.candidateOutputs, id,
                                             &ShaderOutputRequirement::node);
        if (found == region.candidateOutputs.end()) continue;
        auto output = *found;
        output.binding = static_cast<int>(generated.outputs.size());
        output.imageName = "outputImage_" + std::to_string(generated.outputs.size());
        generated.outputs.push_back(std::move(output));
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
        else append("uniform float " + input.uniformName + ";");
    }
    append("void main() {");
    append("    ivec2 p=ivec2(gl_GlobalInvocationID.xy), s=imageSize(" +
           generated.outputs.front().imageName + ");");
    append("    if(any(greaterThanEqual(p,s))) return;");
    append("    vec2 uv=(vec2(p)+0.5)/vec2(s);");
    for (std::size_t index = 0; index < generated.inputs.size(); ++index) {
        const auto& input = generated.inputs[index];
        if (input.kind == ShaderInputKind::Image)
            append("    vec4 " + input.value.name + "=texture(" + input.uniformName + ",uv);");
    }
    for (const auto& instruction : region.instructions) {
        const auto first = lines.size() + 1;
        append("    // node " + std::to_string(instruction.contributor) + ": " +
               commentLabel(instruction.contributorLabel));
        append("    vec4 " + instruction.result.name + "=" + instruction.expression + ";");
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
        generated.specializationKey += std::to_string(output.node) + ",";
    return generated;
}

} // namespace reaction
