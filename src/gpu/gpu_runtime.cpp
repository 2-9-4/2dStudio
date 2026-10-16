#include "reaction/gpu/gpu_runtime.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace reaction {
namespace {

std::size_t reportedLine(std::string_view log) {
    for (auto colon = log.find(':'); colon != std::string_view::npos;
         colon = log.find(':', colon + 1)) {
        std::size_t end = colon + 1;
        while (end < log.size() && std::isdigit(static_cast<unsigned char>(log[end]))) ++end;
        if (end > colon + 1)
            return static_cast<std::size_t>(std::stoul(std::string(log.substr(colon + 1, end - colon - 1))));
    }
    const auto open = log.find('(');
    if (open != std::string_view::npos) {
        std::size_t end = open + 1;
        while (end < log.size() && std::isdigit(static_cast<unsigned char>(log[end]))) ++end;
        if (end > open + 1 && end < log.size() && log[end] == ')')
            return static_cast<std::size_t>(std::stoul(std::string(log.substr(open + 1, end - open - 1))));
    }
    return 0;
}

std::string sourceExcerpt(std::string_view source, std::size_t reported) {
    std::vector<std::string_view> lines;
    std::size_t begin = 0;
    while (begin <= source.size()) {
        const auto end = source.find('\n', begin);
        lines.push_back(source.substr(begin, end == std::string_view::npos ? source.size() - begin : end - begin));
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    if (lines.empty()) return {};
    const std::size_t center = reported > 0 ? std::min(reported, lines.size()) : 1;
    const std::size_t first = center > 3 ? center - 3 : 1;
    const std::size_t last = std::min(lines.size(), center + 3);
    std::string result;
    for (std::size_t line = first; line <= last; ++line) {
        result += line == center ? "> " : "  ";
        result += std::to_string(line) + ": " + std::string(lines[line - 1]) + "\n";
    }
    return result;
}

nlohmann::json withConnectedParameterOverrides(const NodeDescriptor& descriptor,
                                               const std::vector<std::string>& keys,
                                               const std::vector<Value>& inputs,
                                               const nlohmann::json& parameters) {
    nlohmann::json result = parameters;
    for (std::size_t index = 0; index < keys.size() && index < inputs.size(); ++index) {
        const auto* number = std::get_if<float>(&inputs[index]);
        if (!number) continue;
        const bool isParameter = std::ranges::any_of(descriptor.parameters,
            [&](const ParameterDescriptor& parameter) { return parameter.key == keys[index]; });
        if (isParameter) result[keys[index]] = *number;
    }
    return result;
}

void applyUnconnectedParameterInputs(const NodeDescriptor& descriptor,
                                     const std::vector<std::string>& keys,
                                     std::vector<Value>& inputs,
                                     const nlohmann::json& parameters) {
    for (std::size_t index = 0; index < keys.size() && index < inputs.size(); ++index) {
        if (!std::holds_alternative<std::monostate>(inputs[index])) continue;
        const auto found = std::ranges::find(descriptor.parameters, keys[index],
                                             &ParameterDescriptor::key);
        if (found == descriptor.parameters.end()) continue;
        if (found->control != ParameterDescriptor::Control::Float &&
            found->control != ParameterDescriptor::Control::Integer) continue;
        inputs[index] = parameters.contains(found->key) && parameters[found->key].is_number()
            ? parameters[found->key].get<float>() : found->defaultValue;
    }
}

std::string simulationSignature(const SubgraphDefinition& definition) {
    try {
        std::string result = generateSimulationShader(definition, true);
        result.push_back('\0');
        result += generateSimulationShader(definition, false);
        // Init/update endpoints do not include presentation outputs. Include their
        // mappings so changing an exported channel still refreshes the executor.
        for (const auto& item : definition.interface) {
            result.push_back('\0');
            result += item.key + ":" + std::to_string(static_cast<int>(item.kind)) + ":" +
                      std::to_string(static_cast<int>(item.contract)) + ":" +
                      std::to_string(item.defaultValue);
            if (item.kind != SubgraphInterfaceKind::Output) continue;
            const auto endpoint = std::ranges::find_if(definition.body.nodes(), [&](const auto& node) {
                return node.type == "subgraph_output" &&
                       node.parameters.value("key", std::string{}) == item.key;
            });
            if (endpoint == definition.body.nodes().end()) continue;
            const auto link = std::ranges::find_if(definition.body.links(), [&](const auto& candidate) {
                return candidate.toNode == endpoint->id && candidate.toSocket == "value";
            });
            if (link != definition.body.links().end())
                result += ":" + std::to_string(link->fromNode) + ":" + link->fromSocket;
        }
        return result;
    } catch (const std::exception& error) {
        // Keep an invalid editable subgraph addressable by the runtime so its
        // node can retain a local diagnostic while the editor repairs it.
        return "error:" + std::string(error.what());
    }
}

GLuint compile(GLenum kind, std::string_view source, std::string_view label) {
    const auto shader = glCreateShader(kind);
    const auto* text = source.data();
    const auto length = static_cast<GLint>(source.size());
    glShaderSource(shader, 1, &text, &length);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok == GL_FALSE) {
        GLint size = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &size);
        std::string log(static_cast<std::size_t>(std::max(size, 1)), '\0');
        GLsizei written = 0;
        glGetShaderInfoLog(shader, size, &written, log.data());
        log.resize(static_cast<std::size_t>(std::max(written, 0)));
        while (!log.empty() && log.back() == '\0') log.pop_back();
        glDeleteShader(shader);
        const auto line = reportedLine(log);
        throw std::runtime_error("Shader compilation failed [" + std::string(label) + "]:\n" + log +
                                 "\nSource excerpt:\n" + sourceExcerpt(source, line));
    }
    return shader;
}

GLuint link(std::initializer_list<GLuint> shaders, std::string_view label) {
    const auto program = glCreateProgram();
    for (const auto shader : shaders) glAttachShader(program, shader);
    glLinkProgram(program);
    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    for (const auto shader : shaders) {
        glDetachShader(program, shader);
        glDeleteShader(shader);
    }
    if (ok == GL_FALSE) {
        GLint size = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &size);
        std::string log(static_cast<std::size_t>(std::max(size, 1)), '\0');
        GLsizei written = 0;
        glGetProgramInfoLog(program, size, &written, log.data());
        log.resize(static_cast<std::size_t>(std::max(written, 0)));
        while (!log.empty() && log.back() == '\0') log.pop_back();
        glDeleteProgram(program);
        throw std::runtime_error("Program linking failed [" + std::string(label) + "]: " + log);
    }
    return program;
}

constexpr std::string_view kPreviewVertex = R"GLSL(#version 430
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
})GLSL";

constexpr std::string_view kPreviewFragment = R"GLSL(#version 430
in vec2 uv;
layout(location=0) out vec4 color;
uniform sampler2D sourceImage;
void main() { color = vec4(texture(sourceImage, uv).rgb, 1.0); }
)GLSL";

} // namespace

struct GraphRuntime::FusionState {
    struct RegionRuntime {
        ShaderRegion region;
        GeneratedShader generated;
        GLuint program = 0;
        std::vector<GLuint> textures;
        int width = 0;
        int height = 0;
        std::string diagnostic;
        std::size_t errorLine = 0;
        double milliseconds = 0.0;
        bool runtimeFallback = false;
        std::vector<NodeId> materializedNodes;
        std::vector<ShaderInputKind> boundarySignature;
        std::size_t specializationCount = 0;
    };

    ~FusionState() {
        for (auto& item : regions) {
            for (const auto texture : item.textures) if (texture) glDeleteTextures(1, &texture);
        }
    }

    bool enabled = true;
    std::optional<NodeId> previewNode;
    std::string signature;
    std::string planningDiagnostic;
    std::vector<RegionRuntime> regions;
    std::unordered_map<NodeId, std::size_t> regionForNode;
};

GpuRuntime::GpuRuntime() {
    previewProgram_ = link({compile(GL_VERTEX_SHADER, kPreviewVertex, "Preview / vertex"),
                            compile(GL_FRAGMENT_SHADER, kPreviewFragment, "Preview / fragment")},
                           "Preview");
}

GpuRuntime::~GpuRuntime() {
    for (const auto& [_, program] : computeCache_) glDeleteProgram(program);
    if (previewProgram_) glDeleteProgram(previewProgram_);
}

GLuint GpuRuntime::compileCompute(std::string_view source, std::string_view label) const {
    return link({compile(GL_COMPUTE_SHADER, source, label)}, label);
}

GLuint GpuRuntime::compileComputeCached(std::string_view source, std::string_view label) {
    const std::string key(source);
    if (const auto found = computeCache_.find(key); found != computeCache_.end()) return found->second;
    const auto program = compileCompute(source, label);
    computeCache_.emplace(key, program);
    return program;
}

ComputeCompileResult GpuRuntime::tryCompileCompute(std::string_view source,
                                                   std::string_view label) const {
    try {
        return {compileCompute(source, label), {}, 0};
    } catch (const std::exception& error) {
        const std::string log = error.what();
        return {0, log, reportedLine(log)};
    }
}

ComputeCompileResult GpuRuntime::tryCompileComputeCached(std::string_view source,
                                                         std::string_view label) {
    try {
        return {compileComputeCached(source, label), {}, 0};
    } catch (const std::exception& error) {
        const std::string log = error.what();
        return {0, log, reportedLine(log)};
    }
}

int GpuRuntime::maximumComputeTextureInputs() const {
    GLint value = 0;
    glGetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS, &value);
    return std::max(value, 1);
}

int GpuRuntime::maximumComputeUniformComponents() const {
    GLint value = 0;
    glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_COMPONENTS, &value);
    return std::max(value, 1);
}

GLuint GpuRuntime::createTexture(int width, int height, GLenum format) const {
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, format, width, height);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

void GpuRuntime::ensureTexture(GLuint& texture, int& currentWidth, int& currentHeight,
                               int width, int height, GLenum format) const {
    if (texture != 0 && currentWidth == width && currentHeight == height) return;
    if (texture != 0) glDeleteTextures(1, &texture);
    texture = createTexture(width, height, format);
    currentWidth = width;
    currentHeight = height;
}

void GpuRuntime::dispatch(GLuint program, int width, int height) const {
    glUseProgram(program);
    glDispatchCompute(static_cast<GLuint>((width + 15) / 16),
                      static_cast<GLuint>((height + 15) / 16), 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

void GpuRuntime::drawFullscreen(GLuint texture, GLuint vertexArray,
                                int framebufferWidth, int framebufferHeight) const {
    glViewport(0, 0, framebufferWidth, framebufferHeight);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.025F, 0.025F, 0.03F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    if (texture == 0) return;
    glUseProgram(previewProgram_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glUniform1i(glGetUniformLocation(previewProgram_, "sourceImage"), 0);
    glBindVertexArray(vertexArray);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

GraphRuntime::GraphRuntime(Graph& graph, const NodeRegistry& registry, GpuRuntime& gpu)
    : graph_(graph), registry_(registry), gpu_(gpu), fusion_(std::make_unique<FusionState>()) {
    rebuild();
}

GraphRuntime::~GraphRuntime() { clear(); }

void GraphRuntime::clear() {
    fusion_.reset();
    for (const auto& [_, query] : timerQueries_) glDeleteQueries(1, &query);
    timerQueries_.clear();
    values_.clear();
    executionPlan_.clear();
    instances_.clear();
    timings_.clear();
    previousParameters_.clear();
    subgraphSignatures_.clear();
    typeErrors_.clear();
    pendingResets_.clear();
    evaluated_.clear();
    frame_ = 0;
}

void GraphRuntime::rebuild() {
    const auto candidate = graph_.compile(registry_);
    // Keep the compile result invalid for editor diagnostics, but still build
    // instances for reachable nodes. Native evaluation can then report a
    // node-local requiresImage/type error instead of going completely dark.
    compiled_ = candidate;
    std::unordered_map<NodeId, std::unique_ptr<NodeInstance>> next;
    std::unordered_map<NodeId, std::string> instantiationErrors;
    for (const auto& node : graph_.nodes()) {
        auto found = instances_.find(node.id);
        if (found != instances_.end() && node.type != "subgraph" &&
            found->second->descriptor().type == node.type) {
            next.emplace(node.id, std::move(found->second));
        } else if (node.type == "subgraph") {
            if (const auto* definition = resolveSubgraph(graph_, node.subgraphId)) {
                const auto signature = simulationSignature(*definition);
                const auto previous = subgraphSignatures_.find(node.id);
                if (found != instances_.end() && previous != subgraphSignatures_.end() &&
                    previous->second == signature) {
                    next.emplace(node.id, std::move(found->second));
                } else {
                    try {
                        auto instance = createSubgraphInstance(*definition, registry_);
                        pendingResets_.insert(node.id);
                        next.emplace(node.id, std::move(instance));
                    } catch (const std::exception& error) {
                        // A simulation shader is generated when its instance is
                        // created. Keep that failure local to this graph node so
                        // the editor can render its existing red error state.
                        instantiationErrors.emplace(node.id, error.what());
                    }
                }
                subgraphSignatures_[node.id] = signature;
            }
        } else if (auto instance = registry_.create(node.type)) {
            if (instance->descriptor().stateful) pendingResets_.insert(node.id);
            next.emplace(node.id, std::move(instance));
        }
    }
    instances_ = std::move(next);
    rebuildExecutionPlan();
    for (const auto& node : graph_.nodes()) {
        if (node.type != "subgraph") continue;
        const auto instance = instances_.find(node.id);
        if (instance == instances_.end()) continue;
        std::vector<bool> requiredOutputs;
        const auto plan = executionPlan_.find(node.id);
        if (plan != executionPlan_.end()) {
            for (std::size_t output = 0; output < plan->second.outputKeys.size(); ++output) {
                const auto& key = plan->second.outputKeys[output];
                bool required = output == 0; // Keep the normal node thumbnail/output.
                required |= std::ranges::any_of(graph_.links(), [&](const auto& link) {
                    return link.fromNode == node.id && link.fromSocket == key;
                });
                requiredOutputs.push_back(required);
            }
        }
        instance->second->setOutputRequirements(requiredOutputs);
    }
    std::erase_if(subgraphSignatures_, [&](const auto& item) {
        return !instances_.contains(item.first);
    });
    std::erase_if(pendingResets_, [&](NodeId id) { return !instances_.contains(id); });
    std::erase_if(typeErrors_, [&](const auto& item) {
        return !graph_.findNode(item.first);
    });
    for (auto& [id, error] : instantiationErrors) typeErrors_[id] = std::move(error);
    std::erase_if(values_, [&](const auto& item) { return !instances_.contains(item.first); });
    for (auto it = timerQueries_.begin(); it != timerQueries_.end();) {
        if (!instances_.contains(it->first)) {
            glDeleteQueries(1, &it->second);
            it = timerQueries_.erase(it);
        } else ++it;
    }
    for (const auto& [id, _] : instances_) {
        if (!timerQueries_.contains(id)) glGenQueries(1, &timerQueries_[id]);
    }
    if (!fusion_) fusion_ = std::make_unique<FusionState>();
    rebuildFusion();
    forceDirty_ = true;
}

void GraphRuntime::rebuildExecutionPlan() {
    executionPlan_.clear();

    for (const auto& node : graph_.nodes()) {
        NodeDescriptor storage;
        const auto* descriptor = resolveDescriptor(graph_, node, registry_, storage);
        if (!descriptor) continue;

        NodeExecutionPlan plan;
        plan.descriptor = *descriptor;
        for (const auto& socket : descriptor->sockets) {
            if (socket.direction == SocketDirection::Input) {
                const auto index = plan.inputKeys.size();
                plan.inputIndices.emplace(socket.key, index);
                plan.inputKeys.push_back(socket.key);
                plan.inputs.emplace_back();
                if (socket.requiresImage) plan.requiredImageInputs.push_back(index);
            } else {
                const auto index = plan.outputKeys.size();
                plan.outputIndices.emplace(socket.key, index);
                plan.outputKeys.push_back(socket.key);
            }
        }
        executionPlan_.emplace(node.id, std::move(plan));
    }

    for (const auto& link : graph_.links()) {
        const auto source = executionPlan_.find(link.fromNode);
        const auto destination = executionPlan_.find(link.toNode);
        if (source == executionPlan_.end() || destination == executionPlan_.end())
            continue;
        const auto sourceOutput = source->second.outputIndices.find(link.fromSocket);
        const auto destinationInput = destination->second.inputIndices.find(link.toSocket);
        if (sourceOutput == source->second.outputIndices.end() ||
            destinationInput == destination->second.inputIndices.end())
            continue;

        auto& input = destination->second.inputs[destinationInput->second];
        input = PlannedInput{link.fromNode, sourceOutput->second, true};
        if (std::ranges::find(destination->second.upstreamNodes, link.fromNode) ==
            destination->second.upstreamNodes.end()) {
            destination->second.upstreamNodes.push_back(link.fromNode);
        }
    }
}

std::string GraphRuntime::fusionSignature() const {
    std::string result;
    for (const auto id : compiled_.order) {
        const auto* node = graph_.findNode(id);
        NodeDescriptor storage;
        const auto* descriptor = node ? resolveDescriptor(graph_, *node, registry_, storage) : nullptr;
        if (!descriptor || !descriptor->lowerable) continue;
        result += std::to_string(id) + ":" + node->type;
        if (const auto found = instances_.find(id); found != instances_.end()) {
            const auto variant = found->second->shaderVariantKey(node->parameters);
            if (!variant.empty()) result += ":" + variant;
        }
        result += ";";
    }
    result += "preview:";
    if (fusion_ && fusion_->previewNode) result += std::to_string(*fusion_->previewNode);
    return result;
}

void GraphRuntime::rebuildFusion() {
    const bool enabled = fusion_ ? fusion_->enabled : true;
    const auto preview = fusion_ ? fusion_->previewNode : std::optional<NodeId>{};
    if (fusion_) for (const auto& [id, _] : fusion_->regionForNode) values_.erase(id);
    auto next = std::make_unique<FusionState>();
    next->enabled = enabled;
    next->previewNode = preview;
    try {
        auto planned = planShaderRegions(graph_, registry_, compiled_,
            gpu_.maximumComputeTextureInputs(), gpu_.maximumComputeUniformComponents(), enabled);
        for (auto& region : planned) {
            std::vector<NodeId> outputs{region.nodes.back()};
            if (preview && std::ranges::find(region.nodes, *preview) != region.nodes.end() &&
                *preview != region.nodes.back()) outputs.push_back(*preview);
            auto generated = generateComputeShader(region, outputs);
            const auto compiled = region.diagnostic.empty()
                ? gpu_.tryCompileComputeCached(generated.source,
                    "Generated shader region " + std::to_string(region.id))
                : ComputeCompileResult{0, region.diagnostic, 0};
            FusionState::RegionRuntime runtime;
            runtime.region = std::move(region);
            runtime.generated = std::move(generated);
            runtime.program = compiled.program;
            runtime.diagnostic = compiled.log;
            runtime.errorLine = compiled.reportedLine;
            runtime.materializedNodes = outputs;
            for (const auto& input : runtime.generated.inputs)
                runtime.boundarySignature.push_back(input.kind);
            runtime.specializationCount = 1;
            runtime.textures.resize(runtime.generated.outputs.size());
            const auto index = next->regions.size();
            for (const auto id : runtime.region.nodes) next->regionForNode[id] = index;
            next->regions.push_back(std::move(runtime));
        }
    } catch (const std::exception& error) {
        // Planning errors are surfaced as an empty generated set; legacy node
        // evaluation remains valid and therefore remains the safe fallback.
        next->planningDiagnostic = error.what();
    }
    next->signature = fusionSignature();
    fusion_ = std::move(next);
    forceDirty_ = true;
}

namespace {

float shaderParameter(const Graph& graph, const ShaderInputRequirement& input) {
    const auto* node = graph.findNode(input.parameterNode);
    return node && node->parameters.contains(input.parameterKey) &&
                   node->parameters[input.parameterKey].is_number()
        ? node->parameters[input.parameterKey].get<float>() : input.fallback;
}



} // namespace

Value GraphRuntime::outputValue(NodeId node, std::string_view socket) const {
    const auto plan = executionPlan_.find(node);
    const auto foundValues = values_.find(node);
    if (plan == executionPlan_.end() || foundValues == values_.end()) return {};
    const auto slot = plan->second.outputIndices.find(std::string(socket));
    if (slot == plan->second.outputIndices.end() ||
        slot->second >= foundValues->second.size())
        return {};
    return foundValues->second[slot->second];
}

std::optional<std::size_t> GraphRuntime::outputIndex(
    NodeId node, std::string_view socket) const {
    const auto plan = executionPlan_.find(node);
    if (plan == executionPlan_.end()) return std::nullopt;
    const auto slot = plan->second.outputIndices.find(std::string(socket));
    if (slot == plan->second.outputIndices.end()) return std::nullopt;
    return slot->second;
}

bool GraphRuntime::evaluateNativeNode(
    NodeId id, EvaluationContext& context, bool resetState) {
    const auto* record = graph_.findNode(id);
    const auto instance = instances_.find(id);
    const auto planned = executionPlan_.find(id);
    if (!record || instance == instances_.end() || planned == executionPlan_.end())
        return false;

    auto& node = *instance->second;
    const auto& plan = planned->second;
    std::vector<Value> inputs(plan.inputs.size());
    for (std::size_t index = 0; index < plan.inputs.size(); ++index) {
        const auto& source = plan.inputs[index];
        if (!source.connected) continue;
        const auto values = values_.find(source.sourceNode);
        if (values != values_.end() && source.sourceOutput < values->second.size())
            inputs[index] = values->second[source.sourceOutput];
    }

    applyUnconnectedParameterInputs(
        plan.descriptor, plan.inputKeys, inputs, record->parameters);
    auto& outputs = values_[id];
    outputs.resize(plan.outputKeys.size());

    for (const auto inputIndex : plan.requiredImageInputs) {
        const auto* image = inputIndex < inputs.size()
            ? std::get_if<ImageHandle>(&inputs[inputIndex]) : nullptr;
        if (!image || !*image) {
            typeErrors_[id] = plan.descriptor.displayName + ": input '" +
                              plan.inputKeys[inputIndex] +
                              "' requires a field image";
            for (auto& output : outputs) output = {};
            timings_.erase(id);
            return false;
        }
    }

    typeErrors_.erase(id);
    node.setParameters(withConnectedParameterOverrides(
        plan.descriptor, plan.inputKeys, inputs, record->parameters));
    if (resetState && plan.descriptor.stateful) node.reset(context);

    const GLuint query = timerQueries_.at(id);
    glBeginQuery(GL_TIME_ELAPSED, query);
    node.evaluate(context, inputs, outputs);
    glEndQuery(GL_TIME_ELAPSED);
    GLuint64 nanoseconds = 0;
    glGetQueryObjectui64v(query, GL_QUERY_RESULT, &nanoseconds);
    timings_[id] = static_cast<double>(nanoseconds) / 1'000'000.0;

    for (std::size_t output = 0;
         output < plan.outputKeys.size() && output < outputs.size(); ++output) {
        if (auto* image = std::get_if<ImageHandle>(&outputs[output])) {
            if (const auto semantic = compiled_.socketType(id, plan.outputKeys[output])) {
                image->semanticType = isFieldType(*semantic)
                    ? *semantic : fieldTypeForWidth(componentCount(*semantic));
            }
        }
    }
    pendingResets_.erase(id);
    return true;
}

bool GraphRuntime::evaluate(double time, double deltaTime, bool playing) {
    evaluated_.clear();
    // A non-empty partial order is useful for the same best-effort diagnostic
    // path used by rebuild(); an empty order means there is nothing executable.
    if (compiled_.order.empty()) return false;
    if (!fusion_) fusion_ = std::make_unique<FusionState>();
    if (fusion_->signature != fusionSignature()) rebuildFusion();
    if (lastWidth_ != graph_.settings.width || lastHeight_ != graph_.settings.height) {
        lastWidth_ = graph_.settings.width;
        lastHeight_ = graph_.settings.height;
        needsReset_ = true;
        forceDirty_ = true;
    }
    EvaluationContext context{graph_.settings.width, graph_.settings.height,
                              time, deltaTime, frame_, playing, &gpu_};
    std::unordered_set<NodeId> dirtyNodes;
    for (const auto id : compiled_.order) {
        const auto* record = graph_.findNode(id);
        const auto instanceIt = instances_.find(id);
        if (!record || instanceIt == instances_.end()) continue;
        auto& instance = *instanceIt->second;
        instance.setParameters(record->parameters);
        const auto plan = executionPlan_.find(id);
        if (plan == executionPlan_.end()) continue;
        const auto& desc = plan->second.descriptor;

        const auto fused = fusion_->regionForNode.find(id);
        if (fused != fusion_->regionForNode.end()) {
            auto& region = fusion_->regions[fused->second];
            if (id == region.region.nodes.back()) {
                const auto resolveKind = [&](NodeId sourceNode,
                                             std::string_view sourceSocket) {
                    const auto value = outputValue(sourceNode, sourceSocket);
                    if (std::holds_alternative<float>(value)) return ShaderInputKind::Float;
                    if (std::holds_alternative<Vec2>(value)) return ShaderInputKind::Vector;
                    if (const auto* image = std::get_if<ImageHandle>(&value); image && *image)
                        return ShaderInputKind::Field;
                    return ShaderInputKind::Empty;
                };
                std::vector<ShaderInputKind> liveSignature;
                liveSignature.reserve(region.generated.inputs.size());
                for (const auto& input : region.generated.inputs)
                    liveSignature.push_back(input.sourceNode == 0
                        ? ShaderInputKind::Float
                        : resolveKind(input.sourceNode, input.sourceSocket));
                if (liveSignature != region.boundarySignature) {
                    auto lowered = lowerShaderRegion(graph_, registry_, compiled_,
                                                     region.region.nodes, resolveKind);
                    auto generated = generateComputeShader(lowered, region.materializedNodes);
                    const auto compiled = lowered.diagnostic.empty()
                        ? gpu_.tryCompileComputeCached(generated.source,
                            "Generated shader region " + std::to_string(lowered.id))
                        : ComputeCompileResult{0, lowered.diagnostic, 0};
                    region.region = std::move(lowered);
                    region.generated = std::move(generated);
                    region.program = compiled.program;
                    region.diagnostic = compiled.log;
                    region.errorLine = compiled.reportedLine;
                    region.boundarySignature.clear();
                    for (const auto& input : region.generated.inputs)
                        region.boundarySignature.push_back(input.kind);
                    if (region.textures.size() > region.generated.outputs.size()) {
                        for (std::size_t index = region.generated.outputs.size();
                             index < region.textures.size(); ++index)
                            if (region.textures[index]) glDeleteTextures(1, &region.textures[index]);
                    }
                    region.textures.resize(region.generated.outputs.size());
                    ++region.specializationCount;
                    forceDirty_ = true;
                }
            }
            const bool generated = region.program != 0;
            if (generated && id != region.region.nodes.back()) continue;
            if (generated) {
                bool parametersChanged = false;
                bool timeDependent = false;
                for (const auto member : region.region.nodes) {
                    const auto* memberRecord = graph_.findNode(member);
                    if (!memberRecord) continue;
                    parametersChanged |= !previousParameters_.contains(member) ||
                                         previousParameters_[member] != memberRecord->parameters;
                    const auto memberInstance = instances_.find(member);
                    timeDependent |= memberInstance != instances_.end() &&
                                     memberInstance->second->descriptor().timeDependent;
                }
                bool upstreamDirty = false;
                for (const auto& input : region.generated.inputs) {
                    if (input.sourceNode != 0 && dirtyNodes.contains(input.sourceNode)) {
                        upstreamDirty = true;
                        break;
                    }
                }
                bool missingOutput = false;
                for (const auto& output : region.generated.outputs) {
                    const auto slot = outputIndex(output.node, output.socket);
                    const auto found = values_.find(output.node);
                    missingOutput |= !slot || found == values_.end() ||
                                     *slot >= found->second.size() ||
                                     std::holds_alternative<std::monostate>(found->second[*slot]);
                }
                const bool shouldEvaluate = forceDirty_ || parametersChanged || upstreamDirty ||
                                            missingOutput || (timeDependent && playing);
                for (const auto member : region.region.nodes) {
                    if (const auto* memberRecord = graph_.findNode(member))
                        previousParameters_[member] = memberRecord->parameters;
                }
                if (!shouldEvaluate) continue;

                bool missingImageInput = false;
                for (const auto& input : region.generated.inputs) {
                    if (input.kind == ShaderInputKind::Empty) {
                        missingImageInput = true;
                        continue;
                    }
                    if (input.kind != ShaderInputKind::Field) continue;
                    const auto value = outputValue(input.sourceNode, input.sourceSocket);
                    const auto* image = std::get_if<ImageHandle>(&value);
                    missingImageInput |= !image || !*image;
                }
                if (missingImageInput) {
                    // Static type inference can identify an image path whose source is
                    // temporarily empty (for example, an Image node without a file).
                    // Evaluate retained instances in dependency order so their
                    // established missing-input behavior remains exact.
                    region.runtimeFallback = true;
                    region.milliseconds = 0.0;
                    for (const auto member : region.region.nodes) {
                        const bool resetState =
                            needsReset_ || pendingResets_.contains(member);
                        evaluateNativeNode(member, context, resetState);
                        if (const auto timing = timings_.find(member);
                            timing != timings_.end())
                            region.milliseconds += timing->second;
                        dirtyNodes.insert(member);
                    }
                    continue;
                }
                region.runtimeFallback = false;

                const bool materializesImage = std::ranges::any_of(
                    region.generated.outputs,
                    [](const auto& output) { return output.requiresImage; });
                const int dispatchWidth = materializesImage ? context.width : 1;
                const int dispatchHeight = materializesImage ? context.height : 1;
                if (region.width != dispatchWidth || region.height != dispatchHeight) {
                    for (auto& texture : region.textures) {
                        if (texture) glDeleteTextures(1, &texture);
                        texture = gpu_.createTexture(dispatchWidth, dispatchHeight);
                    }
                    region.width = dispatchWidth;
                    region.height = dispatchHeight;
                }
                glUseProgram(region.program);
                for (const auto& input : region.generated.inputs) {
                    if (input.kind == ShaderInputKind::Field) {
                        const auto value = outputValue(input.sourceNode, input.sourceSocket);
                        const auto* image = std::get_if<ImageHandle>(&value);
                        if (image && *image) {
                            glActiveTexture(static_cast<GLenum>(GL_TEXTURE0 + input.binding));
                            glBindTexture(GL_TEXTURE_2D, image->texture);
                        }
                        glUniform1i(glGetUniformLocation(region.program, input.uniformName.c_str()),
                                    input.binding);
                    } else if (input.kind == ShaderInputKind::Vector) {
                        Vec2 value{};
                        if (input.sourceNode != 0) {
                            const auto source = outputValue(input.sourceNode, input.sourceSocket);
                            if (const auto* pair = std::get_if<Vec2>(&source)) value = *pair;
                        }
                        glUniform2f(glGetUniformLocation(region.program, input.uniformName.c_str()),
                                    value.x, value.y);
                    } else if (input.kind == ShaderInputKind::Float) {
                        float value = input.fallback;
                        if (input.sourceNode != 0) {
                            const auto source = outputValue(input.sourceNode, input.sourceSocket);
                            if (const auto* number = std::get_if<float>(&source)) value = *number;
                        } else if (input.parameterKey == "__time") {
                            value = static_cast<float>(context.frame) / 60.0F;
                        } else value = shaderParameter(graph_, input);
                        glUniform1f(glGetUniformLocation(region.program, input.uniformName.c_str()), value);
                    }
                }
                for (std::size_t outputIndex = 0; outputIndex < region.generated.outputs.size();
                     ++outputIndex) {
                    const auto& output = region.generated.outputs[outputIndex];
                    glBindImageTexture(static_cast<GLuint>(output.binding),
                        region.textures[outputIndex], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
                }
                for (const auto member : region.region.nodes) {
                    values_.erase(member);
                    timings_.erase(member);
                    dirtyNodes.insert(member);
                }
                const GLuint query = timerQueries_.at(region.region.nodes.back());
                glBeginQuery(GL_TIME_ELAPSED, query);
                gpu_.dispatch(region.program, dispatchWidth, dispatchHeight);
                glEndQuery(GL_TIME_ELAPSED);
                GLuint64 nanoseconds = 0;
                glGetQueryObjectui64v(query, GL_QUERY_RESULT, &nanoseconds);
                region.milliseconds = static_cast<double>(nanoseconds) / 1'000'000.0;
                timings_[region.region.nodes.back()] = region.milliseconds;
                for (std::size_t generatedIndex = 0;
                     generatedIndex < region.generated.outputs.size(); ++generatedIndex) {
                    const auto& generatedOutput = region.generated.outputs[generatedIndex];
                    const auto slot = outputIndex(generatedOutput.node, generatedOutput.socket);
                    if (!slot) continue;
                    auto& nodeValues = values_[generatedOutput.node];
                    if (nodeValues.size() <= *slot) nodeValues.resize(*slot + 1);
                    if (generatedOutput.requiresImage) {
                        const auto semantic = compiled_.socketType(
                            generatedOutput.node, generatedOutput.socket)
                            .value_or(ValueType::ColorImage);
                        const auto imageSemantic = isFieldType(semantic)
                            ? semantic : fieldTypeForWidth(componentCount(semantic));
                        nodeValues[*slot] = ImageHandle{region.textures[generatedIndex],
                                                        dispatchWidth, dispatchHeight, imageSemantic};
                    } else {
                        std::array<float, 4> folded{};
                        glBindTexture(GL_TEXTURE_2D, region.textures[generatedIndex]);
                        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, folded.data());
                        nodeValues[*slot] = generatedOutput.value.type == ShaderValueType::Vec2
                            ? Value{Vec2{folded[0], folded[1]}} : Value{folded[0]};
                    }
                }
                continue;
            }
        }
        bool upstreamDirty = false;
        const auto planned = executionPlan_.find(id);
        if (planned != executionPlan_.end()) {
            upstreamDirty = std::ranges::any_of(
                planned->second.upstreamNodes,
                [&](NodeId upstream) { return dirtyNodes.contains(upstream); });
        }
        const bool parametersChanged = !previousParameters_.contains(id) ||
                                       previousParameters_[id] != record->parameters;
        const bool nodeReset = pendingResets_.contains(id);
        const bool shouldEvaluate = forceDirty_ || parametersChanged || upstreamDirty || nodeReset ||
                                    (desc.timeDependent && playing) ||
                                    (desc.stateful && needsReset_) || !values_.contains(id);
        previousParameters_[id] = record->parameters;
        if (!shouldEvaluate) continue;
        dirtyNodes.insert(id);
        evaluateNativeNode(id, context, needsReset_ || nodeReset);
    }
    needsReset_ = false;
    forceDirty_ = false;
    evaluated_ = std::move(dirtyNodes);
    if (playing) ++frame_;
    return true;
}

std::vector<GeneratedShaderInfo> GraphRuntime::generatedShaders() const {
    std::vector<GeneratedShaderInfo> result;
    if (!fusion_) return result;
    result.reserve(fusion_->regions.size());
    for (const auto& region : fusion_->regions) {
        result.push_back(GeneratedShaderInfo{region.region.id, region.region.nodes.back(),
            region.region.nodes, region.generated,
            region.program && !region.runtimeFallback ? GeneratedExecutionMode::Generated :
                             GeneratedExecutionMode::LegacyFallback,
            region.diagnostic, region.errorLine, region.milliseconds,
            evaluated_.contains(region.region.nodes.back()), region.specializationCount});
    }
    if (!fusion_->planningDiagnostic.empty()) {
        GeneratedShaderInfo failure;
        failure.mode = GeneratedExecutionMode::LegacyFallback;
        failure.diagnostic = fusion_->planningDiagnostic;
        result.push_back(std::move(failure));
    }
    return result;
}

std::optional<NodeFusionInfo> GraphRuntime::fusionInfo(NodeId id) const {
    if (!fusion_) return std::nullopt;
    const auto found = fusion_->regionForNode.find(id);
    if (found == fusion_->regionForNode.end()) {
        if (typeErrors_.contains(id))
            return NodeFusionInfo{id, id, 1, false, false, true,
                                  GeneratedExecutionMode::LegacyFallback, 0};
        return std::nullopt;
    }
    const auto& region = fusion_->regions[found->second];
    const bool materialized = std::ranges::find(region.generated.outputs, id,
                                                &ShaderOutputRequirement::node) !=
                              region.generated.outputs.end();
    return NodeFusionInfo{region.region.id, region.region.nodes.back(), region.region.nodes.size(),
        id != region.region.nodes.back(), materialized, region.program == 0,
        region.program && !region.runtimeFallback ? GeneratedExecutionMode::Generated :
                         GeneratedExecutionMode::LegacyFallback,
        region.specializationCount};
}

void GraphRuntime::setFusionEnabled(bool enabled) {
    if (!fusion_) fusion_ = std::make_unique<FusionState>();
    if (fusion_->enabled == enabled) return;
    fusion_->enabled = enabled;
    rebuildFusion();
}

bool GraphRuntime::fusionEnabled() const { return !fusion_ || fusion_->enabled; }

void GraphRuntime::setIntermediatePreview(std::optional<NodeId> id) {
    if (!fusion_) fusion_ = std::make_unique<FusionState>();
    if (fusion_->previewNode == id) return;
    fusion_->previewNode = id;
    fusion_->signature.clear();
    forceDirty_ = true;
}

std::optional<NodeId> GraphRuntime::intermediatePreview() const {
    return fusion_ ? fusion_->previewNode : std::optional<NodeId>{};
}

void GraphRuntime::reset() { needsReset_ = true; frame_ = 0; }

void GraphRuntime::resetNode(NodeId id) {
    const auto found = instances_.find(id);
    if (found != instances_.end() && found->second->descriptor().stateful) {
        pendingResets_.insert(id);
    }
}

ImageHandle GraphRuntime::outputImage() const {
    const auto found = values_.find(graph_.activeOutput);
    if (found == values_.end() || found->second.empty()) return {};
    if (const auto* image = std::get_if<ImageHandle>(&found->second.front())) return *image;
    return {};
}

} // namespace reaction
