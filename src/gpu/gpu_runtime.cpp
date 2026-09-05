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

std::string simulationSignature(const SubgraphDefinition& definition) {
    std::string result = generateSimulationShader(definition, true);
    result.push_back('\0');
    result += generateSimulationShader(definition, false);
    // Init/update endpoints do not include presentation outputs. Include their
    // mappings so changing an exported channel still refreshes the executor.
    for (const auto& item : definition.interface) {
        result.push_back('\0');
        result += item.key + ":" + std::to_string(static_cast<int>(item.kind)) + ":" +
                  std::to_string(static_cast<int>(item.type)) + ":" +
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
    };

    ~FusionState() {
        for (auto& item : regions) {
            if (item.program) glDeleteProgram(item.program);
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
    instances_.clear();
    timings_.clear();
    previousParameters_.clear();
    subgraphSignatures_.clear();
    pendingResets_.clear();
    evaluated_.clear();
    frame_ = 0;
}

void GraphRuntime::rebuild() {
    const auto candidate = graph_.compile(registry_);
    if (!candidate.valid) {
        compiled_ = candidate;
        return;
    }
    compiled_ = candidate;
    std::unordered_map<NodeId, std::unique_ptr<NodeInstance>> next;
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
                    auto instance = createSubgraphInstance(*definition, registry_);
                    pendingResets_.insert(node.id);
                    next.emplace(node.id, std::move(instance));
                }
                subgraphSignatures_[node.id] = signature;
            }
        } else if (auto instance = registry_.create(node.type)) {
            if (instance->descriptor().stateful) pendingResets_.insert(node.id);
            next.emplace(node.id, std::move(instance));
        }
    }
    instances_ = std::move(next);
    for (const auto& node : graph_.nodes()) {
        if (node.type != "subgraph") continue;
        const auto instance = instances_.find(node.id);
        if (instance == instances_.end()) continue;
        std::vector<bool> requiredOutputs;
        for (const auto& socket : instance->second->descriptor().sockets) {
            if (socket.direction != SocketDirection::Output) continue;
            bool required = requiredOutputs.empty(); // Keep the normal node thumbnail/output.
            required |= std::ranges::any_of(graph_.links(), [&](const auto& link) {
                return link.fromNode == node.id && link.fromSocket == socket.key;
            });
            requiredOutputs.push_back(required);
        }
        instance->second->setOutputRequirements(requiredOutputs);
    }
    std::erase_if(subgraphSignatures_, [&](const auto& item) {
        return !instances_.contains(item.first);
    });
    std::erase_if(pendingResets_, [&](NodeId id) { return !instances_.contains(id); });
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

std::string GraphRuntime::fusionSignature() const {
    std::string result;
    for (const auto id : compiled_.order) {
        const auto* node = graph_.findNode(id);
        NodeDescriptor storage;
        const auto* descriptor = node ? resolveDescriptor(graph_, *node, registry_, storage) : nullptr;
        if (!descriptor || !descriptor->lowerable || !compiled_.inferredOutputs.contains(id) ||
            compiled_.inferredOutputs.at(id) != ValueType::Image2D) continue;
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
    if (!compiled_.valid) return;
    const bool enabled = fusion_ ? fusion_->enabled : true;
    const auto preview = fusion_ ? fusion_->previewNode : std::optional<NodeId>{};
    if (fusion_) for (const auto& [id, _] : fusion_->regionForNode) values_.erase(id);
    auto next = std::make_unique<FusionState>();
    next->enabled = enabled;
    next->previewNode = preview;
    try {
        auto planned = planShaderRegions(graph_, registry_, compiled_,
            gpu_.maximumComputeTextureInputs(), gpu_.maximumComputeUniformComponents());
        for (auto& region : planned) {
            std::vector<NodeId> outputs{region.nodes.back()};
            if (preview && std::ranges::find(region.nodes, *preview) != region.nodes.end() &&
                *preview != region.nodes.back()) outputs.push_back(*preview);
            auto generated = generateComputeShader(region, outputs);
            const auto compiled = gpu_.tryCompileCompute(generated.source,
                "Generated shader region " + std::to_string(region.id));
            FusionState::RegionRuntime runtime;
            runtime.region = std::move(region);
            runtime.generated = std::move(generated);
            runtime.program = compiled.program;
            runtime.diagnostic = compiled.log;
            runtime.errorLine = compiled.reportedLine;
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

Value outputValue(const Graph& graph, const NodeRegistry& registry,
                  const std::unordered_map<NodeId, std::vector<Value>>& values,
                  NodeId nodeId, std::string_view socketKey) {
    const auto* node = graph.findNode(nodeId);
    NodeDescriptor storage;
    const auto* descriptor = node ? resolveDescriptor(graph, *node, registry, storage) : nullptr;
    const auto foundValues = values.find(nodeId);
    if (!descriptor || foundValues == values.end()) return {};
    std::size_t outputIndex = 0;
    for (const auto& socket : descriptor->sockets) {
        if (socket.direction != SocketDirection::Output) continue;
        if (socket.key == socketKey && outputIndex < foundValues->second.size())
            return foundValues->second[outputIndex];
        ++outputIndex;
    }
    return {};
}

float shaderParameter(const Graph& graph, const ShaderInputRequirement& input) {
    const auto* node = graph.findNode(input.parameterNode);
    return node && node->parameters.contains(input.parameterKey) &&
                   node->parameters[input.parameterKey].is_number()
        ? node->parameters[input.parameterKey].get<float>() : input.fallback;
}

std::optional<std::size_t> descriptorOutputIndex(const Graph& graph, const NodeRegistry& registry,
                                                 NodeId id, std::string_view socketKey) {
    const auto* node = graph.findNode(id);
    NodeDescriptor storage;
    const auto* descriptor = node ? resolveDescriptor(graph, *node, registry, storage) : nullptr;
    if (!descriptor) return std::nullopt;
    std::size_t index = 0;
    for (const auto& socket : descriptor->sockets) {
        if (socket.direction != SocketDirection::Output) continue;
        if (socket.key == socketKey) return index;
        ++index;
    }
    return std::nullopt;
}

} // namespace

bool GraphRuntime::evaluate(double time, double deltaTime, bool playing) {
    evaluated_.clear();
    if (!compiled_.valid) return false;
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
        NodeDescriptor descriptorStorage;
        const auto* resolved = resolveDescriptor(graph_, *record, registry_, descriptorStorage);
        if (!resolved) continue;
        const auto& desc = *resolved;

        const auto fused = fusion_->regionForNode.find(id);
        if (fused != fusion_->regionForNode.end()) {
            auto& region = fusion_->regions[fused->second];
            const bool generated = fusion_->enabled && region.program != 0;
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
                    const auto slot = descriptorOutputIndex(graph_, registry_, output.node, output.socket);
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
                    if (input.kind != ShaderInputKind::Image) continue;
                    const auto value = outputValue(graph_, registry_, values_,
                                                   input.sourceNode, input.sourceSocket);
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
                        const auto* memberRecord = graph_.findNode(member);
                        const auto memberInstance = instances_.find(member);
                        if (!memberRecord || memberInstance == instances_.end()) continue;
                        auto& fallbackInstance = *memberInstance->second;
                        NodeDescriptor fallbackStorage;
                        const auto* fallbackDescriptor = resolveDescriptor(
                            graph_, *memberRecord, registry_, fallbackStorage);
                        if (!fallbackDescriptor) continue;
                        std::vector<std::string> keys;
                        std::size_t outputCount = 0;
                        for (const auto& socket : fallbackDescriptor->sockets) {
                            if (socket.direction == SocketDirection::Input) keys.push_back(socket.key);
                            else ++outputCount;
                        }
                        std::vector<Value> inputs(keys.size());
                        for (std::size_t inputIndex = 0; inputIndex < keys.size(); ++inputIndex) {
                            const auto link = std::ranges::find_if(graph_.links(), [&](const auto& item) {
                                return item.toNode == member && item.toSocket == keys[inputIndex];
                            });
                            if (link != graph_.links().end())
                                inputs[inputIndex] = outputValue(graph_, registry_, values_,
                                                                 link->fromNode, link->fromSocket);
                        }
                        fallbackInstance.setParameters(withConnectedParameterOverrides(
                            *fallbackDescriptor, keys, inputs, memberRecord->parameters));
                        auto& outputs = values_[member];
                        outputs.resize(outputCount);
                        const GLuint query = timerQueries_.at(member);
                        glBeginQuery(GL_TIME_ELAPSED, query);
                        fallbackInstance.evaluate(context, inputs, outputs);
                        glEndQuery(GL_TIME_ELAPSED);
                        GLuint64 nanoseconds = 0;
                        glGetQueryObjectui64v(query, GL_QUERY_RESULT, &nanoseconds);
                        timings_[member] = static_cast<double>(nanoseconds) / 1'000'000.0;
                        region.milliseconds += timings_[member];
                        dirtyNodes.insert(member);
                    }
                    continue;
                }
                region.runtimeFallback = false;

                if (region.width != context.width || region.height != context.height) {
                    for (auto& texture : region.textures) {
                        if (texture) glDeleteTextures(1, &texture);
                        texture = gpu_.createTexture(context.width, context.height);
                    }
                    region.width = context.width;
                    region.height = context.height;
                }
                glUseProgram(region.program);
                for (const auto& input : region.generated.inputs) {
                    if (input.kind == ShaderInputKind::Image) {
                        const auto value = outputValue(graph_, registry_, values_,
                                                       input.sourceNode, input.sourceSocket);
                        const auto* image = std::get_if<ImageHandle>(&value);
                        if (image && *image) {
                            glActiveTexture(static_cast<GLenum>(GL_TEXTURE0 + input.binding));
                            glBindTexture(GL_TEXTURE_2D, image->texture);
                        }
                        glUniform1i(glGetUniformLocation(region.program, input.uniformName.c_str()),
                                    input.binding);
                    } else {
                        float value = input.fallback;
                        if (input.sourceNode != 0) {
                            const auto source = outputValue(graph_, registry_, values_,
                                                            input.sourceNode, input.sourceSocket);
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
                gpu_.dispatch(region.program, context.width, context.height);
                glEndQuery(GL_TIME_ELAPSED);
                GLuint64 nanoseconds = 0;
                glGetQueryObjectui64v(query, GL_QUERY_RESULT, &nanoseconds);
                region.milliseconds = static_cast<double>(nanoseconds) / 1'000'000.0;
                timings_[region.region.nodes.back()] = region.milliseconds;
                for (std::size_t outputIndex = 0; outputIndex < region.generated.outputs.size();
                     ++outputIndex) {
                    const auto& generatedOutput = region.generated.outputs[outputIndex];
                    const auto slot = descriptorOutputIndex(graph_, registry_, generatedOutput.node,
                                                            generatedOutput.socket);
                    if (!slot) continue;
                    auto& nodeValues = values_[generatedOutput.node];
                    if (nodeValues.size() <= *slot) nodeValues.resize(*slot + 1);
                    nodeValues[*slot] = ImageHandle{region.textures[outputIndex],
                                                    context.width, context.height};
                }
                continue;
            }
        }
        bool upstreamDirty = false;
        for (const auto& link : graph_.links()) {
            if (link.toNode == id && dirtyNodes.contains(link.fromNode)) { upstreamDirty = true; break; }
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
        std::vector<Value> inputs;
        std::vector<std::string> inputKeys;
        std::size_t outputCount = 0;
        for (const auto& socket : desc.sockets) {
            if (socket.direction == SocketDirection::Input) {
                inputs.emplace_back();
                inputKeys.push_back(socket.key);
            } else ++outputCount;
        }
        for (std::size_t i = 0; i < inputKeys.size(); ++i) {
            for (const auto& link : graph_.links()) {
                if (link.toNode != id || link.toSocket != inputKeys[i]) continue;
                const auto* source = graph_.findNode(link.fromNode);
                NodeDescriptor sourceStorage;
                const auto* sourceDesc = source ? resolveDescriptor(graph_, *source, registry_, sourceStorage) : nullptr;
                if (!sourceDesc) continue;
                std::size_t outputIndex = 0;
                for (const auto& socket : sourceDesc->sockets) {
                    if (socket.direction != SocketDirection::Output) continue;
                    if (socket.key == link.fromSocket && outputIndex < values_[link.fromNode].size()) {
                        inputs[i] = values_[link.fromNode][outputIndex];
                        break;
                    }
                    ++outputIndex;
                }
            }
        }
        auto& outputs = values_[id];
        outputs.resize(outputCount);
        instance.setParameters(withConnectedParameterOverrides(
            desc, inputKeys, inputs, record->parameters));
        if ((needsReset_ || nodeReset) && desc.stateful) instance.reset(context);
        const GLuint query = timerQueries_.at(id);
        glBeginQuery(GL_TIME_ELAPSED, query);
        instance.evaluate(context, inputs, outputs);
        glEndQuery(GL_TIME_ELAPSED);
        GLuint64 nanoseconds = 0;
        glGetQueryObjectui64v(query, GL_QUERY_RESULT, &nanoseconds);
        timings_[id] = static_cast<double>(nanoseconds) / 1'000'000.0;
        pendingResets_.erase(id);
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
            !fusion_->enabled ? GeneratedExecutionMode::ForcedLegacy :
            region.program && !region.runtimeFallback ? GeneratedExecutionMode::Generated :
                             GeneratedExecutionMode::LegacyFallback,
            region.diagnostic, region.errorLine, region.milliseconds,
            evaluated_.contains(region.region.nodes.back())});
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
    if (found == fusion_->regionForNode.end()) return std::nullopt;
    const auto& region = fusion_->regions[found->second];
    const bool materialized = std::ranges::find(region.generated.outputs, id,
                                                &ShaderOutputRequirement::node) !=
                              region.generated.outputs.end();
    return NodeFusionInfo{region.region.id, region.region.nodes.back(), region.region.nodes.size(),
        id != region.region.nodes.back(), materialized, region.program == 0,
        !fusion_->enabled ? GeneratedExecutionMode::ForcedLegacy :
        region.program && !region.runtimeFallback ? GeneratedExecutionMode::Generated :
                         GeneratedExecutionMode::LegacyFallback};
}

void GraphRuntime::setFusionEnabled(bool enabled) {
    if (!fusion_) fusion_ = std::make_unique<FusionState>();
    if (fusion_->enabled == enabled) return;
    fusion_->enabled = enabled;
    forceDirty_ = true;
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
