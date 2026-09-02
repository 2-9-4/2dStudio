#include "reaction/gpu/gpu_runtime.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace reaction {
namespace {

GLuint compile(GLenum kind, std::string_view source) {
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
        std::string log(static_cast<std::size_t>(size), '\0');
        glGetShaderInfoLog(shader, size, nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error("Shader compilation failed: " + log);
    }
    return shader;
}

GLuint link(std::initializer_list<GLuint> shaders) {
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
        std::string log(static_cast<std::size_t>(size), '\0');
        glGetProgramInfoLog(program, size, nullptr, log.data());
        glDeleteProgram(program);
        throw std::runtime_error("Program linking failed: " + log);
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

GpuRuntime::GpuRuntime() {
    previewProgram_ = link({compile(GL_VERTEX_SHADER, kPreviewVertex),
                            compile(GL_FRAGMENT_SHADER, kPreviewFragment)});
}

GpuRuntime::~GpuRuntime() {
    if (previewProgram_) glDeleteProgram(previewProgram_);
}

GLuint GpuRuntime::compileCompute(std::string_view source) const {
    return link({compile(GL_COMPUTE_SHADER, source)});
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
    : graph_(graph), registry_(registry), gpu_(gpu) { rebuild(); }

GraphRuntime::~GraphRuntime() { clear(); }

void GraphRuntime::clear() {
    for (const auto& [_, query] : timerQueries_) glDeleteQueries(1, &query);
    timerQueries_.clear();
    values_.clear();
    instances_.clear();
    timings_.clear();
    previousParameters_.clear();
    pendingResets_.clear();
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
        if (found != instances_.end() && found->second->descriptor().type == node.type) {
            next.emplace(node.id, std::move(found->second));
        } else if (auto instance = registry_.create(node.type)) {
            if (instance->descriptor().stateful) pendingResets_.insert(node.id);
            next.emplace(node.id, std::move(instance));
        }
    }
    instances_ = std::move(next);
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
    forceDirty_ = true;
}

bool GraphRuntime::evaluate(double time, double deltaTime, bool playing) {
    if (!compiled_.valid) return false;
    if (lastWidth_ != graph_.settings.width || lastHeight_ != graph_.settings.height) {
        lastWidth_ = graph_.settings.width;
        lastHeight_ = graph_.settings.height;
        needsReset_ = true;
        forceDirty_ = true;
    }
    EvaluationContext context{graph_.settings.width, graph_.settings.height,
                              time, deltaTime, playing, &gpu_};
    std::unordered_set<NodeId> dirtyNodes;
    for (const auto id : compiled_.order) {
        const auto* record = graph_.findNode(id);
        const auto instanceIt = instances_.find(id);
        if (!record || instanceIt == instances_.end()) continue;
        auto& instance = *instanceIt->second;
        instance.setParameters(record->parameters);
        const auto& desc = instance.descriptor();
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
                const auto* sourceDesc = source ? registry_.descriptor(source->type) : nullptr;
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
    return true;
}

void GraphRuntime::reset() { needsReset_ = true; }

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
