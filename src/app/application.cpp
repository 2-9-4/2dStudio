#include "application.hpp"
#include "reaction/core/layout.hpp"
#include "reaction/core/persistence.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_node_editor.h>
#include <png.h>
#include <portable-file-dialogs.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace reaction {
namespace ed = ax::NodeEditor;
namespace {

void glfwError(int, const char* description) {
    throw std::runtime_error(description ? description : "GLFW error");
}

std::uintptr_t nodeUiId(NodeId id) { return static_cast<std::uintptr_t>(id * 128); }
std::uintptr_t pinUiId(NodeId id, std::size_t index, SocketDirection direction) {
    return nodeUiId(id) + 2 + index * 2 + (direction == SocketDirection::Output ? 1 : 0);
}
std::uintptr_t linkUiId(LinkId id) { return (std::uintptr_t{1} << 60U) + id; }

struct PinTarget {
    NodeId node = 0;
    std::string socket;
    SocketDirection direction = SocketDirection::Input;
};

ImColor socketColor(ValueType type) {
    switch (type) {
    case ValueType::Float: return ImColor(116, 185, 255);
    case ValueType::Image2D: return ImColor(241, 196, 80);
    case ValueType::AnyNumeric: return ImColor(180, 125, 230);
    }
    return ImColor(180, 180, 180);
}

} // namespace

Application::Application(std::filesystem::path startupProject) {
    std::signal(SIGPIPE, SIG_IGN);
    glfwSetErrorCallback(glfwError);
    if (glfwInit() == GLFW_FALSE) throw std::runtime_error("Unable to initialize GLFW");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    editorWindow_ = glfwCreateWindow(1280, 800, "Reaction Studio — Nodes", nullptr, nullptr);
    if (!editorWindow_) throw std::runtime_error("OpenGL 4.3 is required");
    glfwMakeContextCurrent(editorWindow_);
    glfwSwapInterval(0);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        throw std::runtime_error("Unable to load OpenGL functions");
    }
    if (GLAD_GL_VERSION_4_3 == 0) {
        const auto* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        const auto* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
        throw std::runtime_error(std::string("OpenGL 4.3 compute required; detected ") +
                                 (renderer ? renderer : "unknown renderer") + " / " +
                                 (version ? version : "unknown version"));
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(editorWindow_, true);
    ImGui_ImplOpenGL3_Init("#version 430");
    ed::Config rootEditorConfig;
    // Node positions belong to the project. Do not reload the legacy shared
    // NodeEditor.json camera, which also contains IDs from the old kernel view.
    rootEditorConfig.SettingsFile = nullptr;
    nodeEditor_ = ed::CreateEditor(&rootEditorConfig);
    // A node-editor context owns one viewport transform. Keeping the nested graph's
    // transform separate prevents its pan/zoom from moving the root viewport; both
    // contexts are rendered by the same canvas UI below.
    ed::Config subgraphEditorConfig;
    subgraphEditorConfig.SettingsFile = nullptr;
    subgraphEditor_ = ed::CreateEditor(&subgraphEditorConfig);

    registerBuiltInNodes(registry_);
    gpu_ = std::make_unique<GpuRuntime>();
    createPreviewWindow();
    if (startupProject.empty()) newProject();
    else loadProject(startupProject);
}

Application::~Application() {
    stopRecording(false);
    if (editorWindow_) glfwMakeContextCurrent(editorWindow_);
    runtime_.reset();
    gpu_.reset();
    if (nodeEditor_) ed::DestroyEditor(nodeEditor_);
    if (subgraphEditor_) ed::DestroyEditor(subgraphEditor_);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (previewWindow_) {
        glfwMakeContextCurrent(previewWindow_);
        if (previewVertexArray_) glDeleteVertexArrays(1, &previewVertexArray_);
        glfwDestroyWindow(previewWindow_);
    }
    if (editorWindow_) glfwDestroyWindow(editorWindow_);
    glfwTerminate();
}

void Application::createPreviewWindow() {
    if (previewWindow_) return;
    constexpr int maximumSize = 900;
    const double aspect = static_cast<double>(graph_.settings.width) /
                          static_cast<double>(graph_.settings.height);
    const int previewWidth = aspect >= 1.0 ? maximumSize :
        std::max(1, static_cast<int>(std::lround(maximumSize * aspect)));
    const int previewHeight = aspect >= 1.0 ?
        std::max(1, static_cast<int>(std::lround(maximumSize / aspect))) : maximumSize;
    previewWindow_ = glfwCreateWindow(previewWidth, previewHeight,
                                      "Reaction Studio — Preview", nullptr, editorWindow_);
    if (!previewWindow_) throw std::runtime_error("Unable to create preview window");
    glfwMakeContextCurrent(previewWindow_);
    glfwSwapInterval(0);
    updatePreviewAspectRatio();
    glGenVertexArrays(1, &previewVertexArray_);
    glfwMakeContextCurrent(editorWindow_);
}

void Application::updatePreviewAspectRatio() {
    if (previewWindow_ && graph_.settings.width > 0 && graph_.settings.height > 0)
        glfwSetWindowAspectRatio(previewWindow_, graph_.settings.width, graph_.settings.height);
}

void Application::newProject() {
    stopRecording();
    if (runtime_) runtime_->clear();
    editingSubgraphId_.clear();
    editingSubgraphInstance_ = 0;
    positionedSubgraphId_.clear();
    positionedSubgraph_.clear();
    graph_.clear();
    graph_.settings = {};
    updatePreviewAspectRatio();
    graph_.addNode("perlin", {40, 80});
    const auto reaction = graph_.addNode("reaction_diffusion", {360, 80});
    const auto ramp = graph_.addNode("color_ramp", {700, 80});
    const auto output = graph_.addNode("output", {1000, 80});
    graph_.addLink(reaction, "image", ramp, "value");
    graph_.addLink(ramp, "image", output, "image");
    graph_.activeOutput = output;
    currentPath_.clear();
    positioned_.clear();
    fitRootGraph_ = true;
    elapsed_ = 0;
    rebuildRuntime();
    dirty_ = false;
    setStatus("New project");
}

void Application::rebuildRuntime() {
    glfwMakeContextCurrent(editorWindow_);
    if (!runtime_) runtime_ = std::make_unique<GraphRuntime>(graph_, registry_, *gpu_);
    else runtime_->rebuild();
    const auto& result = runtime_->compileResult();
    if (!result.valid && !result.errors.empty()) setStatus(result.errors.front(), true);
}

void Application::loadProjectDialog() {
    const auto selected = pfd::open_file("Open Reaction project", currentPath_.parent_path().string(),
                                         {"Reaction project", "*.reaction.json", "JSON", "*.json"}).result();
    if (selected.empty()) return;
    try {
        loadProject(selected.front());
    } catch (const std::exception& error) { setStatus(error.what(), true); }
}

void Application::loadProject(const std::filesystem::path& path) {
    stopRecording();
    if (runtime_) runtime_->clear();
    editingSubgraphId_.clear();
    editingSubgraphInstance_ = 0;
    positionedSubgraphId_.clear();
    positionedSubgraph_.clear();
    graph_ = reaction::loadProject(path, registry_);
    updatePreviewAspectRatio();
    currentPath_ = path;
    positioned_.clear();
    fitRootGraph_ = true;
    elapsed_ = 0;
    rebuildRuntime();
    dirty_ = false;
    setStatus("Loaded " + currentPath_.filename().string());
}

void Application::saveProjectDialog(bool forceDialog) {
    if (forceDialog || currentPath_.empty()) {
        auto selected = pfd::save_file("Save Reaction project", "artwork.reaction.json",
                                       {"Reaction project", "*.reaction.json"}).result();
        if (selected.empty()) return;
        currentPath_ = std::move(selected);
    }
    try {
        saveProjectAtomic(graph_, currentPath_);
        dirty_ = false;
        setStatus("Saved " + currentPath_.filename().string());
    } catch (const std::exception& error) { setStatus(error.what(), true); }
}

void Application::exportFrameDialog() {
    const auto image = runtime_->outputImage();
    if (image.texture == 0 || image.width <= 0 || image.height <= 0) {
        setStatus("No output image to export", true);
        return;
    }

    auto filename = currentPath_.empty() ? std::filesystem::path("frame.png")
                                         : currentPath_.filename().replace_extension().replace_extension(".png");
    auto selected = pfd::save_file("Export Current Frame", filename.string(),
                                   {"PNG image", "*.png"}).result();
    if (selected.empty()) return;
    std::filesystem::path path = std::move(selected);
    if (path.extension() != ".png") path += ".png";

    try {
        std::vector<float> source(static_cast<std::size_t>(image.width) *
                                  static_cast<std::size_t>(image.height) * 4U);
        std::vector<png_byte> pixels(static_cast<std::size_t>(image.width) *
                                     static_cast<std::size_t>(image.height) * 3U);
        glBindTexture(GL_TEXTURE_2D, image.texture);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, source.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                const auto sourceIndex = static_cast<std::size_t>((y * image.width + x) * 4);
                const auto pixelIndex = static_cast<std::size_t>(((image.height - 1 - y) * image.width + x) * 3);
                for (int channel = 0; channel < 3; ++channel) {
                    const float value = source[sourceIndex + static_cast<std::size_t>(channel)];
                    pixels[pixelIndex + static_cast<std::size_t>(channel)] = static_cast<png_byte>(
                        std::lround(std::clamp(std::isfinite(value) ? value : 0.0F, 0.0F, 1.0F) * 255.0F));
                }
            }
        }

        png_image png{};
        png.version = PNG_IMAGE_VERSION;
        png.width = static_cast<png_uint_32>(image.width);
        png.height = static_cast<png_uint_32>(image.height);
        png.format = PNG_FORMAT_RGB;
        if (!png_image_write_to_file(&png, path.string().c_str(), 0, pixels.data(),
                                     image.width * 3, nullptr)) {
            throw std::runtime_error(std::string("Cannot write PNG: ") + png.message);
        }
        setStatus("Exported " + path.filename().string());
    } catch (const std::exception& error) {
        setStatus(error.what(), true);
    }
}

void Application::startRecordingDialog() {
    const auto image = runtime_->outputImage();
    if (image.texture == 0 || image.width <= 0 || image.height <= 0) {
        setStatus("No output image to record", true);
        return;
    }
    if ((image.width & 1) != 0 || (image.height & 1) != 0) {
        setStatus("MP4 recording requires an even project width and height", true);
        return;
    }

    auto filename = currentPath_.empty() ? std::filesystem::path("recording.mp4")
                                         : currentPath_.filename().replace_extension().replace_extension(".mp4");
    auto selected = pfd::save_file("Record Video", filename.string(),
                                   {"MP4 video", "*.mp4"}).result();
    if (selected.empty()) return;
    recordingPath_ = std::move(selected);
    if (recordingPath_.extension() != ".mp4") recordingPath_ += ".mp4";

    int pipeEnds[2]{};
    if (pipe(pipeEnds) != 0) {
        setStatus(std::string("Cannot create video pipe: ") + std::strerror(errno), true);
        recordingPath_.clear();
        return;
    }
    const pid_t child = fork();
    if (child == 0) {
        close(pipeEnds[1]);
        if (dup2(pipeEnds[0], STDIN_FILENO) < 0) _exit(127);
        close(pipeEnds[0]);
        const std::string videoSize = std::to_string(image.width) + "x" + std::to_string(image.height);
        const std::string frameRate = std::to_string(graph_.settings.targetFps);
        execlp("ffmpeg", "ffmpeg", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", videoSize.c_str(),
               "-framerate", frameRate.c_str(), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
               recordingPath_.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    close(pipeEnds[0]);
    if (child < 0) {
        close(pipeEnds[1]);
        setStatus(std::string("Cannot start FFmpeg: ") + std::strerror(errno), true);
        recordingPath_.clear();
        return;
    }
    recordingPipe_ = pipeEnds[1];
    recordingProcess_ = static_cast<int>(child);
    recordingWidth_ = image.width;
    recordingHeight_ = image.height;
    recordedFrames_ = 0;
    setStatus("Recording " + recordingPath_.filename().string());
}

void Application::stopRecording(bool reportStatus) {
    if (recordingPipe_ < 0) return;
    close(recordingPipe_);
    recordingPipe_ = -1;
    int processStatus = 0;
    const pid_t result = waitpid(static_cast<pid_t>(recordingProcess_), &processStatus, 0);
    const bool succeeded = result >= 0 && WIFEXITED(processStatus) && WEXITSTATUS(processStatus) == 0;
    if (reportStatus) {
        if (succeeded) {
            const double seconds = static_cast<double>(recordedFrames_) /
                                   static_cast<double>(std::max(graph_.settings.targetFps, 1));
            char duration[32]{};
            std::snprintf(duration, sizeof(duration), "%.1f", seconds);
            setStatus("Saved " + recordingPath_.filename().string() + " (" + duration + "s)");
        } else {
            setStatus("FFmpeg could not finish the video; check that FFmpeg with libx264 is installed", true);
        }
    }
    recordingProcess_ = -1;
    recordingPath_.clear();
    recordingWidth_ = 0;
    recordingHeight_ = 0;
    recordedFrames_ = 0;
    recordingReadback_.clear();
    recordingPixels_.clear();
}

void Application::recordFrame() {
    if (recordingPipe_ < 0 || !playing_) return;
    const auto image = runtime_->outputImage();
    if (image.texture == 0) return;
    if (image.width != recordingWidth_ || image.height != recordingHeight_) {
        stopRecording(false);
        setStatus("Recording stopped because the project resolution changed", true);
        return;
    }

    const std::size_t rowBytes = static_cast<std::size_t>(image.width) * 3U;
    const std::size_t frameBytes = rowBytes * static_cast<std::size_t>(image.height);
    recordingReadback_.resize(frameBytes);
    recordingPixels_.resize(frameBytes);
    glBindTexture(GL_TEXTURE_2D, image.texture);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, recordingReadback_.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    for (int y = 0; y < image.height; ++y) {
        const std::size_t sourceOffset = static_cast<std::size_t>(y) * rowBytes;
        const std::size_t targetOffset = static_cast<std::size_t>(image.height - 1 - y) * rowBytes;
        std::copy_n(recordingReadback_.data() + sourceOffset, rowBytes,
                    recordingPixels_.data() + targetOffset);
    }

    std::size_t written = 0;
    while (written < recordingPixels_.size()) {
        const ssize_t count = write(recordingPipe_, recordingPixels_.data() + written,
                                    recordingPixels_.size() - written);
        if (count > 0) written += static_cast<std::size_t>(count);
        else if (count < 0 && errno == EINTR) continue;
        else {
            stopRecording(false);
            setStatus("Recording failed; check that FFmpeg with libx264 is installed", true);
            return;
        }
    }
    ++recordedFrames_;
}

void Application::setStatus(std::string message, bool error) {
    status_ = std::move(message);
    statusError_ = error;
}

void Application::handleShortcuts() {
    auto& io = ImGui::GetIO();
    if (!io.KeyCtrl) return;
    if (ImGui::IsKeyPressed(ImGuiKey_S, false)) saveProjectDialog(io.KeyShift);
    if (ImGui::IsKeyPressed(ImGuiKey_O, false)) loadProjectDialog();
    if (ImGui::IsKeyPressed(ImGuiKey_N, false)) newProject();
    if (ImGui::IsKeyPressed(ImGuiKey_L, false)) layoutRequested_ = true;
    if (editingSubgraphId_.empty() && !io.WantTextInput &&
        ImGui::IsKeyPressed(ImGuiKey_C, false)) copyRequested_ = true;
    if (editingSubgraphId_.empty() && !io.WantTextInput &&
        ImGui::IsKeyPressed(ImGuiKey_V, false)) pasteRequested_ = true;
}

void Application::copySelectedNodes() {
    const int selectedCount = ed::GetSelectedObjectCount();
    if (selectedCount <= 0) { setStatus("No nodes selected", true); return; }
    std::vector<ed::NodeId> selectedUiIds(static_cast<std::size_t>(selectedCount));
    const int nodeCount = ed::GetSelectedNodes(selectedUiIds.data(), selectedCount);
    std::unordered_set<NodeId> selected;
    std::unordered_set<std::string> selectedSubgraphs;
    nlohmann::json nodes = nlohmann::json::array();
    for (int index = 0; index < nodeCount; ++index) {
        const auto id = static_cast<NodeId>(selectedUiIds[static_cast<std::size_t>(index)].Get() / 128);
        const auto* node = graph_.findNode(id);
        if (!node) continue;
        selected.insert(id);
        nodes.push_back({{"sourceId", id}, {"type", node->type},
                         {"typeVersion", node->typeVersion},
                         {"subgraphId", node->subgraphId},
                         {"position", {node->position.x, node->position.y}},
                         {"parameters", node->parameters}});
        if (node->type == "subgraph" && graph_.findSubgraph(node->subgraphId))
            selectedSubgraphs.insert(node->subgraphId);
    }
    if (nodes.empty()) { setStatus("No nodes selected", true); return; }
    nlohmann::json links = nlohmann::json::array();
    for (const auto& link : graph_.links()) {
        if (selected.contains(link.fromNode) && selected.contains(link.toNode)) {
            links.push_back({{"fromNode", link.fromNode}, {"fromSocket", link.fromSocket},
                             {"toNode", link.toNode}, {"toSocket", link.toSocket}});
        }
    }
    nlohmann::json subgraphs = nlohmann::json::array();
    const auto projectJson = serializeProject(graph_);
    for (const auto& definition : projectJson.value("subgraphs", nlohmann::json::array()))
        if (selectedSubgraphs.contains(definition.value("id", std::string{}))) subgraphs.push_back(definition);
    const auto payload = nlohmann::json{{"kind", "reaction.nodes"}, {"version", 2},
        {"subgraphs", std::move(subgraphs)}, {"nodes", std::move(nodes)}, {"links", std::move(links)}}.dump();
    ImGui::SetClipboardText(payload.c_str());
    setStatus("Copied " + std::to_string(selected.size()) + " node(s)");
}

void Application::pasteNodes() {
    const char* clipboard = ImGui::GetClipboardText();
    if (!clipboard) { setStatus("Clipboard is empty", true); return; }
    try {
        const auto payload = nlohmann::json::parse(clipboard);
        const int clipboardVersion = payload.value("version", 0);
        if (payload.value("kind", "") != "reaction.nodes" ||
            (clipboardVersion != 1 && clipboardVersion != 2)) {
            setStatus("Clipboard does not contain Reaction nodes", true);
            return;
        }
        std::unordered_map<std::string, std::string> remappedSubgraphs;
        if (clipboardVersion >= 2) {
            const auto settings = nlohmann::json{{"width", graph_.settings.width},
                {"height", graph_.settings.height}, {"targetFps", graph_.settings.targetFps}};
            for (const auto& definitionJson : payload.value("subgraphs", nlohmann::json::array())) {
                nlohmann::json document{{"formatVersion", 2}, {"project", settings},
                    {"subgraphs", nlohmann::json::array({definitionJson})},
                    {"nodes", nlohmann::json::array()}, {"links", nlohmann::json::array()}, {"activeOutput", 0}};
                auto parsed = deserializeProject(document, registry_).subgraphs().front();
                const std::string sourceId = parsed.id;
                std::string targetId = sourceId;
                if (const auto* existing = graph_.findSubgraph(parsed.id)) {
                    Graph existingGraph; existingGraph.subgraphs().push_back(*existing);
                    Graph parsedGraph; parsedGraph.subgraphs().push_back(parsed);
                    if (serializeProject(existingGraph)["subgraphs"] != serializeProject(parsedGraph)["subgraphs"]) {
                        int suffix = 2; const std::string base = parsed.id;
                        do parsed.id = base + ".copy" + std::to_string(suffix++);
                        while (resolveSubgraph(graph_, parsed.id));
                        targetId = parsed.id;
                        graph_.subgraphs().push_back(std::move(parsed));
                    }
                } else graph_.subgraphs().push_back(std::move(parsed));
                remappedSubgraphs[sourceId] = targetId;
            }
        }
        std::unordered_map<NodeId, NodeId> remapped;
        pendingSelection_.clear();
        for (const auto& value : payload.at("nodes")) {
            const auto& position = value.at("position");
            const auto id = graph_.addNode(value.at("type").get<std::string>(),
                                           {position.at(0).get<float>() + 30.0F,
                                            position.at(1).get<float>() + 30.0F});
            auto* node = graph_.findNode(id);
            node->typeVersion = value.value("typeVersion", 1);
            node->subgraphId = value.value("subgraphId", std::string{});
            if (const auto mapping = remappedSubgraphs.find(node->subgraphId); mapping != remappedSubgraphs.end())
                node->subgraphId = mapping->second;
            node->parameters = value.value("parameters", nlohmann::json::object());
            NodeDescriptor storage;
            node->missing = resolveDescriptor(graph_, *node, registry_, storage) == nullptr;
            remapped.emplace(value.at("sourceId").get<NodeId>(), id);
            positioned_[id] = false;
            pendingSelection_.push_back(id);
        }
        for (const auto& link : payload.at("links")) {
            const auto from = remapped.find(link.at("fromNode").get<NodeId>());
            const auto to = remapped.find(link.at("toNode").get<NodeId>());
            if (from != remapped.end() && to != remapped.end()) {
                graph_.addLink(from->second, link.at("fromSocket").get<std::string>(),
                               to->second, link.at("toSocket").get<std::string>());
            }
        }
        dirty_ = true;
        rebuildRuntime();
        setStatus("Pasted " + std::to_string(pendingSelection_.size()) + " node(s)");
    } catch (const std::exception& error) {
        setStatus(std::string("Cannot paste nodes: ") + error.what(), true);
    }
}

void Application::autoLayoutBody(GraphBody& body, std::unordered_map<NodeId, bool>& positioned) {
    std::unordered_map<NodeId, Vec2> sizes;
    for (const auto& node : body.nodes()) {
        const auto size = ed::GetNodeSize(ed::NodeId(nodeUiId(node.id)));
        if (size.x > 1.0F && size.y > 1.0F) sizes[node.id] = {size.x, size.y};
    }
    layoutGraph(body, &sizes);
    for (const auto& node : body.nodes()) positioned[node.id] = false;
    dirty_ = true;
}

void Application::renderGraph() {
    ed::SetCurrentEditor(nodeEditor_);
    if (layoutRequested_) {
        layoutRequested_ = false;
        autoLayoutBody(graph_, positioned_);
        fitRootGraph_ = true;
    }
    ed::Begin("Node graph");
    if (copyRequested_) { copySelectedNodes(); copyRequested_ = false; }
    if (pasteRequested_) { pasteNodes(); pasteRequested_ = false; }
    std::unordered_map<std::uintptr_t, PinTarget> pins;

    for (auto& node : graph_.nodes()) {
        NodeDescriptor descriptorStorage;
        const auto* descriptor = resolveDescriptor(graph_, node, registry_, descriptorStorage);
        ed::BeginNode(ed::NodeId(nodeUiId(node.id)));
        ImGui::PushID(static_cast<int>(node.id));
        if (!descriptor) {
            ImGui::TextColored(ImVec4(1, .35F, .35F, 1), "Missing: %s", node.type.c_str());
        } else {
            ImGui::TextUnformatted(descriptor->displayName.c_str());
            ImGui::Separator();
            std::size_t pinIndex = 0;
            for (const auto& socket : descriptor->sockets) {
                const auto id = pinUiId(node.id, pinIndex++, socket.direction);
                pins.emplace(id, PinTarget{node.id, socket.key, socket.direction});
                ed::BeginPin(ed::PinId(id), socket.direction == SocketDirection::Input ? ed::PinKind::Input : ed::PinKind::Output);
                ImGui::TextColored(socketColor(socket.type), "%s%s", socket.direction == SocketDirection::Output ? "● " : "○ ", socket.label.c_str());
                ed::EndPin();
            }
            for (const auto& property : descriptor->parameters) {
                float value = node.parameters.value(property.key, property.defaultValue);
                ImGui::SetNextItemWidth(150);
                bool changed = false;
                if (property.key == "operation" && node.type == "math") {
                    node_widgets::renderMathOperationSelector(node, nodePopup_);
                    continue;
                } else if (property.key == "mode" && node.type == "mix") {
                    node_widgets::renderMixModeSelector(node, nodePopup_);
                    continue;
                } else if (property.control == ParameterDescriptor::Control::Boolean) {
                    bool enabled = value > 0.5F;
                    changed = ImGui::Checkbox(property.label.c_str(), &enabled);
                    value = enabled ? 1.0F : 0.0F;
                } else if (property.control == ParameterDescriptor::Control::Integer ||
                           property.key == "iterations" || property.key == "octaves" || property.key == "seed") {
                    int integer = static_cast<int>(value);
                    changed = ImGui::SliderInt(property.label.c_str(), &integer, static_cast<int>(property.minimum), static_cast<int>(property.maximum));
                    value = static_cast<float>(integer);
                } else {
                    const float speed = std::max((property.maximum - property.minimum) / 500.0F, 0.0001F);
                    changed = ImGui::DragFloat(property.label.c_str(), &value, speed,
                                               property.minimum, property.maximum, "%.6g",
                                               ImGuiSliderFlags_AlwaysClamp);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Drag to adjust; Ctrl+click to type an exact value");
                    }
                }
                if (changed) { node.parameters[property.key] = value; dirty_ = true; }
            }
            if (node.type == "convolution" &&
                node_widgets::renderConvolutionEditor(node, nodePopup_)) dirty_ = true;
            if (node.type == "image" && node_widgets::renderImagePicker(node)) dirty_ = true;
            if (descriptor->stateful && ImGui::Button("Reset Simulation")) {
                runtime_->resetNode(node.id);
                setStatus("Reset simulation node");
            }
            if (node.type == "subgraph") {
                ImGui::TextDisabled("Select + Tab to enter");
                const auto* definition = resolveSubgraph(graph_, node.subgraphId);
                if (definition && definition->immutable) {
                    if (ImGui::Button("Duplicate as Editable")) duplicateSubgraph(node);
                    ImGui::TextDisabled("Built-in (read-only)");
                }
            }
            if (node.type == "output") {
                const bool active = graph_.activeOutput == node.id;
                if (active) ImGui::TextColored(ImVec4(.35F, .9F, .45F, 1), "Active preview output");
                else if (ImGui::Button("Use for Preview")) {
                    graph_.activeOutput = node.id;
                    dirty_ = true;
                }
            }
            const auto values = runtime_->values().find(node.id);
            if (values != runtime_->values().end() && !values->second.empty()) {
                if (node.type == "float_preview") {
                    if (const auto* value = std::get_if<float>(&values->second.front())) {
                        ImGui::Text("Live value: %.6g", *value);
                    }
                } else if (const auto* image = std::get_if<ImageHandle>(&values->second.front()); image && *image) {
                    ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<std::intptr_t>(image->texture)),
                                 ImVec2(150, 100), ImVec2(0, 1), ImVec2(1, 0));
                }
            }
            if (const auto timing = runtime_->gpuMilliseconds().find(node.id); timing != runtime_->gpuMilliseconds().end()) {
                ImGui::TextDisabled("GPU %.3f ms", timing->second);
            }
        }
        ImGui::PopID();
        ed::EndNode();
        if (!positioned_[node.id]) {
            ed::SetNodePosition(ed::NodeId(nodeUiId(node.id)), ImVec2(node.position.x, node.position.y));
            positioned_[node.id] = true;
        } else {
            const auto position = ed::GetNodePosition(ed::NodeId(nodeUiId(node.id)));
            if (node.position.x != position.x || node.position.y != position.y) {
                node.position = {position.x, position.y};
                dirty_ = true;
            }
        }
    }

    if (!pendingSelection_.empty()) {
        ed::ClearSelection();
        for (const auto id : pendingSelection_) ed::SelectNode(ed::NodeId(nodeUiId(id)), true);
        pendingSelection_.clear();
    }

    for (const auto& link : graph_.links()) {
        const auto* from = graph_.findNode(link.fromNode); const auto* to = graph_.findNode(link.toNode);
        NodeDescriptor fromStorage, toStorage;
        const auto* fromDesc = from ? resolveDescriptor(graph_, *from, registry_, fromStorage) : nullptr;
        const auto* toDesc = to ? resolveDescriptor(graph_, *to, registry_, toStorage) : nullptr;
        if (!fromDesc || !toDesc) continue;
        std::size_t fromIndex = 0, toIndex = 0; bool foundFrom = false, foundTo = false;
        for (const auto& socket : fromDesc->sockets) { if (socket.key == link.fromSocket && socket.direction == SocketDirection::Output) { foundFrom = true; break; } ++fromIndex; }
        for (const auto& socket : toDesc->sockets) { if (socket.key == link.toSocket && socket.direction == SocketDirection::Input) { foundTo = true; break; } ++toIndex; }
        if (foundFrom && foundTo) ed::Link(ed::LinkId(linkUiId(link.id)), ed::PinId(pinUiId(link.fromNode, fromIndex, SocketDirection::Output)), ed::PinId(pinUiId(link.toNode, toIndex, SocketDirection::Input)));
    }

    if (ed::BeginCreate()) {
        ed::PinId first, second;
        if (ed::QueryNewLink(&first, &second) && first && second) {
            auto a = pins.find(first.Get()), b = pins.find(second.Get());
            if (a != pins.end() && b != pins.end()) {
                if (a->second.direction == SocketDirection::Input) std::swap(a, b);
                const bool directionOk = a->second.direction == SocketDirection::Output && b->second.direction == SocketDirection::Input;
                if (directionOk && ed::AcceptNewItem()) {
                    graph_.addLink(a->second.node, a->second.socket, b->second.node, b->second.socket);
                    dirty_ = true; rebuildRuntime();
                }
            }
        }
        ed::EndCreate();
    }
    if (ed::BeginDelete()) {
        ed::LinkId linkId;
        while (ed::QueryDeletedLink(&linkId)) if (ed::AcceptDeletedItem()) {
            graph_.removeLink(static_cast<LinkId>(linkId.Get() - (std::uintptr_t{1} << 60U))); dirty_ = true; rebuildRuntime();
        }
        ed::NodeId nodeId;
        while (ed::QueryDeletedNode(&nodeId)) if (ed::AcceptDeletedItem()) {
            graph_.removeNode(static_cast<NodeId>(nodeId.Get() / 128)); dirty_ = true; rebuildRuntime();
        }
        ed::EndDelete();
    }

    static ImVec2 addNodeCanvasPosition{};
    ed::Suspend();
    if (node_widgets::renderPopup(nodePopup_, graph_)) dirty_ = true;
    if (ed::ShowBackgroundContextMenu()) {
        addNodeCanvasPosition = ed::ScreenToCanvas(ImGui::GetMousePos());
        ImGui::OpenPopup("Add node");
    }
    if (ImGui::BeginPopup("Add node")) {
        static char search[96]{};
        ImGui::InputTextWithHint("##search", "Search nodes", search, sizeof(search));
        const std::string filter = search;
        for (const auto* descriptor : registry_.descriptors()) {
            std::string label = descriptor->category + " / " + descriptor->displayName;
            if (!filter.empty() && label.find(filter) == std::string::npos) continue;
            if (ImGui::MenuItem(label.c_str())) {
                const auto id = graph_.addNode(descriptor->type,
                                               {addNodeCanvasPosition.x, addNodeCanvasPosition.y});
                if (descriptor->type == "output") graph_.activeOutput = id;
                positioned_[id] = false;
                dirty_ = true; rebuildRuntime();
            }
        }
        const auto addSubgraph = [&](const SubgraphDefinition& definition) {
            const std::string label = definition.category + " / " + definition.name;
            if (!filter.empty() && label.find(filter) == std::string::npos) return;
            if (ImGui::MenuItem(label.c_str())) {
                const auto id = graph_.addNode("subgraph", {addNodeCanvasPosition.x, addNodeCanvasPosition.y});
                auto* record = graph_.findNode(id);
                record->subgraphId = definition.id;
                record->typeVersion = definition.version;
                for (const auto& item : definition.interface)
                    if (item.kind == SubgraphInterfaceKind::Slider)
                        record->parameters[item.key] = item.defaultValue;
                // A subgraph added from the menu is an editable project asset. The
                // immutable built-in remains only as the pristine source template.
                if (definition.immutable) duplicateSubgraph(*record);
                positioned_[id] = false; dirty_ = true; rebuildRuntime();
            }
        };
        for (const auto& definition : builtInSubgraphs()) addSubgraph(definition);
        for (const auto& definition : graph_.subgraphs()) addSubgraph(definition);
        ImGui::EndPopup();
    }
    ed::Resume();
    if (!ImGui::GetIO().WantTextInput &&
        !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId) &&
        ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
        const int selectedCount = ed::GetSelectedObjectCount();
        if (selectedCount == 1) {
            ed::NodeId selectedUiId;
            if (ed::GetSelectedNodes(&selectedUiId, 1) == 1) {
                const auto selectedId = static_cast<NodeId>(selectedUiId.Get() / 128);
                auto* selected = graph_.findNode(selectedId);
                if (selected && selected->type == "subgraph" &&
                    resolveSubgraph(graph_, selected->subgraphId)) {
                    if (resolveSubgraph(graph_, selected->subgraphId)->immutable)
                        duplicateSubgraph(*selected);
                    editingSubgraphId_ = selected->subgraphId;
                    editingSubgraphInstance_ = selected->id;
                    setStatus("Entered subgraph — press Tab to return");
                }
            }
        }
    }
    if (fitRootGraph_) {
        ed::NavigateToContent(0.0F);
        fitRootGraph_ = false;
    }
    ed::End();
    ed::SetCurrentEditor(nullptr);
}

void Application::duplicateSubgraph(NodeRecord& node) {
    const auto* source = resolveSubgraph(graph_, node.subgraphId);
    if (!source) return;
    SubgraphDefinition copy = *source;
    copy.immutable = false;
    int suffix = 1;
    do {
        copy.id = "project.reaction_diffusion." + std::to_string(graph_.nextNodeId) + "." +
                  std::to_string(suffix++);
    } while (resolveSubgraph(graph_, copy.id));
    const int copyNumber = static_cast<int>(graph_.subgraphs().size()) + 1;
    copy.name = source->name + " (Editable " + std::to_string(copyNumber) + ")";
    graph_.subgraphs().push_back(std::move(copy));
    node.subgraphId = graph_.subgraphs().back().id;
    dirty_ = true;
    rebuildRuntime();
    setStatus("Created editable shared subgraph — press Tab to enter");
}

void Application::renderSubgraphEditor() {
    if (editingSubgraphId_.empty()) return;
    const auto leaveSubgraph = [&] {
        editingSubgraphId_.clear();
        editingSubgraphInstance_ = 0;
        setStatus("Returned to root graph");
    };
    if (!ImGui::GetIO().WantTextInput &&
        !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId) &&
        ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
        leaveSubgraph();
        return;
    }
    const auto* resolved = resolveSubgraph(graph_, editingSubgraphId_);
    if (!resolved) {
        leaveSubgraph();
        return;
    }
    if (resolved->immutable) {
        if (auto* instance = graph_.findNode(editingSubgraphInstance_)) {
            duplicateSubgraph(*instance);
            editingSubgraphId_ = instance->subgraphId;
        }
        return;
    }
    auto* definition = graph_.findSubgraph(editingSubgraphId_);
    if (!definition) { leaveSubgraph(); return; }
    auto& body = definition->body;
    bool changed = false;
    const bool navigateToGraph = positionedSubgraphId_ != editingSubgraphId_;
    if (navigateToGraph) {
        positionedSubgraphId_ = editingSubgraphId_;
        positionedSubgraph_.clear();
    }

    if (ImGui::Button("← Root")) { leaveSubgraph(); return; }
    ImGui::SameLine();
    ImGui::Text("Root / %s", definition->name.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("Tab: return");
    ImGui::SameLine();
    if (ImGui::Button("Subgraph Settings…")) ImGui::OpenPopup("Subgraph Settings");
    if (ImGui::BeginPopupModal("Subgraph Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        char name[128]{};
        std::snprintf(name, sizeof(name), "%s", definition->name.c_str());
        if (ImGui::InputText("Name", name, sizeof(name))) {
            definition->name = name; changed = true;
        }
        ImGui::SeparatorText("Interface");
        for (std::size_t index = 0; index < definition->interface.size(); ++index) {
            auto& item = definition->interface[index];
            ImGui::PushID(static_cast<int>(index));
            ImGui::TextDisabled("%s · %s",
                item.kind == SubgraphInterfaceKind::Input ? "Input" :
                item.kind == SubgraphInterfaceKind::Slider ? "Control" : "Output",
                item.key.c_str());
            char label[128]{}; std::snprintf(label, sizeof(label), "%s", item.label.c_str());
            if (ImGui::InputText("Label", label, sizeof(label))) { item.label = label; changed = true; }
            if (item.kind == SubgraphInterfaceKind::Slider) {
                changed |= ImGui::DragFloat("Default", &item.defaultValue, .001F, item.minimum, item.maximum);
                changed |= ImGui::DragFloat("Minimum", &item.minimum, .001F);
                changed |= ImGui::DragFloat("Maximum", &item.maximum, .001F);
                if (item.minimum > item.maximum) std::swap(item.minimum, item.maximum);
                item.defaultValue = std::clamp(item.defaultValue, item.minimum, item.maximum);
            }
            ImGui::Separator();
            ImGui::PopID();
        }
        if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    // This is the same full-size canvas interaction model as the root graph. The
    // separate context is only an independent camera; it is not a separate UI.
    ed::SetCurrentEditor(subgraphEditor_);
    if (layoutRequested_) {
        layoutRequested_ = false;
        autoLayoutBody(body, positionedSubgraph_);
        fitSubgraphRequested_ = true;
    }
    ed::Begin("Node graph");
    std::unordered_map<std::uintptr_t, PinTarget> pins;
    for (auto& node : body.nodes()) {
        NodeDescriptor descriptorStorage;
        const auto* descriptor = resolveSubgraphBodyDescriptor(*definition, node, registry_,
                                                               descriptorStorage);
        ed::BeginNode(ed::NodeId(nodeUiId(node.id)));
        ImGui::PushID(static_cast<int>(node.id));
        if (!descriptor) {
            ImGui::TextColored(ImVec4(1, .35F, .35F, 1), "Missing: %s", node.type.c_str());
        } else {
            ImGui::TextUnformatted(node.label.empty() ? descriptor->displayName.c_str()
                                                       : node.label.c_str());
            if (!node.label.empty() && node.label != descriptor->displayName)
                ImGui::TextDisabled("%s", descriptor->displayName.c_str());
            ImGui::Separator();
            std::size_t pinIndex = 0;
            for (const auto& socket : descriptor->sockets) {
                const auto id = pinUiId(node.id, pinIndex++, socket.direction);
                pins.emplace(id, PinTarget{node.id, socket.key, socket.direction});
                ed::BeginPin(ed::PinId(id), socket.direction == SocketDirection::Input
                                               ? ed::PinKind::Input : ed::PinKind::Output);
                ImGui::TextColored(socketColor(socket.type), "%s%s",
                    socket.direction == SocketDirection::Output ? "● " : "○ ",
                    socket.label.c_str());
                ed::EndPin();
            }
            for (const auto& property : descriptor->parameters) {
                float value = node.parameters.value(property.key, property.defaultValue);
                ImGui::SetNextItemWidth(150);
                bool propertyChanged = false;
                if (property.key == "operation" && node.type == "math") {
                    node_widgets::renderMathOperationSelector(node, nodePopup_);
                    continue;
                } else if (property.control == ParameterDescriptor::Control::Boolean) {
                    bool enabled = value > .5F;
                    propertyChanged = ImGui::Checkbox(property.label.c_str(), &enabled);
                    value = enabled ? 1.0F : 0.0F;
                } else if (property.control == ParameterDescriptor::Control::Integer) {
                    int integer = static_cast<int>(value);
                    propertyChanged = ImGui::SliderInt(property.label.c_str(), &integer,
                        static_cast<int>(property.minimum), static_cast<int>(property.maximum));
                    value = static_cast<float>(integer);
                } else {
                    const float speed = std::max((property.maximum - property.minimum) / 500.0F,
                                                 .0001F);
                    propertyChanged = ImGui::DragFloat(property.label.c_str(), &value, speed,
                        property.minimum, property.maximum, "%.6g", ImGuiSliderFlags_AlwaysClamp);
                }
                if (propertyChanged) { node.parameters[property.key] = value; changed = true; }
            }
        }
        ImGui::PopID();
        ed::EndNode();
        if (!positionedSubgraph_[node.id]) {
            ed::SetNodePosition(ed::NodeId(nodeUiId(node.id)), ImVec2(node.position.x, node.position.y));
            positionedSubgraph_[node.id] = true;
        } else {
            const auto position = ed::GetNodePosition(ed::NodeId(nodeUiId(node.id)));
            if (node.position.x != position.x || node.position.y != position.y) {
                node.position = {position.x, position.y};
                dirty_ = true;
            }
        }
    }

    for (const auto& link : body.links()) {
        const auto* from = body.findNode(link.fromNode);
        const auto* to = body.findNode(link.toNode);
        NodeDescriptor fromStorage, toStorage;
        const auto* fromDesc = from ? resolveSubgraphBodyDescriptor(*definition, *from, registry_, fromStorage) : nullptr;
        const auto* toDesc = to ? resolveSubgraphBodyDescriptor(*definition, *to, registry_, toStorage) : nullptr;
        if (!fromDesc || !toDesc) continue;
        std::size_t fromIndex = 0, toIndex = 0;
        bool foundFrom = false, foundTo = false;
        for (const auto& socket : fromDesc->sockets) {
            if (socket.direction == SocketDirection::Output && socket.key == link.fromSocket) {
                foundFrom = true; break;
            }
            ++fromIndex;
        }
        for (const auto& socket : toDesc->sockets) {
            if (socket.direction == SocketDirection::Input && socket.key == link.toSocket) {
                foundTo = true; break;
            }
            ++toIndex;
        }
        if (foundFrom && foundTo)
            ed::Link(ed::LinkId(linkUiId(link.id)),
                     ed::PinId(pinUiId(link.fromNode, fromIndex, SocketDirection::Output)),
                     ed::PinId(pinUiId(link.toNode, toIndex, SocketDirection::Input)));
    }

    if (ed::BeginCreate()) {
        ed::PinId first, second;
        if (ed::QueryNewLink(&first, &second) && first && second) {
            auto from = pins.find(first.Get()), to = pins.find(second.Get());
            if (from != pins.end() && to != pins.end()) {
                if (from->second.direction == SocketDirection::Input) std::swap(from, to);
                const bool directionOk = from->second.direction == SocketDirection::Output &&
                                         to->second.direction == SocketDirection::Input;
                if (directionOk && ed::AcceptNewItem()) {
                    body.addLink(from->second.node, from->second.socket,
                                 to->second.node, to->second.socket);
                    changed = true;
                }
            }
        }
        ed::EndCreate();
    }
    if (ed::BeginDelete()) {
        ed::LinkId linkId;
        while (ed::QueryDeletedLink(&linkId)) if (ed::AcceptDeletedItem()) {
            body.removeLink(static_cast<LinkId>(linkId.Get() - (std::uintptr_t{1} << 60U)));
            changed = true;
        }
        ed::NodeId nodeId;
        while (ed::QueryDeletedNode(&nodeId)) if (ed::AcceptDeletedItem()) {
            const auto id = static_cast<NodeId>(nodeId.Get() / 128);
            body.removeNode(id);
            positionedSubgraph_.erase(id);
            changed = true;
        }
        ed::EndDelete();
    }

    static ImVec2 addNodeCanvasPosition{};
    ed::Suspend();
    if (node_widgets::renderPopup(nodePopup_, body)) changed = true;
    if (ed::ShowBackgroundContextMenu()) {
        addNodeCanvasPosition = ed::ScreenToCanvas(ImGui::GetMousePos());
        ImGui::OpenPopup("Add subgraph node");
    }
    if (ImGui::BeginPopup("Add subgraph node")) {
        static char search[96]{};
        ImGui::InputTextWithHint("##search", "Search nodes", search, sizeof(search));
        const std::string filter = search;
        const auto addNode = [&](std::string type, std::string label,
                                 nlohmann::json parameters = nlohmann::json::object()) {
            if (!filter.empty() && label.find(filter) == std::string::npos) return;
            if (ImGui::MenuItem(label.c_str())) {
                const auto id = body.addNode(std::move(type),
                    {addNodeCanvasPosition.x, addNodeCanvasPosition.y});
                auto* node = body.findNode(id);
                node->parameters = std::move(parameters);
                positionedSubgraph_[id] = false;
                changed = true;
            }
        };
        for (const auto* type : {"float", "math", "threshold", "select", "coordinates", "laplacian"}) {
            if (const auto* descriptor = registry_.descriptor(type))
                addNode(type, descriptor->category + " / " + descriptor->displayName);
        }
        addNode("simulation_previous_state", "Simulation / Previous Simulation State");
        addNode("simulation_initial_state", "Simulation / Initial Simulation State");
        addNode("simulation_next_state", "Simulation / Next Simulation State");
        for (const auto& item : definition->interface) {
            if (item.kind == SubgraphInterfaceKind::Output) {
                addNode("subgraph_output", "Subgraph / Output: " + item.label,
                        {{"key", item.key}});
            } else {
                addNode("subgraph_input", "Subgraph / " +
                    std::string(item.kind == SubgraphInterfaceKind::Slider ? "Control: " : "Input: ") +
                    item.label, {{"key", item.key}});
            }
        }
        ImGui::EndPopup();
    }
    ed::Resume();
    if (navigateToGraph || fitSubgraphRequested_) {
        ed::NavigateToContent(0.0F);
        fitSubgraphRequested_ = false;
    }
    ed::End();
    ed::SetCurrentEditor(nullptr);

    if (changed) {
        for (auto& node : graph_.nodes()) if (node.type == "subgraph" && node.subgraphId == editingSubgraphId_) {
            for (const auto& item : definition->interface) if (item.kind == SubgraphInterfaceKind::Slider) {
                const float value = node.parameters.value(item.key, item.defaultValue);
                node.parameters[item.key] = std::clamp(value, item.minimum, item.maximum);
            }
        }
        dirty_ = true; rebuildRuntime();
    }
}

void Application::renderEditor() {
    glfwMakeContextCurrent(editorWindow_);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    handleShortcuts();

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New", "Ctrl+N")) newProject();
            if (ImGui::MenuItem("Open…", "Ctrl+O")) loadProjectDialog();
            if (ImGui::MenuItem("Save", "Ctrl+S")) saveProjectDialog(false);
            if (ImGui::MenuItem("Save As…", "Ctrl+Shift+S")) saveProjectDialog(true);
            if (ImGui::MenuItem("Export Current Frame as PNG…")) exportFrameDialog();
            if (recordingPipe_ < 0) {
                if (ImGui::MenuItem("Start Video Recording…")) startRecordingDialog();
            } else if (ImGui::MenuItem("Stop Video Recording")) stopRecording();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit")) {
            const bool rootGraph = editingSubgraphId_.empty();
            if (ImGui::MenuItem("Copy Nodes", "Ctrl+C", false, rootGraph)) copyRequested_ = true;
            if (ImGui::MenuItem("Paste Nodes", "Ctrl+V", false, rootGraph)) pasteRequested_ = true;
            ImGui::Separator();
            if (ImGui::MenuItem("Auto-Layout", "Ctrl+L")) layoutRequested_ = true;
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Window")) {
            if (ImGui::MenuItem("Open Preview", nullptr, false, previewWindow_ == nullptr)) createPreviewWindow();
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetFrameHeight()));
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y - ImGui::GetFrameHeight()));
    ImGui::Begin("Workspace", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoScrollWithMouse);
    if (ImGui::Button(playing_ ? "Pause" : "Play")) playing_ = !playing_;
    ImGui::SameLine(); if (ImGui::Button("Reset")) { runtime_->reset(); elapsed_ = 0; }
    ImGui::SameLine(); if (ImGui::Button("Auto-Layout")) layoutRequested_ = true;
    ImGui::SameLine();
    if (recordingPipe_ < 0) {
        if (ImGui::Button("Record")) startRecordingDialog();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(.65F, .12F, .12F, 1));
        if (ImGui::Button("Stop Recording")) stopRecording();
        ImGui::PopStyleColor();
    }
    ImGui::SameLine(); ImGui::Text("%.1f FPS", displayedFps_);
    ImGui::BeginDisabled(recordingPipe_ >= 0);
    ImGui::SameLine(); ImGui::SetNextItemWidth(80);
    if (ImGui::SliderInt("Target", &graph_.settings.targetFps, 1, 240)) dirty_ = true;
    ImGui::SameLine(); ImGui::SetNextItemWidth(90); int width = graph_.settings.width;
    if (ImGui::InputInt("W", &width, 0)) { graph_.settings.width = std::clamp(width, 16, 8192); updatePreviewAspectRatio(); runtime_->reset(); dirty_ = true; }
    ImGui::SameLine(); ImGui::SetNextItemWidth(90); int height = graph_.settings.height;
    if (ImGui::InputInt("H", &height, 0)) { graph_.settings.height = std::clamp(height, 16, 8192); updatePreviewAspectRatio(); runtime_->reset(); dirty_ = true; }
    ImGui::EndDisabled();
    ImGui::SameLine(); ImGui::TextColored(statusError_ ? ImVec4(1,.35F,.35F,1) : ImVec4(.55F,.8F,.55F,1), "%s%s", dirty_ ? "* " : "", status_.c_str());
    if (editingSubgraphId_.empty()) renderGraph();
    else renderSubgraphEditor();
    ImGui::End();

    ImGui::Render();
    int widthPixels = 0, heightPixels = 0; glfwGetFramebufferSize(editorWindow_, &widthPixels, &heightPixels);
    glViewport(0, 0, widthPixels, heightPixels); glClearColor(.035F,.035F,.045F,1); glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(editorWindow_);
}

void Application::renderPreview() {
    if (!previewWindow_) return;
    if (glfwWindowShouldClose(previewWindow_)) {
        glfwMakeContextCurrent(previewWindow_);
        if (previewVertexArray_) glDeleteVertexArrays(1, &previewVertexArray_);
        previewVertexArray_ = 0;
        glfwDestroyWindow(previewWindow_);
        previewWindow_ = nullptr;
        glfwMakeContextCurrent(editorWindow_);
        return;
    }
    glfwMakeContextCurrent(previewWindow_);
    int width = 0, height = 0; glfwGetFramebufferSize(previewWindow_, &width, &height);
    gpu_->drawFullscreen(runtime_->outputImage().texture, previewVertexArray_, width, height);
    glfwSwapBuffers(previewWindow_);
}

int Application::run() {
    using Clock = std::chrono::steady_clock;
    auto previous = Clock::now();
    double fpsAccumulator = 0; int fpsFrames = 0;
    while (!glfwWindowShouldClose(editorWindow_)) {
        const auto frameStart = Clock::now();
        const double delta = std::chrono::duration<double>(frameStart - previous).count(); previous = frameStart;
        const double target = 1.0 / static_cast<double>(std::max(graph_.settings.targetFps, 1));
        const double evaluationDelta = recordingPipe_ >= 0 ? target : delta;
        if (playing_) elapsed_ += evaluationDelta;
        glfwPollEvents();
        glfwMakeContextCurrent(editorWindow_);
        runtime_->evaluate(elapsed_, playing_ ? evaluationDelta : 0.0, playing_);
        recordFrame();
        renderEditor();
        renderPreview();
        fpsAccumulator += delta; ++fpsFrames;
        if (fpsAccumulator >= .5) { displayedFps_ = static_cast<double>(fpsFrames) / fpsAccumulator; fpsAccumulator = 0; fpsFrames = 0; }
        const double spent = std::chrono::duration<double>(Clock::now() - frameStart).count();
        if (spent < target) std::this_thread::sleep_for(std::chrono::duration<double>(target - spent));
    }
    return 0;
}

} // namespace reaction
