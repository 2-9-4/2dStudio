#include "application.hpp"
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

std::string humanizeIdentifier(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 8);
    for (std::size_t index = 0; index < value.size(); ++index) {
        const unsigned char current = static_cast<unsigned char>(value[index]);
        if (value[index] == '_' || value[index] == '-') {
            if (!result.empty() && result.back() != ' ') result.push_back(' ');
            continue;
        }
        if (index > 0 && std::isupper(current) && result.back() != ' ') result.push_back(' ');
        result.push_back(value[index]);
    }
    bool capitalize = true;
    for (char& character : result) {
        if (character == ' ') capitalize = true;
        else if (capitalize) {
            character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
            capitalize = false;
        }
    }
    return result;
}

std::string operationLabel(std::string_view operation) {
    static const std::unordered_map<std::string_view, std::string_view> labels{
        {"uv", "Canvas Coordinates"}, {"constant", "Number"}, {"constant2", "2D Value"},
        {"previous_state", "Previous State"}, {"interface", "Subgraph Input"},
        {"connected", "Input Connected"}, {"laplacian", "Laplacian"},
        {"pack2", "Combine Channels"}, {"swizzle", "Extract Channel"},
        {"clamp01", "Clamp 0–1"}, {"output", "Subgraph Output"}
    };
    const auto found = labels.find(operation);
    return found == labels.end() ? humanizeIdentifier(operation) : std::string(found->second);
}

std::string kernelNodeLabel(const SubgraphKernelNode& node,
                            const SubgraphDefinition& definition) {
    if (node.properties.is_object()) {
        const auto explicitLabel = node.properties.value("label", std::string{});
        if (!explicitLabel.empty()) return explicitLabel;
        if (node.operation == "interface" || node.operation == "connected" || node.operation == "output") {
            const auto key = node.properties.value("key", std::string{});
            const auto item = std::ranges::find(definition.interface, key, &SubgraphInterfaceItem::key);
            if (item != definition.interface.end()) return item->label;
        }
    }
    return humanizeIdentifier(node.key);
}

std::string kernelInputLabel(std::string_view operation, std::size_t input) {
    if (operation == "select") {
        static const std::array labels{"Condition", "True", "False"};
        if (input < labels.size()) return labels[input];
    }
    if (operation == "clamp") {
        static const std::array labels{"Value", "Minimum", "Maximum"};
        if (input < labels.size()) return labels[input];
    }
    if (operation == "remap") {
        static const std::array labels{"Value", "Input Min", "Input Max", "Output Min", "Output Max"};
        if (input < labels.size()) return labels[input];
    }
    if (operation == "pack2") return input == 0 ? "Channel A" : "Channel B";
    if (operation == "step") return input == 0 ? "Value" : "Edge";
    if (operation == "subtract" || operation == "divide" || operation == "pow")
        return input == 0 ? "A" : "B";
    return input == 0 ? "Value" : "Value " + std::to_string(input + 1);
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
    nodeEditor_ = ed::CreateEditor();
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
    graph_ = reaction::loadProject(path, registry_);
    updatePreviewAspectRatio();
    currentPath_ = path;
    positioned_.clear();
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
               "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
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

    std::vector<float> source(static_cast<std::size_t>(image.width) *
                              static_cast<std::size_t>(image.height) * 4U);
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(image.width) *
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
                pixels[pixelIndex + static_cast<std::size_t>(channel)] = static_cast<std::uint8_t>(
                    std::lround(std::clamp(std::isfinite(value) ? value : 0.0F, 0.0F, 1.0F) * 255.0F));
            }
        }
    }

    std::size_t written = 0;
    while (written < pixels.size()) {
        const ssize_t count = write(recordingPipe_, pixels.data() + written, pixels.size() - written);
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
    if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_C, false)) copyRequested_ = true;
    if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_V, false)) pasteRequested_ = true;
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

void Application::renderGraph() {
    ed::SetCurrentEditor(nodeEditor_);
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
            node.position = {position.x, position.y};
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
                positioned_[id] = false; dirty_ = true; rebuildRuntime();
            }
        };
        for (const auto& definition : builtInSubgraphs()) addSubgraph(definition);
        for (const auto& definition : graph_.subgraphs()) addSubgraph(definition);
        ImGui::EndPopup();
    }
    ed::Resume();
    if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
        const int selectedCount = ed::GetSelectedObjectCount();
        if (selectedCount == 1) {
            ed::NodeId selectedUiId;
            if (ed::GetSelectedNodes(&selectedUiId, 1) == 1) {
                const auto selectedId = static_cast<NodeId>(selectedUiId.Get() / 128);
                const auto* selected = graph_.findNode(selectedId);
                if (selected && selected->type == "subgraph" &&
                    resolveSubgraph(graph_, selected->subgraphId)) {
                    editingSubgraphId_ = selected->subgraphId;
                    editingSubgraphInstance_ = selected->id;
                    positionedSubgraphId_.clear();
                    setStatus("Entered subgraph — press Tab to return");
                }
            }
        }
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
    copy.name = "Reaction Diffusion Copy " + std::to_string(copyNumber);
    graph_.subgraphs().push_back(std::move(copy));
    node.subgraphId = graph_.subgraphs().back().id;
    dirty_ = true;
    rebuildRuntime();
    setStatus("Created editable shared subgraph — press Tab to enter");
}

void Application::renderSubgraphEditor() {
    if (editingSubgraphId_.empty()) return;
    if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
        editingSubgraphId_.clear();
        editingSubgraphInstance_ = 0;
        positionedSubgraphId_.clear();
        setStatus("Returned to root graph");
        return;
    }
    const auto* resolved = resolveSubgraph(graph_, editingSubgraphId_);
    if (!resolved) {
        editingSubgraphId_.clear();
        editingSubgraphInstance_ = 0;
        return;
    }
    const bool readOnly = resolved->immutable;
    SubgraphDefinition* definition = readOnly ? nullptr : graph_.findSubgraph(editingSubgraphId_);
    bool changed = false;
    ImGui::Text("Root / %s", resolved->name.c_str());
    ImGui::SameLine(); ImGui::TextDisabled("Tab: return to root");
    if (readOnly) {
        ImGui::TextColored(ImVec4(.85F, .7F, .25F, 1),
                           "Built-in template — inspect it here or make an editable project copy.");
        ImGui::SameLine();
        if (ImGui::Button("Make Editable Copy")) {
            if (auto* instance = graph_.findNode(editingSubgraphInstance_)) {
                duplicateSubgraph(*instance);
                editingSubgraphId_ = instance->subgraphId;
                positionedSubgraphId_.clear();
            }
            return;
        }
    }
    const auto& shown = readOnly ? *resolved : *definition;

    const ImVec2 available = ImGui::GetContentRegionAvail();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float leftWidth = std::min(260.0F, available.x * .23F);
    const float rightWidth = std::min(300.0F, available.x * .27F);
    const float canvasWidth = std::max(180.0F, available.x - leftWidth - rightWidth - spacing * 2.0F);

    ImGui::BeginChild("SubgraphInterface", ImVec2(leftWidth, 0), true);
    ImGui::SeparatorText("Interface");
    if (!readOnly) {
        char name[128]{};
        std::snprintf(name, sizeof(name), "%s", definition->name.c_str());
        if (ImGui::InputText("Name", name, sizeof(name))) {
            definition->name = name; changed = true;
        }
    }
    for (std::size_t index = 0; index < shown.interface.size(); ++index) {
        const auto& current = shown.interface[index];
        ImGui::PushID(static_cast<int>(index));
        ImGui::Separator();
        ImGui::TextUnformatted(current.label.c_str());
        ImGui::TextDisabled("%s · %s", current.kind == SubgraphInterfaceKind::Input ? "Input" :
                                      current.kind == SubgraphInterfaceKind::Slider ? "Control" : "Output",
                                      current.key.c_str());
        if (!readOnly) {
            auto& item = definition->interface[index];
            char label[128]{}; std::snprintf(label, sizeof(label), "%s", item.label.c_str());
            if (ImGui::InputText("Label", label, sizeof(label))) { item.label = label; changed = true; }
            if (item.kind == SubgraphInterfaceKind::Slider) {
                changed |= ImGui::DragFloat("Default", &item.defaultValue, .001F, item.minimum, item.maximum);
                changed |= ImGui::DragFloat("Minimum", &item.minimum, .001F);
                changed |= ImGui::DragFloat("Maximum", &item.maximum, .001F);
                if (item.minimum > item.maximum) std::swap(item.minimum, item.maximum);
                item.defaultValue = std::clamp(item.defaultValue, item.minimum, item.maximum);
            }
        }
        ImGui::PopID();
    }
    ImGui::EndChild();
    ImGui::SameLine();

    ImGui::BeginChild("KernelCanvas", ImVec2(canvasWidth, 0), true);
    ed::SetCurrentEditor(subgraphEditor_);
    ed::Begin("Subgraph kernel");
    struct KernelPin { std::size_t node = 0, input = 0; bool output = false; };
    std::unordered_map<std::uintptr_t, KernelPin> kernelPins;
    std::unordered_map<std::string, std::size_t> kernelIndices;
    for (std::size_t index = 0; index < shown.kernel.size(); ++index)
        kernelIndices.emplace(shown.kernel[index].key, index);
    const auto kernelNodeId = [](std::size_t index) { return std::uintptr_t{1000} + index * 128; };
    const auto kernelInputId = [&](std::size_t index, std::size_t input) {
        return kernelNodeId(index) + 2 + input * 2;
    };
    const auto kernelOutputId = [&](std::size_t index) { return kernelNodeId(index) + 1; };
    const bool navigateToGraph = positionedSubgraphId_ != editingSubgraphId_;
    for (std::size_t index = 0; index < shown.kernel.size(); ++index) {
        const auto& kernel = shown.kernel[index];
        ed::BeginNode(ed::NodeId(kernelNodeId(index)));
        ImGui::TextUnformatted(kernelNodeLabel(kernel, shown).c_str());
        ImGui::TextDisabled("%s", operationLabel(kernel.operation).c_str());
        ImGui::Separator();
        for (std::size_t input = 0; input < kernel.inputs.size(); ++input) {
            const auto pin = kernelInputId(index, input);
            kernelPins.emplace(pin, KernelPin{index, input, false});
            ed::BeginPin(ed::PinId(pin), ed::PinKind::Input);
            ImGui::Text("○ %s", kernelInputLabel(kernel.operation, input).c_str());
            ed::EndPin();
        }
        const auto outputPin = kernelOutputId(index);
        kernelPins.emplace(outputPin, KernelPin{index, 0, true});
        ed::BeginPin(ed::PinId(outputPin), ed::PinKind::Output);
        ImGui::TextUnformatted("● Value");
        ed::EndPin();
        ed::EndNode();
        if (navigateToGraph)
            ed::SetNodePosition(ed::NodeId(kernelNodeId(index)), ImVec2(kernel.position.x, kernel.position.y));
        else if (!readOnly) {
            const auto position = ed::GetNodePosition(ed::NodeId(kernelNodeId(index)));
            definition->kernel[index].position = {position.x, position.y};
        }
    }
    positionedSubgraphId_ = editingSubgraphId_;
    std::uintptr_t linkId = std::uintptr_t{1} << 56U;
    for (std::size_t target = 0; target < shown.kernel.size(); ++target) {
        for (std::size_t input = 0; input < shown.kernel[target].inputs.size(); ++input) {
            const auto source = kernelIndices.find(shown.kernel[target].inputs[input]);
            if (source != kernelIndices.end())
                ed::Link(ed::LinkId(linkId++), ed::PinId(kernelOutputId(source->second)),
                         ed::PinId(kernelInputId(target, input)));
        }
    }
    if (!readOnly && ed::BeginCreate()) {
        ed::PinId first, second;
        if (ed::QueryNewLink(&first, &second) && first && second) {
            auto from = kernelPins.find(first.Get()), to = kernelPins.find(second.Get());
            if (from != kernelPins.end() && to != kernelPins.end()) {
                if (!from->second.output) std::swap(from, to);
                const bool valid = from->second.output && !to->second.output &&
                                   from->second.node < to->second.node;
                if (valid && ed::AcceptNewItem()) {
                    definition->kernel[to->second.node].inputs[to->second.input] =
                        definition->kernel[from->second.node].key;
                    changed = true;
                }
            }
        }
        ed::EndCreate();
    }
    int selectedKernelIndex = -1;
    ed::NodeId selectedNode;
    if (ed::GetSelectedNodes(&selectedNode, 1) == 1 && selectedNode.Get() >= 1000) {
        const auto offset = selectedNode.Get() - 1000;
        if (offset % 128 == 0 && offset / 128 < shown.kernel.size())
            selectedKernelIndex = static_cast<int>(offset / 128);
    }
    if (navigateToGraph) ed::NavigateToContent(0.0F);
    ed::End();
    ed::SetCurrentEditor(nullptr);
    ImGui::EndChild();
    ImGui::SameLine();

    ImGui::BeginChild("KernelInspector", ImVec2(rightWidth, 0), true);
    ImGui::SeparatorText("Node Inspector");
    if (selectedKernelIndex < 0) {
        ImGui::TextWrapped("Select a kernel node to inspect its stable key, operation, and inputs.");
        ImGui::Spacing();
        ImGui::TextDisabled("This graph is fused into one compute shader per simulation iteration.");
    } else {
        const auto index = static_cast<std::size_t>(selectedKernelIndex);
        const auto& current = shown.kernel[index];
        ImGui::TextUnformatted(kernelNodeLabel(current, shown).c_str());
        ImGui::TextDisabled("Key: %s", current.key.c_str());
        ImGui::TextDisabled("Operation: %s", operationLabel(current.operation).c_str());
        static const std::array operations{"add", "subtract", "multiply", "divide", "min", "max",
            "abs", "sin", "cos", "pow", "clamp01", "clamp", "remap", "pack2", "swizzle", "length", "step", "select"};
        ImGui::PushID(1000 + selectedKernelIndex);
        if (!readOnly) {
            auto& kernel = definition->kernel[index];
            char label[128]{};
            const auto displayedLabel = kernelNodeLabel(kernel, shown);
            std::snprintf(label, sizeof(label), "%s", displayedLabel.c_str());
            if (ImGui::InputText("Label", label, sizeof(label))) {
                kernel.properties["label"] = label; changed = true;
            }
            if (kernel.operation != "uv" && kernel.operation != "previous_state" &&
                kernel.operation != "interface" && kernel.operation != "connected" &&
                kernel.operation != "laplacian" && kernel.operation != "output") {
                if (ImGui::BeginCombo("Operation", operationLabel(kernel.operation).c_str())) {
                    for (const auto* operation : operations) {
                        if (ImGui::Selectable(operationLabel(operation).c_str(),
                                              kernel.operation == operation)) {
                            kernel.operation = operation; changed = true;
                        }
                    }
                    ImGui::EndCombo();
                }
            }
            for (std::size_t inputIndex = 0; inputIndex < kernel.inputs.size(); ++inputIndex) {
                ImGui::PushID(static_cast<int>(inputIndex));
                const auto inputLabel = kernelInputLabel(kernel.operation, inputIndex);
                if (ImGui::BeginCombo(inputLabel.c_str(), kernel.inputs[inputIndex].c_str())) {
                    for (std::size_t candidate = 0; candidate < index; ++candidate) {
                        const auto& candidateNode = definition->kernel[candidate];
                        const auto& key = candidateNode.key;
                        if (ImGui::Selectable(kernelNodeLabel(candidateNode, *definition).c_str(),
                                              kernel.inputs[inputIndex] == key)) {
                            kernel.inputs[inputIndex] = key; changed = true;
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::PopID();
            }
            if (kernel.operation == "constant" && kernel.properties.contains("value")) {
                float value = kernel.properties.value("value", 0.0F);
                if (ImGui::DragFloat("Value", &value, .001F)) { kernel.properties["value"] = value; changed = true; }
            }
        } else {
            ImGui::Spacing();
            for (std::size_t input = 0; input < current.inputs.size(); ++input)
                ImGui::BulletText("%s: %s", kernelInputLabel(current.operation, input).c_str(),
                                  current.inputs[input].c_str());
        }
        ImGui::PopID();
    }
    ImGui::EndChild();
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
            if (ImGui::MenuItem("Copy Nodes", "Ctrl+C")) copyRequested_ = true;
            if (ImGui::MenuItem("Paste Nodes", "Ctrl+V")) pasteRequested_ = true;
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
    ImGui::Begin("Workspace", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
    if (ImGui::Button(playing_ ? "Pause" : "Play")) playing_ = !playing_;
    ImGui::SameLine(); if (ImGui::Button("Reset")) { runtime_->reset(); elapsed_ = 0; }
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
        if (playing_) elapsed_ += delta;
        glfwPollEvents();
        glfwMakeContextCurrent(editorWindow_);
        runtime_->evaluate(elapsed_, playing_ ? delta : 0.0, playing_);
        renderEditor();
        glfwMakeContextCurrent(editorWindow_);
        recordFrame();
        renderPreview();
        fpsAccumulator += delta; ++fpsFrames;
        if (fpsAccumulator >= .5) { displayedFps_ = static_cast<double>(fpsFrames) / fpsAccumulator; fpsAccumulator = 0; fpsFrames = 0; }
        const double target = 1.0 / static_cast<double>(std::max(graph_.settings.targetFps, 1));
        const double spent = std::chrono::duration<double>(Clock::now() - frameStart).count();
        if (spent < target) std::this_thread::sleep_for(std::chrono::duration<double>(target - spent));
    }
    return 0;
}

} // namespace reaction
