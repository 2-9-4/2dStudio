#include "application.hpp"
#include "node_search.hpp"
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
#include <sstream>
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

std::filesystem::path recoveryPath(const std::filesystem::path& projectPath) {
    return projectPath.string() + ".recovery";
}

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
    case ValueType::Vec2: return ImColor(82, 210, 190);
    case ValueType::Image2D: return ImColor(241, 196, 80);
    case ValueType::AnyNumeric: return ImColor(180, 125, 230);
    case ValueType::AnyVector: return ImColor(112, 200, 145);
    }
    return ImColor(180, 180, 180);
}

void renderSocketPin(NodeId nodeId, const NodeDescriptor& descriptor, std::size_t socketIndex,
                     std::unordered_map<std::uintptr_t, PinTarget>& pins) {
    const auto& socket = descriptor.sockets[socketIndex];
    const auto id = pinUiId(nodeId, socketIndex, socket.direction);
    pins.emplace(id, PinTarget{nodeId, socket.key, socket.direction});
    const bool input = socket.direction == SocketDirection::Input;
    ed::BeginPin(ed::PinId(id), input ? ed::PinKind::Input : ed::PinKind::Output);
    ed::PinPivotAlignment(input ? ImVec2(0.0F, 0.5F) : ImVec2(1.0F, 0.5F));
    if (input) {
        ImGui::TextColored(socketColor(socket.type), "○ %s", socket.label.c_str());
    } else {
        ImGui::TextColored(socketColor(socket.type), "%s", socket.label.c_str());
        ImGui::SameLine(0.0F, 4.0F);
        ImGui::TextColored(socketColor(socket.type), "●");
    }
    ed::EndPin();
}

std::ptrdiff_t inputSocketIndex(const NodeDescriptor& descriptor, std::string_view key) {
    for (std::size_t index = 0; index < descriptor.sockets.size(); ++index) {
        const auto& socket = descriptor.sockets[index];
        if (socket.direction == SocketDirection::Input && socket.key == key)
            return static_cast<std::ptrdiff_t>(index);
    }
    return -1;
}

bool compatible(const SocketDescriptor& from, const SocketDescriptor& to) {
    if (!areSocketTypesCompatible(from.type, to.type)) return false;
    if (!(from.strictType || to.strictType)) return true;
    return !(isNumericType(from.type) && to.type == ValueType::AnyVector) &&
           !(from.type == ValueType::AnyVector && isNumericType(to.type));
}

// Operation-dependent nodes can remove an input or change their result category.
// Keep their stored graph links in sync with that live descriptor rather than
// retaining invisible links that would make the graph invalid.
bool pruneInactiveLinks(Graph& graph, const NodeRegistry& registry) {
    const auto before = graph.links().size();
    std::erase_if(graph.links(), [&](const LinkRecord& link) {
        const auto* from = graph.findNode(link.fromNode);
        const auto* to = graph.findNode(link.toNode);
        if (!from || !to) return true;
        NodeDescriptor fromStorage, toStorage;
        const auto* fromDescriptor = resolveDescriptor(graph, *from, registry, fromStorage);
        const auto* toDescriptor = resolveDescriptor(graph, *to, registry, toStorage);
        if (!fromDescriptor || !toDescriptor) return true;
        const auto output = std::ranges::find_if(fromDescriptor->sockets, [&](const auto& socket) {
                return socket.key == link.fromSocket && socket.direction == SocketDirection::Output;
            });
        const auto input = std::ranges::find_if(toDescriptor->sockets, [&](const auto& socket) {
                return socket.key == link.toSocket && socket.direction == SocketDirection::Input;
            });
        return output == fromDescriptor->sockets.end() || input == toDescriptor->sockets.end() ||
            !compatible(*output, *input);
    });
    return graph.links().size() != before;
}

bool renderParameterRow(NodeId nodeId, const NodeDescriptor& descriptor,
                        const ParameterDescriptor& property, NodeRecord& node,
                        node_widgets::PopupState& popup,
                        std::unordered_map<std::uintptr_t, PinTarget>& pins,
                        bool connected) {
    const auto socketIndex = inputSocketIndex(descriptor, property.key);
    const bool hasSocket = socketIndex >= 0;
    if (hasSocket)
        renderSocketPin(nodeId, descriptor, static_cast<std::size_t>(socketIndex), pins);
    if (connected) ImGui::BeginDisabled();
    ImGui::SetNextItemWidth(150);
    const std::string hiddenLabel = "##" + property.key;
    const char* label = hasSocket ? hiddenLabel.c_str() : property.label.c_str();
    bool changed = false;
    if (property.key == "operation" && node.type == "math") {
        node_widgets::renderMathOperationSelector(node, popup);
    } else if (property.key == "mode" && node.type == "mix") {
        node_widgets::renderMixModeSelector(node, popup);
    } else if (property.control == ParameterDescriptor::Control::Boolean) {
        bool enabled = node.parameters.value(property.key, property.defaultValue) > 0.5F;
        changed = ImGui::Checkbox(label, &enabled);
        if (changed) node.parameters[property.key] = enabled ? 1.0F : 0.0F;
    } else if (property.control == ParameterDescriptor::Control::Enum) {
        node_widgets::renderEnumSelector(property, node, popup);
    } else if (property.control == ParameterDescriptor::Control::Integer ||
               property.key == "iterations" || property.key == "octaves" || property.key == "seed") {
        int integer = static_cast<int>(node.parameters.value(property.key, property.defaultValue));
        changed = ImGui::SliderInt(label, &integer,
                                   static_cast<int>(property.minimum),
                                   static_cast<int>(property.maximum));
        if (changed) node.parameters[property.key] = static_cast<float>(integer);
    } else {
        float value = node.parameters.value(property.key, property.defaultValue);
        const float speed = std::max((property.maximum - property.minimum) / 500.0F, 0.0001F);
        changed = ImGui::DragFloat(label, &value, speed, property.minimum, property.maximum,
                                   "%.6g", ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Drag to adjust; Ctrl+click to type an exact value");
        if (changed) node.parameters[property.key] = value;
    }
    if (connected) ImGui::EndDisabled();
    if (connected && ImGui::IsItemHovered())
        ImGui::SetTooltip("Connected input overrides this value");
    return changed;
}

struct ColorRampPickerState {
    NodeId nodeId = 0;
    bool highColor = false;
    bool openRequested = false;
};

ColorRampPickerState colorRampPicker;

bool renderColorRampColors(NodeId nodeId, NodeRecord& node,
                           const std::unordered_set<std::string>& connectedInputs) {
    const auto isConnected = [&](std::string_view key) {
        return connectedInputs.contains(std::to_string(nodeId) + "\x1f" + std::string(key));
    };
    const bool lowConnected = isConnected("r0") || isConnected("g0") || isConnected("b0");
    const bool highConnected = isConnected("r1") || isConnected("g1") || isConnected("b1");

    std::array<float, 3> low{
        node.parameters.value("r0", .015F), node.parameters.value("g0", .01F),
        node.parameters.value("b0", .04F)};
    std::array<float, 3> high{
        node.parameters.value("r1", 1.0F), node.parameters.value("g1", .35F),
        node.parameters.value("b1", .08F)};

    bool changed = false;
    const auto renderColor = [&](const char* label, std::array<float, 3>& color, bool connected) {
        if (connected) ImGui::BeginDisabled();
        ImGui::TextUnformatted(label);
        ImGui::SameLine();
        ImGui::PushID(label);
        if (ImGui::ColorButton("##swatch", ImVec4(color[0], color[1], color[2], 1.0F),
                               ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoDragDrop,
                               ImVec2(72.0F, 20.0F))) {
            colorRampPicker = {nodeId, label[0] == 'H', true};
        }
        ImGui::PopID();
        if (connected) ImGui::EndDisabled();
        return false;
    };

    if (renderColor("Low color", low, lowConnected)) {
        node.parameters["r0"] = low[0];
        node.parameters["g0"] = low[1];
        node.parameters["b0"] = low[2];
        changed = true;
    }
    if (renderColor("High color", high, highConnected)) {
        node.parameters["r1"] = high[0];
        node.parameters["g1"] = high[1];
        node.parameters["b1"] = high[2];
        changed = true;
    }
    return changed;
}

template <typename GraphLike>
bool renderColorRampPickerPopup(GraphLike& graph) {
    if (colorRampPicker.openRequested) {
        ImGui::OpenPopup("Color ramp color picker");
        colorRampPicker.openRequested = false;
    }
    if (!ImGui::BeginPopup("Color ramp color picker")) return false;

    bool changed = false;
    if (auto* node = graph.findNode(colorRampPicker.nodeId); node && node->type == "color_ramp") {
        const char* prefix = colorRampPicker.highColor ? "High" : "Low";
        std::array<float, 3> color{
            node->parameters.value(colorRampPicker.highColor ? "r1" : "r0", colorRampPicker.highColor ? 1.0F : .015F),
            node->parameters.value(colorRampPicker.highColor ? "g1" : "g0", colorRampPicker.highColor ? .35F : .01F),
            node->parameters.value(colorRampPicker.highColor ? "b1" : "b0", colorRampPicker.highColor ? .08F : .04F)};
        ImGui::Text("%s color", prefix);
        if (ImGui::ColorPicker3("##picker", color.data(),
                                ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs)) {
            node->parameters[colorRampPicker.highColor ? "r1" : "r0"] = color[0];
            node->parameters[colorRampPicker.highColor ? "g1" : "g0"] = color[1];
            node->parameters[colorRampPicker.highColor ? "b1" : "b0"] = color[2];
            changed = true;
        }
    } else {
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
    return changed;
}

ImVec4 contributorColor(NodeId id) {
    const float hue = std::fmod(static_cast<float>(id) * 0.61803398875F, 1.0F);
    return ImColor::HSV(hue, 0.55F, 0.95F);
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
    glfwSetWindowUserPointer(editorWindow_, this);
    glfwSetWindowCloseCallback(editorWindow_, &Application::editorWindowCloseCallback);
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
    if (intermediatePreviewWindow_) {
        glfwMakeContextCurrent(intermediatePreviewWindow_);
        if (intermediatePreviewVertexArray_)
            glDeleteVertexArrays(1, &intermediatePreviewVertexArray_);
        glfwDestroyWindow(intermediatePreviewWindow_);
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

void Application::createIntermediatePreviewWindow(NodeId nodeId) {
    intermediatePreviewNode_ = nodeId;
    if (intermediatePreviewWindow_) {
        glfwFocusWindow(intermediatePreviewWindow_);
        return;
    }

    constexpr int maximumSize = 900;
    const double aspect = static_cast<double>(graph_.settings.width) /
                          static_cast<double>(graph_.settings.height);
    const int width = aspect >= 1.0 ? maximumSize :
        std::max(1, static_cast<int>(std::lround(maximumSize * aspect)));
    const int height = aspect >= 1.0 ?
        std::max(1, static_cast<int>(std::lround(maximumSize / aspect))) : maximumSize;
    intermediatePreviewWindow_ = glfwCreateWindow(width, height,
        "Reaction Studio — Intermediate Preview", nullptr, editorWindow_);
    if (!intermediatePreviewWindow_)
        throw std::runtime_error("Unable to create intermediate preview window");
    glfwMakeContextCurrent(intermediatePreviewWindow_);
    glfwSwapInterval(0);
    updatePreviewAspectRatio();
    glGenVertexArrays(1, &intermediatePreviewVertexArray_);
    glfwMakeContextCurrent(editorWindow_);
}

void Application::updatePreviewAspectRatio() {
    if (previewWindow_ && graph_.settings.width > 0 && graph_.settings.height > 0)
        glfwSetWindowAspectRatio(previewWindow_, graph_.settings.width, graph_.settings.height);
    if (intermediatePreviewWindow_ && graph_.settings.width > 0 && graph_.settings.height > 0)
        glfwSetWindowAspectRatio(intermediatePreviewWindow_, graph_.settings.width, graph_.settings.height);
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
    recoverySnapshot_.clear();
    recoveryCandidate_.clear();
    recoveryPromptDismissed_ = false;
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
    runtime_->setFusionEnabled(executeFusedShaders_);
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
    const auto sidecar = recoveryPath(path);
    std::error_code error;
    const bool recover = std::filesystem::exists(sidecar, error) && !error &&
        (!std::filesystem::exists(path, error) ||
         std::filesystem::last_write_time(sidecar, error) >
             std::filesystem::last_write_time(path, error));
    graph_ = reaction::loadProject(path, registry_);
    updatePreviewAspectRatio();
    currentPath_ = path;
    positioned_.clear();
    fitRootGraph_ = true;
    elapsed_ = 0;
    rebuildRuntime();
    recoverySnapshot_ = serializeProject(graph_).dump();
    recoveryCandidate_ = recover ? sidecar : std::filesystem::path{};
    recoveryPromptDismissed_ = false;
    dirty_ = false;
    setStatus(recover ? "Newer recovery changes are available"
                      : "Loaded " + currentPath_.filename().string());
}

bool Application::saveProjectDialog(bool forceDialog) {
    if (forceDialog || currentPath_.empty()) {
        auto selected = pfd::save_file("Save Reaction project", "artwork.reaction.json",
                                       {"Reaction project", "*.reaction.json"}).result();
        if (selected.empty()) return false;
        currentPath_ = std::move(selected);
    }
    try {
        // Keep a separate, atomically replaced recovery copy. It intentionally
        // stays beside the project after a normal save, so it is also useful if
        // the project file itself is interrupted or damaged during a later save.
        saveProjectAtomic(graph_, recoveryPath(currentPath_));
        // Write the primary project last. Its timestamp then records that it
        // already includes this recovery snapshot.
        saveProjectAtomic(graph_, currentPath_);
        recoverySnapshot_ = serializeProject(graph_).dump();
        recoveryCandidate_.clear();
        recoveryPromptDismissed_ = false;
        dirty_ = false;
        setStatus("Saved " + currentPath_.filename().string());
        return true;
    } catch (const std::exception& error) {
        setStatus(error.what(), true);
        return false;
    }
}

void Application::updateRecoverySidecar() {
    if (closeApproved_ || currentPath_.empty()) return;
    const auto snapshot = serializeProject(graph_).dump();
    if (snapshot == recoverySnapshot_) return;
    try {
        saveProjectAtomic(graph_, recoveryPath(currentPath_));
        recoverySnapshot_ = snapshot;
    } catch (const std::exception& error) {
        setStatus(std::string("Recovery save failed: ") + error.what(), true);
    }
}

void Application::discardRecoverySidecar() {
    if (currentPath_.empty()) return;
    std::error_code error;
    std::filesystem::remove(recoveryPath(currentPath_), error);
    if (error) setStatus(std::string("Could not remove recovery copy: ") + error.message(), true);
    recoveryCandidate_.clear();
    recoveryPromptDismissed_ = false;
}

void Application::requestEditorClose() {
    if (closeApproved_) return;
    glfwSetWindowShouldClose(editorWindow_, GLFW_FALSE);
    if (dirty_) {
        closeConfirmationOpen_ = true;
        return;
    }
    discardRecoverySidecar();
    closeApproved_ = true;
    glfwSetWindowShouldClose(editorWindow_, GLFW_TRUE);
}

void Application::editorWindowCloseCallback(GLFWwindow* window) {
    auto* application = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (application) application->requestEditorClose();
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
    std::unordered_set<std::string> connectedInputs;
    for (const auto& link : graph_.links())
        connectedInputs.insert(std::to_string(link.toNode) + "\x1f" + link.toSocket);

    for (auto& node : graph_.nodes()) {
        NodeDescriptor descriptorStorage;
        const auto* descriptor = resolveDescriptor(graph_, node, registry_, descriptorStorage);
        const auto fusion = runtime_->fusionInfo(node.id);
        const bool selectedFusedRegion = shaderInspectorOpen_ && selectedShaderRegion_ != 0 &&
            fusion && fusion->region == selectedShaderRegion_;
        const bool failedGeneratedRegion = fusion && fusion->compileFailed;
        if (failedGeneratedRegion) {
            ed::PushStyleColor(ed::StyleColor_NodeBorder, ImVec4(1.0F, .18F, .14F, 1.0F));
            ed::PushStyleColor(ed::StyleColor_NodeBg, ImVec4(.24F, .045F, .04F, 1.0F));
            ed::PushStyleVar(ed::StyleVar_NodeBorderWidth, 4.0F);
        } else if (selectedFusedRegion) {
            ed::PushStyleColor(ed::StyleColor_NodeBorder, ImVec4(.3F, .82F, 1.0F, 1.0F));
            ed::PushStyleColor(ed::StyleColor_NodeBg, ImVec4(.06F, .14F, .19F, 1.0F));
            ed::PushStyleVar(ed::StyleVar_NodeBorderWidth, 4.0F);
        }
        ed::BeginNode(ed::NodeId(nodeUiId(node.id)));
        ImGui::PushID(static_cast<int>(node.id));
        if (!descriptor) {
            ImGui::TextColored(ImVec4(1, .35F, .35F, 1), "Missing: %s", node.type.c_str());
        } else {
            ImGui::TextUnformatted(descriptor->displayName.c_str());
            ImGui::Separator();
            ImGui::BeginGroup();
            for (std::size_t socketIndex = 0; socketIndex < descriptor->sockets.size(); ++socketIndex) {
                const auto& socket = descriptor->sockets[socketIndex];
                if (socket.direction != SocketDirection::Input) continue;
                const bool isParameter = std::ranges::any_of(descriptor->parameters,
                    [&](const ParameterDescriptor& parameter) { return parameter.key == socket.key; });
                if (!isParameter)
                    renderSocketPin(node.id, *descriptor, socketIndex, pins);
            }
            for (const auto& property : descriptor->parameters) {
                if (node.type == "color_ramp" &&
                    (property.key == "r0" || property.key == "g0" || property.key == "b0" ||
                     property.key == "r1" || property.key == "g1" || property.key == "b1")) {
                    if (property.key == "r0" && renderColorRampColors(node.id, node, connectedInputs))
                        dirty_ = true;
                    continue;
                }
                const bool connected = connectedInputs.contains(
                    std::to_string(node.id) + "\x1f" + property.key);
                if (renderParameterRow(node.id, *descriptor, property, node, nodePopup_, pins, connected))
                    dirty_ = true;
            }
            ImGui::EndGroup();
            ImGui::SameLine();
            ImGui::BeginGroup();
            if (node.type == "convolution" &&
                node_widgets::renderConvolutionEditor(node, nodePopup_)) dirty_ = true;
            if (node.type == "table" && node_widgets::renderTableEditor(node)) dirty_ = true;
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
            if (fusion) {
                if (fusion->compileFailed)
                    ImGui::TextColored(ImVec4(1.0F, .45F, .25F, 1.0F),
                                       "Type or generated shader error");
                else if (fusion->mode == GeneratedExecutionMode::LegacyFallback)
                    ImGui::TextColored(ImVec4(1.0F, .65F, .25F, 1.0F),
                                       "Input unavailable");
                else if (fusion->nodeCount == 1)
                    ImGui::TextColored(ImVec4(.45F, .8F, 1.0F, 1.0F), "Generated shader");
                else if (fusion->interior)
                    ImGui::TextColored(ImVec4(.45F, .8F, 1.0F, 1.0F), "Fused → node #%llu",
                        static_cast<unsigned long long>(fusion->tail));
                else
                    ImGui::TextColored(ImVec4(.45F, .8F, 1.0F, 1.0F),
                                       "Fused region · %zu Math nodes", fusion->nodeCount);
                if (fusion->mode == GeneratedExecutionMode::Generated && fusion->interior &&
                    !fusion->materialized) {
                    ImGui::TextDisabled("Intermediate texture elided");
                    if (shaderInspectorOpen_ && ImGui::Button("Preview Intermediate"))
                        runtime_->setIntermediatePreview(node.id);
                } else if (runtime_->intermediatePreview() == node.id &&
                           ImGui::Button("Stop Intermediate Preview")) {
                    runtime_->setIntermediatePreview(std::nullopt);
                }
            }
            const auto values = runtime_->values().find(node.id);
            if (values != runtime_->values().end() && !values->second.empty()) {
                if (node.type == "float_preview") {
                    if (const auto* value = std::get_if<float>(&values->second.front())) {
                        ImGui::Text("Live value: %.6g", *value);
                    }
                } else if (const auto* image = std::get_if<ImageHandle>(&values->second.front()); image && *image) {
                    constexpr float maximumPreviewSide = 150.0F;
                    const float projectAspect = static_cast<float>(graph_.settings.width) /
                                                static_cast<float>(graph_.settings.height);
                    const ImVec2 previewSize = projectAspect >= 1.0F
                        ? ImVec2(maximumPreviewSide, maximumPreviewSide / projectAspect)
                        : ImVec2(maximumPreviewSide * projectAspect, maximumPreviewSide);
                    ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<std::intptr_t>(image->texture)),
                                 previewSize, ImVec2(0, 1), ImVec2(1, 0));
                    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                        createIntermediatePreviewWindow(node.id);
                    ImGui::TextDisabled("Double-click for Intermediate Preview");
                }
            }
            const NodeId timingNode = fusion && fusion->mode == GeneratedExecutionMode::Generated
                ? fusion->tail : node.id;
            if (const auto timing = runtime_->gpuMilliseconds().find(timingNode);
                timing != runtime_->gpuMilliseconds().end()) {
                const bool evaluated = runtime_->wasEvaluated(timingNode);
                if (fusion && fusion->mode == GeneratedExecutionMode::Generated)
                    ImGui::TextDisabled(evaluated ? "Group GPU %.3f ms"
                                                  : "Group GPU %.3f ms (cached)",
                                        timing->second);
                else
                    ImGui::TextDisabled(evaluated ? "GPU %.3f ms" : "GPU %.3f ms (cached)",
                                        timing->second);
            }
            ImGui::EndGroup();
            ImGui::SameLine();
            ImGui::BeginGroup();
            for (std::size_t socketIndex = 0; socketIndex < descriptor->sockets.size(); ++socketIndex) {
                if (descriptor->sockets[socketIndex].direction == SocketDirection::Output)
                    renderSocketPin(node.id, *descriptor, socketIndex, pins);
            }
            ImGui::EndGroup();
        }
        ImGui::PopID();
        ed::EndNode();
        if (failedGeneratedRegion || selectedFusedRegion) {
            ed::PopStyleVar();
            ed::PopStyleColor(2);
        }
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
    if (renderColorRampPickerPopup(graph_)) dirty_ = true;
    if (node_widgets::renderPopup(nodePopup_, graph_)) {
        if (pruneInactiveLinks(graph_, registry_))
            setStatus("Removed links incompatible with the selected operation");
        dirty_ = true;
        rebuildRuntime();
    }
    if (ed::ShowBackgroundContextMenu()) {
        addNodeCanvasPosition = ed::ScreenToCanvas(ImGui::GetMousePos());
        ImGui::OpenPopup("Add node");
    }
    if (ImGui::BeginPopup("Add node")) {
        static char search[96]{};
        ImGui::InputTextWithHint("##search", "Search nodes", search, sizeof(search));
        const auto entries = node_search::buildRootEntries(registry_, graph_);
        if (const auto* entry = node_search::renderMenu(entries, search)) {
            const auto id = graph_.addNode(entry->type,
                                           {addNodeCanvasPosition.x, addNodeCanvasPosition.y});
            auto* record = graph_.findNode(id);
            record->parameters = entry->parameters;
            if (entry->type == "subgraph") {
                const auto& definition = *entry->subgraph;
                record->subgraphId = entry->subgraphId;
                record->typeVersion = entry->subgraphVersion;
                for (const auto& item : definition.interface)
                    if (item.kind == SubgraphInterfaceKind::Slider)
                        record->parameters[item.key] = item.defaultValue;
                // A subgraph added from the menu is an editable project asset. The
                // immutable built-in remains only as the pristine source template.
                if (definition.immutable) duplicateSubgraph(*record);
            } else if (entry->type == "output") {
                graph_.activeOutput = id;
            }
            positioned_[id] = false;
            dirty_ = true;
            rebuildRuntime();
        }
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

void Application::renderShaderInspector() {
    if (!shaderInspectorOpen_) {
        runtime_->setIntermediatePreview(std::nullopt);
        shaderInspectorNeedsFocus_ = true;
        return;
    }
    auto shaders = runtime_->generatedShaders();
    ImGui::SetNextWindowSize(ImVec2(980, 680), ImGuiCond_FirstUseEver);
    if (shaderInspectorNeedsFocus_) {
        // Focus once when opened. Re-focusing every frame clears an active widget
        // while a selectable row is being clicked.
        ImGui::SetNextWindowFocus();
        shaderInspectorNeedsFocus_ = false;
    }
    if (!ImGui::Begin("Shader Inspector", &shaderInspectorOpen_)) {
        ImGui::End();
        return;
    }
    if (ImGui::Checkbox("Fuse lowered shader chains", &executeFusedShaders_))
        runtime_->setFusionEnabled(executeFusedShaders_);
    ImGui::SameLine();
    ImGui::TextDisabled("Session only");
    if (shaders.empty()) {
        ImGui::TextDisabled("No lowerable nodes are available to generate.");
        ImGui::End();
        return;
    }
    if (selectedShaderRegion_ == 0 ||
        std::ranges::find(shaders, selectedShaderRegion_, &GeneratedShaderInfo::id) == shaders.end())
        selectedShaderRegion_ = shaders.front().id;

    ImGui::BeginChild("regions", ImVec2(285, 0), true);
    for (const auto& shader : shaders) {
        std::string label = shader.evaluated ? "RAN  " : "CACHED  ";
        label += "Region #" + std::to_string(shader.id) + " · " +
                 std::to_string(shader.nodes.size()) + " node" +
                 (shader.nodes.size() == 1 ? "" : "s") + " · ";
        char duration[32];
        std::snprintf(duration, sizeof(duration), "%.3f ms", shader.gpuMilliseconds);
        label += duration;
        if (shader.mode == GeneratedExecutionMode::LegacyFallback) label += " ⚠";
        // Keep the interactive ID and hit target entirely independent of live
        // text. Fast regions can update their duration between mouse-down/up.
        ImGui::PushID(std::to_string(shader.id).c_str());
        if (ImGui::Selectable("##region", selectedShaderRegion_ == shader.id,
                              0, ImVec2(0, ImGui::GetTextLineHeightWithSpacing())))
            selectedShaderRegion_ = shader.id;
        const ImVec2 rowMin = ImGui::GetItemRectMin();
        const ImVec2 padding = ImGui::GetStyle().FramePadding;
        const ImVec2 textPosition{rowMin.x + padding.x, rowMin.y + padding.y};
        const ImU32 textColor = ImGui::GetColorU32(shader.evaluated
            ? ImVec4(.35F, .9F, .5F, 1) : ImVec4(.45F, .75F, 1.0F, 1));
        ImGui::GetWindowDrawList()->AddText(textPosition, textColor, label.c_str());
        ImGui::PopID();
    }
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginGroup();
    const auto selected = std::ranges::find(shaders, selectedShaderRegion_,
                                            &GeneratedShaderInfo::id);
    if (selected == shaders.end()) {
        ImGui::EndGroup();
        ImGui::End();
        return;
    }
    ImGui::Text("Region #%llu · tail node #%llu",
        static_cast<unsigned long long>(selected->id),
        static_cast<unsigned long long>(selected->tail));
    if (selected->evaluated)
        ImGui::TextColored(ImVec4(.35F, .9F, .5F, 1),
                           "RAN THIS FRAME · GPU %.3f ms", selected->gpuMilliseconds);
    else
        ImGui::TextColored(ImVec4(.45F, .75F, 1.0F, 1),
                           "CACHED · last GPU dispatch %.3f ms", selected->gpuMilliseconds);
    ImGui::TextDisabled("Cached regions reuse their prior output until an input, parameter, or time changes.");
    ImGui::TextDisabled("Selected region nodes are outlined in cyan on the canvas.");
    if (!selected->diagnostic.empty())
        ImGui::TextColored(ImVec4(1, .35F, .25F, 1), "Generated shader error");
    else if (selected->mode == GeneratedExecutionMode::Generated)
        ImGui::TextColored(ImVec4(.4F, .9F, .5F, 1), "Generated execution active");
    else
        ImGui::TextColored(ImVec4(.9F, .75F, .35F, 1), "Input unavailable");

    ImGui::SeparatorText("Contributors");
    for (const auto id : selected->nodes) {
        const auto* node = graph_.findNode(id);
        const std::string label = node && !node->label.empty() ? node->label : "Math";
        ImGui::PushID(static_cast<int>(id));
        ImGui::TextColored(contributorColor(id), "● node #%llu · %s",
            static_cast<unsigned long long>(id), label.c_str());
        if (id != selected->tail) {
            ImGui::SameLine();
            if (runtime_->intermediatePreview() == id) {
                if (ImGui::SmallButton("Stop Preview"))
                    runtime_->setIntermediatePreview(std::nullopt);
            } else if (ImGui::SmallButton("Preview Intermediate")) {
                runtime_->setIntermediatePreview(id);
            }
        }
        ImGui::PopID();
    }
    if (!selected->diagnostic.empty()) {
        ImGui::SeparatorText("Compiler diagnostic");
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, .45F, .35F, 1));
        ImGui::TextWrapped("%s", selected->diagnostic.c_str());
        ImGui::PopStyleColor();
    }
    if (ImGui::Button("Copy Source")) ImGui::SetClipboardText(selected->shader.source.c_str());
    ImGui::SeparatorText("Generated GLSL");
    ImGui::BeginChild("source", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
    std::istringstream source(selected->shader.source);
    std::string line;
    std::size_t lineNumber = 1;
    while (std::getline(source, line)) {
        NodeId contributor = 0;
        for (const auto& annotation : selected->shader.annotations) {
            if (lineNumber >= annotation.firstLine && lineNumber <= annotation.lastLine) {
                contributor = annotation.contributor;
                break;
            }
        }
        ImVec4 color = contributor ? contributorColor(contributor)
                                   : ImVec4(.78F, .8F, .84F, 1.0F);
        if (selected->errorLine == lineNumber) color = ImVec4(1, .2F, .2F, 1);
        ImGui::TextColored(color, "%4zu  %s", lineNumber, line.c_str());
        ++lineNumber;
    }
    ImGui::EndChild();
    ImGui::EndGroup();
    ImGui::End();
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
    bool executionChanged = false;
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
                if (ImGui::DragFloat("Default", &item.defaultValue, .001F,
                                     item.minimum, item.maximum)) {
                    changed = true; executionChanged = true;
                }
                changed |= ImGui::DragFloat("Minimum", &item.minimum, .001F);
                changed |= ImGui::DragFloat("Maximum", &item.maximum, .001F);
                if (item.minimum > item.maximum) std::swap(item.minimum, item.maximum);
                const float clampedDefault = std::clamp(item.defaultValue, item.minimum, item.maximum);
                if (clampedDefault != item.defaultValue) {
                    item.defaultValue = clampedDefault;
                    executionChanged = true;
                }
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
    std::unordered_set<std::string> connectedInputs;
    for (const auto& link : body.links())
        connectedInputs.insert(std::to_string(link.toNode) + "\x1f" + link.toSocket);
    for (auto& node : body.nodes()) {
        if (node.needsAttention && node.type == "simulation_previous_state" &&
            std::ranges::none_of(body.links(), [&](const LinkRecord& link) {
                return link.fromNode == node.id &&
                       (link.fromSocket == "a" || link.fromSocket == "b");
            })) node.needsAttention = false;
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
            if (node.needsAttention)
                ImGui::TextColored(ImVec4(1, .35F, .35F, 1),
                    "Needs update: reconnect via Channel Split");
            ImGui::Separator();
            ImGui::BeginGroup();
            for (std::size_t socketIndex = 0; socketIndex < descriptor->sockets.size(); ++socketIndex) {
                const auto& socket = descriptor->sockets[socketIndex];
                if (socket.direction != SocketDirection::Input) continue;
                const bool isParameter = std::ranges::any_of(descriptor->parameters,
                    [&](const ParameterDescriptor& parameter) { return parameter.key == socket.key; });
                if (!isParameter)
                    renderSocketPin(node.id, *descriptor, socketIndex, pins);
            }
            for (const auto& property : descriptor->parameters) {
                if (node.type == "color_ramp" &&
                    (property.key == "r0" || property.key == "g0" || property.key == "b0" ||
                     property.key == "r1" || property.key == "g1" || property.key == "b1")) {
                    if (property.key == "r0" && renderColorRampColors(node.id, node, connectedInputs)) {
                        changed = true; executionChanged = true;
                    }
                    continue;
                }
                const bool connected = connectedInputs.contains(
                    std::to_string(node.id) + "\x1f" + property.key);
                if (renderParameterRow(node.id, *descriptor, property, node, nodePopup_, pins, connected)) {
                    changed = true; executionChanged = true;
                }
            }
            ImGui::EndGroup();
            ImGui::SameLine();
            ImGui::BeginGroup();
            for (std::size_t socketIndex = 0; socketIndex < descriptor->sockets.size(); ++socketIndex) {
                if (descriptor->sockets[socketIndex].direction == SocketDirection::Output)
                    renderSocketPin(node.id, *descriptor, socketIndex, pins);
            }
            ImGui::EndGroup();
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
                    changed = true; executionChanged = true;
                }
            }
        }
        ed::EndCreate();
    }
    if (ed::BeginDelete()) {
        ed::LinkId linkId;
        while (ed::QueryDeletedLink(&linkId)) if (ed::AcceptDeletedItem()) {
            body.removeLink(static_cast<LinkId>(linkId.Get() - (std::uintptr_t{1} << 60U)));
            changed = true; executionChanged = true;
        }
        ed::NodeId nodeId;
        while (ed::QueryDeletedNode(&nodeId)) if (ed::AcceptDeletedItem()) {
            const auto id = static_cast<NodeId>(nodeId.Get() / 128);
            body.removeNode(id);
            positionedSubgraph_.erase(id);
            changed = true; executionChanged = true;
        }
        ed::EndDelete();
    }

    static ImVec2 addNodeCanvasPosition{};
    ed::Suspend();
    if (renderColorRampPickerPopup(body)) {
        changed = true;
        executionChanged = true;
    }
    if (node_widgets::renderPopup(nodePopup_, body)) {
        changed = true; executionChanged = true;
    }
    if (ed::ShowBackgroundContextMenu()) {
        addNodeCanvasPosition = ed::ScreenToCanvas(ImGui::GetMousePos());
        ImGui::OpenPopup("Add subgraph node");
    }
    if (ImGui::BeginPopup("Add subgraph node")) {
        static char search[96]{};
        ImGui::InputTextWithHint("##search", "Search nodes", search, sizeof(search));
        const auto entries = node_search::buildSubgraphEditorEntries(registry_, *definition);
        if (const auto* entry = node_search::renderMenu(entries, search)) {
            const auto id = body.addNode(entry->type,
                {addNodeCanvasPosition.x, addNodeCanvasPosition.y});
            auto* node = body.findNode(id);
            node->parameters = entry->parameters;
            positionedSubgraph_[id] = false;
            changed = true;
            executionChanged = true;
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
        dirty_ = true;
        if (executionChanged) rebuildRuntime();
    }
}

void Application::renderEditor() {
    glfwMakeContextCurrent(editorWindow_);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    handleShortcuts();

    // This is deliberately a regular top-level window instead of an ImGui popup.
    // A GLFW close request can arrive while another modal (such as recovery) is
    // active, in which case a nested popup may not be presented to the user.
    // Rendering it last below guarantees that the close decision remains visible.
    const auto renderCloseConfirmation = [&] {
        if (!closeConfirmationOpen_) return;
        const auto displaySize = ImGui::GetIO().DisplaySize;
        ImGui::SetNextWindowPos(ImVec2(displaySize.x * 0.5F, displaySize.y * 0.5F),
                                ImGuiCond_Always, ImVec2(0.5F, 0.5F));
        ImGui::SetNextWindowFocus();
        ImGui::Begin("Save changes before closing?###close_confirmation", nullptr,
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoSavedSettings);
        ImGui::TextWrapped("This project has unsaved changes. Save them before closing?");
        ImGui::Spacing();
        if (ImGui::Button("Save and Quit", ImVec2(130, 0))) {
            if (saveProjectDialog(false)) {
                discardRecoverySidecar();
                closeConfirmationOpen_ = false;
                closeApproved_ = true;
                glfwSetWindowShouldClose(editorWindow_, GLFW_TRUE);
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Discard Changes", ImVec2(140, 0))) {
            discardRecoverySidecar();
            closeConfirmationOpen_ = false;
            closeApproved_ = true;
            glfwSetWindowShouldClose(editorWindow_, GLFW_TRUE);
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(80, 0))) {
            closeConfirmationOpen_ = false;
        }
        ImGui::End();
    };

    if (!recoveryCandidate_.empty() && !recoveryPromptDismissed_)
        ImGui::OpenPopup("Recover project changes?");
    if (ImGui::BeginPopupModal("Recover project changes?", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("A newer recovery copy was found beside this project:");
        ImGui::TextWrapped("%s", recoveryCandidate_.filename().string().c_str());
        ImGui::Spacing();
        ImGui::TextWrapped("Recover its unsaved graph changes?");
        if (ImGui::Button("Recover", ImVec2(120, 0))) {
            try {
                graph_ = reaction::loadProject(recoveryCandidate_, registry_);
                editingSubgraphId_.clear();
                editingSubgraphInstance_ = 0;
                positionedSubgraphId_.clear();
                positionedSubgraph_.clear();
                positioned_.clear();
                fitRootGraph_ = true;
                elapsed_ = 0;
                updatePreviewAspectRatio();
                rebuildRuntime();
                recoverySnapshot_ = serializeProject(graph_).dump();
                recoveryCandidate_.clear();
                dirty_ = true;
                setStatus("Recovered unsaved graph changes");
                ImGui::CloseCurrentPopup();
            } catch (const std::exception& error) {
                recoveryPromptDismissed_ = true;
                setStatus(std::string("Cannot recover project: ") + error.what(), true);
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Keep Current File", ImVec2(150, 0))) {
            recoveryPromptDismissed_ = true;
            setStatus("Kept current project; recovery copy remains available", false);
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

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
            ImGui::MenuItem("Shader Inspector", nullptr, &shaderInspectorOpen_);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetFrameHeight()));
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y - ImGui::GetFrameHeight()));
    ImGui::Begin("Workspace", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBringToFrontOnFocus);
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

    const auto generated = runtime_->generatedShaders();
    const auto failed = std::ranges::find_if(generated, [](const auto& shader) {
        return !shader.diagnostic.empty();
    });
    if (failed != generated.end()) {
        const std::string key = std::to_string(failed->id) + failed->diagnostic;
        if (reportedShaderError_ != key) {
            reportedShaderError_ = key;
            setStatus("Generated shader region has a lowering or compile error", true);
        }
    } else if (!reportedShaderError_.empty()) {
        reportedShaderError_.clear();
        setStatus("Generated shader regions active");
    }
    renderShaderInspector();
    renderCloseConfirmation();

    ImGui::Render();
    int widthPixels = 0, heightPixels = 0; glfwGetFramebufferSize(editorWindow_, &widthPixels, &heightPixels);
    glViewport(0, 0, widthPixels, heightPixels); glClearColor(.035F,.035F,.045F,1); glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(editorWindow_);
    updateRecoverySidecar();
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

void Application::renderIntermediatePreview() {
    if (!intermediatePreviewWindow_) return;
    if (glfwWindowShouldClose(intermediatePreviewWindow_)) {
        glfwMakeContextCurrent(intermediatePreviewWindow_);
        if (intermediatePreviewVertexArray_)
            glDeleteVertexArrays(1, &intermediatePreviewVertexArray_);
        intermediatePreviewVertexArray_ = 0;
        glfwDestroyWindow(intermediatePreviewWindow_);
        intermediatePreviewWindow_ = nullptr;
        intermediatePreviewNode_ = 0;
        glfwMakeContextCurrent(editorWindow_);
        return;
    }

    const auto values = runtime_->values().find(intermediatePreviewNode_);
    if (values == runtime_->values().end() || values->second.empty()) return;
    const auto* image = std::get_if<ImageHandle>(&values->second.front());
    if (!image || !*image) return;

    glfwMakeContextCurrent(intermediatePreviewWindow_);
    int width = 0, height = 0;
    glfwGetFramebufferSize(intermediatePreviewWindow_, &width, &height);
    gpu_->drawFullscreen(image->texture, intermediatePreviewVertexArray_, width, height);
    glfwSwapBuffers(intermediatePreviewWindow_);
    glfwMakeContextCurrent(editorWindow_);
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
        renderIntermediatePreview();
        fpsAccumulator += delta; ++fpsFrames;
        if (fpsAccumulator >= .5) { displayedFps_ = static_cast<double>(fpsFrames) / fpsAccumulator; fpsAccumulator = 0; fpsFrames = 0; }
        const double spent = std::chrono::duration<double>(Clock::now() - frameStart).count();
        if (spent < target) std::this_thread::sleep_for(std::chrono::duration<double>(target - spent));
    }
    if (closeApproved_) discardRecoverySidecar();
    return 0;
}

} // namespace reaction
