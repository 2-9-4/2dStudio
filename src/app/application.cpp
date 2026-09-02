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
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

constexpr std::array<const char*, 12> kMathOperationNames = {
    "Add", "Subtract", "Multiply", "Divide", "Power", "Minimum",
    "Maximum", "Absolute", "Sine", "Cosine", "Clamp", "Remap"};

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

    registerBuiltInNodes(registry_);
    gpu_ = std::make_unique<GpuRuntime>();
    createPreviewWindow();
    if (startupProject.empty()) newProject();
    else loadProject(startupProject);
}

Application::~Application() {
    if (editorWindow_) glfwMakeContextCurrent(editorWindow_);
    runtime_.reset();
    gpu_.reset();
    if (nodeEditor_) ed::DestroyEditor(nodeEditor_);
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
    previewWindow_ = glfwCreateWindow(900, 900, "Reaction Studio — Preview", nullptr, editorWindow_);
    if (!previewWindow_) throw std::runtime_error("Unable to create preview window");
    glfwMakeContextCurrent(previewWindow_);
    glfwSwapInterval(0);
    glGenVertexArrays(1, &previewVertexArray_);
    glfwMakeContextCurrent(editorWindow_);
}

void Application::newProject() {
    if (runtime_) runtime_->clear();
    graph_.clear();
    graph_.settings = {};
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
    if (runtime_) runtime_->clear();
    graph_ = reaction::loadProject(path, registry_);
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
    nlohmann::json nodes = nlohmann::json::array();
    for (int index = 0; index < nodeCount; ++index) {
        const auto id = static_cast<NodeId>(selectedUiIds[static_cast<std::size_t>(index)].Get() / 128);
        const auto* node = graph_.findNode(id);
        if (!node) continue;
        selected.insert(id);
        nodes.push_back({{"sourceId", id}, {"type", node->type},
                         {"typeVersion", node->typeVersion},
                         {"position", {node->position.x, node->position.y}},
                         {"parameters", node->parameters}});
    }
    if (nodes.empty()) { setStatus("No nodes selected", true); return; }
    nlohmann::json links = nlohmann::json::array();
    for (const auto& link : graph_.links()) {
        if (selected.contains(link.fromNode) && selected.contains(link.toNode)) {
            links.push_back({{"fromNode", link.fromNode}, {"fromSocket", link.fromSocket},
                             {"toNode", link.toNode}, {"toSocket", link.toSocket}});
        }
    }
    const auto payload = nlohmann::json{{"kind", "reaction.nodes"}, {"version", 1},
                                        {"nodes", std::move(nodes)}, {"links", std::move(links)}}.dump();
    ImGui::SetClipboardText(payload.c_str());
    setStatus("Copied " + std::to_string(selected.size()) + " node(s)");
}

void Application::pasteNodes() {
    const char* clipboard = ImGui::GetClipboardText();
    if (!clipboard) { setStatus("Clipboard is empty", true); return; }
    try {
        const auto payload = nlohmann::json::parse(clipboard);
        if (payload.value("kind", "") != "reaction.nodes" || payload.value("version", 0) != 1) {
            setStatus("Clipboard does not contain Reaction nodes", true);
            return;
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
            node->parameters = value.value("parameters", nlohmann::json::object());
            node->missing = !registry_.contains(node->type);
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
        const auto* descriptor = registry_.descriptor(node.type);
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
                    int op = std::clamp(static_cast<int>(value), 0, 11);
                    ImGui::TextUnformatted(property.label.c_str());
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::Button(kMathOperationNames[static_cast<std::size_t>(op)], ImVec2(150, 0))) {
                        mathOperationPopupNode_ = node.id;
                        openMathOperationPopup_ = true;
                    }
                    continue;
                } else if (property.key == "autoReset" && node.type == "reaction_diffusion") {
                    bool enabled = value > 0.5F;
                    changed = ImGui::Checkbox(property.label.c_str(), &enabled);
                    value = enabled ? 1.0F : 0.0F;
                } else if (property.key == "iterations" || property.key == "octaves" || property.key == "seed") {
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
            if (node.type == "reaction_diffusion" && ImGui::Button("Reset Simulation")) {
                runtime_->resetNode(node.id);
                setStatus("Reset reaction-diffusion node");
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
                if (const auto* image = std::get_if<ImageHandle>(&values->second.front()); image && *image) {
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
        const auto* fromDesc = from ? registry_.descriptor(from->type) : nullptr; const auto* toDesc = to ? registry_.descriptor(to->type) : nullptr;
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
    if (openMathOperationPopup_) {
        ImGui::OpenPopup("Math operation");
        openMathOperationPopup_ = false;
    }
    if (ImGui::BeginPopup("Math operation")) {
        if (auto* mathNode = graph_.findNode(mathOperationPopupNode_)) {
            const int current = std::clamp(
                static_cast<int>(mathNode->parameters.value("operation", 0.0F)), 0, 11);
            for (int operation = 0; operation < static_cast<int>(kMathOperationNames.size()); ++operation) {
                if (ImGui::Selectable(kMathOperationNames[static_cast<std::size_t>(operation)],
                                      operation == current)) {
                    mathNode->parameters["operation"] = static_cast<float>(operation);
                    dirty_ = true;
                }
            }
        } else {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
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
        ImGui::EndPopup();
    }
    ed::Resume();
    ed::End();
    ed::SetCurrentEditor(nullptr);
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
    ImGui::SameLine(); ImGui::Text("%.1f FPS", displayedFps_);
    ImGui::SameLine(); ImGui::SetNextItemWidth(80);
    if (ImGui::SliderInt("Target", &graph_.settings.targetFps, 1, 240)) dirty_ = true;
    ImGui::SameLine(); ImGui::SetNextItemWidth(90); int width = graph_.settings.width;
    if (ImGui::InputInt("W", &width, 0)) { graph_.settings.width = std::clamp(width, 16, 8192); runtime_->reset(); dirty_ = true; }
    ImGui::SameLine(); ImGui::SetNextItemWidth(90); int height = graph_.settings.height;
    if (ImGui::InputInt("H", &height, 0)) { graph_.settings.height = std::clamp(height, 16, 8192); runtime_->reset(); dirty_ = true; }
    ImGui::SameLine(); ImGui::TextColored(statusError_ ? ImVec4(1,.35F,.35F,1) : ImVec4(.55F,.8F,.55F,1), "%s%s", dirty_ ? "* " : "", status_.c_str());
    renderGraph();
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
