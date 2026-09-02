#pragma once

#include "reaction/gpu/gpu_runtime.hpp"
#include "node_widgets.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct GLFWwindow;
namespace ax::NodeEditor { struct EditorContext; }

namespace reaction {

class Application {
public:
    explicit Application(std::filesystem::path startupProject = {});
    ~Application();
    int run();

private:
    void createPreviewWindow();
    void newProject();
    void loadProject(const std::filesystem::path& path);
    void loadProjectDialog();
    void saveProjectDialog(bool forceDialog);
    void exportFrameDialog();
    void rebuildRuntime();
    void renderEditor();
    void renderGraph();
    void renderSubgraphEditor();
    void duplicateSubgraph(NodeRecord& node);
    void renderPreview();
    void handleShortcuts();
    void copySelectedNodes();
    void pasteNodes();
    void setStatus(std::string message, bool error = false);

    GLFWwindow* editorWindow_ = nullptr;
    GLFWwindow* previewWindow_ = nullptr;
    std::uint32_t previewVertexArray_ = 0;
    ax::NodeEditor::EditorContext* nodeEditor_ = nullptr;
    ax::NodeEditor::EditorContext* subgraphEditor_ = nullptr;
    NodeRegistry registry_;
    Graph graph_;
    std::unique_ptr<GpuRuntime> gpu_;
    std::unique_ptr<GraphRuntime> runtime_;
    std::filesystem::path currentPath_;
    std::unordered_map<NodeId, bool> positioned_;
    bool playing_ = true;
    bool dirty_ = false;
    bool statusError_ = false;
    std::string status_ = "Ready";
    double elapsed_ = 0.0;
    double displayedFps_ = 0.0;
    node_widgets::PopupState nodePopup_;
    bool copyRequested_ = false;
    bool pasteRequested_ = false;
    std::vector<NodeId> pendingSelection_;
    std::string editingSubgraphId_;
    std::string positionedSubgraphId_;
};

} // namespace reaction
