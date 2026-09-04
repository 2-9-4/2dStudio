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
    void updateRecoverySidecar();
    void exportFrameDialog();
    void startRecordingDialog();
    void stopRecording(bool reportStatus = true);
    void recordFrame();
    void updatePreviewAspectRatio();
    void rebuildRuntime();
    void renderEditor();
    void renderGraph();
    void renderSubgraphEditor();
    void renderShaderInspector();
    void autoLayoutBody(GraphBody& body, std::unordered_map<NodeId, bool>& positioned);
    void duplicateSubgraph(NodeRecord& node);
    void renderPreview();
    void handleShortcuts();
    void copySelectedNodes();
    void pasteNodes();
    void setStatus(std::string message, bool error = false);

    GLFWwindow* editorWindow_ = nullptr;
    GLFWwindow* previewWindow_ = nullptr;
    std::uint32_t previewVertexArray_ = 0;
    int recordingPipe_ = -1;
    int recordingProcess_ = -1;
    std::filesystem::path recordingPath_;
    int recordingWidth_ = 0;
    int recordingHeight_ = 0;
    std::uint64_t recordedFrames_ = 0;
    std::vector<std::uint8_t> recordingReadback_;
    std::vector<std::uint8_t> recordingPixels_;
    ax::NodeEditor::EditorContext* nodeEditor_ = nullptr;
    ax::NodeEditor::EditorContext* subgraphEditor_ = nullptr;
    NodeRegistry registry_;
    Graph graph_;
    std::unique_ptr<GpuRuntime> gpu_;
    std::unique_ptr<GraphRuntime> runtime_;
    std::filesystem::path currentPath_;
    std::string recoverySnapshot_;
    std::filesystem::path recoveryCandidate_;
    bool recoveryPromptDismissed_ = false;
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
    bool layoutRequested_ = false;
    std::vector<NodeId> pendingSelection_;
    bool fitRootGraph_ = true;
    std::string editingSubgraphId_;
    std::string positionedSubgraphId_;
    std::unordered_map<NodeId, bool> positionedSubgraph_;
    bool fitSubgraphRequested_ = false;
    NodeId editingSubgraphInstance_ = 0;
    bool shaderInspectorOpen_ = false;
    bool executeFusedShaders_ = true;
    std::uint64_t selectedShaderRegion_ = 0;
    std::string reportedShaderError_;
};

} // namespace reaction
