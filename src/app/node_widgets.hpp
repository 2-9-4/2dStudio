#pragma once

#include "reaction/core/graph.hpp"

namespace reaction::node_widgets {

enum class PopupKind {
    None,
    MathOperation,
    ConvolutionPreset,
};

struct PopupState {
    PopupKind kind = PopupKind::None;
    NodeId node = 0;
    bool openRequested = false;

    void request(PopupKind requestedKind, NodeId requestedNode) {
        kind = requestedKind;
        node = requestedNode;
        openRequested = true;
    }
};

void renderMathOperationSelector(NodeRecord& node, PopupState& popup);
bool renderConvolutionEditor(NodeRecord& node, PopupState& popup);

// Must be called while the node editor is suspended. Popups use ImGui screen
// coordinates and must never be rendered inside the transformed node canvas.
bool renderPopup(PopupState& popup, Graph& graph);

} // namespace reaction::node_widgets
