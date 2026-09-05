#pragma once

#include "reaction/core/graph.hpp"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace reaction::node_widgets {

inline constexpr std::array<const char*, 10> kMixModeNames = {
    "Mix", "Add", "Multiply", "Screen", "Overlay", "Difference", "Darken",
    "Lighten", "Color Dodge", "Color Burn"};

[[nodiscard]] constexpr const auto& mixModeNames() { return kMixModeNames; }

enum class PopupKind {
    None,
    MathOperation,
    MixMode,
    ConvolutionPreset,
    ParameterEnum,
};

struct PopupState {
    PopupKind kind = PopupKind::None;
    NodeId node = 0;
    bool openRequested = false;
    std::string parameterKey;
    std::vector<std::string> enumOptions;

    void request(PopupKind requestedKind, NodeId requestedNode) {
        kind = requestedKind;
        node = requestedNode;
        openRequested = true;
    }

    void requestEnum(NodeId requestedNode, std::string key,
                     std::vector<std::string> options) {
        request(PopupKind::ParameterEnum, requestedNode);
        parameterKey = std::move(key);
        enumOptions = std::move(options);
    }
};

void renderMathOperationSelector(NodeRecord& node, PopupState& popup);
void renderMixModeSelector(NodeRecord& node, PopupState& popup);
void renderEnumSelector(const ParameterDescriptor& parameter, NodeRecord& node,
                        PopupState& popup);
bool renderConvolutionEditor(NodeRecord& node, PopupState& popup);
bool renderTableEditor(NodeRecord& node);
bool renderImagePicker(NodeRecord& node);

// Must be called while the node editor is suspended. Popups use ImGui screen
// coordinates and must never be rendered inside the transformed node canvas.
bool renderPopup(PopupState& popup, GraphBody& graph);

} // namespace reaction::node_widgets
