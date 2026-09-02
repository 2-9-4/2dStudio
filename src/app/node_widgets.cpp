#include "node_widgets.hpp"
#include "convolution_presets.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>

namespace reaction::node_widgets {
namespace {

constexpr std::array<const char*, 12> kMathOperationNames = {
    "Add", "Subtract", "Multiply", "Divide", "Power", "Minimum",
    "Maximum", "Absolute", "Sine", "Cosine", "Clamp", "Remap"};

constexpr const char* popupName(PopupKind kind) {
    switch (kind) {
    case PopupKind::MathOperation: return "Math operation";
    case PopupKind::ConvolutionPreset: return "Convolution preset";
    case PopupKind::None: return "";
    }
    return "";
}

} // namespace

void renderMathOperationSelector(NodeRecord& node, PopupState& popup) {
    const int operation = std::clamp(
        static_cast<int>(node.parameters.value("operation", 0.0F)), 0, 11);
    ImGui::TextUnformatted("Operation");
    ImGui::SameLine();
    if (ImGui::Button(kMathOperationNames[static_cast<std::size_t>(operation)], ImVec2(150, 0))) {
        popup.request(PopupKind::MathOperation, node.id);
    }
}

bool renderConvolutionEditor(NodeRecord& node, PopupState& popup) {
    auto& parameters = node.parameters;
    bool changed = false;
    const auto& presetNames = convolution_presets::names();
    const int preset = convolution_presets::current(parameters);
    ImGui::TextUnformatted("Preset");
    ImGui::SameLine();
    if (ImGui::Button(presetNames[static_cast<std::size_t>(preset)], ImVec2(150, 0))) {
        popup.request(PopupKind::ConvolutionPreset, node.id);
    }

    int size = convolution_presets::kernelSize(parameters);
    ImGui::SetNextItemWidth(150);
    if (ImGui::SliderInt("Kernel size", &size, 3, 15, "%d")) {
        size |= 1;
        convolution_presets::resize(parameters, size);
        changed = true;
    }
    ImGui::SameLine();
    ImGui::Text("%d x %d", size, size);
    auto values = convolution_presets::values(parameters, size);
    ImGui::TextUnformatted("Kernel (center is the current pixel)");
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (x > 0) ImGui::SameLine(0, 2);
            const auto index = static_cast<std::size_t>(y * size + x);
            ImGui::PushID(static_cast<int>(index));
            ImGui::SetNextItemWidth(30);
            if (ImGui::DragFloat("##weight", &values[index], 0.02F, -100.0F, 100.0F, "%.2g")) {
                convolution_presets::write(parameters, values, size);
                parameters["preset"] = "Custom";
                parameters["operation"] = 0.0F;
                changed = true;
            }
            ImGui::PopID();
        }
    }
    bool normalize = parameters.value("normalize", 0.0F) > 0.5F;
    if (ImGui::Checkbox("Normalize weights", &normalize)) {
        parameters["normalize"] = normalize ? 1.0F : 0.0F;
        changed = true;
    }
    float bias = parameters.value("bias", 0.0F);
    ImGui::SetNextItemWidth(150);
    if (ImGui::DragFloat("Bias", &bias, 0.01F, -10.0F, 10.0F, "%.4g")) {
        parameters["bias"] = bias;
        changed = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Added to every output channel after convolution");
    return changed;
}

bool renderPopup(PopupState& popup, Graph& graph) {
    if (popup.kind == PopupKind::None) return false;
    const char* name = popupName(popup.kind);
    if (popup.openRequested) {
        ImGui::OpenPopup(name);
        popup.openRequested = false;
    }

    bool changed = false;
    if (ImGui::BeginPopup(name)) {
        auto* node = graph.findNode(popup.node);
        if (!node) {
            ImGui::CloseCurrentPopup();
        } else if (popup.kind == PopupKind::MathOperation) {
            const int current = std::clamp(
                static_cast<int>(node->parameters.value("operation", 0.0F)), 0, 11);
            for (int operation = 0; operation < static_cast<int>(kMathOperationNames.size()); ++operation) {
                if (ImGui::Selectable(kMathOperationNames[static_cast<std::size_t>(operation)],
                                      operation == current)) {
                    node->parameters["operation"] = static_cast<float>(operation);
                    changed = true;
                }
            }
        } else if (popup.kind == PopupKind::ConvolutionPreset) {
            const auto& presetNames = convolution_presets::names();
            const int current = convolution_presets::current(node->parameters);
            for (int preset = 0; preset < convolution_presets::kPresetCount; ++preset) {
                if (ImGui::Selectable(presetNames[static_cast<std::size_t>(preset)],
                                      preset == current)) {
                    convolution_presets::apply(node->parameters, preset);
                    changed = true;
                }
            }
        }
        ImGui::EndPopup();
    } else if (!popup.openRequested) {
        popup.kind = PopupKind::None;
        popup.node = 0;
    }
    return changed;
}

} // namespace reaction::node_widgets
