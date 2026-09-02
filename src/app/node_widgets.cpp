#include "node_widgets.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace reaction::node_widgets {
namespace {

constexpr std::array<const char*, 12> kMathOperationNames = {
    "Add", "Subtract", "Multiply", "Divide", "Power", "Minimum",
    "Maximum", "Absolute", "Sine", "Cosine", "Clamp", "Remap"};

constexpr std::array<const char*, 7> kKernelPresetNames = {
    "Custom", "Identity", "Box Blur", "Gaussian Blur", "Sharpen", "Edge Detect", "Emboss"};

constexpr const char* popupName(PopupKind kind) {
    switch (kind) {
    case PopupKind::MathOperation: return "Math operation";
    case PopupKind::ConvolutionPreset: return "Convolution preset";
    case PopupKind::None: return "";
    }
    return "";
}

int kernelSize(const nlohmann::json& parameters) {
    return std::clamp(static_cast<int>(parameters.value("kernelSize", 3.0F)) | 1, 3, 15);
}

std::vector<float> kernelValues(const nlohmann::json& parameters, int size) {
    std::vector<float> values(static_cast<std::size_t>(size * size), 0.0F);
    values[static_cast<std::size_t>((size / 2) * size + size / 2)] = 1.0F;
    if (const auto it = parameters.find("kernel"); it != parameters.end() && it->is_array()) {
        const auto count = std::min(values.size(), it->size());
        for (std::size_t index = 0; index < count; ++index) {
            if ((*it)[index].is_number()) values[index] = (*it)[index].get<float>();
        }
    }
    return values;
}

void writeKernel(nlohmann::json& parameters, const std::vector<float>& values, int size) {
    parameters["kernelSize"] = size;
    parameters["kernel"] = values;
}

void resizeKernel(nlohmann::json& parameters, int size) {
    const int previousSize = kernelSize(parameters);
    const auto previous = kernelValues(parameters, previousSize);
    std::vector<float> resized(static_cast<std::size_t>(size * size), 0.0F);
    const int overlap = std::min(previousSize, size) / 2;
    for (int y = -overlap; y <= overlap; ++y) {
        for (int x = -overlap; x <= overlap; ++x) {
            resized[static_cast<std::size_t>((y + size / 2) * size + x + size / 2)] =
                previous[static_cast<std::size_t>((y + previousSize / 2) * previousSize + x + previousSize / 2)];
        }
    }
    writeKernel(parameters, resized, size);
}

void applyKernelPreset(nlohmann::json& parameters, int preset) {
    constexpr std::array<std::array<float, 9>, 6> presets = {{
        {{0, 0, 0, 0, 1, 0, 0, 0, 0}},
        {{1, 1, 1, 1, 1, 1, 1, 1, 1}},
        {{1, 2, 1, 2, 4, 2, 1, 2, 1}},
        {{0, -1, 0, -1, 5, -1, 0, -1, 0}},
        {{-1, -1, -1, -1, 8, -1, -1, -1, -1}},
        {{-2, -1, 0, -1, 1, 1, 0, 1, 2}}
    }};
    if (preset < 1 || preset >= static_cast<int>(kKernelPresetNames.size())) return;
    const auto& values = presets[static_cast<std::size_t>(preset - 1)];
    writeKernel(parameters, std::vector<float>(values.begin(), values.end()), 3);
    parameters["normalize"] = (preset == 2 || preset == 3) ? 1.0F : 0.0F;
    parameters["bias"] = 0.0F;
    parameters["preset"] = kKernelPresetNames[static_cast<std::size_t>(preset)];
}

int currentPreset(const nlohmann::json& parameters) {
    const auto name = parameters.value("preset", std::string("Custom"));
    for (int index = 1; index < static_cast<int>(kKernelPresetNames.size()); ++index) {
        if (name == kKernelPresetNames[static_cast<std::size_t>(index)]) return index;
    }
    return 0;
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
    const int preset = currentPreset(parameters);
    ImGui::TextUnformatted("Preset");
    ImGui::SameLine();
    if (ImGui::Button(kKernelPresetNames[static_cast<std::size_t>(preset)], ImVec2(150, 0))) {
        popup.request(PopupKind::ConvolutionPreset, node.id);
    }

    int size = kernelSize(parameters);
    ImGui::SetNextItemWidth(150);
    if (ImGui::SliderInt("Kernel size", &size, 3, 15, "%d")) {
        size |= 1;
        resizeKernel(parameters, size);
        parameters["preset"] = "Custom";
        changed = true;
    }
    ImGui::SameLine();
    ImGui::Text("%d x %d", size, size);
    auto values = kernelValues(parameters, size);
    ImGui::TextUnformatted("Kernel (center is the current pixel)");
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (x > 0) ImGui::SameLine(0, 2);
            const auto index = static_cast<std::size_t>(y * size + x);
            ImGui::PushID(static_cast<int>(index));
            ImGui::SetNextItemWidth(30);
            if (ImGui::DragFloat("##weight", &values[index], 0.02F, -100.0F, 100.0F, "%.2g")) {
                writeKernel(parameters, values, size);
                parameters["preset"] = "Custom";
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
            const int current = currentPreset(node->parameters);
            for (int preset = 0; preset < static_cast<int>(kKernelPresetNames.size()); ++preset) {
                if (ImGui::Selectable(kKernelPresetNames[static_cast<std::size_t>(preset)],
                                      preset == current)) {
                    if (preset == 0) node->parameters["preset"] = "Custom";
                    else applyKernelPreset(node->parameters, preset);
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
