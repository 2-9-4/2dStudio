#include "node_widgets.hpp"
#include "reaction/core/math.hpp"
#include "convolution_presets.hpp"

#include <imgui.h>
#include <portable-file-dialogs.h>

#include <algorithm>
#include <array>
#include <filesystem>

namespace reaction::node_widgets {
namespace {

constexpr const char* popupName(PopupKind kind) {
    switch (kind) {
    case PopupKind::MathOperation: return "Math operation";
    case PopupKind::MixMode: return "Mix mode";
    case PopupKind::ConvolutionPreset: return "Convolution preset";
    case PopupKind::ParameterEnum: return "Parameter enum";
    case PopupKind::None: return "";
    }
    return "";
}

} // namespace

void renderEnumSelector(const ParameterDescriptor& parameter, NodeRecord& node,
                        PopupState& popup) {
    const int maximum = static_cast<int>(parameter.enumOptions.size()) - 1;
    const int current = std::clamp(
        static_cast<int>(node.parameters.value(parameter.key, parameter.defaultValue)),
        0, std::max(maximum, 0));
    ImGui::TextUnformatted(parameter.label.c_str());
    ImGui::SameLine();
    const char* preview = maximum >= 0
        ? parameter.enumOptions[static_cast<std::size_t>(current)].c_str() : "Select";
    ImGui::PushID(parameter.key.c_str());
    if (ImGui::Button(preview, ImVec2(150, 0)))
        popup.requestEnum(node.id, parameter.key, parameter.enumOptions);
    ImGui::PopID();
}

void renderMathOperationSelector(NodeRecord& node, PopupState& popup) {
    const int operation = std::clamp(
        static_cast<int>(node.parameters.value("operation", 0.0F)), 0, 11);
    ImGui::TextUnformatted("Operation");
    ImGui::SameLine();
    const auto name = mathOperationName(mathOperation(static_cast<float>(operation)));
    if (ImGui::Button(std::string(name).c_str(), ImVec2(150, 0))) {
        popup.request(PopupKind::MathOperation, node.id);
    }
}

void renderMixModeSelector(NodeRecord& node, PopupState& popup) {
    const int mode = std::clamp(static_cast<int>(node.parameters.value("mode", 0.0F)), 0, 9);
    const auto& names = mixModeNames();
    ImGui::TextUnformatted("Mode");
    ImGui::SameLine();
    if (ImGui::Button(names[static_cast<std::size_t>(mode)], ImVec2(150, 0))) {
        popup.request(PopupKind::MixMode, node.id);
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

bool renderImagePicker(NodeRecord& node) {
    const auto path = node.parameters.value("path", std::string{});
    const auto label = path.empty() ? std::string("Choose PNG…")
                                    : std::filesystem::path(path).filename().string();
    if (!ImGui::Button(label.c_str(), ImVec2(180, 0))) return false;

    const auto initialDirectory = path.empty()
        ? std::string{} : std::filesystem::path(path).parent_path().string();
    const auto selected = pfd::open_file("Import PNG image", initialDirectory,
                                         {"PNG image", "*.png"}, pfd::opt::none).result();
    if (selected.empty()) return false;
    node.parameters["path"] = selected.front();
    return true;
}

bool renderTableEditor(NodeRecord& node) {
    constexpr std::size_t maximumValues = 64;
    auto values = node.parameters.value("values", nlohmann::json::array());
    if (!values.is_array()) values = nlohmann::json::array();
    if (values.empty()) values.push_back(0.0F);
    while (values.size() > maximumValues) values.erase(values.end() - 1);

    bool changed = false;
    ImGui::TextUnformatted("Values");
    ImGui::SameLine();
    ImGui::TextDisabled("%zu / %zu", values.size(), maximumValues);
    for (std::size_t index = 0; index < values.size(); ++index) {
        float value = values[index].is_number() ? values[index].get<float>() : 0.0F;
        ImGui::PushID(static_cast<int>(index));
        ImGui::SetNextItemWidth(105);
        if (ImGui::DragFloat("##value", &value, 0.01F, 0.0F, 0.0F, "%.6g")) {
            values[index] = value;
            changed = true;
        }
        ImGui::SameLine();
        ImGui::Text("%zu", index);
        ImGui::SameLine();
        if (ImGui::SmallButton("Remove") && values.size() > 1) {
            values.erase(values.begin() + static_cast<nlohmann::json::difference_type>(index));
            changed = true;
            ImGui::PopID();
            break;
        }
        ImGui::PopID();
    }
    if (values.size() < maximumValues && ImGui::Button("Add value")) {
        values.push_back(values.back());
        changed = true;
    }
    if (changed || !node.parameters.contains("values")) node.parameters["values"] = std::move(values);
    return changed;
}

bool renderIntegerMaskEditor(NodeRecord& node) {
    constexpr int maximumBit = 23;
    constexpr int maximumMask = (1 << (maximumBit + 1)) - 1;
    int value = std::clamp(static_cast<int>(node.parameters.value("value", 0.0F)), 0, maximumMask);
    bool changed = false;
    ImGui::SetNextItemWidth(150);
    if (ImGui::InputInt("Decimal", &value)) {
        value = std::clamp(value, 0, maximumMask);
        changed = true;
    }
    std::string bits;
    bits.reserve(maximumBit + 1);
    for (int bit = maximumBit; bit >= 0; --bit) bits += (value & (1 << bit)) ? '1' : '0';
    ImGui::TextUnformatted(bits.c_str());
    for (int bit = maximumBit; bit >= 0; --bit) {
        ImGui::PushID(bit);
        if (ImGui::SmallButton((value & (1 << bit)) ? "1" : "0")) {
            value ^= 1 << bit;
            changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Bit %d", bit);
        ImGui::PopID();
        if (bit != 0) ImGui::SameLine(0.0F, 1.0F);
    }
    if (changed) node.parameters["value"] = static_cast<float>(value);
    return changed;
}

bool renderPopup(PopupState& popup, GraphBody& graph) {
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
                const auto name = mathOperationName(static_cast<MathOperation>(operation));
                if (ImGui::Selectable(std::string(name).c_str(),
                                      operation == current)) {
                    node->parameters["operation"] = static_cast<float>(operation);
                    changed = true;
                }
            }
        } else if (popup.kind == PopupKind::MixMode) {
            const auto& names = mixModeNames();
            const int current = std::clamp(
                static_cast<int>(node->parameters.value("mode", 0.0F)), 0, 9);
            for (int mode = 0; mode < static_cast<int>(names.size()); ++mode) {
                if (ImGui::Selectable(names[static_cast<std::size_t>(mode)], mode == current)) {
                    node->parameters["mode"] = static_cast<float>(mode);
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
        } else if (popup.kind == PopupKind::ParameterEnum) {
            const int current = std::clamp(
                static_cast<int>(node->parameters.value(popup.parameterKey, 0.0F)),
                0, std::max(static_cast<int>(popup.enumOptions.size()) - 1, 0));
            for (int index = 0; index < static_cast<int>(popup.enumOptions.size()); ++index) {
                if (ImGui::Selectable(popup.enumOptions[static_cast<std::size_t>(index)].c_str(),
                                      index == current)) {
                    node->parameters[popup.parameterKey] = static_cast<float>(index);
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
