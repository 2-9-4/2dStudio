#pragma once

#include <array>
#include <nlohmann/json_fwd.hpp>
#include <vector>

namespace reaction::convolution_presets {

constexpr int kMinimumKernelSize = 3;
// Keep this aligned with the GPU convolution specialization limit.
constexpr int kMaximumKernelSize = 101;
constexpr int kPresetCount = 10;
constexpr int kAnnularRingPreset = 9;

const std::array<const char*, kPresetCount>& names();
int kernelSize(const nlohmann::json& parameters);
int current(const nlohmann::json& parameters);
std::vector<float> values(const nlohmann::json& parameters, int size);
void write(nlohmann::json& parameters, const std::vector<float>& values, int size);

// Applies a preset at the kernel's current size. Custom keeps the current weights.
void apply(nlohmann::json& parameters, int preset);

// Regenerates an active preset at the new size, or center-resizes a custom kernel.
void resize(nlohmann::json& parameters, int size);

} // namespace reaction::convolution_presets
