#include "convolution_presets.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <string>

namespace reaction::convolution_presets {
namespace {

constexpr std::array<const char*, kPresetCount> kNames = {
    "Custom", "Identity", "Box Blur", "Gaussian Blur", "Sharpen", "Edge Detect", "Emboss",
    "Erosion", "Dilation"};

int validSize(int size) {
    return std::clamp(size | 1, kMinimumKernelSize, kMaximumKernelSize);
}

std::vector<float> generate(int preset, int size) {
    size = validSize(size);
    const int radius = size / 2;
    const auto center = static_cast<std::size_t>(radius * size + radius);
    std::vector<float> kernel(static_cast<std::size_t>(size * size), 0.0F);

    switch (preset) {
    case 1: // Identity
        kernel[center] = 1.0F;
        break;
    case 2: // Box blur
        std::ranges::fill(kernel, 1.0F);
        break;
    case 7: // Erosion disk structuring element
    case 8: // Dilation disk structuring element
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                if (x * x + y * y <= radius * radius) {
                    kernel[static_cast<std::size_t>((y + radius) * size + x + radius)] = 1.0F;
                }
            }
        }
        break;
    case 3: { // Gaussian blur: outer product of the corresponding Pascal row.
        std::vector<float> row(static_cast<std::size_t>(size), 1.0F);
        const int order = size - 1;
        if (size <= 31) {
            for (int index = 1; index < size; ++index) {
                row[static_cast<std::size_t>(index)] =
                    row[static_cast<std::size_t>(index - 1)] *
                    static_cast<float>(order - index + 1) / static_cast<float>(index);
            }
        } else {
            // Raw Pascal outer products overflow float at larger editable
            // sizes. Relative normalized binomial weights preserve the same
            // Gaussian kernel once Convolution normalizes it.
            const float peak = std::lgamma(static_cast<float>(order + 1)) -
                2.0F * std::lgamma(static_cast<float>(order / 2 + 1));
            for (int index = 0; index < size; ++index) {
                const float logCoefficient = std::lgamma(static_cast<float>(order + 1)) -
                    std::lgamma(static_cast<float>(index + 1)) -
                    std::lgamma(static_cast<float>(order - index + 1));
                row[static_cast<std::size_t>(index)] = std::exp(logCoefficient - peak);
            }
        }
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                kernel[static_cast<std::size_t>(y * size + x)] =
                    row[static_cast<std::size_t>(y)] * row[static_cast<std::size_t>(x)];
            }
        }
        break;
    }
    case 4: // Sharpen: size-aware cross Laplacian plus identity.
        for (int offset = 1; offset <= radius; ++offset) {
            kernel[static_cast<std::size_t>(radius * size + radius - offset)] = -1.0F;
            kernel[static_cast<std::size_t>(radius * size + radius + offset)] = -1.0F;
            kernel[static_cast<std::size_t>((radius - offset) * size + radius)] = -1.0F;
            kernel[static_cast<std::size_t>((radius + offset) * size + radius)] = -1.0F;
        }
        kernel[center] = static_cast<float>(1 + 4 * radius);
        break;
    case 5: // Edge detect: compare the center with the full neighborhood.
        std::ranges::fill(kernel, -1.0F);
        kernel[center] = static_cast<float>(size * size - 1);
        break;
    case 6: // Emboss: directional ramp, retaining the 3x3 preset's orientation.
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                kernel[static_cast<std::size_t>((y + radius) * size + x + radius)] =
                    static_cast<float>(x + y) / static_cast<float>(radius);
            }
        }
        kernel[center] = 1.0F;
        break;
    default:
        kernel[center] = 1.0F;
        break;
    }
    return kernel;
}

} // namespace

const std::array<const char*, kPresetCount>& names() { return kNames; }

int kernelSize(const nlohmann::json& parameters) {
    return validSize(static_cast<int>(parameters.value("kernelSize", 3.0F)));
}

int current(const nlohmann::json& parameters) {
    const auto name = parameters.value("preset", std::string("Custom"));
    for (int index = 1; index < kPresetCount; ++index) {
        if (name == kNames[static_cast<std::size_t>(index)]) return index;
    }
    return 0;
}

std::vector<float> values(const nlohmann::json& parameters, int size) {
    size = validSize(size);
    std::vector<float> result(static_cast<std::size_t>(size * size), 0.0F);
    result[static_cast<std::size_t>((size / 2) * size + size / 2)] = 1.0F;
    if (const auto it = parameters.find("kernel"); it != parameters.end() && it->is_array()) {
        const auto count = std::min(result.size(), it->size());
        for (std::size_t index = 0; index < count; ++index) {
            if ((*it)[index].is_number()) result[index] = (*it)[index].get<float>();
        }
    }
    return result;
}

void write(nlohmann::json& parameters, const std::vector<float>& kernel, int size) {
    parameters["kernelSize"] = validSize(size);
    parameters["kernel"] = kernel;
}

void apply(nlohmann::json& parameters, int preset) {
    if (preset <= 0 || preset >= kPresetCount) {
        parameters["preset"] = "Custom";
        parameters["operation"] = 0.0F;
        return;
    }
    const int size = kernelSize(parameters);
    write(parameters, generate(preset, size), size);
    parameters["normalize"] = (preset == 2 || preset == 3) ? 1.0F : 0.0F;
    parameters["bias"] = 0.0F;
    parameters["operation"] = preset == 7 ? 1.0F : preset == 8 ? 2.0F : 0.0F;
    parameters["preset"] = kNames[static_cast<std::size_t>(preset)];
}

void resize(nlohmann::json& parameters, int size) {
    size = validSize(size);
    const int preset = current(parameters);
    if (preset != 0) {
        write(parameters, generate(preset, size), size);
        return;
    }

    const int previousSize = kernelSize(parameters);
    const auto previous = values(parameters, previousSize);
    std::vector<float> resized(static_cast<std::size_t>(size * size), 0.0F);
    const int overlap = std::min(previousSize, size) / 2;
    for (int y = -overlap; y <= overlap; ++y) {
        for (int x = -overlap; x <= overlap; ++x) {
            resized[static_cast<std::size_t>((y + size / 2) * size + x + size / 2)] =
                previous[static_cast<std::size_t>((y + previousSize / 2) * previousSize + x + previousSize / 2)];
        }
    }
    write(parameters, resized, size);
}

} // namespace reaction::convolution_presets
