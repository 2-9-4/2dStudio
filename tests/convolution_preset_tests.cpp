#include "convolution_presets.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <numeric>

namespace reaction::convolution_presets {

TEST_CASE("convolution presets retain every supported kernel size") {
    for (int preset = 1; preset < kPresetCount; ++preset) {
        for (int size = kMinimumKernelSize; size <= kMaximumKernelSize; size += 2) {
            nlohmann::json parameters = {{"kernelSize", size}};
            apply(parameters, preset);

            INFO("preset=" << names()[static_cast<std::size_t>(preset)] << ", size=" << size);
            REQUIRE(kernelSize(parameters) == size);
            REQUIRE(current(parameters) == preset);
            REQUIRE(parameters.at("kernel").size() == static_cast<std::size_t>(size * size));
        }
    }
}

TEST_CASE("resizing regenerates the active convolution preset") {
    nlohmann::json parameters = {{"kernelSize", 3}};
    apply(parameters, 3);
    const auto small = values(parameters, 3);

    resize(parameters, 15);
    const auto large = values(parameters, 15);

    REQUIRE(kernelSize(parameters) == 15);
    REQUIRE(current(parameters) == 3);
    REQUIRE(parameters.at("preset") == "Gaussian Blur");
    REQUIRE(large.size() == 225);
    REQUIRE(large != small);
    REQUIRE(large.front() == 1.0F);
    REQUIRE(large[112] > large.front());
}

TEST_CASE("size-aware convolution kernels preserve their filter invariants") {
    for (int size = kMinimumKernelSize; size <= kMaximumKernelSize; size += 2) {
        nlohmann::json parameters = {{"kernelSize", size}};
        const auto center = static_cast<std::size_t>((size / 2) * size + size / 2);

        apply(parameters, 1);
        auto kernel = values(parameters, size);
        REQUIRE(kernel[center] == 1.0F);
        REQUIRE(std::accumulate(kernel.begin(), kernel.end(), 0.0F) == 1.0F);

        apply(parameters, 2);
        kernel = values(parameters, size);
        REQUIRE(std::accumulate(kernel.begin(), kernel.end(), 0.0F) ==
                static_cast<float>(size * size));
        REQUIRE(parameters.at("normalize") == 1.0F);

        apply(parameters, 4);
        kernel = values(parameters, size);
        REQUIRE(std::accumulate(kernel.begin(), kernel.end(), 0.0F) == 1.0F);

        apply(parameters, 5);
        kernel = values(parameters, size);
        REQUIRE(std::accumulate(kernel.begin(), kernel.end(), 0.0F) == 0.0F);

        apply(parameters, 6);
        kernel = values(parameters, size);
        REQUIRE(kernel[center] == 1.0F);
        REQUIRE(std::accumulate(kernel.begin(), kernel.end(), 0.0F) ==
                Catch::Approx(1.0F).margin(0.0001F));
    }
}

TEST_CASE("custom convolution kernels remain custom when resized") {
    nlohmann::json parameters = {
        {"kernelSize", 3}, {"preset", "Custom"},
        {"kernel", {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F}}};

    resize(parameters, 5);
    const auto kernel = values(parameters, 5);

    REQUIRE(current(parameters) == 0);
    REQUIRE(kernelSize(parameters) == 5);
    REQUIRE(kernel.size() == 25);
    REQUIRE(kernel[6] == 1.0F);
    REQUIRE(kernel[12] == 5.0F);
    REQUIRE(kernel[18] == 9.0F);
}

TEST_CASE("morphology presets select their GPU operation and retain it on resize") {
    nlohmann::json parameters = {{"kernelSize", 5}};

    apply(parameters, 7);
    REQUIRE(parameters.at("preset") == "Erosion");
    REQUIRE(parameters.at("operation") == 1.0F);
    REQUIRE(parameters.at("normalize") == 0.0F);
    const auto erosionKernel = values(parameters, 5);
    REQUIRE(std::accumulate(erosionKernel.begin(), erosionKernel.end(), 0.0F) == 13.0F);
    REQUIRE(erosionKernel.front() == 0.0F);
    REQUIRE(erosionKernel[2] == 1.0F);
    REQUIRE(erosionKernel[12] == 1.0F);
    resize(parameters, 15);
    REQUIRE(current(parameters) == 7);
    REQUIRE(parameters.at("operation") == 1.0F);

    apply(parameters, 8);
    REQUIRE(parameters.at("preset") == "Dilation");
    REQUIRE(parameters.at("operation") == 2.0F);
    REQUIRE(values(parameters, 15).size() == 225);

    apply(parameters, 0);
    REQUIRE(current(parameters) == 0);
    REQUIRE(parameters.at("operation") == 0.0F);
}

TEST_CASE("morphology footprints are circular at every supported size") {
    for (int size = kMinimumKernelSize; size <= kMaximumKernelSize; size += 2) {
        nlohmann::json parameters = {{"kernelSize", size}};
        const int radius = size / 2;

        for (const int preset : {7, 8}) {
            apply(parameters, preset);
            const auto kernel = values(parameters, size);
            for (int y = -radius; y <= radius; ++y) {
                for (int x = -radius; x <= radius; ++x) {
                    const float expected = x * x + y * y <= radius * radius ? 1.0F : 0.0F;
                    INFO("size=" << size << ", x=" << x << ", y=" << y);
                    REQUIRE(kernel[static_cast<std::size_t>((y + radius) * size + x + radius)] == expected);
                }
            }
        }
    }
}

} // namespace reaction::convolution_presets
