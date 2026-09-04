#pragma once

#include "reaction/core/graph.hpp"

#include <filesystem>

namespace reaction {

inline constexpr int kProjectFormatVersion = 3;

[[nodiscard]] nlohmann::json serializeProject(const Graph& graph);
[[nodiscard]] Graph deserializeProject(const nlohmann::json& document,
                                       const NodeRegistry& registry);
void saveProjectAtomic(const Graph& graph, const std::filesystem::path& path);
[[nodiscard]] Graph loadProject(const std::filesystem::path& path,
                                const NodeRegistry& registry);

} // namespace reaction
