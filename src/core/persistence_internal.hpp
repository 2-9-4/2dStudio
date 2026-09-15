#pragma once

#include "reaction/core/graph.hpp"

namespace reaction::persistence_internal {

// Parse only the canonical current project schema. Historical documents must
// pass through migrateProjectJson() before calling this function.
[[nodiscard]] Graph deserializeCurrentProject(const nlohmann::json& document,
                                              const NodeRegistry& registry);

} // namespace reaction::persistence_internal
