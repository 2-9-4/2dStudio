#pragma once

#include "reaction/core/graph.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace reaction::node_search {

struct AddNodeEntry {
    std::string type;
    std::string label;
    std::string category;
    std::string hint;
    nlohmann::json parameters = nlohmann::json::object();
    std::string subgraphId;
    int subgraphVersion = 1;
    const SubgraphDefinition* subgraph = nullptr;
    std::vector<std::string> keywords;
};

[[nodiscard]] std::vector<AddNodeEntry> buildRootEntries(const NodeRegistry& registry,
                                                         const Graph& graph);
[[nodiscard]] std::vector<AddNodeEntry> buildSubgraphEditorEntries(
    const NodeRegistry& registry, const SubgraphDefinition& definition);

[[nodiscard]] bool matches(const AddNodeEntry& entry, std::string_view query);

// The selected pointer refers to an element in entries and remains valid as long
// as the entries vector is not modified.
[[nodiscard]] const AddNodeEntry* renderMenu(const std::vector<AddNodeEntry>& entries,
                                             std::string_view query);

} // namespace reaction::node_search
