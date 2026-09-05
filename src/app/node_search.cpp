#include "node_search.hpp"

#include "convolution_presets.hpp"
#include "node_widgets.hpp"
#include "reaction/core/math.hpp"
#include "reaction/core/vector_math.hpp"

#ifndef REACTION_NODE_SEARCH_HEADLESS
#include <imgui.h>
#endif

#include <algorithm>
#include <string>
#include <unordered_set>

namespace reaction::node_search {
namespace {

std::string lowercase(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const char character : value)
        result.push_back(character >= 'A' && character <= 'Z'
            ? static_cast<char>(character - 'A' + 'a') : character);
    return result;
}

bool whitespace(char character) {
    return character == ' ' || character == '\t' || character == '\n' ||
           character == '\r' || character == '\f' || character == '\v';
}

AddNodeEntry baseEntry(const NodeDescriptor& descriptor) {
    // An explicit {} in an aggregate initializer constructs a null JSON value,
    // rather than using AddNodeEntry::parameters' object default.  Nodes created
    // from an unconfigured entry (such as plain "Mix") must always start with a
    // parameter object, because the editor reads defaults with json::value().
    return {descriptor.type, descriptor.displayName, descriptor.category, {},
            nlohmann::json::object(), {}, 1,
            nullptr, {descriptor.displayName, descriptor.category, descriptor.type}};
}

void addAliases(std::vector<AddNodeEntry>& entries, const NodeDescriptor& descriptor) {
    entries.push_back(baseEntry(descriptor));

    const auto addAlias = [&](std::string_view hint, std::string_view parameter, int value,
                              nlohmann::json parameters = nlohmann::json::object()) {
        auto entry = baseEntry(descriptor);
        entry.hint = hint;
        parameters[parameter] = value;
        entry.parameters = std::move(parameters);
        entry.keywords.push_back(entry.hint);
        entries.push_back(std::move(entry));
    };

    if (descriptor.type == "math") {
        for (std::size_t index = 0; index < kMathOperationNames.size(); ++index)
            addAlias(kMathOperationNames[index], "operation", static_cast<int>(index));
    } else if (descriptor.type == "vector_math") {
        for (int index = 0; index <= static_cast<int>(VectorMathOperation::ProductComponents);
             ++index) {
            addAlias(vectorMathOperationName(static_cast<VectorMathOperation>(index)), "operation", index);
        }
    } else if (descriptor.type == "mix") {
        const auto& names = node_widgets::mixModeNames();
        for (std::size_t index = 0; index < names.size(); ++index)
            addAlias(names[index], "mode", static_cast<int>(index));
    } else if (descriptor.type == "convolution") {
        const auto& names = convolution_presets::names();
        for (int index = 1; index < convolution_presets::kPresetCount; ++index) {
            nlohmann::json parameters = nlohmann::json::object();
            convolution_presets::apply(parameters, index);
            auto entry = baseEntry(descriptor);
            entry.hint = names[static_cast<std::size_t>(index)];
            entry.parameters = std::move(parameters);
            entry.keywords.push_back(entry.hint);
            entries.push_back(std::move(entry));
        }
    }
}

void addSubgraph(std::vector<AddNodeEntry>& entries, const SubgraphDefinition& definition) {
    entries.push_back({"subgraph", definition.name, definition.category, {}, {}, definition.id,
                       definition.version, &definition, {definition.name, definition.category}});
}

} // namespace

std::vector<AddNodeEntry> buildRootEntries(const NodeRegistry& registry, const Graph& graph) {
    std::vector<AddNodeEntry> entries;
    for (const auto* descriptor : registry.descriptors()) addAliases(entries, *descriptor);
    for (const auto& definition : builtInSubgraphs()) addSubgraph(entries, definition);
    for (const auto& definition : graph.subgraphs()) addSubgraph(entries, definition);
    return entries;
}

std::vector<AddNodeEntry> buildSubgraphEditorEntries(
    const NodeRegistry& registry, const SubgraphDefinition& definition) {
    std::vector<AddNodeEntry> entries;
    for (const auto* type : {"float", "math", "threshold", "select", "coordinates", "laplacian"})
        if (const auto* descriptor = registry.descriptor(type)) addAliases(entries, *descriptor);

    for (const auto* type : {"simulation_previous_state", "simulation_channel",
                             "simulation_initial_state", "simulation_next_state"}) {
        NodeRecord node;
        node.type = type;
        NodeDescriptor storage;
        if (const auto* descriptor = resolveSubgraphBodyDescriptor(
                definition, node, registry, storage))
            addAliases(entries, *descriptor);
    }

    for (const auto& item : definition.interface) {
        const bool output = item.kind == SubgraphInterfaceKind::Output;
        AddNodeEntry entry;
        entry.type = output ? "subgraph_output" : "subgraph_input";
        entry.label = item.label;
        entry.category = "Subgraph Interface";
        entry.parameters = {{"key", item.key}};
        entry.keywords = {entry.label, entry.category, entry.type};
        entries.push_back(std::move(entry));
    }
    return entries;
}

bool matches(const AddNodeEntry& entry, std::string_view query) {
    const auto loweredQuery = lowercase(query);
    std::vector<std::string> keywords;
    keywords.reserve(entry.keywords.size());
    for (const auto& keyword : entry.keywords) keywords.push_back(lowercase(keyword));

    std::size_t position = 0;
    while (position < loweredQuery.size()) {
        while (position < loweredQuery.size() && whitespace(loweredQuery[position]))
            ++position;
        const auto begin = position;
        while (position < loweredQuery.size() && !whitespace(loweredQuery[position]))
            ++position;
        if (begin == position) break;
        const auto token = std::string_view(loweredQuery).substr(begin, position - begin);
        if (std::ranges::none_of(keywords, [&](const std::string& keyword) {
                return keyword.find(token) != std::string::npos;
            }))
            return false;
    }
    return true;
}

const AddNodeEntry* renderMenu(const std::vector<AddNodeEntry>& entries,
                               std::string_view query) {
#ifdef REACTION_NODE_SEARCH_HEADLESS
    (void)entries;
    (void)query;
    return nullptr;
#else
    if (!query.empty()) {
        for (const auto& entry : entries) {
            if (!matches(entry, query)) continue;
            const auto label = entry.hint.empty() ? entry.label : entry.label + " — " + entry.hint;
            ImGui::PushID(&entry);
            const bool selected = ImGui::MenuItem(label.c_str());
            ImGui::PopID();
            if (selected) return &entry;
        }
        return nullptr;
    }

    std::vector<std::string> categories;
    for (const auto& entry : entries)
        if (std::ranges::find(categories, entry.category) == categories.end())
            categories.push_back(entry.category);
    std::ranges::sort(categories);

    std::unordered_set<std::string> rendered;
    for (const auto& category : categories) {
        if (!ImGui::BeginMenu(category.c_str())) continue;
        for (const auto& entry : entries) {
            if (entry.category != category) continue;
            const auto key = entry.type + "\n" + entry.label;
            if (!rendered.insert(key).second) continue;
            ImGui::PushID(&entry);
            const bool selected = ImGui::MenuItem(entry.label.c_str());
            ImGui::PopID();
            if (selected) {
                ImGui::EndMenu();
                return &entry;
            }
        }
        ImGui::EndMenu();
    }
    return nullptr;
#endif
}

} // namespace reaction::node_search
