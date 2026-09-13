#include "node_search.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <memory>

namespace reaction::node_search {
namespace {

void addDescriptor(NodeRegistry& registry, NodeDescriptor descriptor) {
    registry.add(std::move(descriptor), [] { return std::unique_ptr<NodeInstance>{}; });
}

const AddNodeEntry* findEntry(const std::vector<AddNodeEntry>& entries,
                              std::string_view type, std::string_view hint = {}) {
    const auto found = std::ranges::find_if(entries, [&](const AddNodeEntry& entry) {
        return entry.type == type && entry.hint == hint;
    });
    return found == entries.end() ? nullptr : &*found;
}

} // namespace

TEST_CASE("node search is case-insensitive and requires every query token") {
    const AddNodeEntry entry{
        "color_ramp", "Color Ramp", "Color", {}, {}, {}, 1, nullptr,
        {"Color Ramp", "Color", "color_ramp"}};

    REQUIRE(matches(entry, "COLOR ramp"));
    REQUIRE(matches(entry, "ramp col"));
    REQUIRE(matches(entry, "  color   RAMP  "));
    REQUIRE_FALSE(matches(entry, "color blur"));
}

TEST_CASE("root entries include preconfigured math aliases") {
    NodeRegistry registry;
    addDescriptor(registry, {"math", 1, "Math", "Math", {}, {}});

    const Graph graph;
    const auto entries = buildRootEntries(registry, graph);
    const auto* subtract = findEntry(entries, "math", "Subtract");

    REQUIRE(subtract != nullptr);
    REQUIRE(subtract->parameters.at("operation") == 1);
    REQUIRE(matches(*subtract, "subtract"));
}

TEST_CASE("root entries include dither mode aliases") {
    NodeRegistry registry;
    addDescriptor(registry, {"dither", 1, "Dither / Halftone", "Filter", {},
                             {{"mode", "Mode", 0.0F, 0.0F, 1.0F,
                               ParameterDescriptor::Control::Enum,
                               {"Threshold", "Halftone"}}}});

    const Graph graph;
    const auto entries = buildRootEntries(registry, graph);
    const auto* halftone = findEntry(entries, "dither", "Halftone");

    REQUIRE(halftone != nullptr);
    REQUIRE(halftone->parameters.at("mode") == 1);
    REQUIRE(matches(*halftone, "halftone"));
}

TEST_CASE("plain search entries start with an object parameter block") {
    NodeRegistry registry;
    addDescriptor(registry, {"mix", 1, "Mix", "Color", {}, {}});

    const Graph graph;
    const auto entries = buildRootEntries(registry, graph);
    const auto* mix = findEntry(entries, "mix");

    REQUIRE(mix != nullptr);
    REQUIRE(mix->parameters.is_object());
}

TEST_CASE("subgraph editor entries include simulation and keyed interface nodes") {
    NodeRegistry registry;
    SubgraphDefinition definition;
    definition.interface = {
        {"source", "Source", SubgraphInterfaceKind::Input, ValueType::ColorImage},
        {"rate", "Rate", SubgraphInterfaceKind::Input, ValueType::Float, true},
        {"result", "Result", SubgraphInterfaceKind::Output, ValueType::ColorImage},
    };

    const auto entries = buildSubgraphEditorEntries(registry, definition);
    for (const auto* type : {"simulation_previous_state", "simulation_channel",
                             "simulation_iteration_info", "simulation_step_info",
                             "simulation_initial_state", "simulation_next_state"}) {
        const auto* entry = findEntry(entries, type);
        REQUIRE(entry != nullptr);
        REQUIRE(entry->category == "Simulation");
    }

    const auto* input = findEntry(entries, "subgraph_input");
    REQUIRE(input != nullptr);
    REQUIRE(input->label == "Source");
    REQUIRE(input->parameters.at("key") == "source");

    const auto output = std::ranges::find_if(entries, [](const AddNodeEntry& entry) {
        return entry.type == "subgraph_output" && entry.label == "Result";
    });
    REQUIRE(output != entries.end());
    REQUIRE(output->parameters.at("key") == "result");
}

} // namespace reaction::node_search
