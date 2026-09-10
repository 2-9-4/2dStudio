#include "reaction/core/graph.hpp"
#include "reaction/core/math.hpp"
#include "reaction/core/vector_math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace reaction {
namespace {

using Control = ParameterDescriptor::Control;

const SocketDescriptor* socket(const NodeDescriptor& descriptor, std::string_view key,
                               SocketDirection direction) {
    const auto found = std::ranges::find_if(descriptor.sockets, [&](const auto& candidate) {
        return candidate.key == key && candidate.direction == direction;
    });
    return found == descriptor.sockets.end() ? nullptr : &*found;
}

const SubgraphInterfaceItem* interfaceItem(const SubgraphDefinition& definition,
                                            const NodeRecord& node) {
    if (!node.parameters.is_object()) return nullptr;
    const auto key = node.parameters.value("key", std::string{});
    const auto found = std::ranges::find(definition.interface, key, &SubgraphInterfaceItem::key);
    return found == definition.interface.end() ? nullptr : &*found;
}

const SimulationStateSlot* stateSlot(const SubgraphDefinition& definition,
                                     const NodeRecord& node) {
    if (!node.parameters.is_object() || definition.stateSlots.empty()) return nullptr;
    const int index = std::clamp(static_cast<int>(node.parameters.value("slot", 0.0F)), 0,
                                 static_cast<int>(definition.stateSlots.size()) - 1);
    return &definition.stateSlots[static_cast<std::size_t>(index)];
}

std::vector<std::string> stateSlotLabels(const SubgraphDefinition& definition) {
    std::vector<std::string> result;
    for (const auto& slot : definition.stateSlots) result.push_back(slot.label);
    return result;
}

const NodeDescriptor* intrinsicDescriptor(const SubgraphDefinition& definition,
                                          const NodeRecord& node,
                                          NodeDescriptor& storage) {
    if (node.type == "subgraph_input") {
        const auto* item = interfaceItem(definition, node);
        if (!item || item->kind == SubgraphInterfaceKind::Output) return nullptr;
        storage = {"subgraph_input", 1, item->label, "Subgraph Interface", {}, {}};
        storage.sockets.push_back({"value", "Value", item->contract, SocketDirection::Output});
        if (item->kind == SubgraphInterfaceKind::Input) {
            storage.sockets.push_back({"connected", "Connected", ValueType::Float,
                                       SocketDirection::Output});
        }
        return &storage;
    }
    if (node.type == "subgraph_output") {
        const auto* item = interfaceItem(definition, node);
        if (!item || item->kind != SubgraphInterfaceKind::Output) return nullptr;
        storage = {"subgraph_output", 1, item->label, "Subgraph Interface",
                   {{"value", "Value", item->contract, SocketDirection::Input}}, {}};
        return &storage;
    }
    if (node.type == "simulation_previous_state") {
        if (const auto* slot = stateSlot(definition, node)) {
            storage = {"simulation_previous_state", 2, "Previous State", "Simulation",
                       {{"value", "Value", slot->type, SocketDirection::Output}},
                       {{"slot", "State Slot", node.parameters.value("slot", 0.0F), 0.0F,
                         static_cast<float>(definition.stateSlots.size() - 1), Control::Enum,
                         stateSlotLabels(definition)}}};
            return &storage;
        }
        storage = {"simulation_previous_state", 1, "Previous Simulation State", "Simulation",
                   {{"state", "State", ValueType::VectorField, SocketDirection::Output}}, {}};
        return &storage;
    }
    if (node.type == "simulation_channel") {
        storage = {"simulation_channel", 1, "Simulation Channel Split", "Simulation",
                   {{"state", "State", ValueType::VectorField, SocketDirection::Input},
                    {"a", "Chemical A", ValueType::ScalarField, SocketDirection::Output},
                    {"b", "Chemical B", ValueType::ScalarField, SocketDirection::Output}}, {}};
        return &storage;
    }
    if (node.type == "simulation_iteration_info") {
        storage = {"simulation_iteration_info", 1, "Simulation Iteration Info", "Simulation",
                   {{"iterationIndex", "Iteration Index", ValueType::Float, SocketDirection::Output},
                    {"iterationCount", "Iteration Count", ValueType::Float, SocketDirection::Output},
                    {"normalizedIteration", "Normalized Iteration", ValueType::Float, SocketDirection::Output},
                    {"firstIteration", "First Iteration", ValueType::Float, SocketDirection::Output},
                    {"lastIteration", "Last Iteration", ValueType::Float, SocketDirection::Output}}, {}};
        return &storage;
    }
    if (node.type == "simulation_step_info") {
        storage = {"simulation_step_info", 1, "Simulation Step / Frame Info", "Simulation",
                   {{"deltaTime", "Delta Time", ValueType::Float, SocketDirection::Output},
                    {"simulationStep", "Simulation Step", ValueType::Float, SocketDirection::Output},
                    {"simulationTime", "Simulation Time", ValueType::Float, SocketDirection::Output},
                    {"wasReset", "Was Reset", ValueType::Float, SocketDirection::Output},
                    {"frameIndex", "Frame Index", ValueType::Float, SocketDirection::Output}}, {}};
        return &storage;
    }
    if (node.type == "simulation_initial_state") {
        if (const auto* slot = stateSlot(definition, node)) {
            storage = {"simulation_initial_state", 2, "Initial State", "Simulation",
                       {{"value", "Value", slot->type, SocketDirection::Input}},
                       {{"slot", "State Slot", node.parameters.value("slot", 0.0F), 0.0F,
                         static_cast<float>(definition.stateSlots.size() - 1), Control::Enum,
                         stateSlotLabels(definition)}}};
            return &storage;
        }
        storage = {"simulation_initial_state", 1, "Initial Simulation State", "Simulation",
                   {{"a", "Chemical A", SocketContract::Numeric, SocketDirection::Input},
                    {"b", "Chemical B", SocketContract::Numeric, SocketDirection::Input}}, {}};
        return &storage;
    }
    if (node.type == "simulation_next_state") {
        if (const auto* slot = stateSlot(definition, node)) {
            storage = {"simulation_next_state", 2, "Next State", "Simulation",
                       {{"value", "Value", slot->type, SocketDirection::Input},
                        {"value", "Value", slot->type, SocketDirection::Output}},
                       {{"slot", "State Slot", node.parameters.value("slot", 0.0F), 0.0F,
                         static_cast<float>(definition.stateSlots.size() - 1), Control::Enum,
                         stateSlotLabels(definition)}}};
            return &storage;
        }
        storage = {"simulation_next_state", 1, "Next Simulation State", "Simulation",
                   {{"a", "Chemical A", SocketContract::Numeric, SocketDirection::Input},
                    {"b", "Chemical B", SocketContract::Numeric, SocketDirection::Input},
                    {"a", "Chemical A", ValueType::ScalarField, SocketDirection::Output},
                    {"b", "Chemical B", ValueType::ScalarField, SocketDirection::Output}}, {}};
        return &storage;
    }

    return nullptr;
}

NodeId addNode(GraphBody& body, std::string type, std::string label, Vec2 position,
               nlohmann::json parameters = nlohmann::json::object()) {
    const auto id = body.addNode(std::move(type), position);
    auto* record = body.findNode(id);
    record->label = std::move(label);
    record->parameters = std::move(parameters);
    return id;
}

void link(GraphBody& body, NodeId from, std::string fromSocket,
          NodeId to, std::string toSocket) {
    body.addLink(from, std::move(fromSocket), to, std::move(toSocket));
}

NodeId math(GraphBody& body, std::string label, int operation, Vec2 position,
            nlohmann::json parameters = nlohmann::json::object()) {
    parameters["operation"] = static_cast<float>(operation);
    return addNode(body, "math", std::move(label), position, std::move(parameters));
}

NodeId input(GraphBody& body, std::string key, std::string label, Vec2 position) {
    return addNode(body, "subgraph_input", std::move(label), position,
                   {{"key", std::move(key)}});
}

NodeId vector(GraphBody& body, std::string label, Vec2 position, float x, float y) {
    return addNode(body, "vector", std::move(label), position, {{"x", x}, {"y", y}});
}

NodeId vectorMath(GraphBody& body, std::string label, VectorMathOperation operation,
                  Vec2 position) {
    return addNode(body, "vector_math", std::move(label), position,
                   {{"operation", static_cast<float>(operation)}});
}

SubgraphDefinition discreteReaction() {
    SubgraphDefinition result;
    result.id = "builtin.reaction_diffusion.discrete";
    result.name = "Reaction Diffusion (Discrete)";
    result.category = "Simulation";
    result.execution = SubgraphExecution::Simulation;
    result.immutable = true;
    result.stateSlots = {{"chemicals", "Chemicals", ValueType::VectorField}};
    result.interface = {
        {"feedMultiplier", "Feed Multiplier", SubgraphInterfaceKind::Input,
         SocketContract::Numeric, true, 1.0F},
        {"killMultiplier", "Kill Multiplier", SubgraphInterfaceKind::Input,
         SocketContract::Numeric, true, 1.0F},
        {"seed", "Seed", SubgraphInterfaceKind::Input, ValueType::ScalarField, true},
        {"feed", "Feed", SubgraphInterfaceKind::Input, ValueType::Float, true,
         .055F, 0.0F, .1F},
        {"kill", "Kill", SubgraphInterfaceKind::Input, ValueType::Float, true,
         .062F, 0.0F, .1F},
        {"diffA", "Diffusion A", SubgraphInterfaceKind::Input, ValueType::Float, true,
         1.0F, 0.0F, 2.0F},
        {"diffB", "Diffusion B", SubgraphInterfaceKind::Input, ValueType::Float, true,
         .5F, 0.0F, 2.0F},
        {"structureScale", "Structure Scale", SubgraphInterfaceKind::Input,
         ValueType::Float, true, 1.0F, .25F, 8.0F},
        {"dt", "Timestep", SubgraphInterfaceKind::Input, ValueType::Float, true,
         1.0F, .01F, 2.0F},
        {"iterations", "Iterations", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 8.0F, 1.0F, 64.0F, Control::Integer, "iterations"},
        {"autoReset", "Auto Reset", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.0F, 0.0F, 1.0F, Control::Boolean, "autoReset"},
        {"image", "Image", SubgraphInterfaceKind::Output, ValueType::ScalarField},
        {"a", "Chemical A", SubgraphInterfaceKind::Output, ValueType::ScalarField},
        {"b", "Chemical B", SubgraphInterfaceKind::Output, ValueType::ScalarField},
    };

    auto& body = result.body;
    const auto coordinates = addNode(body, "coordinates", "Canvas Coordinates", {0, 20});
    const auto center = vector(body, "Seed Center", {200, 20}, .5F, .5F);
    const auto radius = vectorMath(body, "Distance from Seed Center", VectorMathOperation::Distance,
                                   {420, 20});
    link(body, coordinates, "coordinates", radius, "a");
    link(body, center, "value", radius, "b");
    const auto outsideSeed = addNode(body, "threshold", "Outside Seed Circle", {640, 20},
                                     {{"threshold", .075F}});
    link(body, radius, "result", outsideSeed, "value");
    const auto defaultSeed = math(body, "Default Circular Seed", 1, {1200, 50}, {{"a", 1.0F}});
    link(body, outsideSeed, "result", defaultSeed, "b");

    const auto seed = input(body, "seed", "Seed Input", {800, 190});
    const auto seedMissing = math(body, "Seed Input Not Connected", 1, {1000, 190}, {{"a", 1.0F}});
    link(body, seed, "connected", seedMissing, "b");
    const auto defaultSeedPart = math(body, "Default Seed Contribution", 2, {1400, 30});
    link(body, defaultSeed, "result", defaultSeedPart, "a");
    link(body, seedMissing, "result", defaultSeedPart, "b");
    const auto suppliedSeedPart = math(body, "Supplied Seed Contribution", 2, {1200, 200});
    link(body, seed, "value", suppliedSeedPart, "a");
    link(body, seed, "connected", suppliedSeedPart, "b");
    const auto initialSeed = math(body, "Selected Initial Seed", 0, {1600, 110});
    link(body, defaultSeedPart, "result", initialSeed, "a");
    link(body, suppliedSeedPart, "result", initialSeed, "b");
    const auto halfSeed = math(body, "Half-strength Seed", 2, {1800, 80}, {{"b", .5F}});
    link(body, initialSeed, "result", halfSeed, "a");
    const auto initialA = math(body, "Initial Chemical A", 1, {2000, 40}, {{"a", 1.0F}});
    link(body, halfSeed, "result", initialA, "b");
    const auto initialState = addNode(body, "simulation_initial_state", "Initial Simulation State",
                                      {2420, 100}, {{"slot", 0.0F}});
    const auto initialChemicals = addNode(body, "combine_vector", "Initial Chemicals", {2200, 100});
    link(body, initialA, "result", initialChemicals, "x");
    link(body, initialSeed, "result", initialChemicals, "y");
    link(body, initialChemicals, "value", initialState, "value");

    const auto previous = addNode(body, "simulation_previous_state", "Previous Simulation State",
                                  {0, 460}, {{"slot", 0.0F}});
    const auto channels = addNode(body, "simulation_channel", "Previous State Channels",
                                  {220, 660});
    link(body, previous, "value", channels, "state");
    const auto scale = input(body, "structureScale", "Structure Scale", {0, 720});
    const auto laplacian = addNode(body, "laplacian", "State Laplacian", {220, 400});
    const auto laplacianChannels = addNode(body, "simulation_channel", "Laplacian Channels",
                                           {440, 460});
    link(body, previous, "value", laplacian, "value");
    link(body, scale, "value", laplacian, "scale");
    link(body, laplacian, "result", laplacianChannels, "state");

    const auto bSquared = math(body, "Chemical B Squared", 2, {220, 760});
    link(body, channels, "b", bSquared, "a");
    link(body, channels, "b", bSquared, "b");
    const auto reaction = math(body, "Reaction Rate", 2, {440, 720});
    link(body, channels, "a", reaction, "a");
    link(body, bSquared, "result", reaction, "b");

    const auto feedMultiplier = input(body, "feedMultiplier", "Feed Multiplier", {220, 940});
    const auto killMultiplier = input(body, "killMultiplier", "Kill Multiplier", {220, 1040});
    const auto feedValue = input(body, "feed", "Feed Rate", {440, 940});
    const auto killValue = input(body, "kill", "Kill Rate", {440, 1040});
    const auto feed = math(body, "Effective Feed Rate", 2, {660, 940});
    const auto kill = math(body, "Effective Kill Rate", 2, {660, 1040});
    link(body, feedValue, "value", feed, "a");
    link(body, feedMultiplier, "value", feed, "b");
    link(body, killValue, "value", kill, "a");
    link(body, killMultiplier, "value", kill, "b");

    const auto diffusionAValue = input(body, "diffA", "Chemical A Diffusion Rate", {440, 400});
    const auto diffusionBValue = input(body, "diffB", "Chemical B Diffusion Rate", {440, 560});
    const auto diffusionA = math(body, "Chemical A Diffusion", 2, {660, 400});
    const auto diffusionB = math(body, "Chemical B Diffusion", 2, {660, 560});
    link(body, diffusionAValue, "value", diffusionA, "a");
    link(body, laplacianChannels, "a", diffusionA, "b");
    link(body, diffusionBValue, "value", diffusionB, "a");
    link(body, laplacianChannels, "b", diffusionB, "b");

    const auto availableA = math(body, "Available Chemical A", 1, {660, 700}, {{"a", 1.0F}});
    link(body, channels, "a", availableA, "b");
    const auto feedTerm = math(body, "Chemical A Feed Term", 2, {880, 700});
    link(body, feed, "result", feedTerm, "a");
    link(body, availableA, "result", feedTerm, "b");
    const auto aDiffusionMinusReaction = math(body, "Chemical A Diffusion Minus Reaction", 1,
                                              {880, 440});
    link(body, diffusionA, "result", aDiffusionMinusReaction, "a");
    link(body, reaction, "result", aDiffusionMinusReaction, "b");
    const auto aDelta = math(body, "Chemical A Change", 0, {1100, 520});
    link(body, aDiffusionMinusReaction, "result", aDelta, "a");
    link(body, feedTerm, "result", aDelta, "b");

    const auto feedPlusKill = math(body, "Feed Rate Plus Kill Rate", 0, {880, 980});
    link(body, feed, "result", feedPlusKill, "a");
    link(body, kill, "result", feedPlusKill, "b");
    const auto decay = math(body, "Chemical B Decay", 2, {1100, 940});
    link(body, feedPlusKill, "result", decay, "a");
    link(body, channels, "b", decay, "b");
    const auto bDiffusionPlusReaction = math(body, "Chemical B Diffusion Plus Reaction", 0,
                                             {880, 580});
    link(body, diffusionB, "result", bDiffusionPlusReaction, "a");
    link(body, reaction, "result", bDiffusionPlusReaction, "b");
    const auto bDelta = math(body, "Chemical B Change", 1, {1100, 660});
    link(body, bDiffusionPlusReaction, "result", bDelta, "a");
    link(body, decay, "result", bDelta, "b");

    const auto timestep = input(body, "dt", "Timestep", {1100, 1080});
    const auto aStep = math(body, "Timestep-scaled Chemical A Change", 2, {1320, 500});
    const auto bStep = math(body, "Timestep-scaled Chemical B Change", 2, {1320, 700});
    link(body, aDelta, "result", aStep, "a");
    link(body, timestep, "value", aStep, "b");
    link(body, bDelta, "result", bStep, "a");
    link(body, timestep, "value", bStep, "b");
    const auto aNextUnclamped = math(body, "Unclamped Next Chemical A", 0, {1540, 480});
    const auto bNextUnclamped = math(body, "Unclamped Next Chemical B", 0, {1540, 700});
    link(body, channels, "a", aNextUnclamped, "a");
    link(body, aStep, "result", aNextUnclamped, "b");
    link(body, channels, "b", bNextUnclamped, "a");
    link(body, bStep, "result", bNextUnclamped, "b");
    const auto aNext = math(body, "Next Chemical A", 10, {1760, 480}, {{"b", 0.0F}, {"c", 1.0F}});
    const auto bNext = math(body, "Next Chemical B", 10, {1760, 700}, {{"b", 0.0F}, {"c", 1.0F}});
    link(body, aNextUnclamped, "result", aNext, "a");
    link(body, bNextUnclamped, "result", bNext, "a");
    const auto nextChemicals = addNode(body, "combine_vector", "Next Chemicals", {1980, 580});
    link(body, aNext, "result", nextChemicals, "x");
    link(body, bNext, "result", nextChemicals, "y");
    const auto nextState = addNode(body, "simulation_next_state", "Next Simulation State",
                                   {2200, 580}, {{"slot", 0.0F}});
    link(body, nextChemicals, "value", nextState, "value");
    const auto nextChannels = addNode(body, "simulation_channel", "Next State Channels", {2420, 580});
    link(body, nextState, "value", nextChannels, "state");

    const auto outputImage = addNode(body, "subgraph_output", "Image Output", {2200, 500}, {{"key", "image"}});
    const auto outputA = addNode(body, "subgraph_output", "Chemical A Output", {2200, 620}, {{"key", "a"}});
    const auto outputB = addNode(body, "subgraph_output", "Chemical B Output", {2200, 740}, {{"key", "b"}});
    link(body, nextChannels, "b", outputImage, "value");
    link(body, nextChannels, "a", outputA, "value");
    link(body, nextChannels, "b", outputB, "value");

    return result;
}

SubgraphDefinition genericCellularAutomata() {
    SubgraphDefinition result;
    result.id = "builtin.cellular_automata.generic";
    result.name = "Generic Cellular Automata";
    result.category = "Simulation";
    result.execution = SubgraphExecution::Simulation;
    result.immutable = true;
    result.stateSlots = {{"state", "State", ValueType::ScalarField}};
    result.interface = {
        {"initialState", "Initial State", SubgraphInterfaceKind::Input,
         ValueType::ScalarField},
        {"birthMask", "Birth Mask", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 8.0F, 0.0F, 16777215.0F, Control::Integer},
        {"survivalMask", "Survival Mask", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 12.0F, 0.0F, 16777215.0F, Control::Integer},
        {"preset", "Rule Preset", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.0F, 0.0F, 7.0F, Control::Enum},
        {"reset", "Reset", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.0F, 0.0F, 1.0F, Control::Boolean},
        {"iterations", "Iterations Per Step", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 1.0F, 1.0F, 64.0F, Control::Integer, "iterations"},
        {"state", "State", SubgraphInterfaceKind::Output, ValueType::ScalarField},
    };
    result.interface[3].enumOptions = {"Custom", "Conway's Life", "HighLife", "Seeds",
                                       "Day & Night", "Maze", "Replicator",
                                       "Life without Death"};

    auto& body = result.body;
    const auto initialInput = input(body, "initialState", "Initial State Input", {0, 40});
    const auto initial = addNode(body, "simulation_initial_state", "Initial State",
                                 {260, 40}, {{"slot", 0.0F}});
    link(body, initialInput, "value", initial, "value");

    const auto previous = addNode(body, "simulation_previous_state", "Previous State",
                                  {0, 360}, {{"slot", 0.0F}});
    const auto binary = addNode(body, "threshold", "Binary Current State", {230, 360},
                                {{"threshold", 0.5F}});
    link(body, previous, "value", binary, "value");

    // Eight unit-weight taps and a zero center form the Moore-neighborhood sum.
    const auto neighbors = addNode(body, "convolution", "Moore Neighborhood Count", {480, 360},
        {{"kernelSize", 3.0F}, {"normalize", 0.0F}, {"operation", 0.0F},
         {"iterations", 1.0F}, {"kernel", nlohmann::json::array(
             {1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F})}});
    link(body, binary, "result", neighbors, "image");

    const auto birthMask = input(body, "birthMask", "Custom Birth Mask", {700, 120});
    const auto survivalMask = input(body, "survivalMask", "Custom Survival Mask", {700, 600});
    const auto preset = input(body, "preset", "Rule Preset", {700, 360});
    const auto customGate = addNode(body, "table", "Is Custom Preset", {930, 120},
        {{"values", nlohmann::json::array({1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F})}});
    const auto presetBirth = addNode(body, "table", "Preset Birth Masks", {930, 280},
        {{"values", nlohmann::json::array({8.0F, 8.0F, 72.0F, 4.0F, 460.0F, 8.0F, 170.0F, 8.0F})}});
    const auto presetSurvival = addNode(body, "table", "Preset Survival Masks", {930, 500},
        {{"values", nlohmann::json::array({12.0F, 12.0F, 12.0F, 0.0F, 472.0F, 62.0F, 170.0F, 511.0F})}});
    link(body, preset, "value", customGate, "index");
    link(body, preset, "value", presetBirth, "index");
    link(body, preset, "value", presetSurvival, "index");
    const auto selectedBirth = addNode(body, "select", "Selected Birth Mask", {1170, 220});
    const auto selectedSurvival = addNode(body, "select", "Selected Survival Mask", {1170, 500});
    link(body, customGate, "result", selectedBirth, "condition");
    link(body, birthMask, "value", selectedBirth, "ifTrue");
    link(body, presetBirth, "result", selectedBirth, "ifFalse");
    link(body, customGate, "result", selectedSurvival, "condition");
    link(body, survivalMask, "value", selectedSurvival, "ifTrue");
    link(body, presetSurvival, "result", selectedSurvival, "ifFalse");
    const auto birth = addNode(body, "bit_test", "Birth Rule", {1420, 260});
    const auto survive = addNode(body, "bit_test", "Survival Rule", {1420, 500});
    link(body, selectedBirth, "result", birth, "mask");
    link(body, neighbors, "image", birth, "bit");
    link(body, selectedSurvival, "result", survive, "mask");
    link(body, neighbors, "image", survive, "bit");

    const auto nextValue = addNode(body, "select", "Birth or Survival", {1660, 380});
    link(body, binary, "result", nextValue, "condition");
    link(body, survive, "result", nextValue, "ifTrue");
    link(body, birth, "result", nextValue, "ifFalse");
    const auto next = addNode(body, "simulation_next_state", "Next State", {1900, 380},
                              {{"slot", 0.0F}});
    link(body, nextValue, "result", next, "value");
    const auto output = addNode(body, "subgraph_output", "State Output", {2140, 380},
                                {{"key", "state"}});
    link(body, next, "value", output, "value");
    return result;
}

SubgraphDefinition floodFill() {
    SubgraphDefinition result;
    result.id = "builtin.flood_fill";
    result.name = "Flood Fill";
    result.category = "Simulation";
    result.execution = SubgraphExecution::Simulation;
    result.immutable = true;
    result.stateSlots = {{"region", "Region", ValueType::ScalarField}};
    result.interface = {
        {"passableMask", "Passable Mask", SubgraphInterfaceKind::Input,
         ValueType::ScalarField},
        {"seedMask", "Seed Mask", SubgraphInterfaceKind::Input, ValueType::ScalarField},
        {"initialState", "Initial State", SubgraphInterfaceKind::Input,
         ValueType::ScalarField, true},
        {"connectivity", "Connectivity", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.0F, 0.0F, 1.0F, Control::Enum},
        {"reset", "Reset", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.0F, 0.0F, 1.0F, Control::Boolean},
        {"region", "Region", SubgraphInterfaceKind::Output, ValueType::ScalarField},
    };
    result.interface[3].enumOptions = {"4", "8"};

    auto& body = result.body;
    const auto passable = input(body, "passableMask", "Passable Mask", {0, 40});
    const auto seed = input(body, "seedMask", "Seed Mask", {0, 160});
    const auto initialInput = input(body, "initialState", "Initial State Input", {0, 280});
    // With no explicit initial region, initialize exactly from Seed Mask.  A
    // connected Initial State is a reset-time override, still clipped below.
    const auto initialSeeds = addNode(body, "select", "Initial State or Seeds", {260, 180});
    link(body, initialInput, "connected", initialSeeds, "condition");
    link(body, initialInput, "value", initialSeeds, "ifTrue");
    link(body, seed, "value", initialSeeds, "ifFalse");
    const auto initialValue = math(body, "Initial Region", 2, {500, 180});
    link(body, passable, "value", initialValue, "a");
    link(body, initialSeeds, "result", initialValue, "b");
    const auto initial = addNode(body, "simulation_initial_state", "Initial State",
                                 {740, 180}, {{"slot", 0.0F}});
    link(body, initialValue, "result", initial, "value");

    const auto previous = addNode(body, "simulation_previous_state", "Previous Region",
                                  {0, 600}, {{"slot", 0.0F}});
    const std::array<std::pair<const char*, Vec2>, 8> offsets{{
        {"North", {0.0F, -1.0F}}, {"South", {0.0F, 1.0F}},
        {"West", {-1.0F, 0.0F}}, {"East", {1.0F, 0.0F}},
        {"Northwest", {-1.0F, -1.0F}}, {"Northeast", {1.0F, -1.0F}},
        {"Southwest", {-1.0F, 1.0F}}, {"Southeast", {1.0F, 1.0F}},
    }};
    std::array<NodeId, 8> samples{};
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        const auto offset = vector(body, std::string(offsets[index].first) + " Offset",
                                   {220, 480.0F + static_cast<float>(index) * 110.0F},
                                   offsets[index].second.x, offsets[index].second.y);
        samples[index] = addNode(body, "state_input_sample_offset",
            std::string(offsets[index].first) + " Previous State", {460,
            480.0F + static_cast<float>(index) * 110.0F},
            {{"sampling", 0.0F}, {"addressMode", 0.0F}});
        link(body, previous, "value", samples[index], "source");
        link(body, offset, "value", samples[index], "offset");
    }
    const auto maximum = [&](NodeId first, NodeId second, std::string label, Vec2 position) {
        const auto node = math(body, std::move(label), 6, position);
        link(body, first, "sampled", node, "a");
        link(body, second, "sampled", node, "b");
        return node;
    };
    const auto axialNS = maximum(samples[0], samples[1], "North/South Reached", {700, 520});
    const auto axialEW = maximum(samples[2], samples[3], "East/West Reached", {700, 680});
    const auto axial = math(body, "4-Connected Neighbors", 6, {940, 600});
    link(body, axialNS, "result", axial, "a");
    link(body, axialEW, "result", axial, "b");
    const auto diagonalNWNE = maximum(samples[4], samples[5], "North Diagonals Reached", {700, 960});
    const auto diagonalSWSE = maximum(samples[6], samples[7], "South Diagonals Reached", {700, 1120});
    const auto diagonals = math(body, "Diagonal Neighbors", 6, {940, 1040});
    link(body, diagonalNWNE, "result", diagonals, "a");
    link(body, diagonalSWSE, "result", diagonals, "b");
    const auto allNeighbors = math(body, "8-Connected Neighbors", 6, {1180, 820});
    link(body, axial, "result", allNeighbors, "a");
    link(body, diagonals, "result", allNeighbors, "b");
    const auto connectivity = input(body, "connectivity", "Connectivity", {1180, 1020});
    const auto selectedNeighbors = addNode(body, "select", "Selected Neighbors", {1420, 820});
    link(body, connectivity, "value", selectedNeighbors, "condition");
    link(body, allNeighbors, "result", selectedNeighbors, "ifTrue");
    link(body, axial, "result", selectedNeighbors, "ifFalse");
    const auto reached = math(body, "Reached Region", 6, {1660, 760});
    link(body, previous, "value", reached, "a");
    link(body, selectedNeighbors, "result", reached, "b");
    const auto nextValue = math(body, "Passable Reached Region", 2, {1900, 760});
    link(body, passable, "value", nextValue, "a");
    link(body, reached, "result", nextValue, "b");
    const auto next = addNode(body, "simulation_next_state", "Next Region", {2140, 760},
                              {{"slot", 0.0F}});
    link(body, nextValue, "result", next, "value");
    const auto output = addNode(body, "subgraph_output", "Region Output", {2380, 760},
                                {{"key", "region"}});
    link(body, next, "value", output, "value");
    return result;
}

nlohmann::json leniaKernel(int size = 15) {
    nlohmann::json values = nlohmann::json::array();
    const float radius = static_cast<float>(size / 2);
    for (int y = -size / 2; y <= size / 2; ++y) {
        for (int x = -size / 2; x <= size / 2; ++x) {
            const float distance = std::sqrt(static_cast<float>(x * x + y * y)) / radius;
            // A smooth annular kernel is the conventional compact Lenia seed.
            const float ring = distance <= 1.0F
                ? std::exp(-0.5F * std::pow((distance - 0.5F) / 0.15F, 2.0F)) : 0.0F;
            values.push_back(ring);
        }
    }
    return values;
}

SubgraphDefinition lenia() {
    SubgraphDefinition result;
    result.id = "builtin.lenia";
    result.name = "Lenia";
    result.category = "Simulation";
    result.execution = SubgraphExecution::Simulation;
    result.immutable = true;
    result.stateSlots = {{"state", "State", ValueType::ScalarField}};
    result.interface = {
        {"initialState", "Initial State", SubgraphInterfaceKind::Input, ValueType::ScalarField},
        {"kernelScale", "Kernel Radius / Scale", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 4.0F, 0.01F, 128.0F},
        {"growthCenter", "Growth Center μ", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.15F, 0.0F, 1.0F},
        {"growthWidth", "Growth Width σ", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.015F, 0.001F, 1.0F},
        {"timeStep", "Time Step", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.1F, 0.0F, 2.0F},
        {"reset", "Reset", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.0F, 0.0F, 1.0F, Control::Boolean},
        {"state", "State", SubgraphInterfaceKind::Output, ValueType::ScalarField},
    };

    auto& body = result.body;
    const auto initialInput = input(body, "initialState", "Initial State Input", {0, 120});
    const auto initial = addNode(body, "simulation_initial_state", "Initial State", {250, 120},
                                 {{"slot", 0.0F}});
    link(body, initialInput, "value", initial, "value");

    const auto previous = addNode(body, "simulation_previous_state", "Previous State", {0, 520},
                                  {{"slot", 0.0F}});
    const auto kernelScale = input(body, "kernelScale", "Kernel Radius / Scale", {240, 390});
    // Kernel Size remains an editable structural control on this Convolution
    // node; Kernel Scale is a live public input, so users can tune both.
    const auto potential = addNode(body, "convolution", "Neighborhood Potential", {480, 520},
        {{"kernelSize", 15.0F}, {"kernel", leniaKernel()}, {"normalize", 1.0F},
         {"operation", 0.0F}, {"iterations", 1.0F}, {"scale", 4.0F},
         {"preset", "Annular Ring"}});
    link(body, previous, "value", potential, "image");
    link(body, kernelScale, "value", potential, "scale");

    const auto growthCenter = input(body, "growthCenter", "Growth Center μ", {730, 320});
    const auto centered = math(body, "Potential − μ", static_cast<int>(MathOperation::Subtract),
                               {730, 520});
    link(body, potential, "image", centered, "a");
    link(body, growthCenter, "value", centered, "b");
    const auto growthWidth = input(body, "growthWidth", "Growth Width σ", {970, 390});
    const auto normalized = math(body, "(Potential − μ) / σ", static_cast<int>(MathOperation::Divide),
                                 {970, 520});
    link(body, centered, "result", normalized, "a");
    link(body, growthWidth, "value", normalized, "b");
    const auto squared = math(body, "Normalized Potential Squared", static_cast<int>(MathOperation::Power),
                              {1210, 520}, {{"b", 2.0F}});
    link(body, normalized, "result", squared, "a");
    const auto exponent = math(body, "−0.5 × Squared", static_cast<int>(MathOperation::Multiply),
                               {1450, 520}, {{"b", -0.5F}});
    link(body, squared, "result", exponent, "a");
    const auto bell = math(body, "Exp Growth Bell", static_cast<int>(MathOperation::Exp), {1690, 520});
    link(body, exponent, "result", bell, "a");
    const auto doubled = math(body, "2 × Exp Growth Bell", static_cast<int>(MathOperation::Multiply),
                              {1930, 520}, {{"b", 2.0F}});
    link(body, bell, "result", doubled, "a");
    const auto growth = math(body, "Growth", static_cast<int>(MathOperation::Subtract),
                             {2170, 520}, {{"b", 1.0F}});
    link(body, doubled, "result", growth, "a");
    const auto timeStep = input(body, "timeStep", "Time Step", {2170, 700});
    const auto change = math(body, "Time Step × Growth", static_cast<int>(MathOperation::Multiply),
                             {2410, 520});
    link(body, timeStep, "value", change, "a");
    link(body, growth, "result", change, "b");
    const auto integrated = math(body, "Integrated State", static_cast<int>(MathOperation::Add),
                                 {2650, 520});
    link(body, previous, "value", integrated, "a");
    link(body, change, "result", integrated, "b");
    const auto clamped = math(body, "Clamp State", static_cast<int>(MathOperation::Clamp),
                              {2890, 520}, {{"b", 0.0F}, {"c", 1.0F}});
    link(body, integrated, "result", clamped, "a");
    const auto next = addNode(body, "simulation_next_state", "Next State", {3130, 520},
                              {{"slot", 0.0F}});
    link(body, clamped, "result", next, "value");
    const auto output = addNode(body, "subgraph_output", "State Output", {3370, 520},
                                {{"key", "state"}});
    link(body, next, "value", output, "value");
    return result;
}

std::vector<std::string> validateSubgraphImpl(const SubgraphDefinition& definition,
                                               const NodeRegistry& registry) {
    std::vector<std::string> errors;
    if (definition.id.empty()) errors.push_back("Subgraph has an empty id");
    if (definition.name.empty()) errors.push_back("Subgraph has an empty name");
    if (definition.execution == SubgraphExecution::Pipeline)
        errors.push_back("Pipeline subgraphs are not supported in this milestone");

    std::unordered_set<std::string> interfaceKeys;
    std::unordered_map<std::string, const SubgraphInterfaceItem*> interfaceByKey;
    int outputCount = 0;
    for (const auto& item : definition.interface) {
        if (item.key.empty() || !interfaceKeys.insert(item.key).second)
            errors.push_back("Subgraph interface keys must be non-empty and unique");
        interfaceByKey[item.key] = &item;
        if (item.kind == SubgraphInterfaceKind::Input && fixedType(item.contract) == ValueType::Float &&
            (item.minimum > item.maximum || item.defaultValue < item.minimum ||
             item.defaultValue > item.maximum))
            errors.push_back("Subgraph slider '" + item.label + "' has an invalid range");
        if (item.kind == SubgraphInterfaceKind::Input &&
            item.control == ParameterDescriptor::Control::Enum &&
            (fixedType(item.contract) != ValueType::Float || item.enumOptions.empty() ||
             item.minimum != 0.0F || item.maximum != static_cast<float>(item.enumOptions.size() - 1)))
            errors.push_back("Subgraph dropdown '" + item.label + "' needs Float values and ordered labels");
        if (item.kind == SubgraphInterfaceKind::Output &&
            !std::ranges::any_of(acceptedTypes(item.contract), isFieldType))
            errors.push_back("Subgraph outputs must resolve to a field type");
        if (item.kind == SubgraphInterfaceKind::Output) ++outputCount;
    }
    if (definition.execution == SubgraphExecution::Simulation && (outputCount < 1 || outputCount > 3))
        errors.push_back("Simulation subgraphs support between one and three outputs");

    std::unordered_set<NodeId> nodeIds;
    std::unordered_map<NodeId, const NodeRecord*> nodes;
    std::unordered_map<NodeId, NodeDescriptor> descriptors;
    int previousStates = 0;
    int initialStates = 0;
    int nextStates = 0;
    std::vector<int> previousBySlot(definition.stateSlots.size());
    std::vector<int> initialBySlot(definition.stateSlots.size());
    std::vector<int> nextBySlot(definition.stateSlots.size());
    std::unordered_set<std::string> stateKeys;
    for (const auto& slot : definition.stateSlots) {
        if (slot.key.empty() || !stateKeys.insert(slot.key).second)
            errors.push_back("Simulation state slot keys must be non-empty and unique");
        if (!isFieldType(slot.type))
            errors.push_back("Simulation state slot '" + slot.label + "' must be a field type");
    }
    std::unordered_map<std::string, int> outputEndpoints;
    for (const auto& node : definition.body.nodes()) {
        if (node.id == 0 || !nodeIds.insert(node.id).second) {
            errors.push_back("Subgraph node IDs must be non-zero and unique");
            continue;
        }
        nodes[node.id] = &node;
        if (node.type == "subgraph") errors.push_back("Nested subgraphs are not supported");
        if (node.type == "simulation_previous_state") ++previousStates;
        if (node.type == "simulation_initial_state") ++initialStates;
        if (node.type == "simulation_next_state") ++nextStates;
        if (!definition.stateSlots.empty() &&
            (node.type == "simulation_previous_state" || node.type == "simulation_initial_state" ||
             node.type == "simulation_next_state")) {
            const int slot = static_cast<int>(node.parameters.value("slot", -1.0F));
            if (slot < 0 || slot >= static_cast<int>(definition.stateSlots.size())) {
                errors.push_back("Simulation state endpoint selects a missing state slot");
            } else if (node.type == "simulation_previous_state") ++previousBySlot[slot];
            else if (node.type == "simulation_initial_state") ++initialBySlot[slot];
            else ++nextBySlot[slot];
        }
        if (node.type == "subgraph_input" || node.type == "subgraph_output") {
            const auto key = node.parameters.is_object()
                ? node.parameters.value("key", std::string{}) : std::string{};
            const auto found = interfaceByKey.find(key);
            if (found == interfaceByKey.end()) {
                errors.push_back("Subgraph boundary node references missing interface item '" + key + "'");
            } else if (node.type == "subgraph_input" &&
                       found->second->kind == SubgraphInterfaceKind::Output) {
                errors.push_back("Subgraph input node references an output interface item");
            } else if (node.type == "subgraph_output" &&
                       found->second->kind != SubgraphInterfaceKind::Output) {
                errors.push_back("Subgraph output node references a non-output interface item");
            }
            if (node.type == "subgraph_output") ++outputEndpoints[key];
        }

        NodeDescriptor descriptor;
        const NodeDescriptor* resolved = resolveSubgraphBodyDescriptor(
            definition, node, registry, descriptor);
        if (!resolved) {
            errors.push_back("Unsupported subgraph node type '" + node.type + "'");
        } else {
            descriptors.emplace(node.id, *resolved);
        }
        // Simulation bodies are lowered into one update shader. Convolution's
        // multi-pass mode uses a native ping-pong texture and cannot participate
        // in that shader, so reject legacy/manual values before runtime.
        if (node.type == "convolution" && node.parameters.is_object() &&
            node.parameters.value("iterations", 1.0F) > 1.0F) {
            errors.push_back("Convolution in a simulation subgraph supports one iteration only");
        }
    }

    if (definition.execution == SubgraphExecution::Simulation) {
        if (definition.stateSlots.empty()) {
            if (previousStates != 1) errors.push_back("A simulation subgraph needs exactly one previous-state node");
            if (initialStates != 1) errors.push_back("A simulation subgraph needs exactly one initial-state endpoint");
            if (nextStates != 1) errors.push_back("A simulation subgraph needs exactly one next-state endpoint");
        } else for (std::size_t slot = 0; slot < definition.stateSlots.size(); ++slot) {
            const auto& label = definition.stateSlots[slot].label;
            if (previousBySlot[slot] != 1) errors.push_back("State '" + label + "' needs exactly one Previous State node");
            if (initialBySlot[slot] != 1) errors.push_back("State '" + label + "' needs exactly one Initial State node");
            if (nextBySlot[slot] != 1) errors.push_back("State '" + label + "' needs exactly one Next State node");
        }
    }
    for (const auto& item : definition.interface) {
        if (item.kind == SubgraphInterfaceKind::Output && outputEndpoints[item.key] != 1)
            errors.push_back("Output '" + item.label + "' needs exactly one subgraph output node");
    }

    std::unordered_set<LinkId> linkIds;
    std::unordered_set<std::string> occupiedInputs;
    std::unordered_map<NodeId, int> indegree;
    std::unordered_map<NodeId, std::vector<NodeId>> outgoing;
    const auto bodyValueType = [&](NodeId id, std::string_view outputSocket) {
        std::unordered_set<std::string> visiting;
        std::unordered_map<std::string, ValueType> memo;
        std::function<ValueType(NodeId, std::string_view)> infer =
            [&](NodeId sourceId, std::string_view sourceSocket) -> ValueType {
                const auto found = nodes.find(sourceId);
                const auto visitKey = std::to_string(sourceId) + ":" + std::string(sourceSocket);
                if (const auto known = memo.find(visitKey); known != memo.end())
                    return known->second;
                if (found == nodes.end() || !visiting.insert(visitKey).second)
                    return ValueType::Float;
                const auto descriptor = descriptors.find(sourceId);
                if (descriptor == descriptors.end()) {
                    visiting.erase(visitKey);
                    return ValueType::Float;
                }
                const auto* output = socket(descriptor->second, sourceSocket,
                                            SocketDirection::Output);
                if (!output) {
                    visiting.erase(visitKey);
                    return ValueType::Float;
                }
                const auto result = resolveOutputType(descriptor->second, *output,
                    [&](std::string_view inputKey) -> std::optional<ValueType> {
                        const auto link = std::ranges::find_if(definition.body.links(),
                            [&](const LinkRecord& candidate) {
                                return candidate.toNode == sourceId &&
                                       candidate.toSocket == inputKey;
                            });
                        if (link != definition.body.links().end())
                            return infer(link->fromNode, link->fromSocket);
                        const auto* input = socket(descriptor->second, inputKey,
                                                   SocketDirection::Input);
                        if (!input) return std::nullopt;
                        auto type = disconnectedType(input->contract);
                        if (input->fieldDefault)
                            type = fieldTypeForWidth(componentCount(type));
                        return type;
                    });
                visiting.erase(visitKey);
                memo.emplace(visitKey, result);
                return result;
            };
        return infer(id, outputSocket);
    };
    for (const auto id : nodeIds) indegree[id] = 0;
    for (const auto& linkRecord : definition.body.links()) {
        if (linkRecord.id == 0 || !linkIds.insert(linkRecord.id).second)
            errors.push_back("Subgraph link IDs must be non-zero and unique");
        const auto from = nodes.find(linkRecord.fromNode);
        const auto to = nodes.find(linkRecord.toNode);
        if (from == nodes.end() || to == nodes.end()) {
            errors.push_back("Subgraph link " + std::to_string(linkRecord.id) + " has a missing endpoint");
            continue;
        }
        const auto fromDescriptor = descriptors.find(linkRecord.fromNode);
        const auto toDescriptor = descriptors.find(linkRecord.toNode);
        if (fromDescriptor == descriptors.end() || toDescriptor == descriptors.end()) continue;
        const auto* output = socket(fromDescriptor->second, linkRecord.fromSocket, SocketDirection::Output);
        const auto* inputSocket = socket(toDescriptor->second, linkRecord.toSocket, SocketDirection::Input);
        if (!output || !inputSocket) {
            errors.push_back("Subgraph link " + std::to_string(linkRecord.id) + " names an unknown socket");
            continue;
        }
        const auto actualType = bodyValueType(linkRecord.fromNode, linkRecord.fromSocket);
        if (!contractAccepts(inputSocket->contract, actualType)) {
            errors.push_back("Subgraph link " + std::to_string(linkRecord.id) +
                             " cannot connect " + toString(actualType) + " to " +
                             toDescriptor->second.displayName + "." + inputSocket->key +
                             "; socket accepts " + toString(inputSocket->contract));
        } else {
            const auto targetType = resolveInputType(
                toDescriptor->second, inputSocket->key, actualType,
                [&](std::string_view outputKey) -> std::optional<ValueType> {
                    return bodyValueType(linkRecord.toNode, outputKey);
                });
            if (!coercionBetween(actualType, targetType))
                errors.push_back("Subgraph link " + std::to_string(linkRecord.id) +
                                 " cannot convert " + toString(actualType) + " to " +
                                 toString(targetType));
        }
        const auto inputKey = std::to_string(linkRecord.toNode) + ":" + linkRecord.toSocket;
        if (!occupiedInputs.insert(inputKey).second)
            errors.push_back("Subgraph input has more than one link: " + inputKey);
        ++indegree[linkRecord.toNode];
        outgoing[linkRecord.fromNode].push_back(linkRecord.toNode);
    }

    std::priority_queue<NodeId, std::vector<NodeId>, std::greater<>> ready;
    for (const auto& [id, degree] : indegree) if (degree == 0) ready.push(id);
    std::size_t visited = 0;
    while (!ready.empty()) {
        const auto id = ready.top();
        ready.pop();
        ++visited;
        for (const auto next : outgoing[id]) if (--indegree[next] == 0) ready.push(next);
    }
    if (visited != nodeIds.size()) errors.push_back("Subgraph contains a cycle");

    for (const auto& [id, descriptor] : descriptors) {
        for (const auto& inputSocket : descriptor.sockets) {
            if (inputSocket.direction != SocketDirection::Input || inputSocket.optional) continue;
            if (!occupiedInputs.contains(std::to_string(id) + ":" + inputSocket.key)) {
                errors.push_back("Subgraph node " + std::to_string(id) + " has unconnected input '" +
                                 inputSocket.label + "'");
            }
        }
    }

    NodeId nextStateId = 0;
    for (const auto& node : definition.body.nodes())
        if (node.type == "simulation_next_state") nextStateId = node.id;
    for (const auto& node : definition.body.nodes()) {
        if (node.type != "subgraph_output") continue;
        const auto incoming = std::ranges::find_if(definition.body.links(), [&](const auto& candidate) {
            return candidate.toNode == node.id && candidate.toSocket == "value";
        });
        if (incoming != definition.body.links().end()) {
            const auto* source = nodes.contains(incoming->fromNode) ? nodes[incoming->fromNode] : nullptr;
            const bool genericNext = source && source->type == "simulation_next_state" &&
                incoming->fromSocket == "value";
            const bool genericChannel = source && source->type == "simulation_channel" &&
                (incoming->fromSocket == "a" || incoming->fromSocket == "b") &&
                std::ranges::any_of(definition.body.links(), [&](const LinkRecord& link) {
                    return link.toNode == source->id && link.toSocket == "state" &&
                        std::ranges::any_of(definition.body.nodes(), [&](const NodeRecord& node) {
                            return node.id == link.fromNode && node.type == "simulation_next_state" &&
                                link.fromSocket == "value";
                        });
                });
            if ((!definition.stateSlots.empty() && !genericNext && !genericChannel) ||
                (definition.stateSlots.empty() && incoming->fromNode != nextStateId))
                errors.push_back("Simulation outputs must expose a Next Simulation State value");
        }
    }

    return errors;
}

} // namespace

const std::vector<SubgraphDefinition>& builtInSubgraphs() {
    static const std::vector<SubgraphDefinition> values{
        discreteReaction(), genericCellularAutomata(), floodFill(), lenia()};
    return values;
}

const SubgraphDefinition* resolveSubgraph(const Graph& graph, std::string_view id) {
    if (const auto* local = graph.findSubgraph(id)) return local;
    const auto& builtins = builtInSubgraphs();
    const auto it = std::ranges::find(builtins, id, &SubgraphDefinition::id);
    return it == builtins.end() ? nullptr : &*it;
}

NodeDescriptor describeSubgraph(const SubgraphDefinition& definition) {
    NodeDescriptor result{"subgraph", definition.version, definition.name, definition.category, {}, {}};
    for (const auto& item : definition.interface) {
        if (item.kind == SubgraphInterfaceKind::Input) {
            // A Float input is a normal socket with a local parameter value
            // while disconnected. Do not split this into a second "control"
            // interface category.
            if (fixedType(item.contract) == ValueType::Float)
                result.parameters.push_back({item.key, item.label, item.defaultValue,
                                             item.minimum, item.maximum, item.control,
                                             item.enumOptions});
            result.sockets.push_back({item.key, item.label, item.contract,
                                      SocketDirection::Input,
                                      item.optional || fixedType(item.contract) == ValueType::Float});
        } else if (item.kind == SubgraphInterfaceKind::Output) {
            result.sockets.push_back({item.key, item.label, item.contract,
                                      SocketDirection::Output});
        }
    }
    result.timeDependent = definition.execution == SubgraphExecution::Simulation;
    result.stateful = definition.execution == SubgraphExecution::Simulation;
    return result;
}

const NodeDescriptor* resolveDescriptor(const Graph& graph, const NodeRecord& node,
                                        const NodeRegistry& registry, NodeDescriptor& storage) {
    if (node.type == "vector_math") {
        storage = vectorMathDescriptor(vectorMathOperation(node.parameters.value("operation", 0.0F)));
        return &storage;
    }
    if (node.type != "subgraph") return registry.descriptor(node.type);
    const auto* definition = resolveSubgraph(graph, node.subgraphId);
    if (!definition) return nullptr;
    storage = describeSubgraph(*definition);
    return &storage;
}

bool isSubgraphBodyNodeType(std::string_view type) {
    return type == "float" || type == "vector" || type == "math" || type == "vector_math" ||
           type == "bit_test" ||
           type == "threshold" || type == "select" || type == "coordinates" ||
           type == "laplacian" || type == "subgraph_input" || type == "subgraph_output" ||
           type == "simulation_previous_state" ||
           type == "simulation_channel" ||
           type == "simulation_iteration_info" ||
           type == "simulation_step_info" ||
           type == "simulation_initial_state" ||
           type == "simulation_next_state";
}

const NodeDescriptor* resolveSubgraphBodyDescriptor(const SubgraphDefinition& definition,
                                                    const NodeRecord& node,
                                                    const NodeRegistry& registry,
                                                    NodeDescriptor& storage) {
    if (node.type == "subgraph_input" || node.type == "subgraph_output" ||
        node.type == "simulation_previous_state" ||
        node.type == "simulation_channel" ||
        node.type == "simulation_iteration_info" ||
        node.type == "simulation_step_info" ||
        node.type == "simulation_initial_state" ||
        node.type == "simulation_next_state") {
        return intrinsicDescriptor(definition, node, storage);
    }
    if (node.type == "vector_math") {
        storage = vectorMathDescriptor(vectorMathOperation(node.parameters.value("operation", 0.0F)));
        return &storage;
    }
    const auto* descriptor = registry.descriptor(node.type);
    if (!descriptor || (!descriptor->lowerable && !isSubgraphBodyNodeType(node.type)))
        return nullptr;
    if (node.type != "convolution") return descriptor;

    // The root node exposes a native multi-pass escape hatch. A simulation
    // subgraph has no native execution path, so present only its lowerable
    // single-pass form and do not offer an unusable parameter/input pin.
    storage = *descriptor;
    std::erase_if(storage.parameters, [](const ParameterDescriptor& parameter) {
        return parameter.key == "iterations";
    });
    std::erase_if(storage.sockets, [](const SocketDescriptor& socket) {
        return socket.direction == SocketDirection::Input && socket.key == "iterations";
    });
    return &storage;
}

std::vector<std::string> validateSubgraph(const SubgraphDefinition& definition,
                                          const NodeRegistry& registry) {
    return validateSubgraphImpl(definition, registry);
}

} // namespace reaction
