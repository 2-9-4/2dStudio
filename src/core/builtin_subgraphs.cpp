#include "reaction/core/graph.hpp"
#include "reaction/core/math.hpp"
#include "reaction/core/vector_math.hpp"

#include <array>
#include <cmath>
#include <tuple>

namespace reaction {
namespace {

using Control = ParameterDescriptor::Control;

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

SubgraphDefinition skeletonization() {
    SubgraphDefinition result;
    result.id = "builtin.skeletonization";
    result.name = "Skeletonization (Zhang-Suen)";
    result.category = "Simulation";
    result.execution = SubgraphExecution::Simulation;
    result.immutable = true;
    // The second slot persists the alternating Zhang-Suen sub-pass.  Keeping
    // it in state, rather than deriving it from the per-frame iteration index,
    // makes an odd number of iterations per step correct too.
    result.stateSlots = {
        {"image", "Binary Image", ValueType::ScalarField},
        {"phase", "Thinning Phase", ValueType::ScalarField},
    };
    result.interface = {
        {"initialState", "Initial Binary Image", SubgraphInterfaceKind::Input,
         ValueType::ScalarField},
        {"iterations", "Iterations Per Step", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 2.0F, 1.0F, 64.0F, Control::Integer, "iterations"},
        {"reset", "Reset", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.0F, 0.0F, 1.0F, Control::Boolean},
        {"skeleton", "Skeleton", SubgraphInterfaceKind::Output, ValueType::ScalarField},
    };

    auto& body = result.body;
    const auto initialInput = input(body, "initialState", "Initial Binary Image", {0, 80});
    const auto initialBinary = addNode(body, "threshold", "Binarize Initial Image", {240, 80},
                                       {{"threshold", 0.5F}});
    link(body, initialInput, "value", initialBinary, "value");
    const auto initialImage = addNode(body, "simulation_initial_state", "Initial Binary Image",
                                      {500, 80}, {{"slot", 0.0F}});
    link(body, initialBinary, "result", initialImage, "value");
    // Keep the phase spatially derived from the binary image. State endpoints
    // also admit lossless constant broadcasts, but this value should track the
    // initial image rather than become a uniform-only reset expression.
    const auto initialPhaseValue = math(body, "Initial Thinning Phase", 2, {240, 220},
                                        {{"b", 0.0F}});
    link(body, initialBinary, "result", initialPhaseValue, "a");
    const auto initialPhase = addNode(body, "simulation_initial_state", "Initial Phase",
                                      {500, 220}, {{"slot", 1.0F}});
    link(body, initialPhaseValue, "result", initialPhase, "value");

    const auto previous = addNode(body, "simulation_previous_state", "Previous Binary Image",
                                  {0, 560}, {{"slot", 0.0F}});
    const auto previousPhase = addNode(body, "simulation_previous_state", "Previous Thinning Phase",
                                       {0, 1660}, {{"slot", 1.0F}});
    const auto binary = addNode(body, "threshold", "Binary Current Pixel", {240, 560},
                                {{"threshold", 0.5F}});
    link(body, previous, "value", binary, "value");

    // P2..P9 are the clockwise 8-neighborhood used by the Zhang-Suen rules.
    const std::array<std::tuple<const char*, const char*, Vec2>, 8> neighbors{{
        {"P2", "North", {0.0F, -1.0F}}, {"P3", "Northeast", {1.0F, -1.0F}},
        {"P4", "East", {1.0F, 0.0F}}, {"P5", "Southeast", {1.0F, 1.0F}},
        {"P6", "South", {0.0F, 1.0F}}, {"P7", "Southwest", {-1.0F, 1.0F}},
        {"P8", "West", {-1.0F, 0.0F}}, {"P9", "Northwest", {-1.0F, -1.0F}},
    }};
    std::array<NodeId, 8> samples{};
    for (std::size_t index = 0; index < neighbors.size(); ++index) {
        const auto& [key, direction, offsetValue] = neighbors[index];
        const auto offset = vector(body, std::string(direction) + " Offset",
                                   {220, 700.0F + static_cast<float>(index) * 120.0F},
                                   offsetValue.x, offsetValue.y);
        samples[index] = addNode(body, "state_input_sample_offset",
            std::string(key) + " " + direction + " Neighbor",
            {460, 700.0F + static_cast<float>(index) * 120.0F},
            {{"sampling", 0.0F}, {"addressMode", 3.0F}});
        link(body, previous, "value", samples[index], "source");
        link(body, offset, "value", samples[index], "offset");
    }

    const auto add = [&](NodeId left, std::string_view leftSocket, NodeId right,
                         std::string_view rightSocket, std::string label, Vec2 position) {
        const auto node = math(body, std::move(label), static_cast<int>(MathOperation::Add), position);
        link(body, left, std::string(leftSocket), node, "a");
        link(body, right, std::string(rightSocket), node, "b");
        return node;
    };
    const auto multiply = [&](NodeId left, std::string_view leftSocket, NodeId right,
                              std::string_view rightSocket, std::string label, Vec2 position) {
        const auto node = math(body, std::move(label), static_cast<int>(MathOperation::Multiply), position);
        link(body, left, std::string(leftSocket), node, "a");
        link(body, right, std::string(rightSocket), node, "b");
        return node;
    };
    const auto inverse = [&](NodeId value, std::string_view valueSocket, std::string label, Vec2 position) {
        const auto node = math(body, std::move(label), static_cast<int>(MathOperation::Subtract),
                               position, {{"a", 1.0F}});
        link(body, value, std::string(valueSocket), node, "b");
        return node;
    };
    const auto atLeast = [&](NodeId value, std::string_view valueSocket, float limit,
                             std::string label, Vec2 position) {
        const auto node = math(body, std::move(label), static_cast<int>(MathOperation::Step),
                               position, {{"a", limit}});
        link(body, value, std::string(valueSocket), node, "b");
        return node;
    };

    NodeId neighborCount = samples[0];
    for (std::size_t index = 1; index < samples.size(); ++index)
        neighborCount = add(neighborCount, index == 1 ? "sampled" : "result", samples[index], "sampled",
                            "Neighbor Count", {700, 700.0F + static_cast<float>(index) * 95.0F});
    const auto atLeastTwo = atLeast(neighborCount, "result", 2.0F, "At Least Two Neighbors", {940, 760});
    const auto moreThanSix = atLeast(neighborCount, "result", 6.5F, "More Than Six Neighbors", {940, 900});
    const auto atMostSix = inverse(moreThanSix, "result", "At Most Six Neighbors", {1180, 900});
    const auto validNeighborCount = multiply(atLeastTwo, "result", atMostSix, "result",
                                             "Two Through Six Neighbors", {1420, 820});

    NodeId transitions = 0;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const auto next = (index + 1) % samples.size();
        const auto off = inverse(samples[index], "sampled", std::string("Not ") + std::get<0>(neighbors[index]),
                                 {700, 1780.0F + static_cast<float>(index) * 110.0F});
        const auto transition = multiply(off, "result", samples[next], "sampled",
                                         std::string("Transition ") + std::get<0>(neighbors[index]) +
                                             " to " + std::get<0>(neighbors[next]),
                                         {940, 1780.0F + static_cast<float>(index) * 110.0F});
        transitions = index == 0 ? transition : add(transitions, "result", transition, "result",
            "0-to-1 Transition Count", {1180, 1780.0F + static_cast<float>(index) * 110.0F});
    }
    const auto hasTransition = atLeast(transitions, "result", 0.5F, "Has One Transition", {1420, 2100});
    const auto multipleTransitions = atLeast(transitions, "result", 1.5F, "Has Multiple Transitions", {1420, 2220});
    const auto oneTransition = inverse(multipleTransitions, "result", "Exactly One Transition", {1660, 2220});
    const auto validTransitions = multiply(hasTransition, "result", oneTransition, "result",
                                           "One 0-to-1 Transition", {1900, 2160});

    const auto triple = [&](std::size_t first, std::size_t second, std::size_t third,
                            std::string label, Vec2 position) {
        const auto pair = multiply(samples[first], "sampled", samples[second], "sampled",
                                   label + " Pair", position);
        return multiply(pair, "result", samples[third], "sampled", std::move(label),
                        {position.x + 240.0F, position.y});
    };
    const auto phase0A = inverse(triple(0, 2, 4, "P2 P4 P6", {2140, 740}), "result",
                                 "Phase 1: P2 P4 P6 Has Background", {2620, 740});
    const auto phase0B = inverse(triple(2, 4, 6, "P4 P6 P8", {2140, 900}), "result",
                                 "Phase 1: P4 P6 P8 Has Background", {2620, 900});
    const auto phase0 = multiply(phase0A, "result", phase0B, "result", "Phase 1 Conditions", {2860, 820});
    const auto phase1A = inverse(triple(0, 2, 6, "P2 P4 P8", {2140, 1080}), "result",
                                 "Phase 2: P2 P4 P8 Has Background", {2620, 1080});
    const auto phase1B = inverse(triple(0, 4, 6, "P2 P6 P8", {2140, 1240}), "result",
                                 "Phase 2: P2 P6 P8 Has Background", {2620, 1240});
    const auto phase1 = multiply(phase1A, "result", phase1B, "result", "Phase 2 Conditions", {2860, 1160});
    const auto phaseConditions = addNode(body, "select", "Alternating Phase Conditions", {3100, 980});
    link(body, previousPhase, "value", phaseConditions, "condition");
    link(body, phase1, "result", phaseConditions, "ifTrue");
    link(body, phase0, "result", phaseConditions, "ifFalse");
    const auto preliminary = multiply(validNeighborCount, "result", validTransitions, "result",
                                      "Topology-Preserving Candidate", {2140, 2160});
    const auto candidate = multiply(preliminary, "result", phaseConditions, "result",
                                    "Phase Deletion Candidate", {3340, 1420});
    const auto deletePixel = multiply(binary, "result", candidate, "result", "Delete Pixel", {3580, 1420});
    const auto nextImageValue = math(body, "Skeleton After Sub-pass", static_cast<int>(MathOperation::Subtract),
                                     {3820, 1420});
    link(body, binary, "result", nextImageValue, "a");
    link(body, deletePixel, "result", nextImageValue, "b");
    const auto nextImage = addNode(body, "simulation_next_state", "Next Skeleton", {4060, 1420},
                                   {{"slot", 0.0F}});
    link(body, nextImageValue, "result", nextImage, "value");
    const auto output = addNode(body, "subgraph_output", "Skeleton Output", {4300, 1420},
                                {{"key", "skeleton"}});
    link(body, nextImage, "value", output, "value");

    const auto nextPhaseValue = inverse(previousPhase, "value", "Alternate Thinning Phase", {2640, 1600});
    const auto nextPhase = addNode(body, "simulation_next_state", "Next Thinning Phase", {2880, 1600},
                                   {{"slot", 1.0F}});
    link(body, nextPhaseValue, "result", nextPhase, "value");
    return result;
}

SubgraphDefinition distanceFromNearestWhitePixel() {
    SubgraphDefinition result;
    result.id = "builtin.distance_from_nearest_white_pixel";
    result.name = "Distance from Nearest White Pixel";
    result.category = "Simulation";
    result.execution = SubgraphExecution::Simulation;
    result.immutable = true;
    result.stateSlots = {
        {"nearestSeed", "Nearest White Seed", ValueType::VectorField},
        {"distance", "Distance", ValueType::ScalarField},
    };
    result.interface = {
        {"whiteMask", "White Mask", SubgraphInterfaceKind::Input, ValueType::ScalarField},
        {"maxDistance", "Maximum Distance", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 2048.0F, 1.0F, 65504.0F},
        {"iterations", "Iterations per Frame", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 16.0F, 1.0F, 64.0F, Control::Integer},
        {"reset", "Reset", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.0F, 0.0F, 1.0F, Control::Boolean},
        {"distance", "Distance (Pixels)", SubgraphInterfaceKind::Output,
         ValueType::ScalarField},
    };

    auto& body = result.body;
    const auto whiteMask = input(body, "whiteMask", "White Mask", {0, 100});
    // Treat only a fully white mask pixel as a seed. Convert a color input to
    // a scalar mask explicitly before entering this subgraph.
    const auto white = addNode(body, "threshold", "White Pixel", {220, 100},
                               {{"threshold", 0.999F}});
    link(body, whiteMask, "value", white, "value");
    const auto coordinates = addNode(body, "coordinates", "Pixel Coordinates", {220, 240},
                                     {{"pixels", 1.0F}});
    // The sentinel is deliberately far outside every practical canvas. It
    // represents "no seed has reached this pixel" without a third state slot.
    const auto noSeed = vector(body, "No Seed Sentinel", {220, 380}, -65504.0F, -65504.0F);
    const auto initialSeed = addNode(body, "select", "Initial Nearest Seed", {480, 210});
    link(body, white, "result", initialSeed, "condition");
    link(body, coordinates, "coordinates", initialSeed, "ifTrue");
    link(body, noSeed, "value", initialSeed, "ifFalse");
    const auto initialSeedState = addNode(body, "simulation_initial_state", "Initial Nearest Seed",
                                          {740, 210}, {{"slot", 0.0F}});
    link(body, initialSeed, "result", initialSeedState, "value");
    const auto maxDistance = input(body, "maxDistance", "Maximum Distance", {480, 370});
    const auto initialDistance = addNode(body, "select", "Initial Distance", {740, 370},
                                         {{"ifTrue", 0.0F}});
    link(body, white, "result", initialDistance, "condition");
    link(body, maxDistance, "value", initialDistance, "ifFalse");
    const auto initialDistanceState = addNode(body, "simulation_initial_state", "Initial Distance",
                                              {980, 370}, {{"slot", 1.0F}});
    link(body, initialDistance, "result", initialDistanceState, "value");

    const auto previous = addNode(body, "simulation_previous_state", "Previous Nearest Seed",
                                  {0, 700}, {{"slot", 0.0F}});
    // This slot is output-only, but every simulation state slot has the same
    // explicit Previous/Initial/Next boundary contract.
    (void)addNode(body, "simulation_previous_state", "Previous Distance", {0, 1860},
                  {{"slot", 1.0F}});
    const std::array<std::pair<const char*, Vec2>, 8> offsets{{
        {"North", {0.0F, -1.0F}}, {"South", {0.0F, 1.0F}},
        {"West", {-1.0F, 0.0F}}, {"East", {1.0F, 0.0F}},
        {"Northwest", {-1.0F, -1.0F}}, {"Northeast", {1.0F, -1.0F}},
        {"Southwest", {-1.0F, 1.0F}}, {"Southeast", {1.0F, 1.0F}},
    }};
    std::array<NodeId, 8> samples{};
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        const auto offset = vector(body, std::string(offsets[index].first) + " Offset",
                                   {220, 620.0F + static_cast<float>(index) * 140.0F},
                                   offsets[index].second.x, offsets[index].second.y);
        samples[index] = addNode(body, "state_input_sample_offset",
            std::string(offsets[index].first) + " Neighbor Seed",
            {460, 620.0F + static_cast<float>(index) * 140.0F},
            {{"sampling", 0.0F}, {"addressMode", 0.0F}});
        link(body, previous, "value", samples[index], "source");
        link(body, offset, "value", samples[index], "offset");
    }
    NodeId bestSeed = previous;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const auto candidateDistance = vectorMath(
            body, std::string(offsets[index].first) + " Candidate Distance",
            VectorMathOperation::Distance, {720, 620.0F + static_cast<float>(index) * 140.0F});
        link(body, coordinates, "coordinates", candidateDistance, "a");
        link(body, samples[index], "sampled", candidateDistance, "b");
        const auto bestDistance = vectorMath(
            body, std::string(offsets[index].first) + " Best Distance",
            VectorMathOperation::Distance, {960, 620.0F + static_cast<float>(index) * 140.0F});
        link(body, coordinates, "coordinates", bestDistance, "a");
        link(body, bestSeed, index == 0 ? "value" : "result", bestDistance, "b");
        // Step(edge, value) is one when the candidate is no farther away.
        const auto chooseCandidate = math(body, std::string(offsets[index].first) + " Is Closer",
                                          static_cast<int>(MathOperation::Step),
                                          {1200, 620.0F + static_cast<float>(index) * 140.0F});
        link(body, candidateDistance, "result", chooseCandidate, "a");
        link(body, bestDistance, "result", chooseCandidate, "b");
        const auto selected = addNode(body, "select", std::string(offsets[index].first) + " Best Seed",
                                      {1440, 620.0F + static_cast<float>(index) * 140.0F});
        link(body, chooseCandidate, "result", selected, "condition");
        link(body, samples[index], "sampled", selected, "ifTrue");
        link(body, bestSeed, index == 0 ? "value" : "result", selected, "ifFalse");
        bestSeed = selected;
    }
    const auto pinnedSeed = addNode(body, "select", "White Pixels Keep Their Seed", {1680, 1180});
    link(body, white, "result", pinnedSeed, "condition");
    link(body, coordinates, "coordinates", pinnedSeed, "ifTrue");
    link(body, bestSeed, "result", pinnedSeed, "ifFalse");
    const auto nextSeed = addNode(body, "simulation_next_state", "Next Nearest Seed", {1920, 1180},
                                  {{"slot", 0.0F}});
    link(body, pinnedSeed, "result", nextSeed, "value");
    const auto exactDistance = vectorMath(body, "Euclidean Distance", VectorMathOperation::Distance,
                                          {1920, 1340});
    link(body, coordinates, "coordinates", exactDistance, "a");
    link(body, pinnedSeed, "result", exactDistance, "b");
    const auto cappedDistance = math(body, "Capped Distance", static_cast<int>(MathOperation::Minimum),
                                     {2160, 1340});
    link(body, exactDistance, "result", cappedDistance, "a");
    link(body, maxDistance, "value", cappedDistance, "b");
    const auto nextDistance = addNode(body, "simulation_next_state", "Next Distance", {2400, 1340},
                                      {{"slot", 1.0F}});
    link(body, cappedDistance, "result", nextDistance, "value");
    const auto output = addNode(body, "subgraph_output", "Distance Output", {2640, 1340},
                                {{"key", "distance"}});
    link(body, nextDistance, "value", output, "value");
    return result;
}

// The native SDF Generator uses jump flooding; this template intentionally
// keeps the slower one-pixel propagation visible and editable.  It is useful
// for learning and for rules that want to intervene in either seed flow.
SubgraphDefinition sdfGenerator() {
    SubgraphDefinition result;
    result.id = "builtin.sdf_generator";
    result.name = "SDF Generator";
    result.category = "Simulation";
    result.execution = SubgraphExecution::Simulation;
    result.immutable = true;
    result.stateSlots = {{"foregroundSeed", "Nearest Foreground", ValueType::VectorField},
                         {"backgroundSeed", "Nearest Background", ValueType::VectorField},
                         {"distance", "Signed Distance", ValueType::ScalarField}};
    result.interface = {
        {"mask", "Mask", SubgraphInterfaceKind::Input, ValueType::ScalarField},
        {"threshold", "Threshold", SubgraphInterfaceKind::Input, ValueType::Float,
         true, .5F, 0.F, 1.F},
        {"iterations", "Iterations per Frame", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 16.F, 1.F, 64.F, Control::Integer},
        {"reset", "Reset", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.F, 0.F, 1.F, Control::Boolean},
        {"distance", "Signed Distance (Pixels)", SubgraphInterfaceKind::Output,
         ValueType::ScalarField},
    };
    auto& body = result.body;
    const auto mask = input(body, "mask", "Mask", {0, 100});
    const auto thresholdInput = input(body, "threshold", "Threshold", {0, 240});
    const auto foreground = addNode(body, "threshold", "Foreground", {240, 100});
    link(body, mask, "value", foreground, "value");
    link(body, thresholdInput, "value", foreground, "threshold");
    const auto background = math(body, "Background", static_cast<int>(MathOperation::Subtract),
                                 {480, 100}, {{"a", 1.0F}});
    link(body, foreground, "result", background, "b");
    const auto coordinates = addNode(body, "coordinates", "Pixel Coordinates", {240, 400},
                                     {{"pixels", 1.0F}});
    const auto iteration = addNode(body, "simulation_iteration_info", "Jump Flood Step", {240, 680});
    const auto remaining = math(body, "Remaining Jump Passes", static_cast<int>(MathOperation::Subtract),
                                {480, 680});
    link(body, iteration, "iterationCount", remaining, "a");
    link(body, iteration, "iterationIndex", remaining, "b");
    const auto exponent = math(body, "Jump Exponent", static_cast<int>(MathOperation::Subtract),
                               {720, 680}, {{"b", 1.0F}});
    link(body, remaining, "result", exponent, "a");
    const auto jump = math(body, "Jump Distance (Pixels)", static_cast<int>(MathOperation::Power),
                           {960, 680}, {{"a", 2.0F}});
    link(body, exponent, "result", jump, "b");
    const auto sentinel = vector(body, "No Seed", {240, 540}, -65504.F, -65504.F);
    const auto initialDistance = addNode(body, "simulation_initial_state", "Initial Distance",
                                         {500, 540}, {{"slot", 2.0F}});
    // Initial values are only presented during reset; the first update replaces
    // this with the freshly propagated signed distance.
    link(body, foreground, "result", initialDistance, "value");
    (void)addNode(body, "simulation_previous_state", "Previous Signed Distance",
                  {0, 3240}, {{"slot", 2.0F}});
    const std::array<std::pair<const char*, Vec2>, 8> offsets{{
        {"N", {0.F,-1.F}}, {"S", {0.F,1.F}}, {"W", {-1.F,0.F}}, {"E", {1.F,0.F}},
        {"NW", {-1.F,-1.F}}, {"NE", {1.F,-1.F}}, {"SW", {-1.F,1.F}}, {"SE", {1.F,1.F}},
    }};
    const auto makeSeedFlow = [&](NodeId membership, int slot, std::string prefix, float y) {
        const auto initial = addNode(body, "select", prefix + " Initial Seed", {500, y});
        link(body, membership, "result", initial, "condition");
        link(body, coordinates, "coordinates", initial, "ifTrue");
        link(body, sentinel, "value", initial, "ifFalse");
        const auto initialState = addNode(body, "simulation_initial_state", prefix + " Initial State",
                                          {740, y}, {{"slot", static_cast<float>(slot)}});
        link(body, initial, "result", initialState, "value");
        const auto previous = addNode(body, "simulation_previous_state", prefix + " Previous",
                                      {0, y + 680.F}, {{"slot", static_cast<float>(slot)}});
        NodeId best = previous;
        std::string bestSocket = "value";
        for (std::size_t i = 0; i < offsets.size(); ++i) {
            const auto offset = vector(body, prefix + " " + offsets[i].first + " Offset",
                                       {220, y + 620.F + static_cast<float>(i)*120.F}, offsets[i].second.x, offsets[i].second.y);
            const auto jumpOffset = vectorMath(body, prefix + " " + offsets[i].first + " Jump Offset",
                                               VectorMathOperation::Scale,
                                               {440, y + 620.F + static_cast<float>(i)*120.F});
            link(body, offset, "value", jumpOffset, "a");
            link(body, jump, "result", jumpOffset, "scalar");
            const auto sample = addNode(body, "state_input_sample_offset", prefix + " " + offsets[i].first,
                                        {440, y + 620.F + static_cast<float>(i)*120.F}, {{"sampling",0.F},{"addressMode",0.F}});
            link(body, previous, "value", sample, "source"); link(body, jumpOffset, "result", sample, "offset");
            const auto candidateD = vectorMath(body, prefix + " Candidate Distance", VectorMathOperation::Distance,
                                               {680, y + 620.F + static_cast<float>(i)*120.F});
            link(body, coordinates, "coordinates", candidateD, "a"); link(body, sample, "sampled", candidateD, "b");
            const auto bestD = vectorMath(body, prefix + " Best Distance", VectorMathOperation::Distance,
                                          {920, y + 620.F + static_cast<float>(i)*120.F});
            link(body, coordinates, "coordinates", bestD, "a"); link(body, best, bestSocket, bestD, "b");
            const auto choose = math(body, prefix + " Is Closer", static_cast<int>(MathOperation::Step),
                                     {1160, y + 620.F + static_cast<float>(i)*120.F});
            link(body, candidateD, "result", choose, "a"); link(body, bestD, "result", choose, "b");
            const auto selected = addNode(body, "select", prefix + " Best Seed", {1400, y + 620.F + static_cast<float>(i)*120.F});
            link(body, choose, "result", selected, "condition"); link(body, sample, "sampled", selected, "ifTrue"); link(body, best, bestSocket, selected, "ifFalse");
            best=selected; bestSocket="result";
        }
        const auto pinned=addNode(body,"select",prefix+" Keep Seed",{1640,y+1080.F});
        link(body,membership,"result",pinned,"condition");link(body,coordinates,"coordinates",pinned,"ifTrue");link(body,best,bestSocket,pinned,"ifFalse");
        const auto next=addNode(body,"simulation_next_state",prefix+" Next",{1880,y+1080.F},{{"slot",static_cast<float>(slot)}});link(body,pinned,"result",next,"value");
        return next;
    };
    const auto fg=makeSeedFlow(foreground,0,"Foreground",700.F);
    const auto bg=makeSeedFlow(background,1,"Background",1900.F);
    const auto fgDistance=vectorMath(body,"Distance to Foreground",VectorMathOperation::Distance,{2140,1700});link(body,coordinates,"coordinates",fgDistance,"a");link(body,fg,"value",fgDistance,"b");
    const auto bgDistance=vectorMath(body,"Distance to Background",VectorMathOperation::Distance,{2140,1840});link(body,coordinates,"coordinates",bgDistance,"a");link(body,bg,"value",bgDistance,"b");
    const auto negative=math(body,"Inside Distance",static_cast<int>(MathOperation::Multiply),{2380,1840},{{"b",-1.F}});link(body,bgDistance,"result",negative,"a");
    const auto signedDistance=addNode(body,"select","Signed Distance",{2620,1760});link(body,foreground,"result",signedDistance,"condition");link(body,negative,"result",signedDistance,"ifTrue");link(body,fgDistance,"result",signedDistance,"ifFalse");
    const auto nextDistance=addNode(body,"simulation_next_state","Next Signed Distance",{2860,1760},{{"slot",2.0F}});link(body,signedDistance,"result",nextDistance,"value");
    const auto output=addNode(body,"subgraph_output","SDF Output",{3100,1760},{{"key","distance"}});link(body,nextDistance,"value",output,"value");
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


SubgraphDefinition noiseDrivenParticles() {
    SubgraphDefinition result;
    result.id = "builtin.particles.noise_flow";
    result.version = 3;
    result.name = "Noise-Driven Particles (Sands)";
    result.category = "Simulation";
    result.execution = SubgraphExecution::Simulation;
    result.immutable = true;
    // Coordinates are integrated at subpixel precision. Half-float UV storage
    // otherwise rounds the sub-1px teal increments to zero or unstable jumps.
    result.stateSlots = {
        {"position", "Particle Position", ValueType::VectorField, true},
        {"life", "Remaining Life", ValueType::ScalarField},
        {"orange", "Orange Positions", ValueType::VectorField, true},
        {"teal", "Teal Positions", ValueType::VectorField, true},
    };
    result.interface = {
        {"noiseScale", "Noise Scale", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 5.0F, .05F, 40.0F},
        {"speed", "Particle Speed", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 1.0F, 0.0F, 10.0F},
        {"angleBins", "Direction Quantization", SubgraphInterfaceKind::Input,
         ValueType::Float, true, 1200.0F, 8.0F, 4096.0F, Control::Integer},
        {"maxJump", "Tangent Jump Limit", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 64.0F, 1.0F, 512.0F},
        {"orangeSpeed", "Orange Speed", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 1.0F, 0.0F, 5.0F},
        {"tealSpeed", "Teal Speed", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 1.0F, 0.0F, 5.0F},
        {"lifeRate", "Life Decay", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.1F, 0.001F, 1.0F},
        {"orangeMin", "Orange Angle Start", SubgraphInterfaceKind::Input,
         ValueType::Float, true, 3.0F, 0.0F, 6.28F},
        {"orangeMax", "Orange Angle End", SubgraphInterfaceKind::Input,
         ValueType::Float, true, 6.0F, 0.0F, 6.28F},
        {"noiseOffsetX", "Noise Pan X", SubgraphInterfaceKind::Input,
         ValueType::Float, true, 0.0F, -100.0F, 100.0F},
        {"noiseOffsetY", "Noise Pan Y", SubgraphInterfaceKind::Input,
         ValueType::Float, true, 0.0F, -100.0F, 100.0F},
        {"reset", "Reset", SubgraphInterfaceKind::Input, ValueType::Float,
         true, 0.0F, 0.0F, 1.0F, Control::Boolean},
        {"iterations", "Iterations per Frame", SubgraphInterfaceKind::Input,
         ValueType::Float, true, 1.0F, 1.0F, 64.0F, Control::Integer},
        {"orangePositions", "Orange Positions", SubgraphInterfaceKind::Output,
         ValueType::VectorField},
        {"tealPositions", "Teal Positions", SubgraphInterfaceKind::Output,
         ValueType::VectorField},
    };

    auto& body = result.body;
    using M = MathOperation;
    using V = VectorMathOperation;
    // Small node wiring helpers: all math, branches, random numbers, and
    // state transitions remain genuine editable nodes in the body.
    using Wire = std::pair<NodeId, const char*>;
    const auto w = [](NodeId n, const char* socket = "result") -> Wire {
        return {n, socket};
    };
    const auto connect = [&](Wire from, NodeId to, const char* socket) {
        link(body, from.first, from.second, to, socket);
    };
    const auto calc = [&](std::string name, M operation, int x, int y,
                          Wire a = {}, Wire b = {}, Wire c = {},
                          nlohmann::json defaults = nlohmann::json::object()) -> Wire {
        const auto id = math(body, std::move(name), static_cast<int>(operation),
                             {static_cast<float>(x), static_cast<float>(y)},
                             std::move(defaults));
        if (a.first) connect(a, id, "a");
        if (b.first) connect(b, id, "b");
        if (c.first) connect(c, id, "c");
        return w(id);
    };
    const auto vec = [&](std::string name, Wire x, Wire y, int col, int row) -> Wire {
        const auto id = addNode(body, "combine_vector", std::move(name),
                                {static_cast<float>(col), static_cast<float>(row)});
        connect(x, id, "x");
        connect(y, id, "y");
        return w(id, "value");
    };
    const auto split = [&](std::string name, Wire value, int col, int row) -> NodeId {
        const auto id = addNode(body, "separate_vector", std::move(name),
                                {static_cast<float>(col), static_cast<float>(row)});
        connect(value, id, "value");
        return id;
    };
    const auto choose = [&](std::string name, Wire condition, Wire ifTrue,
                            Wire ifFalse, int col, int row) -> Wire {
        const auto id = addNode(body, "select", std::move(name),
                                {static_cast<float>(col), static_cast<float>(row)});
        connect(condition, id, "condition");
        connect(ifTrue, id, "ifTrue");
        connect(ifFalse, id, "ifFalse");
        return w(id);
    };

    // INIT: independent uniform positions and lifetimes, keyed by agent slot.
    const auto slots = addNode(body, "coordinates", "Agent Slot Identity (UV)", {0, 20});
    const auto slotUv = w(slots, "coordinates");
    const auto spawnAtStart = addNode(body, "hash", "Initial Random Position", {260, 0},
                                      {{"seed", 14.0F}, {"salt", 0.0F}});
    const auto ageAtStart = addNode(body, "hash", "Initial Random Life", {260, 200},
                                    {{"seed", 15.0F}, {"salt", 0.0F}});
    connect(slotUv, spawnAtStart, "position");
    connect(slotUv, ageAtStart, "position");
    const auto initialPos = addNode(body, "simulation_initial_state", "Start Position",
                                    {520, 0}, {{"slot", 0.0F}});
    const auto initialAge = addNode(body, "simulation_initial_state", "Start Life",
                                    {520, 200}, {{"slot", 1.0F}});
    const auto initialOrange = addNode(body, "simulation_initial_state", "Initial Orange Positions",
                                       {520, 400}, {{"slot", 2.0F}});
    const auto initialTeal = addNode(body, "simulation_initial_state", "Initial Teal Positions",
                                       {520, 600}, {{"slot", 3.0F}});
    connect(w(spawnAtStart, "vector"), initialPos, "value");
    connect(w(ageAtStart, "scalar"), initialAge, "value");
    const auto inactive = vector(body, "Inactive Off-Canvas Position", {260, 400}, -1.0F, -1.0F);
    connect(w(inactive,"value"), initialOrange, "value");
    connect(w(inactive,"value"), initialTeal, "value");

    // FEEDBACK: three explicit state slots, with the per-agent positions stored
    // in RG32F rather than quantized RG16F.
    const auto previousPos = addNode(body, "simulation_previous_state", "Previous Position",
                                     {0, 760}, {{"slot", 0.0F}});
    const auto previousLife = addNode(body, "simulation_previous_state", "Previous Life",
                                      {0, 960}, {{"slot", 1.0F}});
    addNode(body, "simulation_previous_state", "Previous Orange Slots",
            {0, 1160}, {{"slot", 2.0F}});
    addNode(body, "simulation_previous_state", "Previous Teal Slots",
            {0, 1260}, {{"slot", 3.0F}});
    const auto pos = w(previousPos, "value");
    const auto life = w(previousLife, "value");

    // Stationary flow field; sample at the *moving particle position*.
    const auto noiseScale = input(body, "noiseScale", "Flow Noise Scale", {0, 1450});
    const auto perlin = addNode(body, "perlin", "Stationary Four-Octave Perlin",
                                {260, 1450},
                                {{"seed", 1.0F}, {"speed", 0.0F}, {"octaves", 4.0F},
                                 {"persistence", 0.5F}, {"lacunarity", 2.0F}});
    connect(w(noiseScale, "value"), perlin, "scale");
    const auto panX = input(body, "noiseOffsetX", "Noise Pan X", {0, 1570});
    const auto panY = input(body, "noiseOffsetY", "Noise Pan Y", {0, 1690});
    connect(w(panX, "value"), perlin, "offsetX");
    connect(w(panY, "value"), perlin, "offsetY");
    const auto noiseSampler = addNode(body, "texture_sample", "Sample Flow at Particle",
                                      {520, 1420},
                                      {{"addressMode", 1.0F}, {"sampling", 1.0F}});
    connect(w(perlin, "image"), noiseSampler, "source");
    connect(pos, noiseSampler, "coordinates");

    // ANGLE = 2*pi * round(noise*1200)/1200.
    const auto bins = input(body, "angleBins", "Direction Bins", {790, 1550});
    const auto scaledNoise = calc("Noise to Angular Bins", M::Multiply, 790, 1370,
                                  w(noiseSampler, "sampled"), w(bins, "value"));
    const auto roundNoise = calc("Quantize Flow Angle", M::Round, 1020, 1370, scaledNoise);
    const auto normalized = calc("Quantized Fraction", M::Divide, 1240, 1370,
                                 roundNoise, w(bins, "value"));
    const auto theta = calc("Flow Angle Radians", M::Multiply, 1460, 1370,
                            normalized, {}, {}, {{"b", 6.28318530718F}});

    // 3 < angle < 6 determines the orange / teal motion branch.
    const auto cutMin = input(body, "orangeMin", "Orange Angular Start", {1460, 1160});
    const auto cutMax = input(body, "orangeMax", "Orange Angular End", {1460, 1300});
    const auto afterThree = calc("Angle After Start", M::Step, 1680, 1210,
                                 w(cutMin,"value"), theta);
    const auto afterSix = calc("Angle After End", M::Step, 1680, 1410,
                               w(cutMax,"value"), theta);
    const auto belowSix = calc("Angle < 6", M::Subtract, 1900, 1410,
                               {}, afterSix, {}, {{"a", 1.0F}});
    const auto orange = calc("Orange Region 3..6", M::Multiply, 2120, 1300,
                             afterThree, belowSix);

    // RANDOM STREAMS: per-slot and per-frame hashes. All seeds are visible,
    // streams are independent, and all randomness is reproducible on replay.
    const auto frame = addNode(body, "simulation_step_info", "Simulation Frame",
                               {0, 1830});
    const auto frameNumber = w(frame, "frameIndex");
    const auto randomFor = [&](std::string name, int seed, int row) -> NodeId {
        const auto id = addNode(body, "hash", std::move(name),
            {270.0F, static_cast<float>(row)},
            {{"seed", static_cast<float>(seed)}, {"salt", 0.0F}});
        connect(slotUv, id, "position");
        connect(frameNumber, id, "salt");
        return id;
    };
    const auto jitter = randomFor("Independent Velocity Jitter", 1001, 1780);
    const auto ageRandomA = randomFor("Life Random Samples A", 1002, 2030);
    const auto ageRandomB = randomFor("Life Random Samples B", 1003, 2280);
    const auto newSpawn = randomFor("Random Respawn XY", 1004, 2530);
    const auto newLife = randomFor("Random Respawn Life", 1005, 2780);
    const auto jitterComponents = split("Movement Random XY", w(jitter, "vector"), 520, 1810);
    const auto ageComponentsA = split("Life Samples 1 and 2", w(ageRandomA, "vector"), 520, 2030);
    const auto ageComponentsB = split("Life Samples 3 and 4", w(ageRandomB, "vector"), 520, 2260);

    // ORANGE: (tan(theta) * random(1,3), tan(theta)).
    const auto rawTan = calc("Tangent Flow", M::Tangent, 1680, 1600, theta);
    const auto jump = input(body, "maxJump", "Tangent Cap", {1460, 1740});
    const auto negativeJump = calc("Negative Tangent Cap", M::Multiply, 1680, 1740,
                                   w(jump, "value"), {}, {}, {{"b", -1.0F}});
    const auto safeTan = calc("Clamp Tangent Singularities", M::Clamp, 1900, 1640,
                              rawTan, negativeJump, w(jump, "value"));
    const auto orangeJitter = calc("Random Orange Speed 1..3", M::Multiply, 800, 1810,
                                   w(jitterComponents, "x"), {}, {}, {{"b", 2.0F}});
    const auto orangeFactor = calc("Orange Speed + 1", M::Add, 1020, 1810,
                                   orangeJitter, {}, {}, {{"b", 1.0F}});
    const auto orangeX = calc("Orange Horizontal Speed", M::Multiply, 2120, 1650,
                              safeTan, orangeFactor);
    const auto orangeStep = vec("Orange Diagonal Motion", orangeX, safeTan, 2340, 1610);

    // TEAL: (-sin(theta) * random(.1,1), -cos(theta)).
    const auto sine = calc("Flow Sine", M::Sine, 1680, 1920, theta);
    const auto cosine = calc("Flow Cosine", M::Cosine, 1680, 2150, theta);
    const auto negSin = calc("Negative Sine", M::Multiply, 1900, 1920,
                             sine, {}, {}, {{"b", -1.0F}});
    const auto negCos = calc("Negative Cosine", M::Multiply, 1900, 2150,
                             cosine, {}, {}, {{"b", -1.0F}});
    const auto tealRandom = calc("Teal Speed 0.1..1", M::Multiply, 800, 1920,
                                 w(jitterComponents, "y"), {}, {}, {{"b", 0.9F}});
    const auto tealFactor = calc("Teal Speed Offset", M::Add, 1020, 1920,
                                 tealRandom, {}, {}, {{"b", 0.1F}});
    const auto tealX = calc("Teal Horizontal Motion", M::Multiply, 2120, 1930,
                            negSin, tealFactor);
    const auto tealStep = vec("Teal Sine Cosine Motion", tealX, negCos, 2340, 2080);

    const auto orangeSpeed = input(body, "orangeSpeed", "Orange Branch Speed", {2350, 1790});
    const auto tealSpeed = input(body, "tealSpeed", "Teal Branch Speed", {2350, 2260});
    const auto scaleOrange = vectorMath(body,"Scale Orange Velocity",V::Scale,{2520,1660});
    connect(orangeStep,scaleOrange,"a");
    connect(w(orangeSpeed,"value"),scaleOrange,"scalar");
    const auto scaleTeal = vectorMath(body,"Scale Teal Velocity",V::Scale,{2520,2070});
    connect(tealStep,scaleTeal,"a");
    connect(w(tealSpeed,"value"),scaleTeal,"scalar");
    const auto selectedStep = choose("Select Orange / Teal Motion", orange,
                                     w(scaleOrange), w(scaleTeal), 2770, 1820);
    const auto speed = input(body, "speed", "Particle Motion Speed", {2590, 2110});
    const auto scaledStep = vectorMath(body, "Apply Motion Speed", V::Scale, {2820, 1840});
    connect(selectedStep, scaledStep, "a");
    connect(w(speed, "value"), scaledStep, "scalar");
    const auto pixelToUv = addNode(body, "pixel_delta_to_uv", "Pixels to UV",
                                   {3050, 1840});
    connect(w(scaledStep), pixelToUv, "delta");
    const auto updatedPosition = vectorMath(body, "Integrate Particle Position",
                                             V::Add, {3290, 1840});
    connect(pos, updatedPosition, "a");
    connect(w(pixelToUv, "uvDelta"), updatedPosition, "b");

    // LIFE: life -= random(random(random(random())))/10. Each nested
    // uniform random input is an independent sample and becomes one factor.
    const auto lifeAB = calc("Nested Random Product 1", M::Multiply, 800, 2180,
                             w(ageComponentsA, "x"), w(ageComponentsA, "y"));
    const auto lifeCD = calc("Nested Random Product 2", M::Multiply, 800, 2370,
                             w(ageComponentsB, "x"), w(ageComponentsB, "y"));
    const auto lifeProduct = calc("Nested Random Product 3", M::Multiply, 1030, 2280,
                                  lifeAB, lifeCD);
    const auto decayRate = input(body,"lifeRate","Life Decay Rate",{1030,2440});
    const auto loss = calc("Life Lost This Step", M::Multiply, 1260, 2280,
                           lifeProduct,w(decayRate,"value"));
    const auto reducedLife = calc("Subtract Life Lost", M::Subtract, 1490, 2280,
                                  life, loss);
    const auto dead = calc("Dead When Life <= 0", M::Step, 1720, 2280,
                           reducedLife, {}, {}, {{"b", 0.0F}});
    const auto respawnedPosition = choose("Respawn or Retain Position", dead,
                                          w(newSpawn, "vector"),
                                          w(updatedPosition, "result"),
                                          3510, 1800);
    const auto updatedLife = choose("Respawn or Retain Life", dead,
                                    w(newLife, "scalar"), reducedLife,
                                    3510, 2230);

    // STORE: visibly separate endpoints. The exported values are next-state
    // texture snapshots; nothing in this body uses a hidden motion kernel.
    const auto nextPos = addNode(body, "simulation_next_state", "Next Position",
                                 {3770, 1800}, {{"slot", 0.0F}});
    const auto nextLife = addNode(body, "simulation_next_state", "Next Life",
                                  {3770, 2230}, {{"slot", 1.0F}});
    const auto orangeOnly = choose("Orange Only / Off Canvas",orange,
                                   respawnedPosition,w(inactive,"value"),3510,1360);
    const auto tealFlag = calc("Invert Orange Mask",M::Subtract,3310,1570,
                              {},orange,{},{{"a",1.0F}});
    const auto tealOnly = choose("Teal Only / Off Canvas",tealFlag,
                                 respawnedPosition,w(inactive,"value"),3510,1630);
    const auto nextOrange = addNode(body, "simulation_next_state", "Next Orange Position",
                                    {3770, 1330}, {{"slot", 2.0F}});
    const auto nextTeal = addNode(body, "simulation_next_state", "Next Teal Position",
                                    {3770, 1590}, {{"slot", 3.0F}});
    connect(respawnedPosition, nextPos, "value");
    connect(updatedLife, nextLife, "value");
    connect(orangeOnly,nextOrange,"value");
    connect(tealOnly,nextTeal,"value");

    const auto orangePositions = addNode(body, "subgraph_output", "Orange Particle Positions",
                                   {4050, 1330}, {{"key", "orangePositions"}});
    const auto tealPositions = addNode(body, "subgraph_output", "Teal Particle Positions",
                                  {4050, 1590}, {{"key", "tealPositions"}});
    connect(w(nextOrange, "value"), orangePositions, "value");
    connect(w(nextTeal, "value"), tealPositions, "value");
    return result;
}

SubgraphDefinition sandsTrailLayers() {
    SubgraphDefinition result;
    result.id = "builtin.particles.sands_layers";
    result.version = 1;
    result.name = "Sands Trails (Two Layers)";
    result.category = "Compositing";
    result.execution = SubgraphExecution::Pipeline;
    result.immutable = true;
    using K = SubgraphInterfaceKind;
    const auto scalar = [&](std::string key,std::string label,float value,
                            float minimum,float maximum) {
        result.interface.emplace_back(
            std::move(key),std::move(label),K::Input,ValueType::Float,
            true,value,minimum,maximum);
    };
    result.interface.emplace_back("orangePositions","Orange Positions",K::Input,
                                   ValueType::VectorField);
    result.interface.emplace_back("tealPositions","Teal Positions",K::Input,
                                   ValueType::VectorField);
    scalar("count","Particle Samples per Layer",1000.0F,1.0F,100000.0F);
    scalar("orangeRetention","Orange Trail Retention",.985F,0.0F,1.0F);
    scalar("tealRetention","Teal Trail Retention",.985F,0.0F,1.0F);
    scalar("orangeOpacity","Orange Point Opacity",.375F,0.0F,1.0F);
    scalar("tealOpacity","Teal Point Opacity",.375F,0.0F,1.0F);
    scalar("orangeSize","Orange Point Size",1.0F,.5F,16.0F);
    scalar("tealSize","Teal Point Size",1.0F,.5F,16.0F);
    scalar("orangeR","Orange Red",.90F,0.0F,1.0F);
    scalar("orangeG","Orange Green",.2925F,0.0F,1.0F);
    scalar("orangeB","Orange Blue",.09F,0.0F,1.0F);
    scalar("tealR","Teal Red",.09F,0.0F,1.0F);
    scalar("tealG","Teal Green",.90F,0.0F,1.0F);
    scalar("tealB","Teal Blue",.8325F,0.0F,1.0F);
    scalar("orangeGain","Orange Layer Gain",1.0F,0.0F,8.0F);
    scalar("tealGain","Teal Layer Gain",1.0F,0.0F,8.0F);
    scalar("backgroundR","Background Red",.05F,0.0F,1.0F);
    scalar("backgroundG","Background Green",.05F,0.0F,1.0F);
    scalar("backgroundB","Background Blue",.05F,0.0F,1.0F);
    result.interface.emplace_back("image","Added Trails",K::Output,
                                   ValueType::ColorImage);

    auto& body=result.body;
    const auto orange=addNode(body,"particle_splat","Orange Splat",
                              {500,70},{{"margin",20.0F},{"roundness",0.0F}});
    const auto teal=addNode(body,"particle_splat","Teal Splat",
                            {500,760},{{"margin",20.0F},{"roundness",0.0F}});
    const auto add=addNode(body,"image_add","Add Independent Layers",
                           {1080,410});
    const auto out=addNode(body,"subgraph_output","Composite Image",
                           {1450,410},{{"key","image"}});
    link(body,orange,"image",add,"a");
    link(body,teal,"image",add,"b");
    link(body,add,"image",out,"value");
    const auto feed=[&](const char* key,NodeId target,const char* pin,int x,int y) {
        const auto source=input(body,key,std::string("Control: ")+key,
                                 {static_cast<float>(x),static_cast<float>(y)});
        link(body,source,"value",target,pin);
    };
    feed("orangePositions",orange,"positions",30,50);
    feed("tealPositions",teal,"positions",30,760);
    feed("count",orange,"count",30,210);
    feed("count",teal,"count",30,920);
    for (const auto& [key,pin,y]:std::array<std::tuple<const char*,const char*,int>,6>{{
            {"orangeRetention","retention",340},
            {"orangeOpacity","opacity",450},
            {"orangeSize","size",560},
            {"orangeR","r",670},
            {"orangeG","g",780},
            {"orangeB","b",890}}})
        feed(key,orange,pin,230,y);
    for (const auto& [key,pin,y]:std::array<std::tuple<const char*,const char*,int>,6>{{
            {"tealRetention","retention",1040},
            {"tealOpacity","opacity",1150},
            {"tealSize","size",1260},
            {"tealR","r",1370},
            {"tealG","g",1480},
            {"tealB","b",1590}}})
        feed(key,teal,pin,230,y);
    feed("orangeGain",add,"gainA",800,200);
    feed("tealGain",add,"gainB",800,320);
    feed("backgroundR",add,"backgroundR",800,700);
    feed("backgroundG",add,"backgroundG",800,820);
    feed("backgroundB",add,"backgroundB",800,940);
    return result;
}

} // namespace

const std::vector<SubgraphDefinition>& builtInSubgraphs() {
    static const std::vector<SubgraphDefinition> values{
        discreteReaction(), genericCellularAutomata(), floodFill(),
        skeletonization(), distanceFromNearestWhitePixel(), sdfGenerator(), lenia(),
        noiseDrivenParticles(), sandsTrailLayers()};
    return values;
}

} // namespace reaction
