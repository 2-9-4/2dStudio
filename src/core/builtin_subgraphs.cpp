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


// Each species has its own independent position/lifetime feedback. Both can
// sample the same external image; neither shares state or modifies that image.
SubgraphDefinition speciesFlow(bool tangent) {
    SubgraphDefinition result;
    result.id = tangent ? "builtin.particles.species_1_flow"
                        : "builtin.particles.species_2_flow";
    result.version = 1;
    result.name = tangent ? "Species 1 Flow (Tangent)"
                          : "Species 2 Flow (Sine/Cosine)";
    result.category = "Simulation";
    result.execution = SubgraphExecution::Simulation;
    result.immutable = true;
    result.stateSlots = {
        {"position", "Particle Position", ValueType::VectorField, true},
        {"life", "Remaining Life", ValueType::ScalarField},
    };
    using K = SubgraphInterfaceKind;
    result.interface = {
        {"field", "Flow Field Image", K::Input, ValueType::ScalarField},
        {"seed", "Random Seed", K::Input, ValueType::Float, true,
         tangent ? 14.0F : 314.0F, 0.0F, 100000.0F, Control::Integer},
        {"speed", "Particle Speed", K::Input, ValueType::Float,
         true, 1.0F, 0.0F, 10.0F},
        {"horizontalMin", "Horizontal Jitter Min", K::Input, ValueType::Float,
         true, tangent ? 1.0F : 0.1F, 0.0F, 10.0F},
        {"horizontalMax", "Horizontal Jitter Max", K::Input, ValueType::Float,
         true, tangent ? 3.0F : 1.0F, 0.0F, 10.0F},
        {"angleBins", "Direction Quantization", K::Input, ValueType::Float,
         true, 1200.0F, 8.0F, 4096.0F, Control::Integer},
        {"lifeRate", "Life Decay", K::Input, ValueType::Float,
         true, 0.1F, 0.001F, 1.0F},
    };
    if (tangent)
        result.interface.push_back({"maxJump", "Tangent Jump Limit", K::Input,
                                    ValueType::Float, true, 64.0F, 1.0F, 512.0F});
    result.interface.insert(result.interface.end(), {
        {"reset", "Reset", K::Input, ValueType::Float,
         true, 0.0F, 0.0F, 1.0F, Control::Boolean},
        {"iterations", "Iterations per Frame", K::Input, ValueType::Float,
         true, 1.0F, 1.0F, 64.0F, Control::Integer},
        {"positions", "Particle Positions", K::Output, ValueType::VectorField},
    });
    auto& body = result.body;
    using M = MathOperation;
    using V = VectorMathOperation;
    using Wire = std::pair<NodeId,const char*>;
    const auto w = [](NodeId id,const char* pin="result")->Wire{return {id,pin};};
    const auto connect = [&](Wire from,NodeId to,const char* key){
        link(body,from.first,from.second,to,key);
    };
    const auto calc = [&](std::string label,M operation,int x,int y,
                          Wire a={},Wire b={},Wire c={},
                          nlohmann::json parameters=nlohmann::json::object())->Wire{
        const auto id=math(body,std::move(label),static_cast<int>(operation),
                           {static_cast<float>(x),static_cast<float>(y)},
                           std::move(parameters));
        if(a.first)connect(a,id,"a");
        if(b.first)connect(b,id,"b");
        if(c.first)connect(c,id,"c");
        return w(id);
    };
    const auto choose = [&](std::string label,Wire condition,Wire yes,Wire no,
                            int x,int y)->Wire{
        const auto id=addNode(body,"select",std::move(label),
                              {static_cast<float>(x),static_cast<float>(y)});
        connect(condition,id,"condition");
        connect(yes,id,"ifTrue");
        connect(no,id,"ifFalse");
        return w(id);
    };
    const auto split = [&](std::string label,Wire value,int x,int y)->NodeId{
        const auto id=addNode(body,"separate_vector",std::move(label),
                              {static_cast<float>(x),static_cast<float>(y)});
        connect(value,id,"value");
        return id;
    };
    const auto vectorFrom = [&](std::string label,Wire x,Wire y,
                                int cx,int cy)->Wire{
        const auto id=addNode(body,"combine_vector",std::move(label),
                              {static_cast<float>(cx),static_cast<float>(cy)});
        connect(x,id,"x");
        connect(y,id,"y");
        return w(id,"value");
    };

    // Initial positions/lifetime vary with this species' own independently
    // configurable seed. Hash randomness uses visible, reusable Hash nodes.
    const auto coords=addNode(body,"coordinates","Agent Slot Identity",{0,40});
    const auto slot=w(coords,"coordinates");
    const auto seed=input(body,"seed","Species Random Seed",{0,260});
    const auto seeded = [&](std::string label,float offset,int x,int y,
                            Wire salt={}) ->NodeId {
        const auto bias=calc(label+" Seed Offset",M::Add,x,y+110,
                             w(seed,"value"),{},{},{{"b",offset}});
        const auto id=addNode(body,"hash",std::move(label),
                              {static_cast<float>(x+230),static_cast<float>(y)});
        connect(slot,id,"position");
        connect(bias,id,"seed");
        if(salt.first)connect(salt,id,"salt");
        return id;
    };
    const auto spawn=seeded("Initial Position",14.0F,270,40);
    const auto startLife=seeded("Initial Lifetime",15.0F,270,350);
    const auto initPos=addNode(body,"simulation_initial_state","Initial Position",
                               {830,40},{{"slot",0.0F}});
    const auto initLife=addNode(body,"simulation_initial_state","Initial Life",
                                {830,350},{{"slot",1.0F}});
    connect(w(spawn,"vector"),initPos,"value");
    connect(w(startLife,"scalar"),initLife,"value");

    const auto prevPos=addNode(body,"simulation_previous_state","Previous Position",
                               {0,760},{{"slot",0.0F}});
    const auto prevLife=addNode(body,"simulation_previous_state","Previous Life",
                                {0,1020},{{"slot",1.0F}});
    const auto position=w(prevPos,"value");
    const auto life=w(prevLife,"value");

    // Field is a required INPUT; any scalar image source may supply it.
    const auto field=input(body,"field","External Flow Field",{0,1320});
    const auto sample=addNode(body,"texture_sample","Sample Field at Particle",
                              {280,1320},{{"addressMode",1.0F},{"sampling",1.0F}});
    connect(w(field,"value"),sample,"source");
    connect(position,sample,"coordinates");
    const auto bins=input(body,"angleBins","Angular Quantization",{280,1550});
    const auto scaled=calc("Field to Angle Bins",M::Multiply,540,1340,
                           w(sample,"sampled"),w(bins,"value"));
    const auto rounded=calc("Quantize Angle",M::Round,780,1340,scaled);
    const auto fraction=calc("Normalized Angle",M::Divide,1010,1340,
                             rounded,w(bins,"value"));
    const auto theta=calc("Angle in Radians",M::Multiply,1240,1340,
                          fraction,{},{},{{"b",6.28318530718F}});

    const auto frame=addNode(body,"simulation_step_info","Simulation Frame",{0,1830});
    const auto frameIndex=w(frame,"frameIndex");
    const auto velocityRandom=seeded("Velocity Jitter",1001.0F,260,1780,frameIndex);
    const auto decayA=seeded("Lifetime Random Pair 1",1002.0F,260,2100,frameIndex);
    const auto decayB=seeded("Lifetime Random Pair 2",1003.0F,260,2410,frameIndex);
    const auto respawn=seeded("Respawn Position",1004.0F,260,2720,frameIndex);
    const auto respawnLife=seeded("Respawn Life",1005.0F,260,3010,frameIndex);
    const auto jitterComponents=split("Velocity Random Channels",
                                       w(velocityRandom,"vector"),790,1780);
    const auto lifeChannelsA=split("Life Random Channels 1",
                                    w(decayA,"vector"),790,2100);
    const auto lifeChannelsB=split("Life Random Channels 2",
                                    w(decayB,"vector"),790,2410);

    const auto minimum=input(body,"horizontalMin","Horizontal Jitter Minimum",{1020,1760});
    const auto maximum=input(body,"horizontalMax","Horizontal Jitter Maximum",{1020,1890});
    const auto spread=calc("Horizontal Jitter Range",M::Subtract,1250,1840,
                           w(maximum,"value"),w(minimum,"value"));
    const auto jitter=calc("Randomized Horizontal Magnitude",M::Multiply,1490,1840,
                           w(jitterComponents,"x"),spread);
    const auto xMagnitude=calc("Horizontal Magnitude + Minimum",M::Add,1710,1840,
                               jitter,w(minimum,"value"));

    Wire flowX,flowY;
    if(tangent) {
        const auto raw=calc("Tangent Flow",M::Tangent,1510,1470,theta);
        const auto cap=input(body,"maxJump","Maximum Tangent Jump",{1510,1610});
        const auto minusCap=calc("Negative Jump Cap",M::Multiply,1740,1600,
                                 w(cap,"value"),{},{},{{"b",-1.0F}});
        const auto safeTan=calc("Clamp Tangent Singularities",M::Clamp,1970,1470,
                                raw,minusCap,w(cap,"value"));
        flowX=calc("Tangent Horizontal Motion",M::Multiply,2210,1780,
                    safeTan,xMagnitude);
        flowY=safeTan;
    } else {
        const auto sine=calc("Flow Sine",M::Sine,1510,1470,theta);
        const auto cosine=calc("Flow Cosine",M::Cosine,1510,1600,theta);
        const auto negSine=calc("Negative Sine",M::Multiply,1770,1470,
                                sine,{},{},{{"b",-1.0F}});
        flowX=calc("Sine Horizontal Motion",M::Multiply,2210,1780,
                    negSine,xMagnitude);
        flowY=calc("Negative Cosine",M::Multiply,1970,1600,
                    cosine,{},{},{{"b",-1.0F}});
    }
    const auto velocity=vectorFrom("Species Velocity",flowX,flowY,2440,1820);
    const auto speed=input(body,"speed","Particle Motion Speed",{2440,2100});
    const auto gain=vectorMath(body,"Apply Species Speed",V::Scale,{2700,1820});
    connect(velocity,gain,"a");
    connect(w(speed,"value"),gain,"scalar");
    const auto uv=addNode(body,"pixel_delta_to_uv","Velocity Pixels to UV",
                          {2930,1820});
    connect(w(gain),uv,"delta");
    const auto integrated=vectorMath(body,"Integrate Position",V::Add,{3170,1820});
    connect(position,integrated,"a");
    connect(w(uv,"uvDelta"),integrated,"b");

    const auto product1=calc("Lifetime Product 1",M::Multiply,1030,2200,
                             w(lifeChannelsA,"x"),w(lifeChannelsA,"y"));
    const auto product2=calc("Lifetime Product 2",M::Multiply,1030,2450,
                             w(lifeChannelsB,"x"),w(lifeChannelsB,"y"));
    const auto product=calc("Four Independent Random Factors",M::Multiply,1260,2330,
                            product1,product2);
    const auto rate=input(body,"lifeRate","Life Decay Rate",{1260,2600});
    const auto decay=calc("Lifetime Decrement",M::Multiply,1510,2330,
                          product,w(rate,"value"));
    const auto remaining=calc("Reduce Life",M::Subtract,1750,2330,life,decay);
    const auto dead=calc("Expired Life",M::Step,1990,2330,
                         remaining,{},{},{{"b",0.0F}});
    const auto nextPosition=choose("Respawn or Retain Position",dead,
                                   w(respawn,"vector"),w(integrated),3420,1820);
    const auto nextLife=choose("Respawn or Retain Lifetime",dead,
                               w(respawnLife,"scalar"),remaining,3420,2330);
    const auto storePosition=addNode(body,"simulation_next_state","Next Position",
                                     {3690,1820},{{"slot",0.0F}});
    const auto storeLife=addNode(body,"simulation_next_state","Next Lifetime",
                                 {3690,2330},{{"slot",1.0F}});
    connect(nextPosition,storePosition,"value");
    connect(nextLife,storeLife,"value");
    const auto output=addNode(body,"subgraph_output","Particle Positions",
                              {3930,1820},{{"key","positions"}});
    connect(w(storePosition,"value"),output,"value");
    return result;
}

} // namespace

const std::vector<SubgraphDefinition>& builtInSubgraphs() {
    static const std::vector<SubgraphDefinition> values{
        discreteReaction(), genericCellularAutomata(), floodFill(),
        skeletonization(), distanceFromNearestWhitePixel(), sdfGenerator(), lenia(),
        speciesFlow(true), speciesFlow(false)};
    return values;
}

} // namespace reaction
