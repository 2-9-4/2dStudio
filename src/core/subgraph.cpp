#include "reaction/core/graph.hpp"
#include "reaction/core/vector_math.hpp"

#include <algorithm>
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
    if (node.type == "simulation_initial_state") {
        storage = {"simulation_initial_state", 1, "Initial Simulation State", "Simulation",
                   {{"a", "Chemical A", SocketContract::Numeric, SocketDirection::Input},
                    {"b", "Chemical B", SocketContract::Numeric, SocketDirection::Input}}, {}};
        return &storage;
    }
    if (node.type == "simulation_next_state") {
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
    result.interface = {
        {"feedMultiplier", "Feed Multiplier", SubgraphInterfaceKind::Input,
         SocketContract::Numeric, true, 1.0F},
        {"killMultiplier", "Kill Multiplier", SubgraphInterfaceKind::Input,
         SocketContract::Numeric, true, 1.0F},
        {"seed", "Seed", SubgraphInterfaceKind::Input, ValueType::ScalarField, true},
        {"feed", "Feed", SubgraphInterfaceKind::Slider, ValueType::Float, false,
         .055F, 0.0F, .1F},
        {"kill", "Kill", SubgraphInterfaceKind::Slider, ValueType::Float, false,
         .062F, 0.0F, .1F},
        {"diffA", "Diffusion A", SubgraphInterfaceKind::Slider, ValueType::Float, false,
         1.0F, 0.0F, 2.0F},
        {"diffB", "Diffusion B", SubgraphInterfaceKind::Slider, ValueType::Float, false,
         .5F, 0.0F, 2.0F},
        {"structureScale", "Structure Scale", SubgraphInterfaceKind::Slider,
         ValueType::Float, false, 1.0F, .25F, 8.0F},
        {"dt", "Timestep", SubgraphInterfaceKind::Slider, ValueType::Float, false,
         1.0F, .01F, 2.0F},
        {"iterations", "Iterations", SubgraphInterfaceKind::Slider, ValueType::Float,
         false, 8.0F, 1.0F, 64.0F, Control::Integer, "iterations"},
        {"autoReset", "Auto Reset", SubgraphInterfaceKind::Slider, ValueType::Float,
         false, 0.0F, 0.0F, 1.0F, Control::Boolean, "autoReset"},
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
                                      {2200, 100});
    link(body, initialA, "result", initialState, "a");
    link(body, initialSeed, "result", initialState, "b");

    const auto previous = addNode(body, "simulation_previous_state", "Previous Simulation State",
                                  {0, 460});
    const auto channels = addNode(body, "simulation_channel", "Previous State Channels",
                                  {220, 660});
    link(body, previous, "state", channels, "state");
    const auto scale = input(body, "structureScale", "Structure Scale", {0, 720});
    const auto laplacian = addNode(body, "laplacian", "State Laplacian", {220, 400});
    const auto laplacianChannels = addNode(body, "simulation_channel", "Laplacian Channels",
                                           {440, 460});
    link(body, previous, "state", laplacian, "value");
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
    const auto nextState = addNode(body, "simulation_next_state", "Next Simulation State", {1980, 580});
    link(body, aNext, "result", nextState, "a");
    link(body, bNext, "result", nextState, "b");

    const auto outputImage = addNode(body, "subgraph_output", "Image Output", {2200, 500}, {{"key", "image"}});
    const auto outputA = addNode(body, "subgraph_output", "Chemical A Output", {2200, 620}, {{"key", "a"}});
    const auto outputB = addNode(body, "subgraph_output", "Chemical B Output", {2200, 740}, {{"key", "b"}});
    link(body, nextState, "b", outputImage, "value");
    link(body, nextState, "a", outputA, "value");
    link(body, nextState, "b", outputB, "value");

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
        if (item.kind == SubgraphInterfaceKind::Slider &&
            (item.minimum > item.maximum || item.defaultValue < item.minimum ||
             item.defaultValue > item.maximum))
            errors.push_back("Subgraph slider '" + item.label + "' has an invalid range");
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
        if (previousStates != 1) errors.push_back("A simulation subgraph needs exactly one previous-state node");
        if (initialStates != 1) errors.push_back("A simulation subgraph needs exactly one initial-state endpoint");
        if (nextStates != 1) errors.push_back("A simulation subgraph needs exactly one next-state endpoint");
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
        if (incoming != definition.body.links().end() && incoming->fromNode != nextStateId)
            errors.push_back("Simulation outputs must expose a Next Simulation State channel");
    }

    return errors;
}

} // namespace

const std::vector<SubgraphDefinition>& builtInSubgraphs() {
    static const std::vector<SubgraphDefinition> values{discreteReaction()};
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
            result.sockets.push_back({item.key, item.label, item.contract,
                                      SocketDirection::Input, item.optional});
        } else if (item.kind == SubgraphInterfaceKind::Output) {
            result.sockets.push_back({item.key, item.label, item.contract,
                                      SocketDirection::Output});
        } else {
            result.parameters.push_back({item.key, item.label, item.defaultValue,
                                         item.minimum, item.maximum, item.control});
            // Slider controls double as optional inputs, mirroring slider-style
            // parameters on built-in nodes. Sockets stay in interface order so the
            // runtime's input layout matches SimulationSubgraphNode::bindInterface.
            if (item.control == ParameterDescriptor::Control::Float ||
                item.control == ParameterDescriptor::Control::Integer) {
                result.sockets.push_back({item.key, item.label, ValueType::Float,
                                          SocketDirection::Input, true});
            }
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
