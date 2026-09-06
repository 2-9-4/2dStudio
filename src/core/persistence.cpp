#include "reaction/core/persistence.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace reaction {
namespace {

SocketContract socketContract(const std::string& value) {
    if (value == "vec2") return SocketContract::Vec2Only;
    if (value == "scalar_field") return SocketContract::ScalarFieldOnly;
    if (value == "vector_field") return SocketContract::VectorFieldOnly;
    if (value == "color_image" || value == "image2d")
        return SocketContract::ColorImageOnly;
    if (value == "numeric") return SocketContract::Numeric;
    if (value == "vector" || value == "vector_numeric")
        return SocketContract::VectorNumeric;
    if (value == "any_field") return SocketContract::AnyField;
    if (value == "any_image_value") return SocketContract::AnyImageValue;
    return SocketContract::FloatOnly;
}

std::optional<ValueType> persistedValueType(std::string_view value) {
    for (const auto type : {ValueType::Float, ValueType::Vec2, ValueType::ScalarField,
                            ValueType::VectorField, ValueType::ColorImage})
        if (toString(type) == value) return type;
    return std::nullopt;
}

nlohmann::json serializeNode(const NodeRecord& node) {
    auto value = node.missing && node.preservedJson.is_object()
        ? node.preservedJson
        : nlohmann::json{{"id", node.id}, {"type", node.type},
            {"typeVersion", node.typeVersion},
            {"position", {node.position.x, node.position.y}},
            {"parameters", node.parameters}};
    value["id"] = node.id;
    value["position"] = {node.position.x, node.position.y};
    if (!node.label.empty()) value["label"] = node.label;
    else value.erase("label");
    if (!node.subgraphId.empty()) value["subgraphId"] = node.subgraphId;
    return value;
}

nlohmann::json serializeLink(const LinkRecord& link) {
    return {{"id", link.id},
            {"from", {{"node", link.fromNode}, {"socket", link.fromSocket}}},
            {"to", {{"node", link.toNode}, {"socket", link.toSocket}}}};
}

NodeRecord deserializeNode(const nlohmann::json& value) {
    NodeRecord node;
    node.id = value.at("id").get<NodeId>();
    node.type = value.at("type").get<std::string>();
    node.subgraphId = value.value("subgraphId", std::string{});
    node.typeVersion = value.value("typeVersion", 1);
    node.label = value.value("label", std::string{});
    const auto& position = value.at("position");
    node.position = {position.at(0).get<float>(), position.at(1).get<float>()};
    // Older builds could save a null parameter block for a node created from a
    // plain search-menu entry.  Treat it as an empty object so loading remains
    // safe for all parameter widgets and runtime nodes.
    const auto parameters = value.value("parameters", nlohmann::json::object());
    node.parameters = parameters.is_object() ? parameters : nlohmann::json::object();
    return node;
}

LinkRecord deserializeLink(const nlohmann::json& value) {
    return {value.at("id").get<LinkId>(),
            value.at("from").at("node").get<NodeId>(),
            value.at("from").at("socket").get<std::string>(),
            value.at("to").at("node").get<NodeId>(),
            value.at("to").at("socket").get<std::string>()};
}

void deserializeBody(GraphBody& body, const nlohmann::json& nodes,
                     const nlohmann::json& links) {
    for (const auto& value : nodes) {
        body.nodes().push_back(deserializeNode(value));
        body.nextNodeId = std::max(body.nextNodeId, body.nodes().back().id + 1);
    }
    for (const auto& value : links) {
        body.links().push_back(deserializeLink(value));
        body.nextLinkId = std::max(body.nextLinkId, body.links().back().id + 1);
    }
}

void markLegacyPreviousStateWiring(SubgraphDefinition& definition) {
    for (auto& node : definition.body.nodes()) {
        if (node.type != "simulation_previous_state") continue;
        node.needsAttention = std::ranges::any_of(
            definition.body.links(), [&](const LinkRecord& link) {
                return link.fromNode == node.id &&
                       (link.fromSocket == "a" || link.fromSocket == "b");
            });
    }
}

struct LegacyKernelNode {
    std::string key;
    std::string operation;
    std::vector<std::string> inputs;
    nlohmann::json properties = nlohmann::json::object();
    Vec2 position;
};

struct LegacyEndpoint {
    NodeId node = 0;
    std::string socket;
    std::optional<float> constant;
};

using LegacyValue = std::vector<LegacyEndpoint>;

std::string humanize(std::string value) {
    std::string result;
    for (std::size_t index = 0; index < value.size(); ++index) {
        const unsigned char character = static_cast<unsigned char>(value[index]);
        if (value[index] == '_' || value[index] == '-') {
            if (!result.empty() && result.back() != ' ') result.push_back(' ');
        } else {
            if (index > 0 && std::isupper(character) && !result.empty() && result.back() != ' ')
                result.push_back(' ');
            result.push_back(value[index]);
        }
    }
    if (!result.empty()) result[0] = static_cast<char>(
        std::toupper(static_cast<unsigned char>(result[0])));
    return result;
}

// Format 2 stored a bespoke key/operation/input DAG. Convert it once on load to
// the same stable-ID node/link representation used by the root graph.
void migrateLegacyKernel(SubgraphDefinition& definition, const nlohmann::json& kernel) {
    std::vector<LegacyKernelNode> legacy;
    legacy.reserve(kernel.size());
    for (const auto& entry : kernel) {
        LegacyKernelNode node;
        node.key = entry.at("key").get<std::string>();
        node.operation = entry.at("operation").get<std::string>();
        node.inputs = entry.value("inputs", std::vector<std::string>{});
        node.properties = entry.value("properties", nlohmann::json::object());
        const auto position = entry.value("position", nlohmann::json::array({0, 0}));
        node.position = {position.at(0).get<float>(), position.at(1).get<float>()};
        legacy.push_back(std::move(node));
    }

    std::unordered_map<std::string, const LegacyKernelNode*> byKey;
    for (const auto& node : legacy) byKey[node.key] = &node;
    std::unordered_map<std::string, LegacyValue> converted;
    std::unordered_map<std::string, int> visiting;
    std::unordered_map<std::string, NodeId> sharedInputs;
    NodeId nextState = 0;
    std::unordered_map<std::string, std::string> nextChannels;

    const auto labelOf = [](const LegacyKernelNode& node) {
        const auto label = node.properties.is_object()
            ? node.properties.value("label", std::string{}) : std::string{};
        return label.empty() ? humanize(node.key) : label;
    };
    const auto add = [&](std::string type, std::string label, Vec2 position,
                         nlohmann::json parameters = nlohmann::json::object()) {
        const auto id = definition.body.addNode(std::move(type), position);
        auto* record = definition.body.findNode(id);
        record->label = std::move(label);
        record->parameters = std::move(parameters);
        return id;
    };
    const auto connect = [&](const LegacyEndpoint& from, NodeId to, std::string socket) {
        if (from.node != 0)
            definition.body.addLink(from.node, from.socket, to, std::move(socket));
    };
    const auto channel = [](const LegacyValue& value, std::size_t index) -> LegacyEndpoint {
        if (value.empty()) return {};
        return value[value.size() == 1 ? 0 : std::min(index, value.size() - 1)];
    };
    const auto makeMath = [&](std::string label, int operation, Vec2 position,
                              nlohmann::json parameters = nlohmann::json::object()) {
        parameters["operation"] = static_cast<float>(operation);
        return add("math", std::move(label), position, std::move(parameters));
    };
    const auto inputNode = [&](const std::string& key, Vec2 position, std::string label) {
        if (const auto found = sharedInputs.find(key); found != sharedInputs.end()) return found->second;
        const auto id = add("subgraph_input", std::move(label), position, {{"key", key}});
        sharedInputs.emplace(key, id);
        return id;
    };

    std::function<LegacyValue(const std::string&)> convert = [&](const std::string& key) -> LegacyValue {
        if (const auto known = converted.find(key); known != converted.end()) return known->second;
        const auto found = byKey.find(key);
        if (found == byKey.end() || visiting[key] == 1) return {};
        visiting[key] = 1;
        const auto& old = *found->second;
        const auto label = labelOf(old);
        const auto inputValue = [&](std::size_t index) {
            return index < old.inputs.size() ? convert(old.inputs[index]) : LegacyValue{};
        };
        LegacyValue result;

        if (old.operation == "constant") {
            const float value = old.properties.value("value", 0.0F);
            const auto id = add("float", label, old.position, {{"value", value}});
            result.push_back({id, "value", value});
        } else if (old.operation == "constant2") {
            const float x = old.properties.value("x", 0.0F);
            const float y = old.properties.value("y", 0.0F);
            const auto xId = add("float", label + " X", old.position, {{"value", x}});
            const auto yId = add("float", label + " Y", {old.position.x, old.position.y + 90},
                                 {{"value", y}});
            result = {{xId, "value", x}, {yId, "value", y}};
        } else if (old.operation == "uv") {
            const auto id = add("coordinates", label, old.position);
            result = {{id, "x", {}}, {id, "y", {}}};
        } else if (old.operation == "previous_state") {
            const auto id = add("simulation_previous_state", label, old.position);
            const auto split = add("simulation_channel", label + " Channels",
                                   {old.position.x + 180.0F, old.position.y});
            definition.body.addLink(id, "state", split, "state");
            result = {{split, "a", {}}, {split, "b", {}}};
        } else if (old.operation == "interface" || old.operation == "connected") {
            const auto interfaceKey = old.properties.value("key", std::string{});
            const auto id = inputNode(interfaceKey, old.position, label);
            if (old.operation == "interface" && old.properties.contains("default"))
                definition.body.findNode(id)->parameters["default"] = old.properties.at("default");
            result.push_back({id, old.operation == "connected" ? "connected" : "value", {}});
        } else if (old.operation == "swizzle") {
            const auto value = inputValue(0);
            result.push_back(channel(value, old.properties.value("channel", 0) == 0 ? 0U : 1U));
        } else if (old.operation == "pack2") {
            const auto a = channel(inputValue(0), 0);
            const auto b = channel(inputValue(1), 0);
            const auto role = old.properties.value("role", std::string{});
            if (role == "initial") {
                const auto id = add("simulation_initial_state", label, old.position);
                connect(a, id, "a");
                connect(b, id, "b");
                result = {a, b};
            } else if (role == "next") {
                nextState = add("simulation_next_state", label, old.position);
                connect(a, nextState, "a");
                connect(b, nextState, "b");
                if (!old.inputs.empty()) nextChannels[old.inputs[0]] = "a";
                if (old.inputs.size() > 1) nextChannels[old.inputs[1]] = "b";
                result = {{nextState, "a", {}}, {nextState, "b", {}}};
            } else {
                result = {a, b};
            }
        } else if (old.operation == "laplacian") {
            const auto value = inputValue(0);
            const auto scaleKey = old.properties.value("scale", std::string("structureScale"));
            const auto scaleId = inputNode(scaleKey, {old.position.x - 140, old.position.y + 120},
                                           humanize(scaleKey));
            const std::size_t count = std::max<std::size_t>(value.size(), 1);
            for (std::size_t index = 0; index < count; ++index) {
                const auto id = add("laplacian", count == 1 ? label : label + (index == 0 ? " A" : " B"),
                                    {old.position.x, old.position.y + static_cast<float>(index) * 80.0F});
                connect(channel(value, index), id, "value");
                definition.body.addLink(scaleId, "value", id, "scale");
                result.push_back({id, "result", {}});
            }
        } else if (old.operation == "length") {
            const auto first = inputValue(0);
            const auto second = inputValue(1);
            LegacyValue deltas;
            const std::size_t count = std::max<std::size_t>(first.size(), 2);
            for (std::size_t index = 0; index < count; ++index) {
                auto component = channel(first, index);
                if (!second.empty() || old.properties.value("subtract", false)) {
                    const auto id = makeMath(label + (index == 0 ? " X" : " Y"), 1,
                                             {old.position.x, old.position.y + static_cast<float>(index) * 70.0F});
                    connect(component, id, "a");
                    connect(channel(second, index), id, "b");
                    component = {id, "result", {}};
                }
                const auto square = makeMath(label + (index == 0 ? " X Squared" : " Y Squared"),
                                             2, {old.position.x + 170, old.position.y + static_cast<float>(index) * 70.0F});
                connect(component, square, "a");
                connect(component, square, "b");
                deltas.push_back({square, "result", {}});
            }
            const auto sum = makeMath(label + " Squared", 0, {old.position.x + 340, old.position.y});
            connect(channel(deltas, 0), sum, "a");
            connect(channel(deltas, 1), sum, "b");
            const auto root = makeMath(label, 4, {old.position.x + 510, old.position.y}, {{"b", .5F}});
            definition.body.addLink(sum, "result", root, "a");
            result.push_back({root, "result", {}});
        } else if (old.operation == "step") {
            const auto value = inputValue(0);
            const auto edge = inputValue(1);
            const bool reverse = old.properties.value("reverse", false);
            const std::size_t count = std::max<std::size_t>({value.size(), edge.size(), 1});
            for (std::size_t index = 0; index < count; ++index) {
                const auto delta = makeMath(label + " Comparison", 1,
                    {old.position.x, old.position.y + static_cast<float>(index) * 100.0F});
                connect(channel(reverse ? edge : value, index), delta, "a");
                connect(channel(reverse ? value : edge, index), delta, "b");
                const auto threshold = add("threshold", label,
                    {old.position.x + 170, old.position.y + static_cast<float>(index) * 100.0F},
                    {{"threshold", 0.0F}});
                definition.body.addLink(delta, "result", threshold, "value");
                result.push_back({threshold, "result", {}});
            }
        } else if (old.operation == "select") {
            const auto condition = channel(inputValue(0), 0);
            const auto yes = inputValue(1);
            const auto no = inputValue(2);
            const std::size_t count = std::max<std::size_t>({yes.size(), no.size(), 1});
            for (std::size_t index = 0; index < count; ++index) {
                const auto select = add("select", count == 1 ? label :
                    label + (index == 0 ? " A" : " B"),
                    {old.position.x, old.position.y + static_cast<float>(index) * 100.0F});
                connect(condition, select, "condition");
                connect(channel(yes, index), select, "ifTrue");
                connect(channel(no, index), select, "ifFalse");
                result.push_back({select, "result", {}});
            }
        } else if (old.operation == "remap") {
            std::vector<LegacyValue> inputs;
            for (const auto& inputKey : old.inputs) inputs.push_back(convert(inputKey));
            if (inputs.size() != 5)
                throw std::runtime_error("Legacy remap node '" + old.key + "' requires five inputs");
            std::size_t count = 1;
            for (const auto& value : inputs) count = std::max(count, value.size());
            for (std::size_t component = 0; component < count; ++component) {
                const float y = old.position.y + static_cast<float>(component) * 140.0F;
                const auto valueMinusMin = makeMath(label + " Value Minus Input Minimum", 1,
                                                     {old.position.x, y});
                connect(channel(inputs[0], component), valueMinusMin, "a");
                connect(channel(inputs[1], component), valueMinusMin, "b");
                const auto inputRange = makeMath(label + " Input Range", 1,
                                                 {old.position.x, y + 55.0F});
                connect(channel(inputs[2], component), inputRange, "a");
                connect(channel(inputs[1], component), inputRange, "b");
                const auto safeRange = makeMath(label + " Safe Input Range", 6,
                                                 {old.position.x + 170, y + 55.0F},
                                                 {{"b", 1.0e-6F}});
                definition.body.addLink(inputRange, "result", safeRange, "a");
                const auto normalized = makeMath(label + " Normalized", 3,
                                                  {old.position.x + 340, y});
                definition.body.addLink(valueMinusMin, "result", normalized, "a");
                definition.body.addLink(safeRange, "result", normalized, "b");
                const auto clamped = makeMath(label + " Clamped", 10,
                                               {old.position.x + 510, y},
                                               {{"b", 0.0F}, {"c", 1.0F}});
                definition.body.addLink(normalized, "result", clamped, "a");
                const auto outputRange = makeMath(label + " Output Range", 1,
                                                  {old.position.x + 340, y + 75.0F});
                connect(channel(inputs[4], component), outputRange, "a");
                connect(channel(inputs[3], component), outputRange, "b");
                const auto scaled = makeMath(label + " Scaled Output Range", 2,
                                              {old.position.x + 680, y});
                definition.body.addLink(outputRange, "result", scaled, "a");
                definition.body.addLink(clamped, "result", scaled, "b");
                const auto remapped = makeMath(count == 1 ? label :
                    label + (component == 0 ? " A" : " B"), 0,
                    {old.position.x + 850, y});
                connect(channel(inputs[3], component), remapped, "a");
                definition.body.addLink(scaled, "result", remapped, "b");
                result.push_back({remapped, "result", {}});
            }
        } else if (old.operation == "pow") {
            const auto base = inputValue(0);
            const auto exponent = inputValue(1);
            const std::size_t count = std::max<std::size_t>({base.size(), exponent.size(), 1});
            for (std::size_t component = 0; component < count; ++component) {
                const float y = old.position.y + static_cast<float>(component) * 100.0F;
                const auto absolute = makeMath(label + " Absolute Base", 7,
                                               {old.position.x, y});
                connect(channel(base, component), absolute, "a");
                const auto power = makeMath(count == 1 ? label :
                    label + (component == 0 ? " A" : " B"), 4,
                    {old.position.x + 170, y});
                definition.body.addLink(absolute, "result", power, "a");
                connect(channel(exponent, component), power, "b");
                result.push_back({power, "result", {}});
            }
        } else if (old.operation == "output") {
            const auto interfaceKey = old.properties.value("key", std::string{});
            const auto id = add("subgraph_output", label, old.position, {{"key", interfaceKey}});
            LegacyEndpoint source;
            if (!old.inputs.empty()) {
                if (const auto next = nextChannels.find(old.inputs[0]); next != nextChannels.end())
                    source = {nextState, next->second, {}};
                else source = channel(inputValue(0), 0);
            }
            connect(source, id, "value");
            result.push_back(source);
        } else {
            static const std::unordered_map<std::string, int> operations{
                {"add", 0}, {"subtract", 1}, {"multiply", 2}, {"divide", 3},
                {"min", 5}, {"max", 6}, {"abs", 7}, {"sin", 8},
                {"cos", 9}, {"clamp01", 10}, {"clamp", 10}};
            const auto operation = operations.find(old.operation);
            if (operation != operations.end()) {
                std::vector<LegacyValue> inputs;
                for (const auto& inputKey : old.inputs) inputs.push_back(convert(inputKey));
                std::size_t count = 1;
                for (const auto& value : inputs) count = std::max(count, value.size());
                for (std::size_t component = 0; component < count; ++component) {
                    nlohmann::json parameters = old.properties;
                    if (old.operation == "clamp01") { parameters["b"] = 0.0F; parameters["c"] = 1.0F; }
                    if (old.operation == "subtract" && old.inputs.size() == 1 && old.properties.contains("from"))
                        parameters["a"] = old.properties.at("from");
                    if (old.operation == "multiply" && old.inputs.size() == 1 && old.properties.contains("value"))
                        parameters["b"] = old.properties.at("value");
                    const auto id = makeMath(count == 1 ? label : label + (component == 0 ? " A" : " B"),
                                             operation->second,
                                             {old.position.x, old.position.y + static_cast<float>(component) * 70.0F},
                                             std::move(parameters));
                    for (std::size_t index = 0; index < inputs.size() && index < 3; ++index) {
                        auto target = std::string(1, static_cast<char>('a' + index));
                        if (old.operation == "subtract" && inputs.size() == 1 && old.properties.contains("from"))
                            target = "b";
                        connect(channel(inputs[index], component), id, target);
                    }
                    result.push_back({id, "result", {}});
                }
            } else {
                const auto id = add(old.operation, label, old.position, old.properties);
                result.push_back({id, "result", {}});
            }
        }

        visiting[key] = 2;
        converted[key] = result;
        return result;
    };

    // Establish the state boundaries before converting outputs so output channels
    // can be mapped explicitly to Next State passthrough sockets.
    for (const auto& node : legacy)
        if (node.properties.value("role", std::string{}) == "initial") convert(node.key);
    for (const auto& node : legacy)
        if (node.properties.value("role", std::string{}) == "next") convert(node.key);
    for (const auto& node : legacy) convert(node.key);
}

nlohmann::json serializeSubgraph(const SubgraphDefinition& definition) {
    nlohmann::json interface = nlohmann::json::array();
    for (const auto& item : definition.interface) {
        const char* kind = item.kind == SubgraphInterfaceKind::Input ? "input" :
                           item.kind == SubgraphInterfaceKind::Slider ? "slider" : "output";
        const char* control = item.control == ParameterDescriptor::Control::Integer ? "integer" :
                              item.control == ParameterDescriptor::Control::Boolean ? "boolean" : "float";
        interface.push_back({{"key", item.key}, {"label", item.label}, {"kind", kind},
            {"type", toString(item.contract)}, {"optional", item.optional},
            {"default", item.defaultValue}, {"minimum", item.minimum}, {"maximum", item.maximum},
            {"control", control}, {"role", item.role}});
    }
    nlohmann::json stateSlots = nlohmann::json::array();
    for (const auto& slot : definition.stateSlots)
        stateSlots.push_back({{"key", slot.key}, {"label", slot.label}, {"type", toString(slot.type)}});
    nlohmann::json nodes = nlohmann::json::array();
    for (const auto& node : definition.body.nodes()) nodes.push_back(serializeNode(node));
    nlohmann::json links = nlohmann::json::array();
    for (const auto& link : definition.body.links()) links.push_back(serializeLink(link));
    return {{"id", definition.id}, {"version", definition.version}, {"name", definition.name},
            {"category", definition.category},
            {"execution", definition.execution == SubgraphExecution::Simulation ? "simulation" : "pipeline"},
            {"interface", std::move(interface)}, {"stateSlots", std::move(stateSlots)}, {"nodes", std::move(nodes)},
            {"links", std::move(links)}};
}

SubgraphDefinition deserializeSubgraph(const nlohmann::json& value,
                                       const NodeRegistry& registry) {
    SubgraphDefinition result;
    result.id = value.at("id").get<std::string>();
    result.version = value.value("version", 1);
    result.name = value.at("name").get<std::string>();
    result.category = value.value("category", std::string("Subgraphs"));
    result.execution = value.value("execution", std::string("pipeline")) == "simulation"
        ? SubgraphExecution::Simulation : SubgraphExecution::Pipeline;
    for (const auto& entry : value.value("interface", nlohmann::json::array())) {
        SubgraphInterfaceItem item;
        item.key = entry.at("key").get<std::string>();
        item.label = entry.value("label", item.key);
        const auto kind = entry.value("kind", std::string("input"));
        item.kind = kind == "slider" ? SubgraphInterfaceKind::Slider :
                    kind == "output" ? SubgraphInterfaceKind::Output : SubgraphInterfaceKind::Input;
        const auto persistedType = entry.value("type", std::string("float"));
        item.contract = socketContract(persistedType);
        // Format 1-3 simulation definitions stored every field as image2d.
        // Known reaction channels are producer-defined scalars; ambiguous
        // interfaces deliberately retain the ColorImage fallback.
        if (persistedType == "image2d" && result.execution == SubgraphExecution::Simulation &&
            (item.key == "seed" || item.key == "image" || item.key == "a" || item.key == "b"))
            item.contract = SocketContract::ScalarFieldOnly;
        item.optional = entry.value("optional", false);
        item.defaultValue = entry.value("default", 0.0F);
        item.minimum = entry.value("minimum", 0.0F);
        item.maximum = entry.value("maximum", 1.0F);
        const auto control = entry.value("control", std::string("float"));
        item.control = control == "integer" ? ParameterDescriptor::Control::Integer :
                       control == "boolean" ? ParameterDescriptor::Control::Boolean :
                       ParameterDescriptor::Control::Float;
        item.role = entry.value("role", std::string{});
        result.interface.push_back(std::move(item));
    }
    for (const auto& entry : value.value("stateSlots", nlohmann::json::array())) {
        SimulationStateSlot slot;
        slot.key = entry.at("key").get<std::string>();
        slot.label = entry.value("label", slot.key);
        const auto type = entry.value("type", std::string("scalar_field"));
        slot.type = persistedValueType(type).value_or(ValueType::ScalarField);
        result.stateSlots.push_back(std::move(slot));
    }
    if (value.contains("nodes")) {
        const auto& nodes = value.at("nodes");
        deserializeBody(result.body, nodes, value.value("links", nlohmann::json::array()));
        markLegacyPreviousStateWiring(result);
        for (std::size_t index = 0; index < result.body.nodes().size(); ++index) {
            auto& node = result.body.nodes()[index];
            NodeDescriptor descriptorStorage;
            node.missing = resolveSubgraphBodyDescriptor(result, node, registry,
                                                         descriptorStorage) == nullptr;
            if (node.missing) node.preservedJson = nodes[index];
        }
    } else if (value.contains("kernel")) {
        migrateLegacyKernel(result, value.at("kernel"));
        for (auto& node : result.body.nodes()) {
            NodeDescriptor descriptorStorage;
            node.missing = resolveSubgraphBodyDescriptor(result, node, registry,
                                                         descriptorStorage) == nullptr;
        }
    }
    return result;
}

} // namespace

nlohmann::json serializeProject(const Graph& graph) {
    nlohmann::json nodes = nlohmann::json::array();
    for (const auto& node : graph.nodes()) nodes.push_back(serializeNode(node));

    nlohmann::json links = nlohmann::json::array();
    for (const auto& link : graph.links()) links.push_back(serializeLink(link));
    nlohmann::json subgraphs = nlohmann::json::array();
    for (const auto& definition : graph.subgraphs()) subgraphs.push_back(serializeSubgraph(definition));
    return {{"formatVersion", kProjectFormatVersion},
            {"project", {{"width", graph.settings.width},
                         {"height", graph.settings.height},
                         {"targetFps", graph.settings.targetFps}}},
            {"subgraphs", std::move(subgraphs)}, {"nodes", std::move(nodes)}, {"links", std::move(links)},
            {"activeOutput", graph.activeOutput}};
}

Graph deserializeProject(const nlohmann::json& document, const NodeRegistry& registry) {
    const int format = document.value("formatVersion", 0);
    if (!document.is_object() || format < 1 || format > kProjectFormatVersion) {
        throw std::runtime_error("Unsupported or missing project formatVersion");
    }
    Graph graph;
    const auto& project = document.at("project");
    graph.settings.width = project.value("width", 1024);
    graph.settings.height = project.value("height", 1024);
    graph.settings.targetFps = project.value("targetFps", 60);
    if (graph.settings.width < 16 || graph.settings.height < 16 ||
        graph.settings.targetFps < 1 || graph.settings.targetFps > 240) {
        throw std::runtime_error("Project settings are out of range");
    }

    if (format >= 2) {
        for (const auto& value : document.value("subgraphs", nlohmann::json::array()))
            graph.subgraphs().push_back(deserializeSubgraph(value, registry));
    }
    for (const auto& value : document.at("nodes")) {
        auto node = deserializeNode(value);
        NodeDescriptor descriptorStorage;
        node.missing = resolveDescriptor(graph, node, registry, descriptorStorage) == nullptr;
        if (node.missing) node.preservedJson = value;
        graph.nodes().push_back(std::move(node));
        graph.nextNodeId = std::max(graph.nextNodeId, graph.nodes().back().id + 1);
    }
    for (const auto& value : document.at("links")) {
        auto link = deserializeLink(value);
        graph.links().push_back(std::move(link));
        graph.nextLinkId = std::max(graph.nextLinkId, graph.links().back().id + 1);
    }
    graph.activeOutput = document.value("activeOutput", NodeId{0});
    if (format < 4) {
        // Old Image2D links silently projected RGBA into scalar/vector node
        // implementations. Make that narrowing explicit during migration.
        const auto typed = graph.compile(registry);
        const auto oldLinks = graph.links();
        for (const auto& old : oldLinks) {
            const auto source = typed.socketType(old.fromNode, old.fromSocket);
            const auto targetNode = graph.findNode(old.toNode);
            NodeDescriptor storage;
            const auto* targetDescriptor = targetNode
                ? resolveDescriptor(graph, *targetNode, registry, storage) : nullptr;
            if (!source || *source != ValueType::ColorImage || !targetDescriptor) continue;
            const auto input = std::ranges::find_if(targetDescriptor->sockets,
                [&](const SocketDescriptor& candidate) {
                    return candidate.direction == SocketDirection::Input &&
                           candidate.key == old.toSocket;
                });
            if (input == targetDescriptor->sockets.end()) continue;
            const bool scalar = input->contract == SocketContract::Numeric ||
                                input->contract == SocketContract::ScalarFieldOnly;
            const bool vector = input->contract == SocketContract::VectorNumeric ||
                                input->contract == SocketContract::VectorFieldOnly;
            if (!scalar && !vector) continue;
            const auto* from = graph.findNode(old.fromNode);
            const auto* to = graph.findNode(old.toNode);
            const Vec2 position = from && to
                ? Vec2{(from->position.x + to->position.x) * 0.5F,
                       (from->position.y + to->position.y) * 0.5F} : Vec2{};
            const auto conversion = graph.addNode(scalar ? "color_r" : "color_rg",
                                                  position);
            const auto link = std::ranges::find(graph.links(), old.id, &LinkRecord::id);
            if (link == graph.links().end()) continue;
            link->toNode = conversion;
            link->toSocket = "color";
            graph.addLink(conversion, "value", old.toNode, old.toSocket);
        }
    }
    return graph;
}

void saveProjectAtomic(const Graph& graph, const std::filesystem::path& path) {
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot open temporary project file");
        output << serializeProject(graph).dump(2) << '\n';
        if (!output) throw std::runtime_error("Failed writing project file");
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
    }
    if (error) throw std::runtime_error("Cannot replace project file: " + error.message());
}

Graph loadProject(const std::filesystem::path& path, const NodeRegistry& registry) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open project file");
    nlohmann::json document;
    input >> document;
    return deserializeProject(document, registry);
}

} // namespace reaction
