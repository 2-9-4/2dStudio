#include "reaction/core/graph.hpp"

#include <algorithm>
#include <set>
#include <functional>
#include <unordered_set>

namespace reaction {
namespace {

SubgraphKernelNode node(std::string key, std::string operation,
                        std::vector<std::string> inputs = {},
                        nlohmann::json properties = nlohmann::json::object(), Vec2 position = {},
                        std::string label = {}) {
    if (!properties.is_object()) properties = nlohmann::json::object();
    if (!label.empty()) properties["label"] = std::move(label);
    return {std::move(key), std::move(operation), std::move(inputs),
            std::move(properties), position};
}

SubgraphDefinition discreteReaction() {
    using Control = ParameterDescriptor::Control;
    SubgraphDefinition result;
    result.id = "builtin.reaction_diffusion.discrete";
    result.name = "Reaction Diffusion (Discrete)";
    result.category = "Simulation";
    result.execution = SubgraphExecution::Simulation;
    result.immutable = true;
    result.interface = {
        {"feedMultiplier", "Feed Multiplier", SubgraphInterfaceKind::Input,
         ValueType::AnyNumeric, true, 1.0F},
        {"killMultiplier", "Kill Multiplier", SubgraphInterfaceKind::Input,
         ValueType::AnyNumeric, true, 1.0F},
        {"seed", "Seed", SubgraphInterfaceKind::Input, ValueType::Image2D, true},
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
        {"image", "Image", SubgraphInterfaceKind::Output, ValueType::Image2D},
        {"a", "A", SubgraphInterfaceKind::Output, ValueType::Image2D},
        {"b", "B", SubgraphInterfaceKind::Output, ValueType::Image2D},
    };

    // This is an editable expression DAG. The simulation compiler lowers the nodes
    // below into one initialization shader and one fused update shader.
    result.kernel = {
        node("uv", "uv", {}, {}, {0, 0}),
        node("center", "constant2", {}, {{"x", .5}, {"y", .5}}, {0, 120}),
        node("radius", "length", {"uv", "center"}, {{"subtract", true}}, {180, 60}),
        node("seedRadius", "constant", {}, {{"value", .075}}, {180, 160}),
        node("defaultSeed", "step", {"radius", "seedRadius"}, {{"reverse", true}}, {360, 100}),
        node("seedInput", "interface", {}, {{"key", "seed"}, {"default", 0.0}}, {360, 200}),
        node("hasSeed", "connected", {}, {{"key", "seed"}}, {360, 280}),
        node("initialSeed", "select", {"hasSeed", "seedInput", "defaultSeed"}, {}, {540, 160}),
        node("halfSeed", "multiply", {"initialSeed"}, {{"value", .5}}, {720, 100}),
        node("initialA", "subtract", {"halfSeed"}, {{"from", 1.0}}, {900, 80}),
        node("initial", "pack2", {"initialA", "initialSeed"}, {{"role", "initial"}}, {1080, 140}, "Initial State"),
        node("state", "previous_state", {}, {}, {0, 440}),
        node("a0", "swizzle", {"state"}, {{"channel", 0}}, {180, 380}, "Current A"),
        node("b0", "swizzle", {"state"}, {{"channel", 1}}, {180, 500}, "Current B"),
        node("lap", "laplacian", {"state"}, {{"scale", "structureScale"}}, {180, 620}, "State Laplacian"),
        node("lapA", "swizzle", {"lap"}, {{"channel", 0}}, {360, 600}),
        node("lapB", "swizzle", {"lap"}, {{"channel", 1}}, {360, 700}),
        node("bb", "multiply", {"b0", "b0"}, {}, {360, 420}, "B Squared"),
        node("reaction", "multiply", {"a0", "bb"}, {}, {540, 420}),
        node("feedMask", "interface", {}, {{"key", "feedMultiplier"}, {"default", 1.0}}, {360, 800}),
        node("killMask", "interface", {}, {{"key", "killMultiplier"}, {"default", 1.0}}, {360, 880}),
        node("feedValue", "interface", {}, {{"key", "feed"}}, {540, 800}),
        node("killValue", "interface", {}, {{"key", "kill"}}, {540, 880}),
        node("f", "multiply", {"feedValue", "feedMask"}, {}, {720, 800}, "Effective Feed"),
        node("k", "multiply", {"killValue", "killMask"}, {}, {720, 880}, "Effective Kill"),
        node("diffAValue", "interface", {}, {{"key", "diffA"}}, {540, 600}),
        node("diffBValue", "interface", {}, {{"key", "diffB"}}, {540, 700}),
        node("diffusionA", "multiply", {"diffAValue", "lapA"}, {}, {720, 580}),
        node("diffusionB", "multiply", {"diffBValue", "lapB"}, {}, {720, 680}),
        node("oneMinusA", "subtract", {"a0"}, {{"from", 1.0}}, {720, 360}),
        node("feedTerm", "multiply", {"f", "oneMinusA"}, {}, {900, 360}),
        node("aDelta0", "subtract", {"diffusionA", "reaction"}, {}, {900, 540}),
        node("aDelta", "add", {"aDelta0", "feedTerm"}, {}, {1080, 500}),
        node("kf", "add", {"k", "f"}, {}, {900, 820}, "Feed + Kill"),
        node("decay", "multiply", {"kf", "b0"}, {}, {1080, 780}),
        node("bDelta0", "add", {"diffusionB", "reaction"}, {}, {900, 680}),
        node("bDelta", "subtract", {"bDelta0", "decay"}, {}, {1260, 720}),
        node("dtValue", "interface", {}, {{"key", "dt"}}, {1080, 900}),
        node("aStep", "multiply", {"aDelta", "dtValue"}, {}, {1260, 480}),
        node("bStep", "multiply", {"bDelta", "dtValue"}, {}, {1440, 700}),
        node("aNext0", "add", {"a0", "aStep"}, {}, {1440, 460}),
        node("bNext0", "add", {"b0", "bStep"}, {}, {1620, 680}),
        node("aNext", "clamp01", {"aNext0"}, {}, {1620, 460}),
        node("bNext", "clamp01", {"bNext0"}, {}, {1800, 680}),
        node("next", "pack2", {"aNext", "bNext"}, {{"role", "next"}}, {1980, 560}, "Next State"),
        node("outputImage", "output", {"bNext"}, {{"key", "image"}}, {2160, 500}),
        node("outputA", "output", {"aNext"}, {{"key", "a"}}, {2160, 580}),
        node("outputB", "output", {"bNext"}, {{"key", "b"}}, {2160, 660}),
    };
    return result;
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
            result.sockets.push_back({item.key, item.label, item.type,
                                      SocketDirection::Input, item.optional});
        } else if (item.kind == SubgraphInterfaceKind::Output) {
            result.sockets.push_back({item.key, item.label, item.type,
                                      SocketDirection::Output});
        } else {
            result.parameters.push_back({item.key, item.label, item.defaultValue,
                                         item.minimum, item.maximum, item.control});
        }
    }
    result.timeDependent = definition.execution == SubgraphExecution::Simulation;
    result.stateful = definition.execution == SubgraphExecution::Simulation;
    return result;
}

const NodeDescriptor* resolveDescriptor(const Graph& graph, const NodeRecord& node,
                                        const NodeRegistry& registry, NodeDescriptor& storage) {
    if (node.type != "subgraph") return registry.descriptor(node.type);
    const auto* definition = resolveSubgraph(graph, node.subgraphId);
    if (!definition) return nullptr;
    storage = describeSubgraph(*definition);
    return &storage;
}

std::vector<std::string> validateSubgraph(const SubgraphDefinition& definition) {
    std::vector<std::string> errors;
    if (definition.id.empty()) errors.push_back("Subgraph has an empty id");
    if (definition.name.empty()) errors.push_back("Subgraph has an empty name");
    if (definition.execution == SubgraphExecution::Pipeline)
        errors.push_back("Pipeline subgraphs are not supported in this milestone");
    std::unordered_set<std::string> keys;
    int outputCount = 0;
    for (const auto& item : definition.interface) {
        if (item.key.empty() || !keys.insert(item.key).second)
            errors.push_back("Subgraph interface keys must be non-empty and unique");
        if (item.kind == SubgraphInterfaceKind::Slider &&
            (item.minimum > item.maximum || item.defaultValue < item.minimum ||
             item.defaultValue > item.maximum))
            errors.push_back("Subgraph slider '" + item.label + "' has an invalid range");
        if (item.kind == SubgraphInterfaceKind::Output && item.type != ValueType::Image2D)
            errors.push_back("Subgraph outputs must be Image2D");
        if (item.kind == SubgraphInterfaceKind::Output) ++outputCount;
    }
    if (definition.execution == SubgraphExecution::Simulation && (outputCount < 1 || outputCount > 3))
        errors.push_back("Simulation subgraphs support between one and three outputs");
    std::unordered_set<std::string> kernelKeys;
    int initial = 0, next = 0;
    for (const auto& item : definition.kernel) {
        if (item.key.empty() || !kernelKeys.insert(item.key).second)
            errors.push_back("Subgraph kernel keys must be non-empty and unique");
        const auto role = item.properties.is_object() ? item.properties.value("role", "") : "";
        if (role == std::string_view("initial")) ++initial;
        if (role == std::string_view("next")) ++next;
        if (item.operation == "subgraph") errors.push_back("Nested subgraphs are not supported");
        if ((item.operation == "interface" || item.operation == "connected") &&
            (!item.properties.is_object() ||
             !keys.contains(item.properties.value("key", std::string{}))))
            errors.push_back("Kernel node '" + item.key + "' references a missing interface item");
        if (item.operation == "connected" && item.properties.is_object()) {
            const auto interfaceKey = item.properties.value("key", std::string{});
            const auto interface = std::ranges::find(definition.interface, interfaceKey,
                                                     &SubgraphInterfaceItem::key);
            if (interface != definition.interface.end() && interface->kind != SubgraphInterfaceKind::Input)
                errors.push_back("Connected nodes must reference an input socket");
        }
        const auto arity = item.inputs.size();
        const auto exactly = [&](std::size_t value) {
            if (arity != value) errors.push_back("Kernel node '" + item.key + "' has the wrong input count");
        };
        if (item.operation == "uv" || item.operation == "constant" || item.operation == "constant2" ||
            item.operation == "previous_state" || item.operation == "interface" || item.operation == "connected") exactly(0);
        else if (item.operation == "abs" || item.operation == "sin" || item.operation == "cos" ||
                 item.operation == "clamp01" || item.operation == "swizzle" || item.operation == "laplacian" ||
                 item.operation == "output") exactly(1);
        else if (item.operation == "add" || item.operation == "divide" || item.operation == "min" ||
                 item.operation == "max" || item.operation == "pow" || item.operation == "pack2" ||
                 item.operation == "step") exactly(2);
        else if (item.operation == "multiply") {
            if (arity != 2 && !(arity == 1 && item.properties.is_object() && item.properties.contains("value")))
                errors.push_back("Kernel node '" + item.key + "' has the wrong input count");
        } else if (item.operation == "subtract") {
            if (arity != 2 && !(arity == 1 && item.properties.is_object() && item.properties.contains("from")))
                errors.push_back("Kernel node '" + item.key + "' has the wrong input count");
        } else if (item.operation == "length") {
            if (arity != 1 && arity != 2) errors.push_back("Kernel node '" + item.key + "' has the wrong input count");
        } else if (item.operation == "select" || item.operation == "clamp") exactly(3);
        else if (item.operation == "remap") exactly(5);
        else if (item.operation != "subgraph") errors.push_back("Unsupported kernel operation '" + item.operation + "'");
    }
    std::unordered_map<std::string, int> visit;
    std::function<void(const std::string&)> walk = [&](const std::string& key) {
        if (visit[key] == 1) { errors.push_back("Subgraph kernel contains a cycle"); return; }
        if (visit[key] == 2) return;
        visit[key] = 1;
        const auto it = std::ranges::find(definition.kernel, key, &SubgraphKernelNode::key);
        if (it != definition.kernel.end()) for (const auto& input : it->inputs) walk(input);
        visit[key] = 2;
    };
    for (const auto& item : definition.kernel) walk(item.key);
    std::unordered_map<std::string, int> outputMappings;
    const SubgraphKernelNode* nextNode = nullptr;
    for (const auto& item : definition.kernel) {
        const auto role = item.properties.is_object() ? item.properties.value("role", "") : "";
        if (role == std::string_view("next")) nextNode = &item;
    }
    for (const auto& item : definition.kernel)
        if (item.operation == "output") {
            ++outputMappings[item.properties.is_object()
                ? item.properties.value("key", std::string{}) : std::string{}];
            if (!nextNode || item.inputs.size() != 1 ||
                std::ranges::find(nextNode->inputs, item.inputs.front()) == nextNode->inputs.end())
                errors.push_back("Simulation outputs must expose a next-state channel");
        }
    for (const auto& item : definition.interface)
        if (item.kind == SubgraphInterfaceKind::Output && outputMappings[item.key] != 1)
            errors.push_back("Output '" + item.label + "' needs exactly one kernel endpoint");
    for (const auto& item : definition.kernel) {
        for (const auto& input : item.inputs) {
            if (!kernelKeys.contains(input))
                errors.push_back("Kernel node '" + item.key + "' references missing input '" + input + "'");
        }
    }
    enum class KernelType { Invalid, Float, Vec2 };
    std::unordered_map<std::string, KernelType> types;
    std::unordered_set<std::string> typing;
    std::function<KernelType(const std::string&)> infer = [&](const std::string& key) -> KernelType {
        if (const auto known = types.find(key); known != types.end()) return known->second;
        if (!typing.insert(key).second) return KernelType::Invalid;
        const auto found = std::ranges::find(definition.kernel, key, &SubgraphKernelNode::key);
        if (found == definition.kernel.end()) return KernelType::Invalid;
        const auto& item = *found;
        KernelType result = KernelType::Float;
        if (item.operation == "uv" || item.operation == "constant2" || item.operation == "previous_state")
            result = KernelType::Vec2;
        else if (item.operation == "pack2") {
            if (item.inputs.size() != 2 || infer(item.inputs[0]) != KernelType::Float ||
                infer(item.inputs[1]) != KernelType::Float) result = KernelType::Invalid;
            else result = KernelType::Vec2;
        } else if (item.operation == "swizzle") {
            result = item.inputs.size() == 1 && infer(item.inputs[0]) == KernelType::Vec2
                ? KernelType::Float : KernelType::Invalid;
        } else if (item.operation == "laplacian") {
            result = item.inputs.size() == 1 && infer(item.inputs[0]) == KernelType::Vec2
                ? KernelType::Vec2 : KernelType::Invalid;
        } else if (item.operation == "length") {
            result = !item.inputs.empty() && infer(item.inputs[0]) == KernelType::Vec2
                ? KernelType::Float : KernelType::Invalid;
            if (item.inputs.size() == 2 && infer(item.inputs[1]) != KernelType::Vec2)
                result = KernelType::Invalid;
        } else if (item.operation == "select" && item.inputs.size() == 3) {
            const auto yes = infer(item.inputs[1]), no = infer(item.inputs[2]);
            result = infer(item.inputs[0]) == KernelType::Float && yes == no ? yes : KernelType::Invalid;
        } else if (item.operation == "output") {
            result = item.inputs.size() == 1 ? infer(item.inputs[0]) : KernelType::Invalid;
            if (result != KernelType::Float) result = KernelType::Invalid;
        } else if (!item.inputs.empty()) {
            result = KernelType::Float;
            for (const auto& input : item.inputs) {
                const auto inputType = infer(input);
                if (inputType == KernelType::Invalid) result = KernelType::Invalid;
                else if (inputType == KernelType::Vec2) result = KernelType::Vec2;
            }
        }
        typing.erase(key); types[key] = result; return result;
    };
    for (const auto& item : definition.kernel)
        if (infer(item.key) == KernelType::Invalid)
            errors.push_back("Kernel node '" + item.key + "' has incompatible input types");
    for (const auto& item : definition.kernel) {
        const auto role = item.properties.is_object() ? item.properties.value("role", "") : "";
        if ((role == std::string_view("initial") || role == std::string_view("next")) &&
            infer(item.key) != KernelType::Vec2)
            errors.push_back("Simulation state endpoints must produce a two-channel value");
    }
    if (definition.execution == SubgraphExecution::Simulation && (initial != 1 || next != 1))
        errors.push_back("A simulation subgraph needs exactly one initial and next state endpoint");
    return errors;
}

} // namespace reaction
