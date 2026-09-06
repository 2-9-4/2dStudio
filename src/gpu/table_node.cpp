#include "nodes_internal.hpp"

#include "reaction/gpu/shader_ir.hpp"
#include "node_support.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <vector>

namespace reaction {
namespace {

constexpr std::size_t kMaximumTableValues = 64;

std::vector<float> tableValues(const nlohmann::json& parameters) {
    std::vector<float> result;
    if (const auto found = parameters.find("values"); found != parameters.end() && found->is_array()) {
        result.reserve(std::min(found->size(), kMaximumTableValues));
        for (const auto& value : *found) {
            const float number = value.is_number() ? value.get<float>() : 0.0F;
            result.push_back(std::isfinite(number) ? number : 0.0F);
            if (result.size() == kMaximumTableValues) break;
        }
    }
    if (result.empty()) result.push_back(0.0F);
    return result;
}

std::string glslFloat(float value) {
    std::ostringstream result;
    result.imbue(std::locale::classic());
    result << std::setprecision(9) << value;
    std::string text = result.str();
    if (text.find_first_of(".eE") == std::string::npos) text += ".0";
    return text;
}

class TableNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"table", 1, "Table", "Math",
            {{"index", "Index", SocketContract::Numeric, SocketDirection::Input, true},
             {"result", "Result", SocketContract::Numeric, SocketDirection::Output}},
            {{"index", "Index", 0.0F, -100.0F, 100.0F},
             {"sampling", "Sampling", 0.0F, 0.0F, 1.0F, ParameterDescriptor::Control::Enum,
              {"Nearest", "Linear"}},
             {"address", "Address", 0.0F, 0.0F, 2.0F, ParameterDescriptor::Control::Enum,
              {"Clamp", "Repeat", "Mirror"}},
             {"indexUnits", "Index Units", 0.0F, 0.0F, 1.0F, ParameterDescriptor::Control::Enum,
              {"Direct Index", "Normalized 0-1"}}}};
        result.lowerable = true;
        return result;
    }

    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    std::string shaderVariantKey(const nlohmann::json& parameters) const override {
        std::string result;
        for (const auto value : tableValues(parameters)) result += glslFloat(value) + ',';
        return result + "sampling=" + std::to_string(std::clamp(static_cast<int>(
            node_support::parameter(parameters, "sampling", 0.0F)), 0, 1)) +
            ";address=" + std::to_string(std::clamp(static_cast<int>(
            node_support::parameter(parameters, "address", 0.0F)), 0, 2)) +
            ";indexUnits=" + std::to_string(std::clamp(static_cast<int>(
            node_support::parameter(parameters, "indexUnits", 0.0F)), 0, 1));
    }

    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto values = tableValues(parameters_);
        const int sampling = std::clamp(static_cast<int>(node_support::parameter(
            parameters_, "sampling", 0.0F)), 0, 1);
        const int address = std::clamp(static_cast<int>(node_support::parameter(
            parameters_, "address", 0.0F)), 0, 2);
        const int indexUnits = std::clamp(static_cast<int>(node_support::parameter(
            parameters_, "indexUnits", 0.0F)), 0, 1);

        std::string initializer;
        for (const auto value : values) {
            if (!initializer.empty()) initializer += ',';
            initializer += glslFloat(value);
        }
        const auto size = std::to_string(values.size());
        std::string source =
            "int tableAddress(int value){"
            "if(" + std::to_string(address) + "==0)return clamp(value,0," + size + "-1);"
            "if(" + std::to_string(address) + "==1){int m=value%" + size + ";return m<0?m+" + size + ":m;}"
            "if(" + size + "==1)return 0;int p=2*(" + size + "-1);int m=value%p;"
            "if(m<0)m+=p;return m<" + size + "?m:p-m;}\n"
            "float tableLookup(float index){const float values[" + size + "]=float[" + size + "]("
            + initializer + ");float position=index"
            + (indexUnits == 1 ? "*float(" + size + "-1)" : "") + ";"
            + (sampling == 0
                ? "return values[tableAddress(int(round(position)))];}"
                : "int lower=int(floor(position));return mix(values[tableAddress(lower)],values[tableAddress(lower+1)],fract(position));}");
        const auto helper = context.helper("table", std::move(source));
        const auto index = context.scalar("index", "index", 0.0F);
        (void)context.emitTyped(helper + "Lookup(" + index.name + ")", ShaderValueType::Scalar,
                                "result");
        return true;
    }
};

} // namespace

void registerTableNode(NodeRegistry& registry) {
    registry.add(TableNode::describe(), [] { return std::make_unique<TableNode>(); });
}

} // namespace reaction
