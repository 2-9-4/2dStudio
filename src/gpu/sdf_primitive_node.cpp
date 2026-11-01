#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "reaction/core/node_builder.hpp"
#include "reaction/gpu/shader_ir.hpp"

#include <algorithm>
#include <memory>
#include <string>

namespace reaction {
namespace {

// Distances are signed normalized-canvas distances: negative inside, positive
// outside, zero on the boundary. ScalarField output can be transformed or
// combined with ordinary Math/Select nodes to construct other arenas.
class SdfPrimitiveNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe(){
        return NodeDescriptorBuilder{"sdf_primitive",1,"SDF Primitive","Generator"}
            .optionalInput("coordinates","Coordinates",SocketContract::VectorNumeric)
            .fieldDefault("coordinates")
            .optionalInput("center","Center",SocketContract::VectorNumeric)
            .output("distance","Signed Distance",ValueType::ScalarField)
            .enumParameter("shape","Shape",0,{"Circle","Box","Rounded Box","Diamond"})
            .floatParameter("radius","Circle Radius (UV)",0.44F,0.001F,2.0F)
            .floatParameter("halfWidth","Half Width (UV)",0.4F,0.001F,2.0F)
            .floatParameter("halfHeight","Half Height (UV)",0.32F,0.001F,2.0F)
            .floatParameter("rounding","Corner Radius (UV)",0.06F,0.0F,1.0F)
            .lowerable().producedField().build();
    }
    const NodeDescriptor& descriptor()const override{
        static const auto d=describe();return d;
    }
    std::string shaderVariantKey(const nlohmann::json& values) const override{
        return "shape="+std::to_string(std::clamp(int(node_support::parameter(
            values,"shape",0.0F)),0,3));
    }
    bool lowerShader(ShaderLoweringContext& context)const override{
        const auto p=context.vector("coordinates","coordinates",0.0F);
        const auto center=context.vector("center","center",0.5F);
        const auto radius=context.parameter("radius",0.44F);
        const auto w=context.parameter("halfWidth",0.4F);
        const auto h=context.parameter("halfHeight",0.32F);
        const auto rounding=context.parameter("rounding",0.06F);
        const std::string d="("+p.name+"-"+center.name+")";
        const std::string q="(abs("+d+")-vec2("+w.name+","+h.name+"))";
        std::string expression;
        switch(std::clamp(int(node_support::parameter(parameters_,"shape",0)),0,3)){
        case 0: expression="(length("+d+")-"+radius.name+")";break;
        case 1: expression="(length(max("+q+",vec2(0.0)))+min(max("+q+".x,"+q+".y),0.0))";break;
        case 2:{
            const std::string rr="clamp("+rounding.name+",0.0,min("+w.name+","+h.name+"))";
            const std::string qr="(abs("+d+")-vec2("+w.name+","+h.name+")+vec2("+rr+"))";
            expression="(length(max("+qr+",vec2(0.0)))+min(max("+qr+".x,"+qr+".y),0.0)-"+rr+")";
            break;
        }
        default: expression="((abs("+d+".x)/max("+w.name+",1e-5)+abs("+d+
                     ".y)/max("+h.name+",1e-5)-1.0)*min("+w.name+","+h.name+"))";
        }
        (void)context.emitTyped(expression,ShaderValueType::Scalar,"distance");
        return true;
    }
};

class SdfOutlineNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe(){
        return NodeDescriptorBuilder{"sdf_outline",1,"SDF Outline","Generator"}
            .input("distance","Signed Distance",SocketContract::Numeric)
            .output("lines","Outline Mask",SocketContract::Numeric)
            .floatParameter("width","Stroke Width (UV)",0.002F,0.00001F,0.1F)
            .typePolicy("lines",SocketDescriptor::TypePolicy::NumericPromotion,{"distance","width"})
            .lowerable().build();
    }
    const NodeDescriptor& descriptor()const override{
        static const auto d=describe();return d;
    }
    bool lowerShader(ShaderLoweringContext& context)const override{
        const auto distance=context.scalar("distance","distance",0.0F);
        const auto width=context.scalar("width","width",0.002F);
        (void)context.emitTyped("(1.0-smoothstep(max(0.0,("+width.name+")-fwidth("+
            distance.name+")),("+width.name+")+fwidth("+distance.name+
            "),abs("+distance.name+")))",ShaderValueType::Scalar,"lines");
        return true;
    }
};
}
void registerSdfPrimitiveNodes(NodeRegistry& registry){
    registry.add(SdfPrimitiveNode::describe(),[]{return std::make_unique<SdfPrimitiveNode>();});
    registry.add(SdfOutlineNode::describe(),[]{return std::make_unique<SdfOutlineNode>();});
}
} // namespace reaction
