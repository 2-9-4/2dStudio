#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "reaction/core/motion_primitives.hpp"
#include "reaction/gpu/shader_ir.hpp"
#include <memory>
#include <string>

namespace reaction {
namespace {

// No hidden collision logic: these are independent, shader-lowerable
// operations. The graph decides how to combine or repeat them.
class BallisticAdvance final : public node_support::ParameterNode {
public:
    const NodeDescriptor& descriptor() const override {
        static const auto d=ballisticAdvanceDescriptor();return d;
    }
    bool lowerShader(ShaderLoweringContext& c) const override {
        const auto p=c.vector("position","position",0.5F);
        const auto v=c.vector("velocity","velocity",0.0F);
        const auto a=c.vector("acceleration","acceleration",0.0F);
        const auto t=c.scalar("time","time",1.0F/600.0F);
        const std::string dt="max(0.0,"+t.name+")";
        const std::string pos="("+p.name+"+"+v.name+"*"+dt+"+0.5*"+a.name+
                              "*"+dt+"*"+dt+")";
        const std::string vel="("+v.name+"+"+a.name+"*"+dt+")";
        (void)c.emitTyped(pos,ShaderValueType::Vec2,"position");
        (void)c.emitTyped(vel,ShaderValueType::Vec2,"velocity");
        return true;
    }
};

class SdfSegmentCrossing final : public node_support::ParameterNode {
public:
    const NodeDescriptor& descriptor() const override {
        static const auto d=sdfSegmentCrossingDescriptor();return d;
    }
    bool lowerShader(ShaderLoweringContext& c) const override {
        const auto sdf=c.inputAt("source","clamp(q,vec2(0.0),vec2(1.0))",
                                 "source",0.0F);
        const auto start=c.vector("start","start",0.5F);
        const auto end=c.vector("end","end",0.5F);
        const auto radius=c.scalar("offset","offset",0.0F);
        std::string code=R"GLSL(
float sdfSegmentSample(vec2 q) { return SDF_EXPRESSION; }
vec2 sdfSegmentSolve(vec2 a,vec2 b,float offset) {
    float d0=sdfSegmentSample(a)+offset;
    float d1=sdfSegmentSample(b)+offset;
    // The caller owns recovery from an already-penetrating initial state.
    // For an interior start and interior end there is no outward crossing.
    if (d0>0.0 || d1<=0.0) return vec2(1.0,0.0);
    float lo=0.0,hi=1.0;
    for(int k=0;k<19;k++) {
        float mid=0.5*(lo+hi);
        if(sdfSegmentSample(mix(a,b,mid))+offset>0.0)hi=mid;
        else lo=mid;
    }
    return vec2(0.5*(lo+hi),1.0);
}
)GLSL";
        const auto at=code.find("SDF_EXPRESSION");
        code.replace(at,std::string("SDF_EXPRESSION").size(),sdf.name);
        const auto fn=c.helper("sdfSegment",std::move(code));
        const std::string call=fn+"Solve("+start.name+","+end.name+","+
                               radius.name+")";
        (void)c.emitTyped("("+call+").x",ShaderValueType::Scalar,"fraction");
        (void)c.emitTyped("("+call+").y",ShaderValueType::Scalar,"hit");
        return true;
    }
};

class SdfSurfaceNormal final : public node_support::ParameterNode {
public:
    const NodeDescriptor& descriptor() const override {
        static const auto d=sdfSurfaceNormalDescriptor();return d;
    }
    bool lowerShader(ShaderLoweringContext& c) const override {
        const auto sdf=c.inputAt("source","clamp(q,vec2(0.0),vec2(1.0))",
                                 "source",0.0F);
        const auto p=c.vector("point","point",0.5F);
        const auto step=c.scalar("step","step",0.001F);
        std::string code=R"GLSL(
float sdfNormalSample(vec2 q) { return SDF_EXPRESSION; }
vec2 sdfNormalCompute(vec2 p,float stepSize) {
    float e=max(stepSize,0.000001);
    vec2 grad=vec2(
        sdfNormalSample(p+vec2(e,0.0))-sdfNormalSample(p-vec2(e,0.0)),
        sdfNormalSample(p+vec2(0.0,e))-sdfNormalSample(p-vec2(0.0,e)));
    float g2=dot(grad,grad);
    return g2>1e-16 ? grad*inversesqrt(g2) : vec2(0.0,-1.0);
}
)GLSL";
        const auto at=code.find("SDF_EXPRESSION");
        code.replace(at,std::string("SDF_EXPRESSION").size(),sdf.name);
        const auto fn=c.helper("sdfNormal",std::move(code));
        (void)c.emitTyped(fn+"Compute("+p.name+","+step.name+")",
                          ShaderValueType::Vec2,"normal");
        return true;
    }
};
} // namespace

void registerMotionPrimitives(NodeRegistry& registry) {
    registry.add(ballisticAdvanceDescriptor(),[]{return std::make_unique<BallisticAdvance>();});
    registry.add(sdfSegmentCrossingDescriptor(),[]{return std::make_unique<SdfSegmentCrossing>();});
    registry.add(sdfSurfaceNormalDescriptor(),[]{return std::make_unique<SdfSurfaceNormal>();});
}

} // namespace reaction
