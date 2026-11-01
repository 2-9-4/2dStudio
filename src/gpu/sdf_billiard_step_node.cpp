#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "reaction/core/node_builder.hpp"
#include "reaction/gpu/shader_ir.hpp"

#include <memory>
#include <string>

namespace reaction {
namespace {

// Reusable symplectic-free ballistic integrator with continuous SDF boundary
// detection. Motion is exact between impacts under constant acceleration;
// contacts use a signed-distance field (negative on the permitted side).
// All inputs/outputs are in normalized canvas units and seconds.
// No geometry, spawn policy, trail renderer, randomness or species mode is
// embedded in the motion law.
class SdfBilliardStepNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe(){
        return NodeDescriptorBuilder{"sdf_billiard_step",1,"SDF Billiard Step","Simulation"}
            .input("position","Previous Position",SocketContract::VectorNumeric)
            .input("velocity","Previous Velocity",SocketContract::VectorNumeric)
            .input("boundary","Boundary SDF (negative inside)",ValueType::ScalarField)
            .neighborhood("boundary")
            .optionalInput("acceleration","Acceleration (UV/s²)",SocketContract::VectorNumeric)
            .output("position","Next Position",SocketContract::VectorNumeric)
            .output("velocity","Next Velocity",SocketContract::VectorNumeric)
            .output("impacts","Impacts This Step",SocketContract::Numeric)
            .floatParameter("dt","Time Step (seconds)",1.0F/600.0F,0.00001F,0.05F)
            .floatParameter("radius","Particle Radius (UV)",0.0F,0.0F,0.1F)
            .floatParameter("restitution","Restitution",1.0F,0.0F,1.0F)
            .floatParameter("normalStep","SDF Normal Sample Step (UV)",0.001F,0.00001F,0.02F)
            .typePolicy("position",SocketDescriptor::TypePolicy::VectorPromotion,
                        {"position","velocity","acceleration"})
            .fieldIf("position",{"boundary","dt","radius","restitution","normalStep"})
            .typePolicy("velocity",SocketDescriptor::TypePolicy::VectorPromotion,
                        {"position","velocity","acceleration"})
            .fieldIf("velocity",{"boundary","dt","radius","restitution","normalStep"})
            .typePolicy("impacts",SocketDescriptor::TypePolicy::NumericPromotion,
                        {"dt","radius","restitution","normalStep"})
            .fieldIf("impacts",{"position","velocity","acceleration","boundary"})
            .lowerable().build();
    }
    const NodeDescriptor& descriptor()const override{
        static const auto d=describe();return d;
    }
    bool lowerShader(ShaderLoweringContext& context)const override{
        // The source is an arbitrary-coordinate image sampler. Sampling must
        // occur at the agent's position, NOT at the agent-slot UV.
        const auto source=context.inputAt("boundary","clamp(q,vec2(0.0),vec2(1.0))",
                                          "boundary",0.0F);
        const auto position=context.vector("position","position",0.5F);
        const auto velocity=context.vector("velocity","velocity",0.0F);
        const auto acceleration=context.vector("acceleration","acceleration",0.0F);
        const auto dt=context.scalar("dt","dt",1.0F/600.0F);
        const auto radius=context.scalar("radius","radius",0.0F);
        const auto restitution=context.scalar("restitution","restitution",1.0F);
        const auto normalStep=context.scalar("normalStep","normalStep",0.001F);

        const auto function=context.helper("sdfBilliard",R"GLSL(
float sdfBilliard(vec2 q){
    return BOUNDARY_EXPRESSION;
}
vec2 sdfBilliardNormal(vec2 q,float e){
    e=max(e,0.000001);
    float dx=sdfBilliard(q+vec2(e,0.0))-sdfBilliard(q-vec2(e,0.0));
    float dy=sdfBilliard(q+vec2(0.0,e))-sdfBilliard(q-vec2(0.0,e));
    vec2 grad=vec2(dx,dy);
    return dot(grad,grad)>1e-16?normalize(grad):vec2(0.0,-1.0);
}
struct sdfBilliardResult {
    vec2 position;
    vec2 velocity;
    float impacts;
};
sdfBilliardResult sdfBilliardAdvance(vec2 initialPosition,vec2 initialVelocity,
                                     vec2 acceleration,float dt,float radius,
                                     float restitution,float normalStep){
    sdfBilliardResult result;
    result.position=initialPosition;
    result.velocity=initialVelocity;
    result.impacts=0.0;
    float remaining=clamp(dt,0.0,0.05);
    radius=max(0.0,radius);
    restitution=clamp(restitution,0.0,1.0);
    // Correct an initial condition that is already outside. No random kick
    // and no artificial energy correction.
    float startDistance=sdfBilliard(result.position)+radius;
    if(startDistance>0.0){
        vec2 n=sdfBilliardNormal(result.position,normalStep);
        result.position-=n*(startDistance+2e-6);
        float normalVelocity=dot(result.velocity,n);
        if(normalVelocity>0.0)result.velocity-=(1.0+restitution)*normalVelocity*n;
    }
    for(int collision=0;collision<4;collision++){
        if(remaining<=1e-8)break;
        vec2 endPosition=result.position+result.velocity*remaining+
                         0.5*acceleration*remaining*remaining;
        if(sdfBilliard(endPosition)+radius<=0.0){
            result.position=endPosition;
            result.velocity+=acceleration*remaining;
            remaining=0.0;
            break;
        }
        // First boundary crossing in this fixed time step. Continuous
        // ballistic root-finding avoids overshoot and timestep bias.
        float lo=0.0,hi=remaining;
        for(int i=0;i<19;i++){
            float mid=0.5*(lo+hi);
            vec2 q=result.position+result.velocity*mid+0.5*acceleration*mid*mid;
            if(sdfBilliard(q)+radius>0.0)hi=mid;else lo=mid;
        }
        float impactTime=0.5*(lo+hi);
        vec2 contact=result.position+result.velocity*impactTime+
                     0.5*acceleration*impactTime*impactTime;
        vec2 impactVelocity=result.velocity+acceleration*impactTime;
        vec2 normal=sdfBilliardNormal(contact,normalStep);
        float normalVelocity=dot(impactVelocity,normal);
        if(normalVelocity>0.0)
            impactVelocity-=(1.0+restitution)*normalVelocity*normal;
        result.position=contact-normal*2e-6;
        result.velocity=impactVelocity;
        result.impacts+=1.0;
        remaining-=impactTime;
    }
    return result;
}
)GLSL");
        // 'helper' names are hygienically rewritten per node. Source fields
        // can be external textures or any lowerable SDF graph composition.
        // Do not retain references to simulation compiler frame-stack storage.
        (void)source;
        const std::string call=function+"Advance("+position.name+","+velocity.name+","+
            acceleration.name+","+dt.name+","+radius.name+","+
            restitution.name+","+normalStep.name+")";
        (void)context.emitTyped("("+call+").position",ShaderValueType::Vec2,"position");
        (void)context.emitTyped("("+call+").velocity",ShaderValueType::Vec2,"velocity");
        (void)context.emitTyped("("+call+").impacts",ShaderValueType::Scalar,"impacts");
        return true;
    }
};
} // namespace
void registerSdfBilliardStepNode(NodeRegistry& registry){
    registry.add(SdfBilliardStepNode::describe(),
                 []{return std::make_unique<SdfBilliardStepNode>();});
}
} // namespace reaction
