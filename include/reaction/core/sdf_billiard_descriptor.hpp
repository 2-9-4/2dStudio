#pragma once

#include "reaction/core/node_builder.hpp"

namespace reaction {

// Shared structural contract for the core-only built-in catalog validator and
// the GPU implementation. Keep a single descriptor source of truth even when
// core tests intentionally do not link OpenGL.
inline NodeDescriptor sdfBilliardStepDescriptor() {
    return NodeDescriptorBuilder{"sdf_billiard_step",1,"Legacy SDF Billiard Step","Legacy"}
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

} // namespace reaction
