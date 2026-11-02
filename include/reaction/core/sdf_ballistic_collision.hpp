#pragma once
#include "reaction/core/node_builder.hpp"

namespace reaction {
// General collision against arbitrary externally supplied negative-inside SDF.
// Integrates ballistic motion through multiple continuous contacts, and
// projects a penetrating starting/ending state safely back into the domain.
inline NodeDescriptor sdfBallisticCollisionDescriptor() {
    return NodeDescriptorBuilder{"sdf_ballistic_collision",1,
                                 "SDF Ballistic Collision","Simulation"}
        .input("position","Position",SocketContract::VectorNumeric)
        .input("velocity","Velocity",SocketContract::VectorNumeric)
        .input("boundary","Boundary SDF (negative inside)",ValueType::ScalarField)
        .neighborhood("boundary")
        .optionalInput("acceleration","Acceleration",SocketContract::VectorNumeric)
        .output("position","Next Position",SocketContract::VectorNumeric)
        .output("velocity","Next Velocity",SocketContract::VectorNumeric)
        .output("impacts","Impacts This Step",SocketContract::Numeric)
        .floatParameter("dt","Time Step (s)",1.0F/600.0F,0.00001F,0.05F)
        .floatParameter("radius","Particle Radius (UV)",0.0F,0.0F,0.1F)
        .floatParameter("restitution","Restitution",1.0F,0.0F,1.0F)
        .floatParameter("normalStep","Surface Normal Step (UV)",0.001F,0.00001F,0.02F)
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
}
