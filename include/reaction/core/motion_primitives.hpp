#pragma once
#include "reaction/core/node_builder.hpp"

namespace reaction {

// Free-flight integration is independent of collision geometry and is useful
// for particles, projectiles, trajectories and other constant-force systems.
inline NodeDescriptor ballisticAdvanceDescriptor() {
    return NodeDescriptorBuilder{"ballistic_advance",1,"Ballistic Advance","Simulation"}
        .input("position","Start Position",SocketContract::VectorNumeric)
        .input("velocity","Start Velocity",SocketContract::VectorNumeric)
        .optionalInput("acceleration","Acceleration",SocketContract::VectorNumeric)
        .optionalInput("time","Duration (s)",SocketContract::Numeric)
        .output("position","End Position",SocketContract::VectorNumeric)
        .output("velocity","End Velocity",SocketContract::VectorNumeric)
        .floatParameter("time","Duration (s)",1.0F/600.0F,0.0F,1.0F)
        .typePolicy("position",SocketDescriptor::TypePolicy::VectorPromotion,
                    {"position","velocity","acceleration"})
        .fieldIf("position",{"time"})
        .typePolicy("velocity",SocketDescriptor::TypePolicy::VectorPromotion,
                    {"position","velocity","acceleration"})
        .fieldIf("velocity",{"time"})
        .lowerable().build();
}

// Finds the first transition to forbidden (positive-distance) space along
// a straight segment. Fraction=1, hit=0 for segments remaining inside.
// Does not integrate motion, reflect velocity, or own simulation state.
inline NodeDescriptor sdfSegmentCrossingDescriptor() {
    return NodeDescriptorBuilder{"sdf_segment_crossing",1,"SDF Segment Crossing","Geometry"}
        .input("source","Signed Distance Field",ValueType::ScalarField)
        .neighborhood("source")
        .input("start","Segment Start",SocketContract::VectorNumeric)
        .input("end","Segment End",SocketContract::VectorNumeric)
        .optionalInput("offset","Distance Offset / Radius",SocketContract::Numeric)
        .output("fraction","Crossing Fraction",ValueType::ScalarField)
        .output("hit","Crosses Boundary",ValueType::ScalarField)
        .floatParameter("offset","Distance Offset / Radius",0.0F,0.0F,0.1F)
        .lowerable().build();
}

// Geometric outward unit normal of any sampled scalar distance field.
inline NodeDescriptor sdfSurfaceNormalDescriptor() {
    return NodeDescriptorBuilder{"sdf_surface_normal",1,"SDF Surface Normal","Geometry"}
        .input("source","Signed Distance Field",ValueType::ScalarField)
        .neighborhood("source")
        .input("point","Sample Point",SocketContract::VectorNumeric)
        .optionalInput("step","Finite Difference Step (UV)",SocketContract::Numeric)
        .output("normal","Outward Normal",ValueType::VectorField)
        .floatParameter("step","Finite Difference Step (UV)",0.001F,0.00001F,0.02F)
        .lowerable().build();
}

} // namespace reaction
