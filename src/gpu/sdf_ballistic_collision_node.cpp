#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "reaction/core/sdf_ballistic_collision.hpp"
#include "reaction/gpu/shader_ir.hpp"

#include <memory>
#include <string>

namespace reaction {
namespace {
class SdfBallisticCollision final : public node_support::ParameterNode {
public:
    const NodeDescriptor& descriptor()const override{
        static const auto d=sdfBallisticCollisionDescriptor();return d;
    }
    bool lowerShader(ShaderLoweringContext& c)const override{
        const auto field=c.inputAt("boundary","clamp(q,vec2(0.0),vec2(1.0))",
                                   "boundary",0.0F);
        const auto p=c.vector("position","position",0.5F);
        const auto v=c.vector("velocity","velocity",0.0F);
        const auto a=c.vector("acceleration","acceleration",0.0F);
        const auto dt=c.scalar("dt","dt",1.0F/600.0F);
        const auto radius=c.scalar("radius","radius",0.0F);
        const auto restitution=c.scalar("restitution","restitution",1.0F);
        const auto normalStep=c.scalar("normalStep","normalStep",0.001F);
        std::string helper=R"GLSL(
float sdfCollideSample(vec2 q){return SDF_EXPRESSION;}
vec2 sdfCollideNormal(vec2 p,float stepSize) {
    float e=max(stepSize,0.000001);
    vec2 g=vec2(
        sdfCollideSample(p+vec2(e,0.0))-sdfCollideSample(p-vec2(e,0.0)),
        sdfCollideSample(p+vec2(0.0,e))-sdfCollideSample(p-vec2(0.0,e)));
    return dot(g,g)>1e-15?normalize(g):vec2(0.0,-1.0);
}
vec2 sdfCollideAt(vec2 p,vec2 v,vec2 a,float t){
    return p+v*t+0.5*a*t*t;
}
vec2 sdfCollideFix(vec2 p,float radius,float stepSize) {
    // Handles particles already outside because of previous numerical error,
    // SDF editing, or a step that exhausted the collision budget.
    for(int k=0;k<12;k++){
        float d=sdfCollideSample(p)+radius;
        if(d<=-0.000002)break;
        p-=sdfCollideNormal(p,stepSize)*clamp(d+0.000004,0.000004,0.5);
    }
    return p;
}
struct sdfCollideResult{vec2 p;vec2 v;float hits;};
sdfCollideResult sdfCollideAdvance(vec2 p,vec2 v,vec2 a,float dt,
                                   float radius,float restitution,float stepSize){
    sdfCollideResult r;
    r.p=p;r.v=v;r.hits=0.0;
    radius=max(0.0,radius);restitution=clamp(restitution,0.0,1.0);
    float remaining=clamp(dt,0.0,0.05);
    if(sdfCollideSample(r.p)+radius>-0.000002){
        vec2 n=sdfCollideNormal(r.p,stepSize);
        r.p=sdfCollideFix(r.p,radius,stepSize);
        if(dot(r.v,n)>0.0)r.v-=(1.0+restitution)*dot(r.v,n)*n;
    }
    // More than one wall contact in a timestep is possible, especially at
    // sharp corners. After every impact, integrate ONLY the unused time.
    for(int bounce=0;bounce<12 && remaining>0.00000001;bounce++){
        vec2 start=r.p,initialVelocity=r.v;
        float lo=0.0,hi=remaining,previousT=0.0;
        bool found=false;
        float prevD=sdfCollideSample(start)+radius;
        // Conservative advancement: bound path speed by |v|+|a|*dt and
        // use SDF clearance to avoid tunneling through a thin obstacle even
        // when both ends of a frame are inside the allowed region.
        float speedBound=max(0.000001,length(initialVelocity)+length(a)*remaining);
        float t=0.0;
        for(int sweep=0;sweep<192;sweep++){
            if(t>=remaining)break;
            float clearance=max(-prevD,0.0);
            float safeStep=0.7*clearance/speedBound;
            float interval=max(0.0000004,min(safeStep,remaining-t));
            float nextT=min(remaining,t+interval);
            if(nextT<=t)break;
            float d=sdfCollideSample(sdfCollideAt(start,initialVelocity,a,nextT))+radius;
            if(d>0.0){
                lo=t;hi=nextT;found=true;break;
            }
            if(d>=-0.000002){
                vec2 at=sdfCollideAt(start,initialVelocity,a,nextT);
                vec2 n=sdfCollideNormal(at,stepSize);
                // Outward motion near the zero contour is itself a contact.
                if(dot(initialVelocity+a*nextT,n)>0.0){
                    lo=t;hi=nextT;found=true;break;
                }
            }
            t=nextT;prevD=d;
        }
        // For poorly conditioned SDFs or extreme velocities, supplement
        // conservative advancement with an exhaustive curved-path scan.
        if(!found){
            float before=0.0;
            for(int k=1;k<=128;k++){
                float after=remaining*(float(k)/128.0);
                float d=sdfCollideSample(sdfCollideAt(start,initialVelocity,a,after))+radius;
                if(d>0.0){lo=before;hi=after;found=true;break;}
                before=after;
            }
        }
        if(!found){
            r.p=sdfCollideAt(start,initialVelocity,a,remaining);
            r.v=initialVelocity+a*remaining;
            remaining=0.0;break;
        }
        for(int root=0;root<22;root++){
            float mid=0.5*(lo+hi);
            if(sdfCollideSample(sdfCollideAt(start,initialVelocity,a,mid))+radius>0.0)
                hi=mid;
            else lo=mid;
        }
        float contactTime=0.5*(lo+hi);
        vec2 contact=sdfCollideAt(start,initialVelocity,a,contactTime);
        vec2 impactVelocity=initialVelocity+a*contactTime;
        vec2 normal=sdfCollideNormal(contact,stepSize);
        float vn=dot(impactVelocity,normal);
        if(vn>0.0){
            impactVelocity-=(1.0+restitution)*vn*normal;
            r.hits+=1.0;
        }
        r.p=sdfCollideFix(contact-normal*0.000004,radius,stepSize);
        r.v=impactVelocity;
        remaining-=contactTime;
        // A near-zero contact must still consume a tiny amount of remaining
        // time to avoid repeatedly detecting the same wall infinitely.
        if(contactTime<0.0000005 && remaining>0.0){
            float tiny=min(remaining,0.000002);
            r.p=sdfCollideAt(r.p,r.v,a,tiny);
            r.v+=a*tiny;
            remaining-=tiny;
            r.p=sdfCollideFix(r.p,radius,stepSize);
        }
    }
    if(remaining>0.0){
        r.p=sdfCollideAt(r.p,r.v,a,remaining);
        r.v+=a*remaining;
    }
    // Hard invariant: at the end of an update no state may remain on
    // the positive (forbidden) side of a usable SDF.
    if(sdfCollideSample(r.p)+radius>-0.000002){
        vec2 n=sdfCollideNormal(r.p,stepSize);
        r.p=sdfCollideFix(r.p,radius,stepSize);
        float vn=dot(r.v,n);
        if(vn>0.0){r.v-=(1.0+restitution)*vn*n;r.hits+=1.0;}
    }
    return r;
}
)GLSL";
        const auto where=helper.find("SDF_EXPRESSION");
        helper.replace(where,std::string("SDF_EXPRESSION").size(),field.name);
        const auto name=c.helper("sdfCollide",std::move(helper));
        const std::string call=name+"Advance("+p.name+","+v.name+","+a.name+","+
           dt.name+","+radius.name+","+restitution.name+","+normalStep.name+")";
        (void)c.emitTyped("("+call+").p",ShaderValueType::Vec2,"position");
        (void)c.emitTyped("("+call+").v",ShaderValueType::Vec2,"velocity");
        (void)c.emitTyped("("+call+").hits",ShaderValueType::Scalar,"impacts");
        return true;
    }
};
} // namespace
void registerSdfBallisticCollisionNode(NodeRegistry& registry){
    registry.add(sdfBallisticCollisionDescriptor(),
                 []{return std::make_unique<SdfBallisticCollision>();});
}
} // namespace reaction
