#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "procedural_node_support.hpp"
#include "reaction/core/node_builder.hpp"

#include <algorithm>
#include <memory>

namespace reaction {
namespace {
constexpr std::string_view kLattice=R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba16f,binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D coordinatesImage;
uniform int hasCoordinates,mode,columns,rows,rings,spokes;
uniform vec2 coordinatesConstant;
uniform float thickness,ringRadius,ringSpacing;
vec2 coord(vec2 uv) {
    return hasCoordinates>0?texture(coordinatesImage,clamp(uv,vec2(0.0),vec2(1.0))).rg:
           (hasCoordinates<0?uv:coordinatesConstant);
}
float line(float position,float spacing,float widthPixels) {
    float distanceToLine=abs(fract(position+0.5)-0.5);
    return 1.0-smoothstep(max(0.0,spacing*(widthPixels-1.0)*0.5),
                          max(1e-6,spacing*(widthPixels+1.0)*0.5),distanceToLine);
}
void main(){
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy),size=imageSize(outputImage);
    if(any(greaterThanEqual(pixel,size)))return;
    vec2 uv=(vec2(pixel)+0.5)/vec2(size);
    vec2 ex=vec2(1.0/float(size.x),0.0),ey=vec2(0.0,1.0/float(size.y));
    vec2 p=coord(uv),px=coord(uv+ex),py=coord(uv+ey);
    float result=0.0;
    if(mode==0) {
        vec2 scale=vec2(float(columns),float(rows));
        vec2 u=p*scale,ux=px*scale,uy=py*scale;
        float sx=max(1e-6,length(vec2(ux.x-u.x,uy.x-u.x)));
        float sy=max(1e-6,length(vec2(ux.y-u.y,uy.y-u.y)));
        result=max(line(u.x,sx,thickness),line(u.y,sy,thickness));
    } else {
        // Radial rings are independent of spoke count (0 = rings only).
        // Use isotropic pixel-space coordinates on non-square canvases.
        float aspect=float(size.x)/float(size.y);
        vec2 q=(p-0.5)*vec2(aspect,1.0);
        vec2 qx=(px-0.5)*vec2(aspect,1.0);
        vec2 qy=(py-0.5)*vec2(aspect,1.0);
        float r=length(q),dr=max(1e-6,length(vec2(length(qx)-r,length(qy)-r)));
        for(int i=0;i<128;i++){
            if(i>=rings)break;
            float target=ringRadius+float(i)*ringSpacing;
            result=max(result,1.0-smoothstep(max(0.0,(thickness-1.0)*dr*0.5),
                         (thickness+1.0)*dr*0.5+1e-7,abs(r-target)));
        }
        if(spokes>0 && r>1e-5){
            float a=atan(q.y,q.x)/(6.28318530718)*float(spokes);
            float ax=atan(qx.y,qx.x)/(6.28318530718)*float(spokes);
            float ay=atan(qy.y,qy.x)/(6.28318530718)*float(spokes);
            float da=max(1e-6,length(vec2(ax-a,ay-a)));
            // Avoid discontinuities at -pi/pi in angular derivatives.
            if(da<0.5)result=max(result,line(a,da,thickness));
        }
    }
    imageStore(outputImage,pixel,vec4(clamp(result,0.0,1.0),0.0,0.0,1.0));
})GLSL";

class LatticeLinesNode final : public node_support::TextureNode {
public:
    static NodeDescriptor describe(){
        return NodeDescriptorBuilder{"lattice_lines",1,"Lattice Lines","Generator"}
            .optionalInput("coordinates","Coordinates",SocketContract::VectorNumeric)
            .fieldDefault("coordinates")
            .output("lines","Line Mask",ValueType::ScalarField)
            .enumParameter("mode","Lattice",0,{"Rectangular Grid","Polar Rings / Spokes"})
            .integerParameter("columns","Columns",32,1,256)
            .integerParameter("rows","Rows",32,1,256)
            .integerParameter("rings","Rings",1,0,128)
            .integerParameter("spokes","Spokes",0,0,128)
            .floatParameter("ringRadius","First Ring Radius",0.44F,0.0F,2.0F)
            .floatParameter("ringSpacing","Ring Separation",0.08F,0.001F,2.0F)
            .floatParameter("thickness","Line Width (pixels)",1.25F,0.5F,12.0F).build();
    }
    const NodeDescriptor& descriptor()const override{
        static const auto d=describe();return d;
    }
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,
                  std::span<Value> outputs)override{
        ensure(context);
        auto& gpu=*static_cast<GpuRuntime*>(context.gpu);
        if(!program_)program_=gpu.compileCompute(kLattice,"Lattice Lines");
        glUseProgram(program_);
        glBindImageTexture(0,texture_,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F);
        procedural::bindVectorInput(program_,0,"coordinatesImage","hasCoordinates",
            "coordinatesConstant",procedural::coordinateInput(inputs,0));
        const auto param=[&](std::size_t index,const char* name,float fallback){
            return node_support::floatAt(inputs,index,node_support::parameter(parameters_,name,fallback));
        };
        node_support::uniform(program_,"mode",std::clamp(int(param(0,"mode",0)),0,1));
        node_support::uniform(program_,"columns",std::clamp(int(param(1,"columns",32)),1,256));
        node_support::uniform(program_,"rows",std::clamp(int(param(2,"rows",32)),1,256));
        node_support::uniform(program_,"rings",std::clamp(int(param(3,"rings",1)),0,128));
        node_support::uniform(program_,"spokes",std::clamp(int(param(4,"spokes",0)),0,128));
        node_support::uniform(program_,"ringRadius",param(5,"ringRadius",0.44F));
        node_support::uniform(program_,"ringSpacing",param(6,"ringSpacing",0.08F));
        node_support::uniform(program_,"thickness",param(7,"thickness",1.25F));
        gpu.dispatch(program_,context.width,context.height);
        outputs[0]=ImageHandle{texture_,context.width,context.height,ValueType::ScalarField};
    }
};
}
void registerLatticeLinesNode(NodeRegistry& registry){
    registry.add(LatticeLinesNode::describe(),[]{return std::make_unique<LatticeLinesNode>();});
}
} // namespace reaction
