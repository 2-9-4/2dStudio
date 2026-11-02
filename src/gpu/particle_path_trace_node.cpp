#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "reaction/core/node_builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace reaction {
namespace {

struct PathPoint { float x=0.0F,y=0.0F; };
struct PathSegment { std::vector<PathPoint> points; };
struct ParticlePath { std::vector<PathSegment> strokes; };

std::array<std::uint8_t,3> hsvRgb(float hue) {
    const auto channel=[&](float offset)->std::uint8_t {
        float u=hue+offset;
        u-=std::floor(u);
        return static_cast<std::uint8_t>(std::lround(
            std::clamp(std::abs(u*6.0F-3.0F)-1.0F,0.0F,1.0F)*255.0F));
    };
    return {channel(0.0F),channel(2.0F/3.0F),channel(1.0F/3.0F)};
}

class ParticlePathTrace final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe() {
        return NodeDescriptorBuilder{"particle_path_trace",1,"Particle Path Trace","Simulation"}
            .input("positions","Particle Positions",ValueType::VectorField)
            .output("image","Vector Trajectory Preview",ValueType::ColorImage)
            .integerParameter("count","Tracked Particles",200,1,4096)
            .integerParameter("sampleEvery","Record Every N Frames",1,1,100)
            .integerParameter("maxPoints","Max Points per Particle",10000,2,100000)
            .floatParameter("minDistance","Minimum Point Separation (UV)",0.0F,0.0F,0.1F)
            .floatParameter("strokeWidth","SVG Stroke Width (pixels)",0.8F,0.1F,20.0F)
            .enumParameter("palette","Color Mode",1,{"White","Rainbow by Particle"})
            .stateful().timeDependent().build();
    }
    const NodeDescriptor& descriptor()const override {
        static const auto d=describe();return d;
    }
    ~ParticlePathTrace()override{ if(texture_)glDeleteTextures(1,&texture_); }
    void reset(EvaluationContext&) override {
        histories_.clear(); recorded_=0; previousFrame_=~std::uint64_t{0};
        lastPoints_.clear(); clearPending_=true;
    }
    std::string vectorSvg(int width,int height)const override{
        if(histories_.empty() || recorded_<2) return {};
        std::ostringstream out;
        out<<std::fixed<<std::setprecision(5);
        out<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        out<<"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\""<<width
           <<"\" height=\""<<height<<"\" viewBox=\"0 0 "<<width<<" "<<height<<"\">\n";
        out<<"<title>Particle trajectories</title>\n";
        const float stroke=std::clamp(node_support::parameter(parameters_,
            "strokeWidth",0.8F),0.1F,20.0F);
        const bool rainbow=int(node_support::parameter(parameters_,"palette",1.0F))==1;
        for(std::size_t i=0;i<histories_.size();++i){
            const auto rgb=rainbow?hsvRgb(float(i)/float(std::max<std::size_t>(histories_.size(),1)))
                                  :std::array<std::uint8_t,3>{255,255,255};
            std::ostringstream hex;
            hex<<'#'<<std::hex<<std::setfill('0')<<std::setw(2)<<int(rgb[0])
               <<std::setw(2)<<int(rgb[1])<<std::setw(2)<<int(rgb[2]);
            for(const auto& strokePoints:histories_[i].strokes){
                if(strokePoints.points.size()<2)continue;
                out<<"<path data-particle=\""<<i<<"\" fill=\"none\" stroke=\""
                   <<hex.str()<<"\" stroke-width=\""<<stroke
                   <<"\" stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"";
                bool first=true;
                for(const auto& p:strokePoints.points){
                    out<<(first?"M ":" L ")<<std::clamp(p.x,0.0F,1.0F)*width<<" "
                       <<(1.0F-std::clamp(p.y,0.0F,1.0F))*height;
                    first=false;
                }
                out<<"\"/>\n";
            }
        }
        out<<"</svg>\n";
        return out.str();
    }
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,
                  std::span<Value> outputs)override {
        const auto positions=node_support::imageAt(inputs,0);
        if(!positions){outputs[0]=Value{};return;}
        if(width_!=context.width || height_!=context.height || !texture_){
            if(texture_)glDeleteTextures(1,&texture_);
            auto& gpu=*static_cast<GpuRuntime*>(context.gpu);
            texture_=gpu.createTexture(context.width,context.height,GL_RGBA8);
            width_=context.width;height_=context.height;
            bitmap_.resize(std::size_t(width_)*std::size_t(height_)*4);
            histories_.clear();lastPoints_.clear();recorded_=0;
            clearPending_=true;
        }
        const int available=std::max(0,positions.width*positions.height);
        const int selected=std::clamp(int(node_support::floatAt(inputs,1,
                        node_support::parameter(parameters_,"count",200.0F))),1,4096);
        const int count=std::min(selected,available);
        const int sampleEvery=std::clamp(int(node_support::floatAt(inputs,2,
                        node_support::parameter(parameters_,"sampleEvery",1.0F))),1,100);
        const int maxPoints=std::clamp(int(node_support::floatAt(inputs,3,
                        node_support::parameter(parameters_,"maxPoints",10000.0F))),2,100000);
        const float minDistance=node_support::floatAt(inputs,4,
                         node_support::parameter(parameters_,"minDistance",0.0F));
        if(clearPending_ || histories_.size()!=std::size_t(count)){
            histories_.assign(static_cast<std::size_t>(count),{});
            lastPoints_.assign(static_cast<std::size_t>(count),std::nullopt);
            std::fill(bitmap_.begin(),bitmap_.end(),std::uint8_t(0));
            for(std::size_t i=3;i<bitmap_.size();i+=4)bitmap_[i]=255;
            recorded_=0;clearPending_=false;
            textureDirty_=true;
        }
        // Paused previews record their first state but never duplicate points
        // repeatedly. A new input field can also be sampled while paused.
        const bool sample=(recorded_==0 || (context.playing &&
             context.frame!=previousFrame_ && context.frame%std::uint64_t(sampleEvery)==0));
        if(sample && count>0 && recorded_<std::size_t(maxPoints)) {
            const std::size_t total=static_cast<std::size_t>(available);
            std::vector<float> pixels(total*2);
            GLint oldTexture=0,oldPack=0;
            glGetIntegerv(GL_TEXTURE_BINDING_2D,&oldTexture);
            glGetIntegerv(GL_PACK_ALIGNMENT,&oldPack);
            glBindTexture(GL_TEXTURE_2D,positions.texture);
            glPixelStorei(GL_PACK_ALIGNMENT,1);
            glGetTexImage(GL_TEXTURE_2D,0,GL_RG,GL_FLOAT,pixels.data());
            glBindTexture(GL_TEXTURE_2D,static_cast<GLuint>(oldTexture));
            glPixelStorei(GL_PACK_ALIGNMENT,oldPack);
            const int palette=std::clamp(int(node_support::parameter(parameters_,"palette",1.0F)),0,1);
            for(int i=0;i<count;i++){
                // Same stable stratified slot mapping as Particle Splat.
                const int slot=std::min(available-1,int((float(i)+0.5F)*
                                                   float(available)/float(count)));
                const PathPoint p{pixels[std::size_t(slot)*2],
                                  pixels[std::size_t(slot)*2+1]};
                auto& path=histories_[std::size_t(i)];
                const bool valid=std::isfinite(p.x)&&std::isfinite(p.y)&&
                                 p.x>=0.0F&&p.x<=1.0F&&p.y>=0.0F&&p.y<=1.0F;
                if(!valid){
                    lastPoints_[std::size_t(i)].reset();
                    continue;
                }
                const auto old=lastPoints_[std::size_t(i)];
                if(old && std::hypot(p.x-old->x,p.y-old->y)<minDistance)continue;
                if(path.strokes.empty() || !old)path.strokes.push_back({});
                auto& points=path.strokes.back().points;
                if(points.size()<static_cast<std::size_t>(maxPoints)){
                    points.push_back(p);
                    if(old){
                        const auto rgb=palette==1 ? hsvRgb(float(i)/float(count)) :
                             std::array<std::uint8_t,3>{255,255,255};
                        rasterSegment(*old,p,rgb);
                    }
                }
                lastPoints_[std::size_t(i)]=p;
            }
            ++recorded_;
            previousFrame_=context.frame;
        }
        if(textureDirty_){
            GLint oldTexture=0,oldUnpack=0;
            glGetIntegerv(GL_TEXTURE_BINDING_2D,&oldTexture);
            glGetIntegerv(GL_UNPACK_ALIGNMENT,&oldUnpack);
            glBindTexture(GL_TEXTURE_2D,texture_);
            glPixelStorei(GL_UNPACK_ALIGNMENT,1);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width_,height_,GL_RGBA,
                            GL_UNSIGNED_BYTE,bitmap_.data());
            glBindTexture(GL_TEXTURE_2D,static_cast<GLuint>(oldTexture));
            glPixelStorei(GL_UNPACK_ALIGNMENT,oldUnpack);
            textureDirty_=false;
        }
        outputs[0]=ImageHandle{texture_,width_,height_,ValueType::ColorImage};
    }
private:
    void rasterSegment(PathPoint a,PathPoint b,std::array<std::uint8_t,3> rgb){
        const float x0=a.x*float(width_-1),y0=a.y*float(height_-1);
        const float x1=b.x*float(width_-1),y1=b.y*float(height_-1);
        const int steps=std::max(1,int(std::ceil(std::max(std::abs(x1-x0),
                                                       std::abs(y1-y0)))));
        for(int s=0;s<=steps;s++){
            const float t=float(s)/float(steps);
            const int x=int(std::round(x0+(x1-x0)*t));
            const int y=int(std::round(y0+(y1-y0)*t));
            if(x<0||y<0||x>=width_||y>=height_)continue;
            auto* pixel=bitmap_.data()+(std::size_t(y)*std::size_t(width_)+
                                        std::size_t(x))*4;
            pixel[0]=std::max(pixel[0],rgb[0]);
            pixel[1]=std::max(pixel[1],rgb[1]);
            pixel[2]=std::max(pixel[2],rgb[2]);
        }
        textureDirty_=true;
    }
    GLuint texture_=0;
    int width_=0,height_=0;
    std::vector<std::uint8_t> bitmap_;
    std::vector<ParticlePath> histories_;
    std::vector<std::optional<PathPoint>> lastPoints_;
    std::size_t recorded_=0;
    std::uint64_t previousFrame_=~std::uint64_t{0};
    bool clearPending_=true,textureDirty_=false;
};
}
void registerParticlePathTraceNode(NodeRegistry& registry){
    registry.add(ParticlePathTrace::describe(),[]{return std::make_unique<ParticlePathTrace>();});
}
} // namespace reaction
