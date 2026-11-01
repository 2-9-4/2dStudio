#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "reaction/core/node_builder.hpp"
#include "reaction/core/circular_billiards.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace reaction {
namespace {

// Physical simulation and visualization deliberately separate. The output
// is a 1-by-N agent state image ready for any particle renderer, not an image
// painted into the canvas. State persists at double precision on CPU; only
// display positions are converted to 32-bit RGBA on upload.
class CircularBilliardsNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe(){
        return NodeDescriptorBuilder{"circular_billiards",1,
                "Circular Gravity Billiards","Simulation"}
            .output("positions","Ball Positions (UV)",ValueType::VectorField)
            .output("spread","Max Separation (m)",ValueType::Float)
            .output("impacts","Total Wall Impacts",ValueType::Float)
            .output("energyError","Max Energy Error (J/kg)",ValueType::Float)
            .integerParameter("count","Ball Count",200,1,2000)
            .floatParameter("fanMm","Launch Fan Width (mm)",0.1F,0.0F,10.0F)
            .floatParameter("radius","Arena Radius (m)",1.0F,0.05F,100.0F)
            .floatParameter("ballRadius","Ball Radius (m)",0.0F,0.0F,0.5F)
            .floatParameter("gravity","Gravity (m/s²)",9.81F,0.01F,100.0F)
            .floatParameter("renderRadius","Display Radius (UV)",0.44F,0.05F,0.49F)
            .floatParameter("speed","Simulation Speed",1.0F,0.0F,10.0F)
            .stateful().timeDependent().build();
    }
    const NodeDescriptor& descriptor()const override{
        static const auto d=describe();return d;
    }
    ~CircularBilliardsNode()override{
        if(texture_)glDeleteTextures(1,&texture_);
    }
    void reset(EvaluationContext&)override{
        balls_.clear();
        accumulator_=0.0;
        simulated_=0.0;
    }
    void setParameters(const nlohmann::json& values)override{
        node_support::ParameterNode::setParameters(values);
        // Reset only when physics setup changes, not on display radius/speed.
        const auto setup=setupSignature();
        if(setup!=lastSetup_){
            lastSetup_=setup;
            balls_.clear();
            accumulator_=0.0;
            simulated_=0.0;
        }
    }
    void evaluate(EvaluationContext& context,std::span<const Value> inputs,
                  std::span<Value> outputs)override{
        auto param=[&](std::size_t i,const char* name,float fallback){
            return node_support::floatAt(inputs,i,
                node_support::parameter(parameters_,name,fallback));
        };
        const int count=std::clamp(int(param(0,"count",200)),1,2000);
        const double fan=std::clamp(double(param(1,"fanMm",0.1F)),0.0,10.0)*1e-3;
        const double radius=std::clamp(double(param(2,"radius",1.0F)),0.05,100.0);
        const double ballRadius=std::clamp(double(param(3,"ballRadius",0.0F)),0.0,radius*0.5);
        const double gravity=std::clamp(double(param(4,"gravity",9.81F)),0.01,100.0);
        const double renderRadius=std::clamp(double(param(5,"renderRadius",0.44F)),0.05,0.49);
        const double speed=std::clamp(double(param(6,"speed",1.0F)),0.0,10.0);
        // Parameter connections also affect reset identity, not just sliders.
        const std::vector<double> setup={double(count),fan,radius,ballRadius,gravity};
        if(setup!=activeSetup_ || balls_.size()!=static_cast<std::size_t>(count)){
            activeSetup_=setup;
            accumulator_=simulated_=0.0;
            balls_.clear();
            balls_.reserve(static_cast<std::size_t>(count));
            for(int i=0;i<count;i++)
                balls_.push_back(billiards::initialBall(i,count,fan));
        }
        const double usable=radius-ballRadius;
        // Simulation time is fixed-step, independent of render frame rate.
        // Time gaps above 0.2s are bounded to prevent UI stalls after pauses.
        if(context.playing){
            accumulator_+=std::clamp(context.deltaTime*speed,0.0,0.2);
            constexpr double fixedDt=1.0/600.0;
            int steps=0;
            while(accumulator_+1e-12>=fixedDt && steps<120){
                for(auto& ball:balls_) billiards::advance(ball,fixedDt,gravity,usable);
                simulated_+=fixedDt;
                accumulator_-=fixedDt;
                ++steps;
            }
        }
        if(!texture_ || textureWidth_!=count){
            if(texture_)glDeleteTextures(1,&texture_);
            auto& gpu=*static_cast<GpuRuntime*>(context.gpu);
            texture_=gpu.createTexture(count,1,GL_RGBA32F);
            textureWidth_=count;
        }
        pixels_.resize(static_cast<std::size_t>(count)*4);
        double energyError=0;
        std::uint64_t impacts=0;
        for(int i=0;i<count;i++){
            const auto& b=balls_[static_cast<std::size_t>(i)];
            const auto j=static_cast<std::size_t>(i)*4;
            pixels_[j]=float(0.5+renderRadius*b.x/radius);
            pixels_[j+1]=float(0.5+renderRadius*b.y/radius);
            pixels_[j+2]=0.0F;
            pixels_[j+3]=1.0F;
            energyError=std::max(energyError,std::abs(billiards::energy(b,gravity)));
            impacts+=b.impacts;
        }
        // Max separation is a diameter of the agent point cloud, not merely
        // its horizontal extent; handles later overlapping/focusing trajectories.
        double spread=0.0;
        for(std::size_t i=0;i<balls_.size();i++)
            for(std::size_t j=i+1;j<balls_.size();j++)
                spread=std::max(spread,std::hypot(
                    balls_[i].x-balls_[j].x,balls_[i].y-balls_[j].y));
        glBindTexture(GL_TEXTURE_2D,texture_);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,count,1,GL_RGBA,GL_FLOAT,pixels_.data());
        glBindTexture(GL_TEXTURE_2D,0);
        outputs[0]=ImageHandle{texture_,count,1,ValueType::VectorField};
        outputs[1]=float(spread);
        outputs[2]=float(impacts);
        outputs[3]=float(energyError);
    }
private:
    [[nodiscard]] std::string setupSignature()const{
        std::string signature;
        for(const char* key:{"count","fanMm","radius","ballRadius","gravity"}){
            signature+=key;
            signature+='=';
            if(parameters_.contains(key))signature+=parameters_[key].dump();
            signature+=';';
        }
        return signature;
    }
    std::vector<billiards::Ball> balls_;
    std::vector<double> activeSetup_;
    std::vector<float> pixels_;
    std::string lastSetup_;
    GLuint texture_=0;
    int textureWidth_=0;
    double accumulator_=0.0,simulated_=0.0;
};
}
void registerCircularBilliardsNode(NodeRegistry& registry){
    registry.add(CircularBilliardsNode::describe(),
        []{return std::make_unique<CircularBilliardsNode>();});
}
} // namespace reaction
