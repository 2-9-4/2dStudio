#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "reaction/core/node_builder.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>

namespace reaction {
namespace {

using node_support::bindTexture;
using node_support::floatAt;
using node_support::imageAt;
using node_support::uniform;

constexpr const char* kFade = R"GLSL(#version 430
layout(local_size_x=16,local_size_y=16)in;
layout(binding=0)uniform sampler2D previousImage;
layout(rgba16f,binding=0)writeonly uniform image2D nextImage;
uniform float retention;
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy),s=imageSize(nextImage);
    if(any(greaterThanEqual(p,s)))return;
    vec3 old=texelFetch(previousImage,p,0).rgb;
    vec3 color=mix(vec3(0.05),old,retention);
    imageStore(nextImage,p,vec4(color,1.0));
})GLSL";

constexpr const char* kVertex = R"GLSL(#version 430
layout(binding=0)uniform sampler2D positions;
layout(binding=1)uniform sampler2D branches;
uniform int particleCount;
uniform vec2 canvasSize;
out float branch;
out float weight;
uint hash(uint h){
    h ^= h>>16u;h*=0x7feb352du;h^=h>>15u;h*=0x846ca68bu;
    return h^(h>>16u);
}
void main(){
    ivec2 size=textureSize(positions,0);
    // Deterministically stratify the requested count over the state image;
    // its resolution does not change the number of rendered particles.
    int total=size.x*size.y;
    int i=min(total-1,int((float(gl_VertexID)+0.5)*float(total)/float(max(1,particleCount))));
    ivec2 cell=ivec2(i%size.x,i/size.x);
    vec2 uv=texelFetch(positions,cell,0).rg;
    branch=texelFetch(branches,cell,0).r;
    uint h=hash(uint(gl_VertexID)+uint(total)*17u);
    weight=0.25+0.75*float(h>>8u)*(1.0/16777216.0);
    vec2 inset=vec2(20.0)/canvasSize;
    uv=clamp(uv,inset,vec2(1.0)-inset);
    gl_Position=vec4(uv*2.0-1.0,0.0,1.0);
    gl_PointSize=1.0;
})GLSL";

constexpr const char* kFragment = R"GLSL(#version 430
in float branch;
in float weight;
uniform float opacity;
out vec4 color;
void main(){
    // Converted from the original HSB colors (15,90,90) and (175,90,90).
    vec3 orange=vec3(0.90,0.2925,0.09);
    vec3 teal=vec3(0.09,0.90,0.8325);
    color=vec4(mix(teal,orange,step(0.5,branch)),opacity*weight);
})GLSL";

GLuint shader(GLenum kind,const char* source) {
    const GLuint id=glCreateShader(kind);
    glShaderSource(id,1,&source,nullptr);
    glCompileShader(id);
    GLint ok=GL_FALSE;
    glGetShaderiv(id,GL_COMPILE_STATUS,&ok);
    if(ok!=GL_TRUE){
        GLint length=0;glGetShaderiv(id,GL_INFO_LOG_LENGTH,&length);
        std::string message(static_cast<std::size_t>(std::max(length,1)),0);
        glGetShaderInfoLog(id,length,nullptr,message.data());
        glDeleteShader(id);
        throw std::runtime_error("Particle Splat shader compile: "+message);
    }
    return id;
}

GLuint makeProgram() {
    GLuint vertex=shader(GL_VERTEX_SHADER,kVertex);
    GLuint fragment=0;
    try { fragment=shader(GL_FRAGMENT_SHADER,kFragment); }
    catch (...) { glDeleteShader(vertex);throw; }
    const GLuint program=glCreateProgram();
    glAttachShader(program,vertex);glAttachShader(program,fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);glDeleteShader(fragment);
    GLint ok=GL_FALSE;glGetProgramiv(program,GL_LINK_STATUS,&ok);
    if(ok!=GL_TRUE){
        GLint length=0;glGetProgramiv(program,GL_INFO_LOG_LENGTH,&length);
        std::string message(static_cast<std::size_t>(std::max(length,1)),0);
        glGetProgramInfoLog(program,length,nullptr,message.data());
        glDeleteProgram(program);
        throw std::runtime_error("Particle Splat program link: "+message);
    }
    return program;
}

class ParticleSplatNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe() {
        return NodeDescriptorBuilder{"particle_splat",1,"Particle Splat / Trails","Simulation"}
            .input("positions","Particle Positions (UV)",ValueType::VectorField)
            .input("branches","Orange / Teal Branch",ValueType::ScalarField)
            .output("image","Accumulated Trails",ValueType::ColorImage)
            .integerParameter("count","Visible Particles",1000,1,100000)
            .floatParameter("retention","Frame Retention",0.985F,0.0F,1.0F)
            .floatParameter("opacity","Point Opacity",0.375F,0.0F,1.0F)
            .stateful().timeDependent()
            .build();
    }

    const NodeDescriptor& descriptor()const override{
        static const auto d=describe();return d;
    }

    ~ParticleSplatNode() override {
        if(canvas_[0]) glDeleteTextures(2,canvas_.data());
        if(framebuffer_)glDeleteFramebuffers(1,&framebuffer_);
        if(vertexArray_)glDeleteVertexArrays(1,&vertexArray_);
        if(pointProgram_)glDeleteProgram(pointProgram_);
        if(fadeProgram_)glDeleteProgram(fadeProgram_);
    }

    void reset(EvaluationContext&) override { clearPending_=true; }

    void evaluate(EvaluationContext& context,std::span<const Value> inputs,
                  std::span<Value> outputs)override{
        if(outputs.empty())return;
        const auto positions=imageAt(inputs,0);
        const auto branches=imageAt(inputs,1);
        if(!positions||!branches){
            outputs[0]=Value{};return;
        }
        auto& gpu=*static_cast<GpuRuntime*>(context.gpu);
        ensureResources(gpu,context.width,context.height);
        const int capacity=std::min({100000,positions.width*positions.height,
                                      branches.width*branches.height});
        if(capacity<=0){outputs[0]=Value{};return;}
        const int count=std::clamp(static_cast<int>(floatAt(inputs,2,1000.0F)),1,capacity);
        const float retention=std::clamp(floatAt(inputs,3,0.985F),0.0F,1.0F);
        const float opacity=std::clamp(floatAt(inputs,4,0.375F),0.0F,1.0F);

        // GL is shared with ImGui and the preview. Do not leak framebuffer,
        // VAO, viewport, or blend/depth/scissor state into the UI.
        GLint savedFramebuffer=0,savedProgram=0,savedVao=0,savedActive=0;
        GLint savedViewport[4]{};
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&savedFramebuffer);
        glGetIntegerv(GL_CURRENT_PROGRAM,&savedProgram);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING,&savedVao);
        glGetIntegerv(GL_ACTIVE_TEXTURE,&savedActive);
        glGetIntegerv(GL_VIEWPORT,savedViewport);
        const GLboolean wasBlend=glIsEnabled(GL_BLEND);
        const GLboolean wasDepth=glIsEnabled(GL_DEPTH_TEST);
        const GLboolean wasScissor=glIsEnabled(GL_SCISSOR_TEST);
        const GLboolean wasPointSize=glIsEnabled(GL_PROGRAM_POINT_SIZE);
        GLint srcRgb=0,dstRgb=0,srcAlpha=0,dstAlpha=0,eqRgb=0,eqAlpha=0;
        glGetIntegerv(GL_BLEND_SRC_RGB,&srcRgb);glGetIntegerv(GL_BLEND_DST_RGB,&dstRgb);
        glGetIntegerv(GL_BLEND_SRC_ALPHA,&srcAlpha);glGetIntegerv(GL_BLEND_DST_ALPHA,&dstAlpha);
        glGetIntegerv(GL_BLEND_EQUATION_RGB,&eqRgb);glGetIntegerv(GL_BLEND_EQUATION_ALPHA,&eqAlpha);

        auto attach=[&](GLuint texture){
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER,framebuffer_);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,
                                    GL_TEXTURE_2D,texture,0);
            if(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
                throw std::runtime_error("Particle Splat framebuffer incomplete");
        };
        if(clearPending_){
            glDisable(GL_SCISSOR_TEST);
            const GLfloat clear[4]{0.05F,0.05F,0.05F,1.0F};
            for(GLuint texture:canvas_){
                attach(texture);
                glClearBufferfv(GL_COLOR,0,clear);
            }
            clearPending_=false;
        }
        if(context.playing){
            const int next=1-index_;
            glUseProgram(fadeProgram_);
            bindTexture(0,canvas_[index_]);
            uniform(fadeProgram_,"previousImage",0);
            uniform(fadeProgram_,"retention",retention);
            glBindImageTexture(0,canvas_[next],0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA16F);
            gpu.dispatch(fadeProgram_,context.width,context.height);
            glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
            attach(canvas_[next]);
            glViewport(0,0,context.width,context.height);
            glDisable(GL_DEPTH_TEST);glDisable(GL_SCISSOR_TEST);
            glEnable(GL_PROGRAM_POINT_SIZE);
            glEnable(GL_BLEND);
            glBlendEquation(GL_FUNC_ADD);
            glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
            glUseProgram(pointProgram_);
            bindTexture(0,positions.texture);
            bindTexture(1,branches.texture);
            uniform(pointProgram_,"positions",0);
            uniform(pointProgram_,"branches",1);
            uniform(pointProgram_,"particleCount",count);
            glUniform2f(glGetUniformLocation(pointProgram_,"canvasSize"),
                        static_cast<float>(context.width),static_cast<float>(context.height));
            uniform(pointProgram_,"opacity",opacity);
            glBindVertexArray(vertexArray_);
            glDrawArrays(GL_POINTS,0,count);
            glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_FETCH_BARRIER_BIT);
            index_=next;
        }
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER,static_cast<GLuint>(savedFramebuffer));
        glViewport(savedViewport[0],savedViewport[1],savedViewport[2],savedViewport[3]);
        glUseProgram(static_cast<GLuint>(savedProgram));
        glBindVertexArray(static_cast<GLuint>(savedVao));
        glActiveTexture(static_cast<GLenum>(savedActive));
        glBlendFuncSeparate(static_cast<GLenum>(srcRgb),static_cast<GLenum>(dstRgb),
                            static_cast<GLenum>(srcAlpha),static_cast<GLenum>(dstAlpha));
        glBlendEquationSeparate(static_cast<GLenum>(eqRgb),static_cast<GLenum>(eqAlpha));
        if(wasBlend)glEnable(GL_BLEND);else glDisable(GL_BLEND);
        if(wasDepth)glEnable(GL_DEPTH_TEST);else glDisable(GL_DEPTH_TEST);
        if(wasScissor)glEnable(GL_SCISSOR_TEST);else glDisable(GL_SCISSOR_TEST);
        if(wasPointSize)glEnable(GL_PROGRAM_POINT_SIZE);else glDisable(GL_PROGRAM_POINT_SIZE);
        outputs[0]=ImageHandle{canvas_[index_],context.width,context.height,
                               ValueType::ColorImage};
    }

private:
    void ensureResources(GpuRuntime& gpu,int width,int height){
        if(!fadeProgram_)fadeProgram_=gpu.compileCompute(kFade,"Particle Splat / fade");
        if(!pointProgram_)pointProgram_=makeProgram();
        if(!framebuffer_)glGenFramebuffers(1,&framebuffer_);
        if(!vertexArray_)glGenVertexArrays(1,&vertexArray_);
        if(width_!=width||height_!=height||!canvas_[0]){
            if(canvas_[0])glDeleteTextures(2,canvas_.data());
            for(auto& texture:canvas_)texture=gpu.createTexture(width,height);
            width_=width;height_=height;index_=0;clearPending_=true;
        }
    }

    std::array<GLuint,2> canvas_{};
    GLuint framebuffer_=0,vertexArray_=0,pointProgram_=0,fadeProgram_=0;
    int width_=0,height_=0,index_=0;
    bool clearPending_=true;
};

} // namespace

void registerParticleSplatNode(NodeRegistry& registry){
    registry.add(ParticleSplatNode::describe(),
                 [] { return std::make_unique<ParticleSplatNode>(); });
}
} // namespace reaction
