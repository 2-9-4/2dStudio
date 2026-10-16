#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "reaction/core/node_builder.hpp"
#include "reaction/gpu/shader_ir.hpp"

#include <algorithm>
#include <memory>

namespace reaction {
namespace {

// A pure, independently reusable particle integrator. Every pixel of the
// simulation state represents one particle; image resolution is NOT the
// number of visible particles (the renderer selects an evenly spread subset).
class ParticleStepNode final : public node_support::ParameterNode {
public:
    static NodeDescriptor describe() {
        return NodeDescriptorBuilder{"particle_step", 1, "Particle Step", "Simulation"}
            .input("position", "Position (UV)", ValueType::VectorField)
            .input("life", "Remaining Life", ValueType::ScalarField)
            .input("noise", "Noise at Position", ValueType::ScalarField)
            .optionalInput("frame", "Frame / Step", ValueType::Float)
            .output("position", "Next Position", ValueType::VectorField)
            .output("life", "Next Life", ValueType::ScalarField)
            .output("branch", "Orange Region", ValueType::ScalarField)
            .floatParameter("speed", "Speed (px/step)", 1.0F, 0.0F, 10.0F)
            .floatParameter("maxJump", "Max Tangent Jump (px)", 64.0F, 1.0F, 512.0F)
            .integerParameter("seed", "Random Seed", 1, 0, 1000000)
            .lowerable()
            .producedField()
            .build();
    }

    const NodeDescriptor& descriptor() const override {
        static const auto d = describe();
        return d;
    }

    bool lowerShader(ShaderLoweringContext& context) const override {
        const auto position = context.vector("position", "position", 0.0F);
        const auto life = context.scalar("life", "life", 1.0F);
        const auto noise = context.scalar("noise", "noise", 0.5F);
        const auto frame = context.scalar("frame", "frame", 0.0F);
        const auto speed = context.scalar("speed", "speed", 1.0F);
        const auto maxJump = context.scalar("maxJump", "maxJump", 64.0F);
        const auto seed = context.scalar("seed", "seed", 1.0F);
        const auto helper = context.helper("particleRng", R"GLSL(
uint particleRng_mix(uint v) {
    v ^= v >> 16u; v *= 0x7feb352du;
    v ^= v >> 15u; v *= 0x846ca68bu;
    return v ^ (v >> 16u);
}
float particleRng_unit(uvec2 id, uint frame, uint seed, uint stream) {
    uint h = particleRng_mix(id.x * 0x9e3779b9u ^ id.y * 0x85ebca6bu
                               ^ frame * 0xc2b2ae35u ^ seed ^ stream * 0x27d4eb2du);
    return float(h >> 8u) * (1.0 / 16777216.0);
}
)GLSL");
        // The random sequence is stable per particle and simulation step;
        // exact p5.js random()/Perlin sequences are not claimed.
        const std::string args = "uvec2(ivec2(floor(uv/pixelSize))),"
            "uint(max(floor(" + frame.name + "),0.0)),uint(max(floor(" + seed.name + "),0.0))";
        const auto random = [&](int stream) {
            return helper + "_unit(" + args + "," + std::to_string(stream) + "u)";
        };
        const auto angle = "(6.28318530718*round(clamp(" + noise.name +
            ",0.0,1.0)*1200.0)/1200.0)";
        const auto orange = "((" + angle + ">3.0&&" + angle + "<6.0)?1.0:0.0)";
        const auto tangent = "clamp(tan(" + angle + "),-max(" + maxJump.name +
            ",1.0),max(" + maxJump.name + ",1.0))";
        const auto delta = "mix(vec2(-sin(" + angle + ")*(0.1+0.9*" + random(1) +
            "),-cos(" + angle + ")),vec2(" + tangent + "*(1.0+2.0*" +
            random(2) + ")," + tangent + ")," + orange + ")*" + speed.name;
        // Nested random(random(random(random()))) is a product of four
        // independent uniform variates. Life decrement follows the original.
        const auto lost = "(" + random(3) + "*" + random(4) + "*" +
            random(5) + "*" + random(6) + "*0.1)";
        const auto died = "((" + life.name + "-" + lost + ")<=0.0)";
        const auto nextPos = "(" + died + "?vec2(" + random(7) + "," +
            random(8) + "):(" + position.name + "+(" + delta + ")*pixelSize))";
        const auto nextLife = "(" + died + "?" + random(9) + ":max(0.0," +
            life.name + "-" + lost + "))";
        (void)context.emitTyped(nextPos, ShaderValueType::Vec2, "position");
        (void)context.emitTyped(nextLife, ShaderValueType::Scalar, "life");
        (void)context.emitTyped(orange, ShaderValueType::Scalar, "branch");
        return true;
    }
};

} // namespace

void registerParticleStepNode(NodeRegistry& registry) {
    registry.add(ParticleStepNode::describe(),
                 [] { return std::make_unique<ParticleStepNode>(); });
}
} // namespace reaction
