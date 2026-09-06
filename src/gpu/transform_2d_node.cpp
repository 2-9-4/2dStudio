#include "nodes_internal.hpp"
#include "node_support.hpp"
#include "procedural_node_support.hpp"

#include <cmath>
#include <memory>

namespace reaction {
namespace {

using node_support::TextureNode;
using procedural::VectorInput;
using procedural::bindVectorInput;
using procedural::safeSignedDenominator;
using procedural::vectorInput;

constexpr std::string_view kTransform2DShader = R"GLSL(#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba16f, binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D coordinatesImage;
layout(binding=1) uniform sampler2D translationImage;
layout(binding=2) uniform sampler2D rotationImage;
layout(binding=3) uniform sampler2D scaleImage;
layout(binding=4) uniform sampler2D shearImage;
layout(binding=5) uniform sampler2D pivotImage;
uniform int hasCoordinates, hasTranslation, hasRotation, hasScale, hasShear, hasPivot;
uniform vec2 coordinatesConstant, translationConstant, scaleConstant, shearConstant, pivotConstant;
uniform float rotationConstant;
const float epsilon = 1e-6;
float safeDenominator(float value) {
    return abs(value) < epsilon ? (value < 0.0 ? -epsilon : epsilon) : value;
}
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(outputImage);
    if (any(greaterThanEqual(pixel, size))) return;
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
    vec2 p = hasCoordinates > 0 ? texture(coordinatesImage, uv).rg :
             (hasCoordinates < 0 ? uv : coordinatesConstant);
    vec2 translation = hasTranslation != 0 ? texture(translationImage, uv).rg : translationConstant;
    float rotation = hasRotation != 0 ? texture(rotationImage, uv).r : rotationConstant;
    vec2 scale = hasScale != 0 ? texture(scaleImage, uv).rg : scaleConstant;
    vec2 shear = hasShear != 0 ? texture(shearImage, uv).rg : shearConstant;
    vec2 pivot = hasPivot != 0 ? texture(pivotImage, uv).rg : pivotConstant;

    // Invert R * H * S so the result is suitable as a Texture Sample domain.
    vec2 v = p - pivot - translation;
    float c = cos(rotation), s = sin(rotation);
    v = vec2(c * v.x + s * v.y, -s * v.x + c * v.y);
    float determinant = safeDenominator(1.0 - shear.x * shear.y);
    v = vec2(v.x - shear.x * v.y, v.y - shear.y * v.x) / determinant;
    vec2 q = pivot + vec2(v.x / safeDenominator(scale.x), v.y / safeDenominator(scale.y));
    imageStore(outputImage, pixel, vec4(q, 0.0, 1.0));
})GLSL";

Vec2 transformInverse(Vec2 p, Vec2 translation, float rotation, Vec2 scale,
                      Vec2 shear, Vec2 pivot) {
    float x = p.x - pivot.x - translation.x;
    float y = p.y - pivot.y - translation.y;
    const float cosine = std::cos(rotation);
    const float sine = std::sin(rotation);
    // inverse(Rotation)
    const float rotatedX = cosine * x + sine * y;
    const float rotatedY = -sine * x + cosine * y;
    // inverse(Shear), including the singular-matrix safeguard.
    const float determinant = safeSignedDenominator(1.0F - shear.x * shear.y);
    const float shearedX = (rotatedX - shear.x * rotatedY) / determinant;
    const float shearedY = (rotatedY - shear.y * rotatedX) / determinant;
    return {pivot.x + shearedX / safeSignedDenominator(scale.x),
            pivot.y + shearedY / safeSignedDenominator(scale.y)};
}

class Transform2DNode final : public TextureNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"transform_2d", 1, "2D Transform", "Coordinates",
            {{"coordinates", "Coordinates", SocketContract::VectorNumeric, SocketDirection::Input, true},
             {"translation", "Translation", SocketContract::VectorNumeric, SocketDirection::Input, true},
             {"rotation", "Rotation", SocketContract::Numeric, SocketDirection::Input, true},
             {"scale", "Scale", SocketContract::VectorNumeric, SocketDirection::Input, true},
             {"shear", "Shear", SocketContract::VectorNumeric, SocketDirection::Input, true},
             {"pivot", "Pivot", SocketContract::VectorNumeric, SocketDirection::Input, true},
             {"coordinates", "Coordinates", SocketContract::VectorNumeric, SocketDirection::Output}},
            {{"rotation", "Rotation", 0.0F, -6.2831853F, 6.2831853F}}};
        result.sockets[0].fieldDefault = true;
        result.sockets.back().typePolicy = SocketDescriptor::TypePolicy::VectorPromotion;
        result.sockets.back().typeInputs = {"coordinates", "translation", "rotation",
                                           "scale", "shear", "pivot"};
        return result;
    }

    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }

    void evaluate(EvaluationContext& context, std::span<const Value> inputs,
                  std::span<Value> outputs) override {
        // An omitted Coordinates socket denotes the canvas coordinate *field*,
        // not a constant vec2. This preserves the documented default behavior.
        const VectorInput coordinates = procedural::coordinateInput(inputs, 0);
        const VectorInput translation = vectorInput(inputs, 1, {});
        const auto rotation = procedural::numericParameterInput(
            inputs, 2, parameters_, "rotation", 0.0F);
        const VectorInput scale = vectorInput(inputs, 3, {1.0F, 1.0F});
        const VectorInput shear = vectorInput(inputs, 4, {});
        const VectorInput pivot = vectorInput(inputs, 5, {0.5F, 0.5F});
        const bool outputIsField = coordinates.isField() ||
            translation.isField() || rotation.isField() || scale.isField() ||
            shear.isField() || pivot.isField();

        if (!outputIsField) {
            outputs[0] = transformInverse(coordinates.constant, translation.constant,
                rotation.constant, scale.constant, shear.constant, pivot.constant);
            return;
        }

        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kTransform2DShader, "2D Transform / compute");
        glUseProgram(program_);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        // The shader's constant coordinate is unused for the normal canvas-default
        // path; use it only when an explicit Float2 is supplied.
        bindVectorInput(program_, 0, "coordinatesImage", "hasCoordinates", "coordinatesConstant", coordinates);
        bindVectorInput(program_, 1, "translationImage", "hasTranslation", "translationConstant", translation);
        procedural::bindNumericInput(program_, 2, "rotationImage", "hasRotation", "rotationConstant", rotation);
        bindVectorInput(program_, 3, "scaleImage", "hasScale", "scaleConstant", scale);
        bindVectorInput(program_, 4, "shearImage", "hasShear", "shearConstant", shear);
        bindVectorInput(program_, 5, "pivotImage", "hasPivot", "pivotConstant", pivot);
        // Override the unconnected-coordinate constant with canvas UV handling.
        // (The branch is written directly to keep the common helper simple.)
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height, ValueType::VectorField};
    }
};

} // namespace

void registerTransform2DNode(NodeRegistry& registry) {
    registry.add(Transform2DNode::describe(), [] { return std::make_unique<Transform2DNode>(); });
}

} // namespace reaction
