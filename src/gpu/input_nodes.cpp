#include "nodes_internal.hpp"
#include "node_support.hpp"

#include <glad/glad.h>
#include <png.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace reaction {
namespace {

using node_support::ParameterNode;
using node_support::TextureNode;
using node_support::bindTexture;
using node_support::uniform;

constexpr std::string_view kImageShader = R"GLSL(#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba16f, binding=0) writeonly uniform image2D outputImage;
layout(binding=0) uniform sampler2D sourceImage;
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(outputImage);
    if (any(greaterThanEqual(pixel, size))) return;
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
    imageStore(outputImage, pixel, texture(sourceImage, vec2(uv.x, 1.0 - uv.y)));
})GLSL";

class ImageNode final : public TextureNode {
public:
    ~ImageNode() override {
        if (sourceTexture_ != 0) glDeleteTextures(1, &sourceTexture_);
    }

    static NodeDescriptor describe() {
        return {"image", 1, "Image", "Input",
                {{"image", "Image", ValueType::Image2D, SocketDirection::Output}}, {}};
    }
    const NodeDescriptor& descriptor() const override {
        static const auto value = describe();
        return value;
    }
    void evaluate(EvaluationContext& context, std::span<const Value>,
                  std::span<Value> outputs) override {
        const auto path = parameters_.contains("path") && parameters_["path"].is_string()
            ? parameters_["path"].get<std::string>() : std::string{};
        if (path != loadedPath_) load(path);
        if (sourceTexture_ == 0) {
            outputs[0] = {};
            return;
        }

        ensure(context);
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        if (!program_) program_ = gpu.compileCompute(kImageShader, "Image / compute");
        glUseProgram(program_);
        bindTexture(0, sourceTexture_);
        uniform(program_, "sourceImage", 0);
        glBindImageTexture(0, texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        gpu.dispatch(program_, context.width, context.height);
        outputs[0] = ImageHandle{texture_, context.width, context.height};
    }

private:
    void load(const std::string& path) {
        loadedPath_ = path;
        if (sourceTexture_ != 0) {
            glDeleteTextures(1, &sourceTexture_);
            sourceTexture_ = 0;
        }
        if (path.empty()) return;

        png_image image{};
        image.version = PNG_IMAGE_VERSION;
        if (!png_image_begin_read_from_file(&image, path.c_str())) return;
        image.format = PNG_FORMAT_RGBA;
        if (image.width > static_cast<png_uint_32>(std::numeric_limits<int>::max()) ||
            image.height > static_cast<png_uint_32>(std::numeric_limits<int>::max())) {
            png_image_free(&image);
            return;
        }
        std::vector<png_byte> pixels(PNG_IMAGE_SIZE(image));
        if (!png_image_finish_read(&image, nullptr, pixels.data(), 0, nullptr)) {
            png_image_free(&image);
            return;
        }

        glGenTextures(1, &sourceTexture_);
        glBindTexture(GL_TEXTURE_2D, sourceTexture_);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(image.width),
                     static_cast<GLsizei>(image.height), 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     pixels.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);
        png_image_free(&image);
    }

    GLuint sourceTexture_ = 0;
    std::string loadedPath_;
};
using node_support::floatAt;
using node_support::parameter;

class FloatNode final : public ParameterNode {
public:
    static NodeDescriptor describe() {
        return {"float", 1, "Float", "Input",
                {{"value", "Value", ValueType::Float, SocketDirection::Output}},
                {{"value", "Value", 0.5F, -10.0F, 10.0F}}};
    }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    void evaluate(EvaluationContext&, std::span<const Value>, std::span<Value> outputs) override {
        outputs[0] = parameter(parameters_, "value", 0.5F);
    }
};

class TimeNode final : public ParameterNode {
public:
    static NodeDescriptor describe() {
        auto result = NodeDescriptor{"time", 1, "Time", "Input",
            {{"time", "Time", ValueType::Float, SocketDirection::Output},
             {"delta", "Delta", ValueType::Float, SocketDirection::Output}},
            {{"speed", "Speed", 1.0F, -4.0F, 4.0F}}};
        result.timeDependent = true;
        return result;
    }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    void evaluate(EvaluationContext& context, std::span<const Value>, std::span<Value> outputs) override {
        const float speed = parameter(parameters_, "speed", 1.0F);
        outputs[0] = static_cast<float>(context.time) * speed;
        outputs[1] = static_cast<float>(context.deltaTime) * speed;
    }
};

class FloatPreviewNode final : public ParameterNode {
public:
    static NodeDescriptor describe() {
        return {"float_preview", 1, "Float Preview", "Utility",
                {{"value", "Value", ValueType::Float, SocketDirection::Input, true},
                 {"value", "Value", ValueType::Float, SocketDirection::Output}},
                {{"value", "Value", 0.0F, -10.0F, 10.0F}}};
    }
    const NodeDescriptor& descriptor() const override { static const auto value = describe(); return value; }
    void evaluate(EvaluationContext&, std::span<const Value> inputs, std::span<Value> outputs) override {
        outputs[0] = floatAt(inputs, 0, parameter(parameters_, "value", 0.0F));
    }
};

template <typename T> void addNode(NodeRegistry& registry) {
    registry.add(T::describe(), [] { return std::make_unique<T>(); });
}

} // namespace

void registerInputNodes(NodeRegistry& registry) {
    addNode<ImageNode>(registry);
    addNode<FloatNode>(registry);
    addNode<TimeNode>(registry);
    addNode<FloatPreviewNode>(registry);
}

} // namespace reaction
