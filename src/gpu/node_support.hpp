#pragma once

#include "reaction/gpu/gpu_runtime.hpp"

namespace reaction::node_support {

inline float parameter(const nlohmann::json& values, const char* key, float fallback) {
    return values.contains(key) && values[key].is_number()
        ? values[key].get<float>() : fallback;
}

inline ImageHandle imageAt(std::span<const Value> values, std::size_t index) {
    if (index >= values.size()) return {};
    if (const auto* image = std::get_if<ImageHandle>(&values[index])) return *image;
    return {};
}

inline float floatAt(std::span<const Value> values, std::size_t index, float fallback = 0.0F) {
    if (index >= values.size()) return fallback;
    if (const auto* number = std::get_if<float>(&values[index])) return *number;
    if (const auto* vector = std::get_if<Vec2>(&values[index])) return vector->x;
    return fallback;
}

inline void uniform(GLuint program, const char* name, float value) {
    glUniform1f(glGetUniformLocation(program, name), value);
}

inline void uniform(GLuint program, const char* name, int value) {
    glUniform1i(glGetUniformLocation(program, name), value);
}

inline void bindTexture(int unit, GLuint texture) {
    glActiveTexture(static_cast<GLenum>(GL_TEXTURE0 + unit));
    glBindTexture(GL_TEXTURE_2D, texture);
}

class ParameterNode : public NodeInstance {
public:
    // Fully lowered nodes inherit this empty fallback. Native nodes override it;
    // an unavailable generated input therefore propagates an empty value without
    // requiring a second per-node implementation.
    void evaluate(EvaluationContext&, std::span<const Value>,
                  std::span<Value> outputs) override {
        for (auto& output : outputs) output = {};
    }
    [[nodiscard]] nlohmann::json parameters() const override { return parameters_; }
    void setParameters(const nlohmann::json& values) override { parameters_ = values; }

protected:
    nlohmann::json parameters_ = nlohmann::json::object();
};

class TextureNode : public ParameterNode {
public:
    ~TextureNode() override {
        if (texture_ != 0) glDeleteTextures(1, &texture_);
        if (program_ != 0) glDeleteProgram(program_);
    }

protected:
    void ensure(EvaluationContext& context, GLenum format = GL_RGBA16F) {
        auto& gpu = *static_cast<GpuRuntime*>(context.gpu);
        gpu.ensureTexture(texture_, width_, height_, context.width, context.height, format);
    }

    GLuint texture_ = 0;
    GLuint program_ = 0;
    int width_ = 0;
    int height_ = 0;
};

} // namespace reaction::node_support
