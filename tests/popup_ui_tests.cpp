#include "node_widgets.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_node_editor.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>

namespace ed = ax::NodeEditor;

namespace {

// No pixel display or OpenGL rendering required: exercise the REAL ImGui
// node-editor suspend/resume and the same popup handler as the application.
struct UiFixture {
    UiFixture() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        auto& io=ImGui::GetIO();
        io.DisplaySize={1280,900};
        io.DeltaTime=1.0F/60.0F;
        unsigned char* pixels=nullptr;
        int width=0,height=0;
        io.Fonts->GetTexDataAsRGBA32(&pixels,&width,&height);
        context=ed::CreateEditor();
        ed::SetCurrentEditor(context);
    }
    ~UiFixture() {
        ed::SetCurrentEditor(nullptr);
        ed::DestroyEditor(context);
        ImGui::DestroyContext();
    }
    ed::EditorContext* context=nullptr;
};

} // namespace

TEST_CASE("node-editor splat picker opens at screen-space mouse position while suspended") {
    using namespace reaction;
    UiFixture ui;
    Graph graph;
    const auto id=graph.addNode("particle_splat",{-900.0F,1700.0F});
    graph.findNode(id)->parameters={{"r",.9F},{"g",.2925F},{"b",.09F}};
    node_widgets::PopupState popup;
    popup.request(node_widgets::PopupKind::SplatInkColor,id);

    auto& io=ImGui::GetIO();
    io.MousePos={740.0F,415.0F};
    ImGui::NewFrame();
    ImGui::SetNextWindowPos({25,25},ImGuiCond_Always);
    ImGui::SetNextWindowSize({1130,780},ImGuiCond_Always);
    ImGui::Begin("Host window");
    ed::SetCurrentEditor(ui.context);
    ed::Begin("Graph");
    ed::BeginNode(ed::NodeId(128));
    ImGui::TextUnformatted("Remote node");
    ed::EndNode();

    // The node editor rewrites canvas-space inputs and window coordinates.
    // Real controls are drawn in BeginNode, but full popups MUST be deferred
    // until this overlay phase, where ImGui::GetMousePos() is screen-space.
    ed::Suspend();
    const ImVec2 physicalMouse=ImGui::GetMousePos();
    REQUIRE(physicalMouse.x==Catch::Approx(740.0F));
    REQUIRE(physicalMouse.y==Catch::Approx(415.0F));
    REQUIRE_FALSE(node_widgets::renderPopup(popup,graph));
    REQUIRE(ImGui::IsPopupOpen("Splat ink color"));

    const auto& stack=ImGui::GetCurrentContext()->OpenPopupStack;
    REQUIRE_FALSE(stack.empty());
    REQUIRE(stack.back().Window!=nullptr);
    const auto position=stack.back().Window->Pos;
    INFO("popup="<<position.x<<","<<position.y<<" mouse="
         <<physicalMouse.x<<","<<physicalMouse.y);
    // A popup opened in canvas coordinates (e.g., -900,+1700) would be
    // nowhere near this mouse position; suspension must remove pan/zoom.
    REQUIRE(std::abs(position.x-physicalMouse.x)<180.0F);
    REQUIRE(std::abs(position.y-physicalMouse.y)<180.0F);
    ed::Resume();
    ed::End();
    ImGui::End();
    ImGui::Render();

    // The exact node and its RGB values must remain unchanged merely from
    // opening the picker; selection must never mutate another canvas node.
    REQUIRE(graph.findNode(id)->parameters.at("r")==.9F);
    REQUIRE(graph.findNode(id)->parameters.at("g")==.2925F);
    REQUIRE(graph.findNode(id)->parameters.at("b")==.09F);
}
