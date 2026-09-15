cmake_minimum_required(VERSION 3.24)
project(reaction_studio VERSION 0.1.0 LANGUAGES CXX)

option(REACTION_BUILD_APP "Build the desktop application" ON)
option(REACTION_BUILD_TESTS "Build automated tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
find_package(nlohmann_json 3.11 REQUIRED)

add_library(reaction_core
    src/core/graph.cpp
    src/core/layout.cpp
    src/core/math.cpp
    src/core/vector_math.cpp
    src/core/persistence.cpp
    src/core/registry.cpp
    src/core/builtin_subgraphs.cpp
    src/core/subgraph.cpp)
target_include_directories(reaction_core PUBLIC include)
target_link_libraries(reaction_core PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(reaction_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

if(REACTION_BUILD_APP)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 3.3 REQUIRED)
    find_package(PNG REQUIRED)

    FetchContent_Declare(glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
        GIT_TAG v0.1.36
        GIT_SHALLOW TRUE)
    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.89.9
        GIT_SHALLOW TRUE)
    FetchContent_Declare(node_editor
        GIT_REPOSITORY https://github.com/thedmd/imgui-node-editor.git
        GIT_TAG v0.9.3
        GIT_SHALLOW TRUE)
    FetchContent_Declare(portable_file_dialogs
        GIT_REPOSITORY https://github.com/samhocevar/portable-file-dialogs.git
        GIT_TAG 0.1.0
        GIT_SHALLOW TRUE)
    set(GLAD_PROFILE "core" CACHE STRING "" FORCE)
    set(GLAD_API "gl=4.3" CACHE STRING "" FORCE)
    FetchContent_MakeAvailable(glad imgui node_editor portable_file_dialogs)

    add_library(reaction_imgui
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
        ${node_editor_SOURCE_DIR}/imgui_node_editor.cpp
        ${node_editor_SOURCE_DIR}/imgui_node_editor_api.cpp
        ${node_editor_SOURCE_DIR}/imgui_canvas.cpp
        ${node_editor_SOURCE_DIR}/crude_json.cpp)
    target_include_directories(reaction_imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${node_editor_SOURCE_DIR})
    target_link_libraries(reaction_imgui PUBLIC glfw glad OpenGL::GL)

    # Focused built-in node families use the *_node.cpp naming convention.
    # CONFIGURE_DEPENDS keeps new family sources discoverable after reconfigure,
    # while registration itself remains explicit in builtin_node_families.inc.
    file(GLOB REACTION_GPU_NODE_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/*_node.cpp")

    add_library(reaction_gpu
        src/gpu/gpu_runtime.cpp
        src/gpu/shader_ir.cpp
        src/gpu/input_nodes.cpp
        src/gpu/nodes.cpp
        ${REACTION_GPU_NODE_SOURCES}
        )
    target_include_directories(reaction_gpu PUBLIC include PRIVATE shaders)
    target_link_libraries(reaction_gpu PUBLIC reaction_core glad OpenGL::GL PNG::PNG)
    target_compile_options(reaction_gpu PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(reaction_studio
        src/app/application.cpp
        src/app/convolution_presets.cpp
        src/app/node_search.cpp
        src/app/node_widgets.cpp
        src/app/main.cpp)
    target_include_directories(reaction_studio PRIVATE
        include
        ${portable_file_dialogs_SOURCE_DIR})
    target_link_libraries(reaction_studio PRIVATE reaction_core reaction_gpu reaction_imgui)
    target_compile_options(reaction_studio PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(REACTION_BUILD_TESTS)
    enable_testing()
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.1
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(Catch2)
    add_executable(reaction_core_tests tests/core_tests.cpp)
    target_link_libraries(reaction_core_tests PRIVATE reaction_core Catch2::Catch2WithMain)
    add_executable(reaction_convolution_preset_tests
        tests/convolution_preset_tests.cpp
        src/app/convolution_presets.cpp)
    target_include_directories(reaction_convolution_preset_tests PRIVATE src/app)
    target_link_libraries(reaction_convolution_preset_tests PRIVATE
        nlohmann_json::nlohmann_json Catch2::Catch2WithMain)
    add_executable(reaction_node_search_tests
        tests/node_search_tests.cpp
        src/app/node_search.cpp
        src/app/convolution_presets.cpp)
    target_include_directories(reaction_node_search_tests PRIVATE src/app)
    target_compile_definitions(reaction_node_search_tests PRIVATE REACTION_NODE_SEARCH_HEADLESS)
    target_link_libraries(reaction_node_search_tests PRIVATE
        reaction_core Catch2::Catch2WithMain)
    include(${catch2_SOURCE_DIR}/extras/Catch.cmake)
    catch_discover_tests(reaction_core_tests)
    catch_discover_tests(reaction_convolution_preset_tests)
    catch_discover_tests(reaction_node_search_tests)
if(REACTION_BUILD_APP)
    add_executable(layout_probe tools/layout_probe.cpp)
    target_link_libraries(layout_probe PRIVATE reaction_core nlohmann_json::nlohmann_json)
    target_compile_options(layout_probe PRIVATE -Wall -Wextra -Wpedantic)
        add_executable(reaction_gpu_tests tests/gpu_tests.cpp)
        target_link_libraries(reaction_gpu_tests PRIVATE
            reaction_gpu glfw glad OpenGL::GL Catch2::Catch2WithMain)
        catch_discover_tests(reaction_gpu_tests)
    endif()
endif()
