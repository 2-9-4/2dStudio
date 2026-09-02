#include "application.hpp"

#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char* argv[]) {
    try {
        if (argc > 2) {
            std::cerr << "Usage: " << argv[0] << " [project.reaction.json]\n";
            return 2;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0] << " [project.reaction.json]\n";
            return 0;
        }
        reaction::Application application(argc == 2 ? argv[1] : "");
        return application.run();
    } catch (const std::exception& error) {
        std::cerr << "Reaction Studio: " << error.what() << '\n';
        return 1;
    }
}
