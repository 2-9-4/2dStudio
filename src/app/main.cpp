#include "application.hpp"

#include <exception>
#include <iostream>

int main() {
    try {
        reaction::Application application;
        return application.run();
    } catch (const std::exception& error) {
        std::cerr << "Reaction Studio: " << error.what() << '\n';
        return 1;
    }
}

