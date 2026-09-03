#include "ImageStreamIO.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr int kNumberOfSemaphores = 10;
constexpr int kCred1Width = 320;
constexpr int kCred1Height = 256;
constexpr int kCred1Reads = 5;

fs::path executable_directory(const char* argv0) {
    std::error_code error;
    const fs::path proc_executable = fs::read_symlink("/proc/self/exe", error);
    if (!error) {
        return proc_executable.parent_path();
    }

    const fs::path executable = fs::absolute(argv0, error);
    if (error) {
        throw std::runtime_error("could not determine the executable directory");
    }
    return executable.parent_path();
}

fs::path default_config_path(const char* argv0) {
    return executable_directory(argv0) / "fake_configs" / "cred1_split.json";
}

json read_config(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("could not open configuration file: " + path.string());
    }

    json config;
    try {
        input >> config;
    } catch (const json::exception& error) {
        throw std::runtime_error(
            "could not parse configuration file " + path.string() + ": " + error.what());
    }
    return config;
}

std::array<int, 2> subframe_size(const json& config, const std::string& key) {
    try {
        const int width = config.at(key).at("xsz").get<int>();
        const int height = config.at(key).at("ysz").get<int>();
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("invalid " + key + ": dimensions must be positive");
        }
        return {width, height};
    } catch (const json::exception& error) {
        throw std::runtime_error("invalid " + key + " configuration: " + error.what());
    }
}

template <typename T>
void zero_image(T* data, std::size_t element_count) {
    std::fill_n(data, element_count, T{});
}

int create_image(
    const std::string& name,
    int width,
    int height,
    int depth,
    int datatype) {
    IMAGE image{};
    std::array<uint32_t, 3> dimensions{
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        static_cast<uint32_t>(depth),
    };
    const uint8_t number_of_axes = depth > 1 ? 3 : 2;

    std::cout << "Creating " << static_cast<int>(number_of_axes) << "D image " << name
              << " (" << width << " x " << height;
    if (number_of_axes == 3) {
        std::cout << " x " << depth;
    }
    std::cout << ")\n";

    const errno_t result = ImageStreamIO_createIm_gpu(
        &image, name.c_str(), number_of_axes, dimensions.data(), datatype,
        -1, 1, kNumberOfSemaphores, 0, 0);
    if (result != 0) {
        std::cerr << "ERROR: Could not create " << name
                  << " (ImageStreamIO error " << result << ")\n";
        return EXIT_FAILURE;
    }

    const std::size_t element_count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
        static_cast<std::size_t>(depth);
    switch (datatype) {
        case _DATATYPE_UINT8:  zero_image(image.array.UI8, element_count); break;
        case _DATATYPE_UINT16: zero_image(image.array.UI16, element_count); break;
        case _DATATYPE_UINT32: zero_image(image.array.UI32, element_count); break;
        case _DATATYPE_INT16:  zero_image(image.array.SI16, element_count); break;
        case _DATATYPE_INT32:  zero_image(image.array.SI32, element_count); break;
        case _DATATYPE_FLOAT:  zero_image(image.array.F, element_count); break;
        case _DATATYPE_DOUBLE: zero_image(image.array.D, element_count); break;
        default:
            std::cerr << "WARNING: Unknown datatype; " << name
                      << " was not zero-initialised\n";
    }
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [cred1_split.json]\n";
        return EXIT_FAILURE;
    }

    try {
        const fs::path config_path =
            argc == 2 ? fs::absolute(argv[1]) : default_config_path(argv[0]);
        std::cout << "Using configuration: " << config_path << '\n';
        const json config = read_config(config_path);

        bool creation_failed = false;
        for (int beam = 1; beam <= 4; ++beam) {
            const std::string name = "baldr" + std::to_string(beam);
            const auto [width, height] = subframe_size(config, name);
            creation_failed |=
                create_image(name, width, height, 1, _DATATYPE_INT32) != EXIT_SUCCESS;
        }
        creation_failed |= create_image(
                               "cred1", kCred1Width, kCred1Height, kCred1Reads,
                               _DATATYPE_UINT16) != EXIT_SUCCESS;
        return creation_failed ? EXIT_FAILURE : EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
