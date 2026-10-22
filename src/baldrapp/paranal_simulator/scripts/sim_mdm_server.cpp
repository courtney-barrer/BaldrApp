#include "ImageStreamIO.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <time.h>

#include <pthread.h>

namespace {

constexpr int kDmCount = 4;
constexpr int kDmSize = 12;
constexpr int kVirtualActuatorCount = kDmSize * kDmSize;
constexpr int kChannelCount = 5;
constexpr int kKeywordCount = 10;

using ImageRow = std::unique_ptr<IMAGE[]>;
using ImageArray = std::array<ImageRow, kDmCount>;

ImageArray images;
std::array<pthread_t, kDmCount> control_threads{};
std::array<unsigned int, kDmCount> thread_dm_ids{1, 2, 3, 4};
std::atomic<bool> keep_running{false};
bool time_logging = false;

int create_one_image(IMAGE& image, const std::string& name) {
    constexpr long number_of_axes = 2;
    constexpr uint8_t datatype = _DATATYPE_DOUBLE;
    constexpr int shared = 1;
    std::array<uint32_t, 2> dimensions{kDmSize, kDmSize};

    const errno_t result = ImageStreamIO_createIm_gpu(
        &image, name.c_str(), number_of_axes, dimensions.data(), datatype,
        -1, shared, IMAGE_NB_SEMAPHORE, kKeywordCount, MATH_DATA);
    if (result != 0) {
        std::cerr << "ERROR: Could not create " << name
                  << " (ImageStreamIO error " << result << ")\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int create_shared_memory_images() {
    for (int dm = 0; dm < kDmCount; ++dm) {
        images[dm] = std::make_unique<IMAGE[]>(kChannelCount + 1);

        for (int channel = 0; channel < kChannelCount; ++channel) {
            const std::string name =
                "dm" + std::to_string(dm + 1) + "disp" +
                (channel < 10 ? "0" : "") + std::to_string(channel);
            if (create_one_image(images[dm][channel], name) != EXIT_SUCCESS) {
                return EXIT_FAILURE;
            }
        }

        const std::string combined_name = "dm" + std::to_string(dm + 1);
        if (create_one_image(images[dm][kChannelCount], combined_name) != EXIT_SUCCESS) {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

void* dm_control_loop(void* argument) {
    const unsigned int dm_id = *static_cast<unsigned int*>(argument);
    IMAGE& combined = images.at(dm_id - 1)[kChannelCount];
    std::array<double, kVirtualActuatorCount> combined_map{};

    FILE* timing_file = nullptr;
    if (time_logging) {
        const std::string filename = "speed_log_" + std::to_string(dm_id) + ".log";
        timing_file = std::fopen(filename.c_str(), "w");
    }

    while (keep_running.load()) {
        ImageStreamIO_semwait(&combined, 1);
        if (!keep_running.load()) {
            break;
        }

        for (int actuator = 0; actuator < kVirtualActuatorCount; ++actuator) {
            double value = 0.0;
            for (int channel = 0; channel < kChannelCount; ++channel) {
                value += images[dm_id - 1][channel].array.D[actuator];
            }
            combined_map[actuator] = std::clamp(value, 0.0, 1.0);
        }

        combined.md->write = 1;
        std::copy(combined_map.begin(), combined_map.end(), combined.array.D);
        combined.md->cnt1 = 0;
        ++combined.md->cnt0;
        combined.md->write = 0;

        if (timing_file != nullptr) {
            timespec now{};
            clock_gettime(CLOCK_REALTIME, &now);
            std::fprintf(timing_file, "%f\n", now.tv_sec + 1e-9 * now.tv_nsec);
        }
    }

    if (timing_file != nullptr) {
        std::fclose(timing_file);
    }
    return nullptr;
}

int start_control_threads() {
    keep_running.store(true);
    for (int dm = 0; dm < kDmCount; ++dm) {
        const int result = pthread_create(
            &control_threads[dm], nullptr, dm_control_loop, &thread_dm_ids[dm]);
        if (result != 0) {
            std::cerr << "ERROR: Could not start DM " << dm + 1
                      << " control thread: " << std::strerror(result) << '\n';
            keep_running.store(false);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

}  // namespace

int main() {
    std::cout << "-----------------------------------------------------------------------------\n"
              << "Simulated DM scenario: no drivers connected\n";

    if (create_shared_memory_images() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (start_control_threads() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    // The startup script keeps stdin open with `tail -f /dev/null`.
    std::cout << "DM control loop running. Press Enter to stop.\n";
    std::cin.get();

    // Normally the launcher terminates this process with SIGTERM. Retaining
    // process-lifetime SHM ownership avoids changing that established flow.
    keep_running.store(false);
    return EXIT_SUCCESS;
}
