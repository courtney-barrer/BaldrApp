cmake_minimum_required(VERSION 3.16)

project(baldr_paranal_simulator LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(IMAGESTREAMIO_ROOT "" CACHE PATH "ImageStreamIO installation or source directory")

find_package(Threads REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

find_path(IMAGESTREAMIO_INCLUDE_DIR NAMES ImageStreamIO.h
          HINTS ${IMAGESTREAMIO_ROOT} ${IMAGESTREAMIO_ROOT}/include)
find_library(IMAGESTREAMIO_LIBRARY NAMES ImageStreamIO
             HINTS ${IMAGESTREAMIO_ROOT} ${IMAGESTREAMIO_ROOT}/lib ${IMAGESTREAMIO_ROOT}/lib64)

if(IMAGESTREAMIO_INCLUDE_DIR AND IMAGESTREAMIO_LIBRARY)
    add_library(ImageStreamIO::ImageStreamIO UNKNOWN IMPORTED)
    set_target_properties(ImageStreamIO::ImageStreamIO PROPERTIES
        IMPORTED_LOCATION "${IMAGESTREAMIO_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${IMAGESTREAMIO_INCLUDE_DIR}")
elseif(IMAGESTREAMIO_INCLUDE_DIR AND EXISTS "${IMAGESTREAMIO_INCLUDE_DIR}/ImageStreamIO.c")
    add_library(baldr_imagestreamio STATIC "${IMAGESTREAMIO_INCLUDE_DIR}/ImageStreamIO.c")
    target_include_directories(baldr_imagestreamio PUBLIC "${IMAGESTREAMIO_INCLUDE_DIR}")
    add_library(ImageStreamIO::ImageStreamIO ALIAS baldr_imagestreamio)
else()
    message(FATAL_ERROR
        "ImageStreamIO was not found. Install it system-wide or configure with "
        "-DIMAGESTREAMIO_ROOT=/path/to/libImageStreamIO. The directory must contain "
        "ImageStreamIO.h and either libImageStreamIO or ImageStreamIO.c.")
endif()

function(add_baldr_simulator executable source)
    add_executable(${executable} ${source})
    target_link_libraries(${executable} PRIVATE ImageStreamIO::ImageStreamIO Threads::Threads)
    target_compile_options(${executable} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
    # Preserve the paths expected by heimbal_simulation_servers.sh.
    set_target_properties(${executable} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endfunction()

add_baldr_simulator(shm_creator_sim shm_creator_sim.cpp)
target_link_libraries(shm_creator_sim PRIVATE nlohmann_json::nlohmann_json)

add_baldr_simulator(sim_mdm_server sim_mdm_server.cpp)
