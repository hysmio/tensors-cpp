cmake_minimum_required(VERSION 3.16)

project(llm-cpp
    VERSION 1.0.0
    LANGUAGES CXX)

# Set build directory structure
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-fsanitize=address -g -O0 -I src -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(PkgConfig REQUIRED)

file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.c"
    "src/**/*.cpp"
    "src/**/*.c"
)

file(GLOB_RECURSE HEADERS
    "src/*.hpp"
    "src/*.h"
    "src/**/*.hpp"
    "src/**/*.h"
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE src)

target_precompile_headers(${PROJECT_NAME} PRIVATE src/pch.hpp)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_BUILD)
endif()

# clang-format targets
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
    set(ALL_SOURCE_FILES ${SOURCES} ${HEADERS})

    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i ${ALL_SOURCE_FILES}
        COMMENT "Formatting source files with clang-format"
        VERBATIM
    )

    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT} --dry-run --Werror ${ALL_SOURCE_FILES}
        COMMENT "Checking code formatting with clang-format"
        VERBATIM
    )

    add_custom_target(format-diff
        COMMAND bash -c "for file in ${ALL_SOURCE_FILES}; do if ! ${CLANG_FORMAT} \"$$file\" | diff -u \"$$file\" - >/dev/null 2>&1; then echo \"=== $$file ===\"; ${CLANG_FORMAT} \"$$file\" | diff -u \"$$file\" - || true; echo; fi; done"
        COMMENT "Showing formatting differences"
        VERBATIM
    )

    message(STATUS "clang-format found: ${CLANG_FORMAT}")
    message(STATUS "Added formatting targets: format, format-check, format-diff")
else()
    message(WARNING "clang-format not found - formatting targets will not be available")
endif()

# clang-tidy targets
find_program(CLANG_TIDY "clang-tidy")
if(CLANG_TIDY)
    set(CPP_SOURCE_FILES ${SOURCES})

    add_custom_target(tidy
        COMMAND ${CLANG_TIDY} ${CPP_SOURCE_FILES} --
        COMMENT "Running clang-tidy linting"
        VERBATIM
    )

    add_custom_target(tidy-fix
        COMMAND ${CLANG_TIDY} --fix ${CPP_SOURCE_FILES} --
        COMMENT "Running clang-tidy with auto-fix"
        VERBATIM
    )

    add_custom_target(tidy-check
        COMMAND ${CLANG_TIDY} --warnings-as-errors='*' ${CPP_SOURCE_FILES} --
        COMMENT "Running clang-tidy with warnings as errors"
        VERBATIM
    )

    message(STATUS "clang-tidy found: ${CLANG_TIDY}")
    message(STATUS "Added linting targets: tidy, tidy-fix, tidy-check")
else()
    message(WARNING "clang-tidy not found - linting targets will not be available")
endif()
