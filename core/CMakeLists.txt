cmake_minimum_required(VERSION 3.28)
project(spartan_core CXX) #package name is just core

# --- Compiler Configuration ---

if(MSVC)
    # MSVC (Visual Studio) Configuration
    add_compile_options(/std:c++latest /utf-8)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)

elseif(MINGW OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # MinGW (GCC) Configuration

    #Enforce C++26 Standard
    set(CMAKE_CXX_STANDARD 26)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # experimental contracts (not used atm, until full support is available)
    add_compile_options(-fcontracts)

    # runtime in dll, not static
    add_link_options(-static-libgcc -static-libstdc++ -static)

    # optimize for native architecture
    add_compile_options(-O3 -march=native)

    message(STATUS "Configuring for MinGW with C++26 and Contracts support.")
endif()

# --- Source Definitions ---

set(SRC_PATH src/org/spartan)

set(API_PATH ${SRC_PATH}/api)

set(CORE_PATH ${SRC_PATH}/core)

set(MATH_PATH ${CORE_PATH}/math)

set(MEMORY_PATH ${CORE_PATH}/memory)



add_library(spartan_core SHARED
        ${API_PATH}/SpartanApi.cpp
        ${CORE_PATH}/SpartanCore.hpp
        ${MATH_PATH}/fuzzy/SpartanFuzzyMath.h
        ${MATH_PATH}/reinforcement/SpartanReinforcement.h
        ${MEMORY_PATH}/ArrayCleaners.h
)

# --- target libraries ---

target_link_libraries(spartan_core PRIVATE stdc++exp)

# --- Include Directories ---

target_include_directories(spartan_core PRIVATE ${SRC_PATH})

# --- Export Configuration ---

set_target_properties(spartan_core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)