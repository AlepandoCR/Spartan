cmake_minimum_required(VERSION 3.28)
project(spartan_core CXX)

# --- Compiler Configuration ---

# Prefer C++26, but allow fallback to C++2b on compilers without full CXX26 support.
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS ON)

if(MSVC)
    # MSVC (Visual Studio) Configuration
    add_compile_options(/std:c++latest /utf-8)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)

elseif(MINGW)
    # MinGW Configuration (Windows only)

    # experimental contracts (not used atm, until full support is available)
    add_compile_options(-fcontracts)

    # runtime in dll, not static (only for MinGW on Windows)
    add_link_options(-static-libgcc -static-libstdc++ -static)

    # optimize for native architecture
    add_compile_options(-O3 -march=native)

    # GCC 13/14: use C++2b flag if CXX26 is not known
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14")
        add_compile_options(-std=gnu++2b)
    endif()

    message(STATUS "Configuring for MinGW with C++2b fallback.")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # GCC Configuration (Linux/Unix)

    # experimental contracts (not used atm, until full support is available)
    add_compile_options(-fcontracts)

    # optimize for native architecture
    add_compile_options(-O3 -march=native)

    # GCC 13/14: use C++2b flag if CXX26 is not known
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14")
        add_compile_options(-std=gnu++2b)
    endif()

    # Use -fPIC for shared libraries on Linux
    add_compile_options(-fPIC)

    message(STATUS "Configuring for GCC with C++2b fallback and PIC.")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang/LLVM Configuration (macOS, Linux)

    # optimize for native architecture
    add_compile_options(-O3 -march=native)

    # Use -fPIC for shared libraries
    add_compile_options(-fPIC)

    message(STATUS "Configuring for Clang/LLVM.")
endif()

# --- Source Definitions ---

set(SRC_PATH src/org/spartan)

set(API_PATH ${SRC_PATH}/api)

set(INTERNAL_PATH ${SRC_PATH}/internal)

set(MATH_PATH ${INTERNAL_PATH}/math)

set(MEMORY_PATH ${INTERNAL_PATH}/memory)

set(LOGGING_PATH ${INTERNAL_PATH}/logging)

set(ML_PATH ${INTERNAL_PATH}/machinelearning)



add_library(spartan_core SHARED
        # --- API ---
        ${API_PATH}/SpartanApi.cpp

        # --- Engine ---
        ${INTERNAL_PATH}/SpartanEngine.h
        ${INTERNAL_PATH}/SpartanEngine.cpp

        # --- Logging ---
        ${LOGGING_PATH}/SpartanLogger.h
        ${LOGGING_PATH}/SpartanLogger.cpp

        # --- Math: Fuzzy ---
        ${MATH_PATH}/fuzzy/SpartanFuzzyMath.h
        ${MATH_PATH}/fuzzy/SpartanFuzzyMath.cpp

        # --- Math: Reinforcement ---
        ${MATH_PATH}/reinforcement/SpartanReinforcement.h
        ${MATH_PATH}/reinforcement/SpartanReinforcement.cpp

        # --- Math: Metrics ---
        ${MATH_PATH}/metric/SpartanMetrics.h
        ${MATH_PATH}/metric/SpartanMetrics.cpp

        # --- Memory ---
        ${MEMORY_PATH}/ArrayCleaners.h
        ${MEMORY_PATH}/ArrayCleaners.cpp

        # --- Machine Learning: Config ---
        ${ML_PATH}/ModelHyperparameterConfig.h

        # --- Machine Learning: Internal Networks (CRTP — Frontier B) ---
        ${ML_PATH}/network/SpartanAbstractCritic.h
        ${ML_PATH}/network/ContinuousQNetwork.h
        ${ML_PATH}/network/GaussianPolicyNetwork.h
        ${ML_PATH}/network/GatedRecurrentUnitLayer.h
        ${ML_PATH}/network/NestedAutoEncoderUnit.h

        # --- Machine Learning: Temporal Credit Assignment ---
        ${ML_PATH}/trace/RemorseTraceBuffer.h
        ${ML_PATH}/trace/RemorseTraceBuffer.cpp

        # --- Machine Learning: Experience Replay ---
        ${ML_PATH}/replay/ExperienceReplayBuffer.h

        # --- Machine Learning: Persistence ---
        ${ML_PATH}/persistence/SpartanPersistence.h
        ${ML_PATH}/persistence/SpartanPersistence.cpp

        # --- Machine Learning: Model Hierarchy (Frontier A) ---
        ${ML_PATH}/model/SpartanModel.h
        ${ML_PATH}/model/SpartanModel.cpp
        ${ML_PATH}/model/SpartanAgent.h
        ${ML_PATH}/model/SpartanCritic.h
        ${ML_PATH}/model/SpartanCompressor.h
        ${ML_PATH}/model/DefaultSpartanAgent.h
        ${ML_PATH}/model/RecurrentSoftActorCriticSpartanModel.h
        ${ML_PATH}/model/RecurrentSoftActorCriticSpartanModel.cpp
        ${ML_PATH}/model/DoubleDeepQNetworkSpartanModel.h
        ${ML_PATH}/model/DoubleDeepQNetworkSpartanModel.cpp
        ${ML_PATH}/model/AutoEncoderCompressorSpartanModel.h
        ${ML_PATH}/model/AutoEncoderCompressorSpartanModel.cpp
        ${ML_PATH}/model/CuriosityDrivenRecurrentSoftActorCriticSpartanModel.h
        ${ML_PATH}/model/CuriosityDrivenRecurrentSoftActorCriticSpartanModel.cpp

        # --- Machine Learning: Registry ---
        ${ML_PATH}/registry/SpartanModelRegistry.h
        ${ML_PATH}/registry/SpartanModelRegistry.cpp
        src/org/spartan/internal/math/tensor/SpartanTensorMath.h
        src/org/spartan/internal/math/tensor/SpartanTensorMath.cpp
        src/org/spartan/internal/simd/SpartanSimd.h
)

# --- target libraries ---

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_libraries(spartan_core PRIVATE stdc++exp)
endif()

# --- Include Directories ---

target_include_directories(spartan_core PRIVATE ${SRC_PATH})

# --- Export Configuration ---

set_target_properties(spartan_core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)