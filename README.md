# Spartan

![Java 25+](https://img.shields.io/badge/Java-25%2B-ED8B00?style=flat&logo=openjdk&logoColor=white)
![C++ 26](https://img.shields.io/badge/C%2B%2B-26-00599C?style=flat&logo=c%2B%2B&logoColor=white)
[![Maven Central API](https://img.shields.io/maven-central/v/io.github.alepandocr/spartan-api?color=brightgreen&label=spartan-api&logo=apachemaven)](https://central.sonatype.com/artifact/io.github.alepandocr/spartan-api)
[![Maven Central Internal](https://img.shields.io/maven-central/v/io.github.alepandocr/spartan-internal?color=blue&label=spartan-internal&logo=apachemaven)](https://central.sonatype.com/artifact/io.github.alepandocr/spartan-internal)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A high-performance **reinforcement learning and machine learning framework** for Java, combining modern RL algorithms with native C++26 performance and Java 25+ Foreign Function Memory (FFM) interoperability. Spartan provides memory-efficient, low-latency neural network training through shared FFM memory bridges and GRU-based recurrent architectures.

## Features

 - **Multiple RL Algorithms**: DDQN (discrete control), SAC (continuous control), CD-SAC (curiosity-driven exploration), PPO (on-policy optimization), and AutoEncoder (feature compression)
- **High-Performance Native Core**: C++26 implementation with native neural network primitives (GRU, Dense, Gaussian policy networks)
- **Java FFM Integration**: Zero-copy interoperability via Foreign Function Memory for efficient Java-to-C++ communication
- **Shared Memory Architecture**: Pre-allocated memory arenas for low-latency tensor operations
- **Production-Ready**: Published to Maven Central with comprehensive Javadoc
- **Cross-Platform**: Supports Windows (MinGW/MSVC), Linux (GCC), and macOS

## Quick Start

### Maven Dependency

Add Spartan to your `pom.xml`:

```xml
<dependency>
    <groupId>io.github.alepandocr</groupId>
    <artifactId>spartan-api</artifactId>
    <version>version</version>
</dependency>
<dependency>
    <groupId>io.github.alepandocr</groupId>
    <artifactId>spartan-internal</artifactId>
    <version>version</version>
    <classifier>OS</classifier> // change OS to windows, linux, or macos
</dependency>
```

Or Gradle:

```gradle
dependencies {

    implementation("io.github.alepandocr:spartan-api:version") // replace version with latest
    
    runtimeOnly("io.github.alepandocr:spartan-internal:version:OS") // change OS to windows, linux, or macos
}
```

### Basic Usage (Try-with-Resources)

```java
import org.spartan.api.SpartanApi;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.config.DoubleDeepQNetworkConfig;


try (SpartanApi api = new SpartanApiImpl()) {
    // Create a context for managing state tensors
    SpartanContext context = api.createContext("my-configuration");                     
    
    // Create an action manager for discrete action spaces
    SpartanActionManager actions = api.createActionManager();
    
    // Configure a DDQN model for discrete control
    var config = DoubleDeepQNetworkConfig.builder()
        .hiddenLayerNeuronCount(128)
        .learningRate(0.001)
        .build();
    
    var ddqnModel = api.createDoubleDeepQNetwork("ddqn-agent", config, context, actions);
    
    // Train the model (model automatically manages native memory via FFM)
    // ... training loop ...
    
} // Automatic cleanup: all native objects freed, arenas closed
```

### Continuous Control (SAC Example)

```java
import org.spartan.api.SpartanApi;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.config.RecurrentSoftActorCriticConfig;

try (SpartanApi api = SpartanApi.create()) {
    SpartanContext context = api.createContext("sac-agent");
    SpartanActionManager actions = api.createActionManager();
    
    var config = RecurrentSoftActorCriticConfig.builder()
        .actorHiddenLayerNeuronCount(256)
        .learningRate(0.0003)
        .entropyTemperatureAlpha(0.2)
        .build();
    
    var sacModel = api.createRecurrentSoftActorCritic("sac-agent", config, context, actions);
    
    // SAC is naturally suited for continuous action spaces
    // ... training loop ...
}
```

## Architecture

Spartan is organized in three layers:

### **API Module** (`spartan-api`)
Pure Java interfaces and configuration classes:
- `SpartanApi`: Main factory for creating contexts, action managers, and models
- `SpartanContext`: Manages shared memory arena and state tensors
- `SpartanAction`: Atomic action contract used by agents
- `SpartanModel`: Base interface for all algorithm models
- Configuration classes for each algorithm

**Maven Central:** [spartan-api](https://central.sonatype.com/artifact/io.github.alepandocr/spartan-api)

### **Internal Module** (`spartan-internal`)
Java implementation layer with FFM bindings:
- `SpartanApiImpl`: Concrete implementation of the API
- `SpartanNative`: Auto-generated FFM bindings to C++26 core
- Python-based code generator for FFM bridge compilation

**Maven Central:** [spartan-internal](https://central.sonatype.com/artifact/io.github.alepandocr/spartan-internal)

### 3. **C++26 Core** (`core/`)
High-performance neural network implementations:
 - **Proximal Policy Optimization (PPO)**: On-policy policy-gradient optimizer for stable updates
 - **Recurrent Soft Actor-Critic (SAC)**: Twin Q-network critics, Gaussian policy actor, GRU hidden state
 - **Double Deep Q-Network (DDQN)**: Experience replay, target network, ε-greedy exploration
 - **Curiosity-Driven SAC (CD-SAC)**: Intrinsic motivation via prediction error, exploration-exploitation tradeoff
 - **AutoEncoder**: Feature compression, bottleneck learning
- **GRU Cells**: Efficient recurrent memory for temporal dependencies
- **Optimizers**: Adam optimization with learning rate schedules

## Memory Sharing: Java ↔ C++ via FFM

Spartan uses **Foreign Function Memory (FFM)** for zero-copy data exchange between Java and C++:

### Data Flow

**Java Side: SpartanContext**
   - You register `SpartanContextElement`s (sensors) in the context
   - Each element provides its data (e.g., health, position, velocities)
   - On `context.update()`, Java writes all element values into a shared `MemorySegment` buffer (contiguous doubles)
   - Variable-length elements record their valid sizes in a parallel int array

**Shared Memory (MemorySegment)**
   - Allocated via `Arena.ofShared()` 
   - Single contiguous buffer of 64-byte aligned doubles
   - C++ gets direct pointer to this buffer via FFM bindings
   - **Zero-copy**: No serialization, no defensive copies

**C++ Side: Neural Networks**
   - Receive memory address and size from Java via generated FFM bridge (`SpartanNative`)
   - Read observation data directly from the shared buffer
   - Compute forward/backward passes over GRU, Dense, and policy networks
   - Write model weights, gradients, and action outputs back to shared memory

**Java Side: Model Output**
   - Read action values/logits from shared buffer
   - Apply exploration strategy (ε-greedy, SAC entropy sampling)
   - Return selected actions to your game/environment

### Zero-GC Hot Path

- `context.update()` avoids allocations in high-frequency updates
- Pre-cached element arrays indexed by position
- Direct `MemorySegment` writes bypass GC pressure
- Suitable for aggressive game loops (60+ Hz)

## ML Algorithms

### Proximal Policy Optimization (PPO)
**Best for:** On-policy learning for continuous and discrete control with stable policy updates

- Trust-region style clipping to stabilize gradient steps
- Mini-batch, epoch-based optimization for sample-efficient on-policy updates
- Compatible with recurrent policies (GRU) for partially observable environments

### Double Deep Q-Network (DDQN)
**Best for:** Discrete action spaces, deterministic environments

- Twin Q-networks reduce overestimation bias
- Experience replay for sample efficiency
- ε-greedy exploration strategy
- Gated Recurrent Unit (GRU) for temporal dependency learning

### Recurrent Soft Actor-Critic (SAC)
**Best for:** Continuous action spaces, stochastic policies, sample efficiency

- Twin Q-critics for stability
- Gaussian policy actor with learned log-std deviation
- Automatic entropy regularization (α learning)
- Monte Carlo policy gradient with importance sampling
- GRU for temporal modeling

### Curiosity-Driven SAC (CD-SAC)
**Best for:** Sparse reward environments, exploration challenges

- Intrinsic motivation via prediction error (curiosity bonus)
- Combines SAC with curiosity module
- Learns world model in latent space
- Balances extrinsic and intrinsic rewards

### AutoEncoder Compressor
**Best for:** Dimensionality reduction, feature learning, state preprocessing

- Unsupervised feature compression
- Variational bottleneck for regularization
- Encoder-decoder architecture with GRU layers
- Suitable for pre-training representations

## Installation & Building from Source

### Prerequisites

- **Java 25+** (for FFM preview API)
- **C++26 compiler** (or C++2b fallback)
  - Windows: MSVC 2022+, MinGW 13+
  - Linux/macOS: GCC 13+, Clang 17+
- **CMake 3.28+**
- **Gradle 8.5+**
- **Python 3.8+** (for FFM bridge generation)

### Build Steps

```bash
# Clone and enter repo
git clone https://github.com/AlepandoCR/Spartan.git
cd Spartan

# Full build (Java + C++ core)
./gradlew build

# Run tests
./gradlew test

# Publish to local Maven repository
./gradlew publishToMavenLocal
```

## API Documentation

Complete Javadoc is available on Maven Central:

- **spartan-api**: [org.spartan.api.*](https://central.sonatype.com/artifact/io.github.alepandocr/spartan-api) documentation
- **spartan-internal**: [org.spartan.internal.*](https://central.sonatype.com/artifact/io.github.alepandocr/spartan-internal) documentation

Key classes:
- `org.spartan.api.SpartanApi` - Main entry point
- `org.spartan.api.engine.context.SpartanContext` - Memory management
- `org.spartan.api.engine.action.SpartanActionManager` - Action space management
- `org.spartan.api.engine.config.*Config` - Algorithm configurations

## Contributing

Contributions welcome! Please note:

1. **AGPL-3.0 Compliance**: All modifications must remain under AGPL-3.0 (copyleft). Derivative works must share source code.
2. **Java Layer**: Follow standard Gradle conventions; add tests in `src/test/java/`
3. **C++ Extensions**: 
   - Use C++26 features; provide C++2b fallback if needed
   - Add tests in `core/src/` with GTest framework
   - Update CMakeLists.txt for new dependencies
4. **FFM Bridge Updates**: Run `./gradlew :internal:generateNativeBindings` after C++ changes
5. **Testing**: `./gradlew test` runs all Java/C++ tests

## License

Spartan is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).

See [LICENSE](LICENSE) for full terms.

**Summary**: You can use and modify Spartan freely, but any network distribution of a modified version requires making source code available to users.

---

**Repository**: [github.com/AlepandoCR/Spartan](https://github.com/AlepandoCR/Spartan)  
**Author**: Alepando

