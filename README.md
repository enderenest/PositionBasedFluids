# Position-Based Fluids (PBF)

> A high-performance, GPU-accelerated Position-Based Fluids simulator written in C++ with GLSL compute shaders. Real-time fluid simulation with advanced features like radix sorting, adaptive LOD, vorticity confinement, and interactive particle manipulation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)]()
[![OpenGL 4.3+](https://img.shields.io/badge/OpenGL-4.3+-green.svg)]()

## Overview

**Position-Based Fluids** is a production-ready fluid simulation system that combines the stability and efficiency of position-based dynamics with the scalability of GPU computing. It simulates incompressible fluids at interactive framerates with up to **131,072 particles** (2^17), suitable for games, visual effects, and scientific visualization.

The simulator implements the classical PBF algorithm (Macklin & Müller, 2013) with modern GPU optimizations:
- **Fully GPU-accelerated** particle updates via compute shaders
- **Radix sort** for O(n) spatial sorting instead of O(n log n)
- **Adaptive LOD** for view-dependent solver iterations
- **Surface tension + viscosity** for realistic fluid behavior
- **Vorticity confinement** to preserve rotational energy
- **Interactive mouse manipulation** with real-time parameter tuning

## Key Features

### 🚀 Performance
- **Real-time simulation** at 60 FPS with 65K–131K particles
- **GPU-driven pipeline** using OpenGL 4.3+ compute shaders
- **Radix sort** for O(n) neighbor search acceleration (replacing O(n log n) bitonic sort)
- **Adaptive solver LOD** with per-particle iteration assignment based on camera distance
- **SIMD optimizations** (AVX2) on CPU for grid initialization and collision handling

### 🌊 Physics & Visuals
- **Incompressibility constraint** via lambda-based density correction
- **Surface tension correction (s-Corr)** to suppress clustering and improve surface quality
- **XSPH viscosity** for smooth energy dissipation
- **Vorticity confinement** to counteract numerical damping and preserve vortices
- **Cohesion** for droplet formation and surface dynamics
- **Mouse interaction** — pull/push particles in a radius-based region
- **Boundary handling** with AABB collisions and damping control

### 🎮 Scene Presets
Six configurable scenes demonstrate different fluid behaviors:
1. **Default Test** — balanced parameters for general exploration
2. **Dam Break** — dramatic height-differential collapse (classic benchmark)
3. **Dripping Faucet** — steady droplet formation with fine spacing
4. **Large Tank** — 131K particles scaling test with large domain
5. **High-Speed Impact** — fast-moving fluid collision (invokes viscosity importance)
6. **Viscous Goo** — high-viscosity non-Newtonian fluid (gel/slime)

### 🎚️ Real-Time Control
All simulation parameters are tunable from the ImGui overlay:
- Solver iterations & substeps
- Particle count & spawn region
- Gravity, viscosity, surface tension
- Solver enablement (vorticity, viscosity, cohesion, s-Corr)
- Camera pan/orbit and particle visualization settings

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Fluid Simulation                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │      CPU: Particle Management            │
        │  ├─ Force application & prediction      │
        │  ├─ Parameter updates                   │
        │  └─ UI/Scene management (ImGui)        │
        └─────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │    GPU: Compute Shader Pipeline         │
        │  1. PredictAndHash                      │
        │  2. RadixSort (Histogram → Prefix)      │
        │  3. BuildOffsets                        │
        │  4. ComputeLambdas (pressure)           │
        │  5. ComputeDeltaP (corrections)         │
        │  6. ApplyDeltaP & collisions            │
        │  7. Integrate (velocity → position)     │
        │  8. VorticityConfinement                │
        │  9. ComputeLOD (adaptive iterations)    │
        └─────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │   OpenGL: Rendering (point sprites)     │
        │  ├─ Per-particle vertex/fragment        │
        │  └─ Depth-based shading                │
        └─────────────────────────────────────────┘
```

### Module Breakdown

#### **Core** (`include/core/`)
- `config.h` — Global configuration structures for fluid, deformation (stub), and viewer settings
- `Scene.h` — Scene presets with pre-tuned parameters
- `Types.h` — Type aliases (PVec3, F32, etc.) for clarity

#### **Fluid Solver** (`include/fluid/`)
- **`PBFluids.h`** — Core solver orchestrating all simulation steps:
  - Force application & prediction
  - Neighbor search integration
  - Solver iterations (lambda computation, constraint solving)
  - Velocity updates & position commitment
  
- **`Particle.h`** — CPU-side particle structure (position, predicted position, velocity) + GPU-side layout
  
- **`NeighborSearch.h`** — Spatial hash grid for neighbor queries:
  - 3D cell hashing with configurable grid resolution
  - O(1) neighbor lookup after GPU sorting
  
- **`Kernels.h`** — SPH kernel functions:
  - Poly6 (for density)
  - Spiky Pow2/Pow3 (for pressure)
  - Cubic (alternative kernel)
  - s-Corr & XSPH variants

#### **OpenGL Abstraction** (`include/opengl/`)
- `SSBO.h` — Shader Storage Buffer Objects (read/write GPU memory)
- `UBO.h` — Uniform Buffer Objects (read-only parameter blocks)
- `ComputeShader.h` — Wrapper for compute shader dispatch
- `VAO.h`, `VBO.h`, `EBO.h` — Vertex/element buffer abstractions for rendering
- `openglErrorReporting.h` — Debug utilities for GPU errors

#### **Compute Shaders** (`resources/shaders/compute/`)
- **`CS_PredictAndHash.glsl`** — Apply forces, predict positions, compute spatial hash keys
- **`CS_RadixHistogram.glsl`** — Count sort distribution (part of radix sort)
- **`CS_RadixPrefixSum.glsl`** — Inclusive prefix sum (parallel scan)
- **`CS_RadixScatter.glsl`** — Reorder particles by sorted hash (completes radix sort)
- **`CS_BuildOffsets.glsl`** — Mark grid cell start/end indices
- **`CS_ComputeLambdas.glsl`** — Density constraint multipliers (λ_i)
- **`CS_ComputeDeltaP.glsl`** — Position corrections (Δp_i) for density satisfaction
- **`CS_ApplyDeltaP.glsl`** — Apply corrections + AABB collisions
- **`CS_Integrate.glsl`** — Update velocity from position change, commit new position
- **`CS_VorticityConfinement.glsl`** — Reintroduce rotational energy
- **`CS_ComputeLOD.glsl`** — Per-particle solver iteration assignment (APBF)

#### **Graphics Shaders** (`resources/shaders/graphics/`)
- **`VS_Particle.glsl`** — Vertex shader (camera transform + point size)
- **`FS_Particle.glsl`** — Fragment shader (depth-based shading of spheres)

## Technical Deep Dive

### Position-Based Fluids Algorithm

The solver executes the following loop each frame:

```
Inputs: particles {x_i, v_i}, dt, parameters {h, rho0, k, viscosity, ...}

1. PREDICT
   for each particle i:
      v_i += dt * g
      x*_i = x_i + dt * v_i

2. NEIGHBOR SEARCH (GPU)
   Hash particles to grid cells
   Radix sort by hash key (O(n))
   Build offset table for grid cell lookups

3. SOLVER LOOP (GPU, repeat 3–7 times)
   a. COMPUTE DENSITY
      for each particle i:
         ρ_i = ∑_j m_j * W(x*_i - x*_j, h)
      
   b. COMPUTE LAMBDA (density correction multiplier)
      for each particle i:
         λ_i = -ρ_i / (∑_j [∇W(...)]² + ε)
      
   c. COMPUTE CORRECTION
      for each particle i:
         Δp_i = (1/ρ0) * ∑_j (λ_i + λ_j) * ∇W(...)
      
   d. HANDLE COLLISIONS
      Clamp x* to AABB bounds (with damping)
      
   e. APPLY CORRECTION
      x*_i += Δp_i

4. UPDATE VELOCITY & POSITION
   for each particle i:
      v_i = (x*_i - x_i) / dt
      Apply XSPH viscosity smoothing
      x_i = x*_i

5. OPTIONAL: VORTICITY CONFINEMENT
   Reintroduce rotational energy to counteract dissipation
```

**Why Position-Based?**
- Unconditionally stable (no CFL constraints)
- Direct incompressibility enforcement
- Simple collision handling
- Fast convergence (3–7 iterations suffice)

### Radix Sort Implementation

Instead of GPU bitonic sort (O(n log²n)), we use **radix sort** for O(n) complexity:

1. **Histogram** — count particles in each bit-bucket
2. **Prefix Sum** — compute exclusive scan (parallel)
3. **Scatter** — reorder particles by sorted keys

**Benefit**: For 65K particles, radix sort is ~4× faster than bitonic sort. Enables real-time sorting each frame.

### Adaptive LOD (APBF)

Camera-distance-dependent solver iterations:
- **Near particles** (surface, close to camera) → 6–7 iterations (fine detail)
- **Far particles** (interior, far from camera) → 2–3 iterations (coarse LOD)
- Per-particle LOD computed once per frame based on distance to camera

Reduces compute cost for distant fluid regions while maintaining visual quality near the viewer.

### Surface Tension (s-Corr)

Suppresses particle clustering through a scaled kernel-based correction:
```
s_corr = -(k * W_corr(x_ij) / W_corr(q * h))^n
```

where `q = 0.3` and `n = 4`. Prevents artificial fluid clumping at low densities.

### Vorticity Confinement

Counteracts damping from viscosity by reintroducing rotational energy:
```
ω = ∇ × v  (vorticity)
f_conf = ε * (∇|ω| × ω)
```

Essential for preserving swirls and eddies in the fluid.

### Neighbor Search: Spatial Hash Grid

Efficient O(1) neighbor queries:

1. **Hash function**: 3D cell coordinates → hash key
   ```
   cell = ⌊position / cellSize⌋
   hash = (x * P1 XOR y * P2 XOR z * P3) mod hashSize
   ```
   where P1, P2, P3 are large primes (reduces collisions)

2. **GPU sort**: Reorder particles by hash key → contiguous blocks per cell

3. **Offset table**: Mark start/end indices for each grid cell

4. **Query**: Access neighbors in O(27) operations (3³ neighboring cells)

## GPU Memory Layout

### Particle Data (SSBO Binding 0)
```glsl
struct GpuParticle {
    vec4 pos;  // xyz = position, w = unused
    vec4 vel;  // xyz = velocity, w = unused
};
```

### Solver Data (SSBO Binding 1)
```glsl
vec4 solverData[particleCount];
// xy = predicted position, z = λ (lambda), w = ρ (density)
```

### Hash Grid (SSBO Binding 2–3)
- Binding 2: Hash grid (sorted pairs: cell hash + particle index)
- Binding 3: Offsets (per-cell start/end indices)

### Radix Sort (SSBO Binding 4–5)
- Binding 4: Ping-pong buffer for radix sort
- Binding 5: Histogram (block-level sort tallies)

### LOD Data (SSBO Binding 7)
- Per-particle solver iteration count (1–7)

### Uniform Config (UBO)
- Bounds, gravity, kernels, solver parameters (FluidConfigUBO in std140)

## Building

### Prerequisites
- **C++ 17 compiler** (MSVC, GCC, Clang)
- **OpenGL 4.3+** support
- **CMake 3.16+**

### Dependencies (included)
- **GLFW 3.3.2** — windowing & input
- **GLAD** — OpenGL loader
- **GLM** — math library (header-only)
- **Eigen** — linear algebra (header-only)
- **ImGui (docking branch)** — real-time UI overlay
- **stb_image, stb_truetype** — asset loading

### Compile

```bash
# Clone and build
git clone https://github.com/enderenest/PositionBasedFluids.git
cd PositionBasedFluids
mkdir build && cd build

# Windows (MSVC)
cmake -G "Visual Studio 17 2022" ..
cmake --build . --config Release

# Linux/macOS
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Run

```bash
./HybridSim
```

The binary will start in windowed mode with a default fluid scene.

## Controls

### Camera
- **Right Mouse Drag** — Orbit camera around fluid
- **Middle Mouse Scroll** — Zoom in/out
- **W/A/S/D** — Pan camera (when holding Alt)

### Interaction
- **Left Mouse Drag** — Pull particles toward cursor (mouse interaction)
- **Shift + Left Drag** — Push particles away from cursor

### UI (ImGui)
- **Scene Select** — Dropdown to switch presets
- **Solver Iterations** — Adjust constraint solving passes
- **Gravity/Viscosity/Cohesion** — Real-time parameter tuning
- **Enable/Disable Effects** — Toggle vorticity, viscosity, surface tension
- **Camera Controls** — Adjust pitch, yaw, distance, point size


## Scenes Explained

### 1. **Default Test**
- **Purpose**: Baseline for parameter tuning
- **Setup**: ~65K particles in a 3×5×3 block
- **Physics**: Moderate gravity, 3 solver iterations
- **Use case**: Learning the system, safe starting point

### 2. **Dam Break**
- **Purpose**: Benchmark classic fluid dynamics test case
- **Setup**: Tall column (0.1–2.5 in X, 0.1–5.8 in Y/Z) collapses across floor
- **Physics**: 4 solver iterations, realistic damping
- **Expected behavior**: Column spreads radially, forms surface waves

### 3. **Dripping Faucet**
- **Purpose**: Surface tension & cohesion demonstration
- **Setup**: Small, dense spawn region (fine 0.02 spacing)
- **Physics**: Weak gravity, high cohesion → droplet formation
- **Visual**: Steady stream breaks into spheres

### 4. **Large Tank**
- **Purpose**: Scalability test (131K particles, 2^17 particles)
- **Setup**: Full tank fill (10×6×10 domain)
- **Physics**: 4 iterations, large hash grid (2^19 cells)
- **Performance**: Stress test for GPU memory & bandwidth

### 5. **High-Speed Impact**
- **Purpose**: Collision & viscosity importance
- **Setup**: Fast-moving block (vx=8) hits stationary pool
- **Physics**: 6 solver iterations, higher viscosity dampens splashing
- **Visual**: Impact wave, splashing suppression

### 6. **Viscous Goo**
- **Purpose**: Non-Newtonian fluid behavior (gel/slime)
- **Setup**: High rest density (2000), high viscosity (0.19)
- **Physics**: Reduced gravity, strong cohesion
- **Visual**: Sluggish, cohesive flow (like honey)

## Configuration & Tuning

### Critical Parameters

| Parameter | Role | Typical Range | Notes |
|-----------|------|---------------|-------|
| `h` (smoothing radius) | Kernel support | 0.1–0.2 | Rule of thumb: h ≈ 2× spacing |
| `rho0` (rest density) | Target density | 800–2000 | Higher = stiffer incompressibility |
| `solverIterations` | Constraint passes | 2–7 | More = stability but slower |
| `substepIterations` | Time subdivision | 1–3 | Larger dt requires more substeps |
| `viscosity` | Energy damping | 0.05–0.3 | Higher = smoother but stiff |
| `cohesionStrength` | Surface tension | 0.0–0.02 | Higher = more droplets |
| `k_Corr` | Surface corr weight | 0.0001–0.001 | Tuning s-Corr impact |

### Stability Tuning

**If particles explode:**
1. Increase `solverIterations` (to 5–6)
2. Decrease `dt` or increase `substepIterations`
3. Check `h` >= 2× spacing

**If particles stick to ceiling:**
1. Decrease `cohesionStrength`
2. Increase `boundDamping` (up to 1.0)
3. Reduce `kCorr` for s-Corr (surface tension dampens gravity)

**If flow looks stiff/jittery:**
1. Increase `viscosity` slightly
2. Enable vorticity confinement (if off)
3. Reduce `solverIterations` to 3–4

## Code Quality & Organization

### Design Principles
- **Modular**: Fluid solver, neighbor search, rendering are independent
- **GPU-first**: Computation offloaded to compute shaders where feasible
- **Type-safe**: Custom types (PVec3, F32, etc.) document intent
- **Debug-ready**: GPU error reporting, configurable parameters, visualization controls

### Notable Implementation Details
- **std140 layout** in UBO (explicit padding for alignment)
- **Work group synchronization** (barriers) in compute shaders to ensure memory safety
- **Ping-pong buffers** for radix sort scratch space
- **Implicit AABB bounds** (no rigid body integration, just clamping)
- **Per-particle LOD via compute** (avoids CPU scheduling overhead)

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Author

**Ender Enes TAN** - Computer Engineering Student & Researcher specializing in real-time simulation and GPU programming.  
- GitHub: [@enderenest](https://github.com/enderenest)
- Email: enderenest@gmail.com

## Contributing

Contributions are welcome! Please open an issue or pull request for:
- Bug fixes & performance improvements
- Additional scenes or physics features
- Documentation clarifications
- Platform support (Linux, macOS)

---

**Status**: ✅ Stable, real-time capable, actively maintained.

For questions or issues, please open a GitHub issue. Enjoy the fluid simulation!
