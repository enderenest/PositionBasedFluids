#include "fluid/PBFluids.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// ------------------------------------------------------------
// Construction / setup
// ------------------------------------------------------------

PBFluids::PBFluids(const FluidConfig& p)
    : _uboConfig(GL_DYNAMIC_DRAW)
    , _ssboParticles(0)
    , _ssboSolver(0)
    , _ssboHashGrid(0)
    , _ssboOffsets(0)
    , _csPredictAndHash("resources/shaders/compute/CS_PredictAndHash.glsl")
    , _csBitonicSort("resources/shaders/compute/CS_BitonicSort.glsl")
    , _csBuildOffsets("resources/shaders/compute/CS_BuildOffsets.glsl")
    , _csComputeLambdas("resources/shaders/compute/CS_ComputeLambdas.glsl")
    , _csComputeDeltaP("resources/shaders/compute/CS_ComputeDeltaP.glsl")
    , _csIntegrate("resources/shaders/compute/CS_Integrate.glsl")
{
    setParams(p);
}

void PBFluids::setParams(const FluidConfig& p)
{
    _params = p;
    setBounds(_params.boundsMin, _params.boundsMax, _params.boundDamping);

    // Precompute wq exactly once
    const F32 dq = _params.deltaQ * _params.h;
    _wq = calcSCorrKernel(make_pvec3(dq, 0.0f, 0.0f), _params.h);

    // Pack and upload UBO
    FluidConfigUBO uboData;
    uboData.boundsMin = { _minBound.x, _minBound.y, _minBound.z, 0.0f };
    uboData.boundsMax = { _maxBound.x, _maxBound.y, _maxBound.z, 0.0f };
    uboData.gravity_dt = { _params.gravity.x, _params.gravity.y, _params.gravity.z, _params.dt / (F32)_params.substepIterations };

    uboData.h = _params.h;
    uboData.rho0 = _params.rho0;
    uboData.eps = _params.eps;
    uboData.wq = _wq;

    uboData.kCorr = _params.kCorr;
    uboData.nCorr = _params.nCorr;
    uboData.viscosity = _params.viscosity;
    uboData.boundDamping = _boundDamping;

    uboData.hashSize = _params.hashSize;
    uboData.particleCount = _params.particleCount;
    uboData.pad1 = 0u;
    uboData.pad2 = 0u;

    _uboConfig.upload(uboData);
}

void PBFluids::setParticles(const std::vector<Particle>& particles)
{
    _particles = particles;
    const size_t N = _particles.size();

    // Update particle count in params and UBO
    _params.particleCount = (U32)N;
    setParams(_params);

    // Allocate and upload initial particle data to SSBO 0
    _ssboParticles.upload(_particles);

    // Allocate volatile solver buffers
    std::vector<PVec4> dummySolver(N * 2); // 2 vec4s per particle (predPos_lambda, deltaP_rho)
    _ssboSolver.upload(dummySolver);

    std::vector<UVec2> dummyHash(N); // 1 uvec2 per particle (hash, original_id)
    _ssboHashGrid.upload(dummyHash);

    // 1 ivec2 per cell (start, end). Initialize both to -1 (empty)
    std::vector<IVec2> dummyOffsets(_params.hashSize, { -1, -1 });
    _ssboOffsets.upload(dummyOffsets);
}

// ------------------------------------------------------------
// Bounds
// ------------------------------------------------------------

void PBFluids::setBounds(const PVec3& minBound, const PVec3& maxBound, F32 damping)
{
    const PVec3 pad = make_pvec3(_collisionPadding, _collisionPadding, _collisionPadding);

    _minBound = minBound + pad;
    _maxBound = maxBound - pad;
    _boundDamping = std::clamp(damping, 0.0f, 1.0f);
}

// ------------------------------------------------------------
// Main step (GPU Compute Pipeline)
// ------------------------------------------------------------

void PBFluids::step()
{
    if (_particles.empty()) return;

    const U32 N = _params.particleCount;

    // Standard compute shader workgroup size (256 threads per group)
    const U32 workgroupSize = 256;
    const U32 numGroups = (N + workgroupSize - 1) / workgroupSize;
    const U32 gridGroups = (_params.hashSize + workgroupSize - 1) / workgroupSize;

    // Bind all buffers to their respective binding points
    _uboConfig.bindTo(0);

    _ssboParticles.bindTo(0);
    _ssboSolver.bindTo(1);
    _ssboHashGrid.bindTo(2);
    _ssboOffsets.bindTo(3);

    // XPBD Substepping Loop
    for (U32 s = 0; s < _params.substepIterations; ++s) {

        // 1. Predict & Hash
        _csPredictAndHash.use();
        _csPredictAndHash.dispatch(numGroups);
        _csPredictAndHash.wait();

        // 2. Bitonic Sort (Sorting the HashGridBuffer)
        // Bitonic sort requires a power-of-two size
        _csBitonicSort.use();
        _csBitonicSort.setUint("particleCount", N);

        for (U32 k = 2; k <= N; k <<= 1) {
            for (U32 j = k >> 1; j > 0; j >>= 1) {
                _csBitonicSort.setUint("k", k);
                _csBitonicSort.setUint("j", j);
                _csBitonicSort.dispatch(numGroups);
                _csBitonicSort.wait();
            }
        }

        // 3. Build Grid Offsets
        _csBuildOffsets.use();

        // Clear offsets first
        _csBuildOffsets.setUint("clearMode", 1u);
        _csBuildOffsets.dispatch(gridGroups);
        _csBuildOffsets.wait();

        // Build offsets
        _csBuildOffsets.setUint("clearMode", 0u);
        _csBuildOffsets.dispatch(numGroups);
        _csBuildOffsets.wait();

        // 4. Constraint Solving
        for (U32 iter = 0; iter < _params.solverIterations; ++iter) {

            _csComputeLambdas.use();
            _csComputeLambdas.dispatch(numGroups);
            _csComputeLambdas.wait();

            _csComputeDeltaP.use();
            _csComputeDeltaP.dispatch(numGroups);
            _csComputeDeltaP.wait();
        }

        // 5. Integrate & Handle Collisions
        _csIntegrate.use();
        _csIntegrate.dispatch(numGroups);
        _csIntegrate.wait();
    }

    // ------------------------------------------------------------
    // GPU to CPU Readback (For Polyscope Rendering)
    // ------------------------------------------------------------
    Particle* mappedData = _ssboParticles.map(GL_READ_ONLY);
    if (mappedData) {
        // Copy the updated pos/vel data back into the CPU vector
        std::copy(mappedData, mappedData + N, _particles.begin());
        _ssboParticles.unmap();
    }
}