#include "fluid/PBFluids.h"

#include <algorithm>
#include <cmath>


// ------------------------------------------------------------
// Construction / setup
// ------------------------------------------------------------

PBFluids::PBFluids(const FluidConfig& p)
    : _params(p)
    , _neighborSearch(_params.h, (U32)_params.hashSize)
{
    setBounds(_params.boundsMin, _params.boundsMax, _params.boundDamping);
}

void PBFluids::setParams(const FluidConfig& p)
{
    _params = p;
    _neighborSearch = NeighborSearch(_params.h, (U32)_params.hashSize);
}

void PBFluids::setParticles(const std::vector<Particle>& particles)
{
    _particles = particles;

    const I32 N = (I32)_particles.size();
    _lambda.assign(N, 0.0f);
    _deltaP.assign(N, make_pvec3(0.0f, 0.0f, 0.0f));
}

// ------------------------------------------------------------
// Bounds
// ------------------------------------------------------------

void PBFluids::setBounds(const PVec3& minBound, const PVec3& maxBound, F32 damping)
{
    const PVec3 pad = make_pvec3(_collisionPadding, _collisionPadding, _collisionPadding);

    _minBound = minBound + pad;
    _maxBound = maxBound - pad;
    _boundDamping = std::max(0.0f, damping);
}

// ------------------------------------------------------------
// Main step
// ------------------------------------------------------------

void PBFluids::step()
{
    if (_particles.empty()) return;

    // Slice the macro time step into smaller sub-steps
    const F32 macroDt = _params.dt;
    const I32 substeps = _params.substepIterations; 
    const F32 subDt = macroDt / (F32)substeps;

    // Temporarily overwrite the parameter dt so all helper functions use subDt
    _params.dt = subDt;

    // XPBD Substepping Loop
    for (I32 s = 0; s < substeps; ++s) {

        applyForcesAndPredict();

        _neighborSearch.build(_particles);

        // Constraint Solving 
        solverIterationLoop();

        // Update velocity and finalize positions for THIS substep
        updateVelocityFromPred();
        applyViscosityXSPH();
        commitPositions();
    }

    // Restore the original macro dt for the next frame
    _params.dt = macroDt;
}

// ------------------------------------------------------------
// Step subroutines
// ------------------------------------------------------------

void PBFluids::applyForcesAndPredict()
{
    const F32 dt = _params.dt;

    for (auto& p : _particles) {
        if (p.invMass <= 0.0f) {
            p.predPos = p.pos;
            continue;
        }

        // gravity
        p.vel += _params.gravity * dt;

        // predict
        p.predPos = p.pos + p.vel * dt;
    }
}

void PBFluids::solverIterationLoop()
{
    for (I32 iter = 0; iter < _params.solverIterations; ++iter) {
        computeLambdas();
        computeDeltaP();
        applyDeltaP();
        handleCollisions();
    }
}

// ------------------------------------------------------------
// Helpers (math)
// ------------------------------------------------------------

F32 PBFluids::computeDensity(I32 i) const
{
    const F32 h = _params.h;
    const PVec3 xi = _particles[i].predPos;
    const F32 invM = _particles[i].invMass;

    if (invM <= 0.0f) return 0.0f;
    const F32 m = 1.0f / invM;

    F32 rho = 0.0f;

    // 1. Particle's own density contribution
    rho += m * calcDensityKernel(make_pvec3(0.0f, 0.0f, 0.0f), h);

    // 2. Real fluid neighbors contribution
    const auto& neigh = _neighborSearch.getNeighbors(i);
    for (I32 j : neigh) {
        if (i == j) continue;

        if (_particles[j].invMass > 0.0f) {
            const F32 mj = 1.0f / _particles[j].invMass;
            rho += mj * calcDensityKernel(xi - _particles[j].predPos, h);
        }
    }

    /* std::vector<PVec3> ghostVectors = getGhostRelativeVectors(xi);
    for (const PVec3& rij : ghostVectors) {
        rho += m * calcDensityKernel(rij, h);
    } */

    return rho;
}

F32 PBFluids::computeSCorr(const PVec3& dpos) const
{
    if (!_params.enableSCorr) return 0.0f;

    const F32 h = _params.h;
    const F32 k = _params.kCorr;
    const F32 n = _params.nCorr;

    const F32 dq = _params.deltaQ * h;
    const PVec3 rq = make_pvec3(dq, 0.0f, 0.0f);

    const F32 w = calcDensityKernel(dpos, h);
    const F32 wq = calcDensityKernel(rq, h);

    if (wq <= 1e-12f) return 0.0f;

    const F32 ratio = w / wq;
    return -k * std::pow(ratio, n);
}

/* std::vector<PVec3> PBFluids::getGhostRelativeVectors(const PVec3& pos) const
{
    std::vector<PVec3> r_ghosts;
    const F32 h = _params.h;

    // Check X axis walls
    if (pos.x - _minBound.x < h) r_ghosts.push_back(make_pvec3(2.0f * (pos.x - _minBound.x), 0.0f, 0.0f));
    if (_maxBound.x - pos.x < h) r_ghosts.push_back(make_pvec3(-2.0f * (_maxBound.x - pos.x), 0.0f, 0.0f));

    // Check Y axis walls
    if (pos.y - _minBound.y < h) r_ghosts.push_back(make_pvec3(0.0f, 2.0f * (pos.y - _minBound.y), 0.0f));
    if (_maxBound.y - pos.y < h) r_ghosts.push_back(make_pvec3(0.0f, -2.0f * (_maxBound.y - pos.y), 0.0f));

    // Check Z axis walls
    if (pos.z - _minBound.z < h) r_ghosts.push_back(make_pvec3(0.0f, 0.0f, 2.0f * (pos.z - _minBound.z)));
    if (_maxBound.z - pos.z < h) r_ghosts.push_back(make_pvec3(0.0f, 0.0f, -2.0f * (_maxBound.z - pos.z)));

    return r_ghosts;
} */

// ------------------------------------------------------------
// Lambda computation (weighted by invMass)
// ------------------------------------------------------------

void PBFluids::computeLambdas()
{
    const I32 N = (I32)_particles.size();
    const F32 h = _params.h;
    const F32 rho0 = _params.rho0;
    const F32 eps = _params.eps;

    for (I32 i = 0; i < N; ++i) {
        if (_particles[i].invMass <= 0.0f) {
            _lambda[i] = 0.0f;
            continue;
        }

        const PVec3 xi = _particles[i].predPos;
        const F32 rho_i = computeDensity(i);
        const F32 C_i = (rho_i / rho0) - 1.0f;

        if (C_i <= 0.0f) {
            _lambda[i] = 0.0f;
            continue;
        }

        F32 sum_grad_Ci_sq = 0.0f;
        PVec3 grad_Ci_i = make_pvec3(0.0f, 0.0f, 0.0f);

        const auto& neigh = _neighborSearch.getNeighbors(i);
        for (I32 j : neigh) {
            if (i == j) continue;

            const F32 wj = _particles[j].invMass;
            if (wj <= 0.0f) continue;

            // DÜZELTME: Kütle (mj) deðiþkeni Lambda hesaplamasýna eklendi!
            const F32 mj = 1.0f / wj;
            const PVec3 rij = xi - _particles[j].predPos;
            PVec3 gradW = calcDensityDerivative(rij, h);

            // mj çarpaný eklendi
            PVec3 grad_Ci_j = gradW * (-mj / rho0);
            sum_grad_Ci_sq += dot(grad_Ci_j, grad_Ci_j);

            // mj çarpaný eklendi
            grad_Ci_i += gradW * (mj / rho0);
        }

        sum_grad_Ci_sq += dot(grad_Ci_i, grad_Ci_i);
        _lambda[i] = -C_i / (sum_grad_Ci_sq + eps);
    }
}

// ------------------------------------------------------------
// DeltaP computation
// ------------------------------------------------------------

void PBFluids::computeDeltaP()
{
    const I32 N = (I32)_particles.size();
    const F32 h = _params.h;
    const F32 rho0 = _params.rho0;

    for (I32 i = 0; i < N; ++i) {
        _deltaP[i] = make_pvec3(0.0f, 0.0f, 0.0f);
    }

    for (I32 i = 0; i < N; ++i) {

        const F32 wi = _particles[i].invMass;
        if (wi <= 0.0f) continue;

        const PVec3 xi = _particles[i].predPos;
        PVec3 dp = make_pvec3(0.0f, 0.0f, 0.0f);

        const auto& neigh = _neighborSearch.getNeighbors(i);
        for (I32 j : neigh) {
            if (i == j) continue;

            const F32 wj = _particles[j].invMass;
            if (wj <= 0.0f) continue;

            const F32 mj = 1.0f / wj;

            const PVec3 rij = xi - _particles[j].predPos;
            const PVec3 gradW = calcDensityDerivative(rij, h);

            const F32 scorr = computeSCorr(rij);
            const F32 s = (_lambda[i] + _lambda[j] + scorr);

            dp += gradW * (s * mj);
        }

        _deltaP[i] = dp * (1.0f / rho0);
    }
}

void PBFluids::applyDeltaP()
{
    const I32 N = (I32)_particles.size();
    for (I32 i = 0; i < N; ++i) {
        if (_particles[i].invMass <= 0.0f) continue;
        _particles[i].predPos += _deltaP[i];
    }
}

// ------------------------------------------------------------
// XSPH viscosity
// ------------------------------------------------------------

void PBFluids::applyViscosityXSPH()
{
    const F32 h = _params.h;
    const F32 c = _params.viscosity;
    const I32 N = (I32)_particles.size();

	if (_params.enableViscosity == 0) return;

    std::vector<PVec3> newVelocities(N);

    for (I32 i = 0; i < N; ++i) {
        if (_particles[i].invMass <= 0.0f) {
            newVelocities[i] = _particles[i].vel;
            continue;
        }

        PVec3 v_i = _particles[i].vel;
        PVec3 viscosityForce = make_pvec3(0.0f, 0.0f, 0.0f);

        const auto& neigh = _neighborSearch.getNeighbors(i);
        for (I32 j : neigh) {
            if (i == j) continue;

            const F32 wj = _particles[j].invMass;
            if (wj <= 0.0f) continue;

            PVec3 v_j = _particles[j].vel;

            PVec3 rij = _particles[i].predPos - _particles[j].predPos;
            F32 w = calcViscosityKernel(rij, h);

            const F32 mj = 1.0f / wj;
            viscosityForce += (v_j - v_i) * w * (mj / _params.rho0);
        }

        newVelocities[i] = v_i + viscosityForce * c;
    }

    for (I32 i = 0; i < N; ++i) {
        _particles[i].vel = newVelocities[i];
    }
}

// ------------------------------------------------------------
// Collisions
// ------------------------------------------------------------

void PBFluids::handleCollisions()
{
    for (auto& p : _particles) {
        if (p.invMass <= 0.0f) continue;

        // X Axis
        if (p.predPos.x < _minBound.x) {
            p.predPos.x = _minBound.x;
        }
        else if (p.predPos.x > _maxBound.x) {
            p.predPos.x = _maxBound.x;
        }

        // Y Axis
        if (p.predPos.y < _minBound.y) {
            p.predPos.y = _minBound.y;
        }
        else if (p.predPos.y > _maxBound.y) {
            p.predPos.y = _maxBound.y;
        }

        // Z Axis
        if (p.predPos.z < _minBound.z) {
            p.predPos.z = _minBound.z;
        }
        else if (p.predPos.z > _maxBound.z) {
            p.predPos.z = _maxBound.z;
        }
    }
}

// ------------------------------------------------------------
// Finalize
// ------------------------------------------------------------

void PBFluids::updateVelocityFromPred()
{
    const F32 dt = _params.dt;
    const F32 invDt = (dt > 0.0f) ? (1.0f / dt) : 0.0f;

    for (auto& p : _particles) {
        if (p.invMass <= 0.0f) continue;

        const bool collidedX = (p.predPos.x <= _minBound.x || p.predPos.x >= _maxBound.x);
        const bool collidedY = (p.predPos.y <= _minBound.y || p.predPos.y >= _maxBound.y);
        const bool collidedZ = (p.predPos.z <= _minBound.z || p.predPos.z >= _maxBound.z);

        p.vel = (p.predPos - p.pos) * invDt;

        if (collidedX) p.vel.x *= -_boundDamping;
        if (collidedY) p.vel.y *= -_boundDamping;
        if (collidedZ) p.vel.z *= -_boundDamping;
    }
}

void PBFluids::commitPositions()
{
    for (auto& p : _particles) {
        p.pos = p.predPos;
    }
}