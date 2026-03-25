#ifndef PBFLUIDS_H
#define PBFLUIDS_H

#include <vector>
#include "core/Types.h"
#include "core/Config.h"
#include "fluid/Particle.h"
#include "fluid/Kernels.h"
#include "fluid/NeighborSearch.h"


// ============================================================
// PBFluids
// ------------------------------------------------------------
// CPU prototype of Position-Based Fluids (PBF)
// Step structure follows the standard PBF loop:
//
// 1) for all i: v_i += dt * fext;  x*_i = x_i + dt * v_i
// 2) for all i: find neighbors N_i(x*) -> it will be implemented in NeighborSearch.h / .cpp
// 3) repeat solverIterations:
//      a) for all i: compute lambda_i
//      b) for all i: compute deltaP_i
//      c) collisions (on x*)
//      d) for all i: x*_i += deltaP_i
// 4) for all i: v_i = (x*_i - x_i)/dt;  x_i = x*_i
// ============================================================

class PBFluids {
public:
    // ----------------------------
    // Construction
    // ----------------------------
    PBFluids() = default;
    explicit PBFluids(const FluidConfig& p);

    void setParams(const FluidConfig& p);
    const FluidConfig& params() const { return _params; }

    // Particles
    void setParticles(const std::vector<Particle>& particles);
    std::vector<Particle>& particles() { return _particles; }
    const std::vector<Particle>& particles() const { return _particles; }

	// visual radius for collisions (early CPU prototype, Akinci boundary problem exists here)
    void setCollisionPadding(F32 p) { _collisionPadding = p; }
    F32 _collisionPadding = 0.0f;

    // simple AABB bounds for collisions (early CPU prototype, Akinci boundary problem exists here)
    void setBounds(const PVec3& minBound, const PVec3& maxBound, F32 damping = 0.0f);

    // ----------------------------
    // Simulation step
    // ----------------------------
    void step();

private:
    // apply forces + predict x*
    void applyForcesAndPredict();

	// the neighbor search will be implemented in NeighborSearch.h / .cpp

    // solver iterations
    void solverIterationLoop();
    void computeLambdas();  
    void computeDeltaP();    
    void handleCollisions(); 
    void applyDeltaP();     

    // update v and commit positions
    void updateVelocityFromPred();
	void applyViscosityXSPH();
    void commitPositions();

    // ----------------------------
    // Math Helpers
    // ----------------------------
    F32  computeDensity(I32 i) const;
    F32  computeSCorr(const PVec3& dpos) const; // returns 0 if disabled

private:
	// Simulation parameters
    FluidConfig _params;

    // Particle storage
    std::vector<Particle> _particles;

    // Per-step / per-iteration buffers
    std::vector<F32>   _lambda;     // size = N
    std::vector<PVec3> _deltaP;     // size = N

	// Neighbor search structure
    NeighborSearch _neighborSearch{ /*cellSize*/ 0.08f, /*hashSize*/ 1u << 16 };

    // AABB collision
    PVec3 _minBound{ 0,0,0 };   // _minBound = (xmin, ymin, zmin)
    PVec3 _maxBound{ 0,0,0 };   // _maxBound = (xmax, ymax, zmax)
    F32 _boundDamping = 0.0f;

    // Solution to density problem on boundaries
    std::vector<PVec3> getGhostRelativeVectors(const PVec3& pos) const;

    // Derived precomputed constants
    F32 _wq;
};

#endif