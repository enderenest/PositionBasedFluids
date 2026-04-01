#ifndef CONFIG_H
#define CONFIG_H

#include "core/Types.h"

// ---------------------------
// Fluid config
// ---------------------------
struct FluidConfig {
	// time step
	F32 dt = 1.0f / 60.0f;						// macro timestep: 60 FPS
	PVec3 gravity = { 0.0f, -9.81f, 0.0f };

	// PBF / SPH kernels
	F32 h = 0.15f;								// smoothing radius: h/spacing = 2.0 (minimum stable PBF ratio)
	F32 rho0 = 1000.0f;						// rest density for m=1, h=0.2, spacing=0.1
	U32 solverIterations = 2;					// 4 iters compensates for single substep
	U32 substepIterations = 2;					// single substep: halves all GPU work for 40+ FPS
	F32 eps = 0.00001f;							// CFM regularization (scales with neighbor count * rho0)

	// Spawn configuration — 65K particles, dam-break block
	U32  particleCount = 1 << 15;				// 65536 particles
	PVec3 spawnMin = { 0.3f, 0.3f, 0.3f };		// lower corner of fluid block
	PVec3 spawnMax = { 4.4f, 4.4f, 4.4f };		// ~41^3 = 68921 grid slots > 65536
	bool spawnRandom = false;					// grid spawn: uniform spacing, no overlaps
	F32  spacing = 0.1f;						// particle spacing = h/2 (stable packing)
	PVec3 initialVelocity = { 0.0f, 0.0f, 0.0f };

	// Boundary handling
	PVec3 boundsMin = { 0.0f, 0.0f, 0.0f };
	PVec3 boundsMax = { 7.0f, 7.0f, 7.0f };
	F32 boundDamping = 0.5f;					// moderate: prevents energy buildup at walls

	// Neighbor search hash grid
	U32 hashSize = 1 << 17;					// 262144 cells: 4:1 ratio reduces hash collisions

	// Surface tension correction (sCorr)
	bool enableSCorr = true;					// suppress particle clustering at surface
	F32 kCorr = 0.001f;						// tensile correction strength
	F32 nCorr = 4.0f;							// standard exponent
	F32 deltaQ = 0.3f;							// reference distance ratio (0.3 * h)

	// XSPH viscosity
	bool enableViscosity = true;				// energy dissipation for stability
	F32 viscosity = 0.01f;						// light viscosity: damps jitter without killing flow

	// Vorticity confinement
	bool enableVorticity = false;
	F32 vorticityEpsilon = 0.05f;				// re-injects rotational energy lost to damping
};

// ---------------------------
// Deformation config (later)
// ---------------------------
struct DeformConfig {
	bool enabled = false;
	F32 stiffness = 1.0f;  // placeholder
	// add Laplacian/constraints params later
};

// ---------------------------
// Viewer config
// ---------------------------
struct ViewerConfig {
	bool showParticles = true;
	F32 pointRadius = 0.02f;
};

// ---------------------------
// Global config: combines all sub-configs
// ---------------------------
struct Config {
	FluidConfig fluid;
	DeformConfig deform;
	ViewerConfig viewer;
};

#endif