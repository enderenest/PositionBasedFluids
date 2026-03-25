#ifndef CONFIG_H
#define CONFIG_H

#include "core/Types.h"

// ---------------------------
// Fluid config (CPU PBF now, later convert it to GPU format)
// ---------------------------
struct FluidConfig {
	// time step - use fixed small dt with substeps for stability
	F32 dt = 1.0f / 60.0f;						// macro timestep: 60 FPS
	PVec3 gravity = { 0.0f, -9.81f, 0.0f };		// realistic gravity

	// PBF / SPH kernels
	F32 h = 0.20f;								// smoothing radius: h/spacing = 2.0 is the minimum stable ratio for PBF
	F32 rho0 = 1000.0f;							// rest density calibrated to natural packing: ~1024 for m=1, h=0.2, spacing=0.1
	I32 solverIterations = 3;					// PBF needs 5-10 iterations to converge
	I32 substepIterations = 2;					// smaller substeps improve stability
	F32 eps = 0.0001f;							// constraint regularization

	// Spawn configuration
	I32  particleCount = 1024;					// 64^3 cube = reasonable for CPU testing
	PVec3 spawnMin = { 0.25f, 0.25f, 0.25f };	// centered region
	PVec3 spawnMax = { 1.75f, 1.75f, 1.75f };	// 2 unit cube
	bool spawnRandom = false;					// grid spawn: ensures uniform spacing, no initial overlaps
	F32  spacing = 0.1f;						// matches smoothing radius (critical for neighbor search)
	PVec3 initialVelocity = { 0.0f, 0.0f, 0.0f };

	// Boundary handling
	PVec3 boundsMin = { 0.0f, 0.0f, 0.0f };		// lower corner
	PVec3 boundsMax = { 2.0f, 2.0f, 2.0f };		// upper corner (1 unit margin on each side)
	F32 boundDamping = 0.5f;					

	// Neighbor search hash grid
	U32 hashSize = 1 << 16;						// 65536 cells (increased from 1<<14 for 4K particles)

	// Surface tension correction (sCorr) - essential for surface stability
	bool enableSCorr = true;					// enable to suppress particle clustering
	F32 kCorr = 0.001f;							// tensile correction strength (needs to be non-trivial with unclamped lambdas)
	F32 nCorr = 4.0f;							// standard exponent
	F32 deltaQ = 0.3f;							// reference distance ratio: 0.5 * h = 0.1 = particle spacing

	// XSPH viscosity - moderate damping
	bool enableViscosity = true;				// enable for energy dissipation
	F32 viscosity = 0.0f;						// low viscosity to preserve fluid dynamics
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