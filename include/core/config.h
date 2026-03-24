#ifndef CONFIG_H
#define CONFIG_H

#include "core/Types.h"

// ---------------------------
// Fluid config (CPU PBF now, later convert it to GPU format)
// ---------------------------
struct FluidConfig {
	// time
	F32 dt = 1.0f / 60.0f;
	PVec3 gravity = { 0.0f, -9.81f, 0.0f };

	// PBF / SPH
	F32 h = 0.2f;				// smoothing radius
	F32 rho0 = 2000.0f;			// rest density
	I32 solverIterations = 3;	// number of PBF solver iterations per step
	I32 substepIterations = 2;	// number of substeps (internal steps with smaller dt for stability)
	F32 eps = 0.001;			// small epsilon to prevent division by zero in lambda computation

	// Spawn
	I32  particleCount = 500;
	PVec3 spawnMin = { 0.0f, 0.0f, 0.0f };
	PVec3 spawnMax = { 1.0f, 1.0f, 1.0f };
	bool spawnRandom = true;	// false=grid with spacing, true=random with particleCount
	F32  spacing = 0.08f;		// used if spawnRandom == false
	PVec3 initialVelocity = { 0.0f, 0.0f, 0.0f };
	
	// Boundary handling (simple AABB for now, later convert to signed distance field)
	PVec3 boundsMin = { 0.0f, 0.0f, 0.0f };
	PVec3 boundsMax = { 1.0f, 1.0f, 1.0f };
	F32 boundDamping = 0.95f;

	// Neighbor search hash grid size
	U32 hashSize = 1 << 14;
		 
	// sCorr
	bool enableSCorr = false;
	F32 kCorr = 0.01f;
	F32 nCorr = 4.0f;
	F32 deltaQ = 0.3f;

	// XSPH viscosity
	bool enableViscosity = false;
	F32 viscosity = 0.2f;
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