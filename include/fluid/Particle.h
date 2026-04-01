#ifndef PARTICLE_H
#define PARTICLE_H

#include "core/Types.h"

struct Particle {
	PVec3 pos;
	PVec3 predPos;
	PVec3 vel;

	// FUTURE IMPLEMENTATION:
	// F32 mass;		// It is needed fot computations
	// F32 invMass;	// it allows us to easily represent static particles with invMass = 0.0f
	// I32 phase;		// 0 for fluid, 1 for rigid (rigid will be implemented in the future)
};

// GPU-side particle layout: matches GLSL struct { vec4 pos; vec4 vel; }
struct GpuParticle {
	PVec4 pos;  // xyz = position,  w = unused
	PVec4 vel;  // xyz = velocity,  w = unused
};

#endif