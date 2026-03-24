#ifndef PARTICLE_H
#define PARTICLE_H

#include "core/Types.h"

struct Particle {
	PVec3 pos;
	PVec3 predPos;
	PVec3 vel;
	F32 invMass;
	I32 phase; // 0 for fluid, 1 for rigid (rigid will be implemented in the future)
};

#endif