#version 430 core

// Match the workgroup size we defined in C++ (256)
layout(local_size_x = 256) in;

// =========================================================================
// UBO: Global Fluid Configuration (Binding 0)
// =========================================================================
layout(std140, binding = 0) uniform FluidConfig {
    vec4 boundsMin;
    vec4 boundsMax;
    vec4 gravity_dt; // xyz: gravity, w: subDt
    
    float h;
    float rho0;
    float eps;
    float wq;
    
    float kCorr;
    float nCorr;
    float viscosity;
    float boundDamping;
    
    uint hashSize;
    uint particleCount;
    uint enableSCorr;
    uint enableViscosity;
} ubo;

// =========================================================================
// SSBOs (Bindings 0, 1, 2)
// =========================================================================

struct Particle {
    vec4 pos;
    vec4 vel;
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

struct SolverData {
    vec4 predPos_lambda;
    vec4 deltaP_rho;
};

layout(std430, binding = 1) buffer SolverBuffer {
    SolverData solver[];
};

layout(std430, binding = 2) buffer HashGridBuffer {
    uvec2 hashGrid[]; // x: grid hash, y: particle ID
};

// =========================================================================
// Hash Function
// =========================================================================
// Standard spatial hashing using large primes. 
uint getGridHash(ivec3 cell) {
    const int p1 = 73856093;
    const int p2 = 19349663;
    const int p3 = 83492791;
    
    int n = (cell.x * p1) ^ (cell.y * p2) ^ (cell.z * p3);
    
    // Modulo ensures the hash fits perfectly into our hashSize buckets
    return uint(n) % ubo.hashSize;
}

// =========================================================================
// MAIN KERNEL
// =========================================================================
void main() {
    uint id = gl_GlobalInvocationID.x;
    
    // Safety check: Don't process out-of-bounds threads
    if (id >= ubo.particleCount) return;

    // 1. Read Particle State
    vec3 pos = particles[id].pos.xyz;
    vec3 vel = particles[id].vel.xyz;
    
    float dt = ubo.gravity_dt.w;
    vec3 gravity = ubo.gravity_dt.xyz;

    // 2. Apply Forces & Predict (Exactly matching CPU)
    vel += gravity * dt;
    vec3 predPos = pos + vel * dt;

    // Update velocity in ParticleBuffer so it persists 
    // (We will use this updated velocity in the XSPH Viscosity step later)
    particles[id].vel.xyz = vel;
    
    // Write prediction to SolverBuffer (and reset lambda to 0 just to be clean)
    solver[id].predPos_lambda = vec4(predPos, 0.0);

    // 3. Spatial Hashing
    // Divide space into cubes of size 'h'
    ivec3 cellCoord = ivec3(floor(predPos / ubo.h));
    uint cellHash = getGridHash(cellCoord);

    // 4. Write Unsorted Data to Hash Grid
    hashGrid[id] = uvec2(cellHash, id);
}