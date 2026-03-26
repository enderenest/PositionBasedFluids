#version 430 core

layout(local_size_x = 256) in;

const float PI = 3.14159265358979323846;

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
// SSBOs (Bindings 0, 1, 2, 3)
// =========================================================================
struct Particle {
    vec4 pos;
    vec4 vel;
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

struct SolverData {
    vec4 predPos_lambda; // xyz: predPos, w: lambda
    vec4 deltaP_rho;     // xyz: deltaP, w: rho
};

layout(std430, binding = 1) buffer SolverBuffer {
    SolverData solver[];
};

layout(std430, binding = 2) buffer HashGridBuffer {
    uvec2 hashGrid[]; // x: grid hash, y: particle ID
};

layout(std430, binding = 3) buffer CellOffsetBuffer {
    ivec2 offsets[];  // x: startIndex, y: endIndex
};

// =========================================================================
// KERNEL & HASH FUNCTIONS
// =========================================================================
float calcPoly6Kernel(vec3 r, float h) {
    float h2 = h * h;
    float r2 = dot(r, r);

    if (r2 > h2) return 0.0;

    float h4 = h2 * h2;
    float h9 = h4 * h4 * h;
    float diff = h2 - r2;
    float diff3 = diff * diff * diff;
    
    float coeff = 315.0 / (64.0 * PI);
    return (coeff / h9) * diff3;
}

uint getGridHash(ivec3 cell) {
    const int p1 = 73856093;
    const int p2 = 19349663;
    const int p3 = 83492791;
    int n = (cell.x * p1) ^ (cell.y * p2) ^ (cell.z * p3);
    return uint(n) % ubo.hashSize;
}

// =========================================================================
// MAIN KERNEL
// =========================================================================
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= ubo.particleCount) return;

    // 1. Read necessary state
    vec3 oldPos = particles[id].pos.xyz;
    vec3 oldVel = particles[id].vel.xyz;
    vec3 predPos = solver[id].predPos_lambda.xyz;
    float h = ubo.h;
    
    // =========================================================================
    // XSPH VISCOSITY (Optional)
    // =========================================================================
    vec3 vel = oldVel;
    
    if (ubo.enableViscosity == 1u) {
        vec3 viscosityForce = vec3(0.0);
        ivec3 cellCoord = ivec3(floor(predPos / h));

        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    ivec3 neighborCell = cellCoord + ivec3(x, y, z);
                    uint neighborHash = getGridHash(neighborCell);
                    ivec2 startEnd = offsets[neighborHash];
                    
                    if (startEnd.x != -1) {
                        for (int k = startEnd.x; k < startEnd.y; k++) {
                            uint j = hashGrid[k].y;
                            if (id == j) continue;
                            
                            vec3 xj = solver[j].predPos_lambda.xyz;
                            vec3 vj = particles[j].vel.xyz; // Read neighbor's old velocity
                            
                            vec3 rij = predPos - xj;
                            float w = calcPoly6Kernel(rij, h);
                            
                            viscosityForce += (vj - oldVel) * w * (1.0 / ubo.rho0);
                        }
                    }
                }
            }
        }
        
        // Apply viscosity to the working velocity
        vel += viscosityForce * ubo.viscosity;
    }

    // =========================================================================
    // BOUNDARY COLLISIONS & VELOCITY UPDATE
    // =========================================================================
    // Check collisions on the predicted position
    bool collidedX = false;
    bool collidedY = false;
    bool collidedZ = false;

    if (predPos.x <= ubo.boundsMin.x) { predPos.x = ubo.boundsMin.x; collidedX = true; }
    if (predPos.x >= ubo.boundsMax.x) { predPos.x = ubo.boundsMax.x; collidedX = true; }
    
    if (predPos.y <= ubo.boundsMin.y) { predPos.y = ubo.boundsMin.y; collidedY = true; }
    if (predPos.y >= ubo.boundsMax.y) { predPos.y = ubo.boundsMax.y; collidedY = true; }
    
    if (predPos.z <= ubo.boundsMin.z) { predPos.z = ubo.boundsMin.z; collidedZ = true; }
    if (predPos.z >= ubo.boundsMax.z) { predPos.z = ubo.boundsMax.z; collidedZ = true; }

    // Derive final velocity from the corrected position difference
    float subDt = ubo.gravity_dt.w;
    float invDt = (subDt > 0.0) ? (1.0 / subDt) : 0.0;
    
    // Base velocity update derived from XPBD integration
    vel = (predPos - oldPos) * invDt;

    // Apply boundary damping if a collision occurred
    if (collidedX) vel.x *= -ubo.boundDamping;
    if (collidedY) vel.y *= -ubo.boundDamping;
    if (collidedZ) vel.z *= -ubo.boundDamping;

    // =========================================================================
    // COMMIT POSITIONS
    // =========================================================================
    // Write the finalized state back to the main ParticleBuffer
    particles[id].pos.xyz = predPos;
    particles[id].vel.xyz = vel;
    
    // Clear lambda for the next macro frame (optional, but clean)
    solver[id].predPos_lambda.w = 0.0;
}