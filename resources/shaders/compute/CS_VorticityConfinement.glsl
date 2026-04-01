#version 430 core

layout(local_size_x = 256) in;

const float PI = 3.14159265358979323846;

// =========================================================================
// UBO: Global Fluid Configuration (Binding 0)
// =========================================================================
layout(std140, binding = 0) uniform FluidConfig {
    vec4 boundsMin;
    vec4 boundsMax;
    vec4 gravity_dt;

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
// SSBOs (Binding 0, 1, 2, 3))
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
    vec4 deltaP_rho;     // .xyz reused for curl storage between passes
};

layout(std430, binding = 1) buffer SolverBuffer {
    SolverData solver[];
};

layout(std430, binding = 2) buffer HashGridBuffer {
    uvec2 hashGrid[];
};

layout(std430, binding = 3) buffer CellOffsetBuffer {
    ivec2 offsets[];
};

// =========================================================================
// Uniforms
// =========================================================================
uniform uint  pass;             // 0 = compute curl, 1 = apply vorticity force
uniform float vorticityEpsilon;

// =========================================================================
// Hash + Kernel
// =========================================================================
uint getGridHash(ivec3 cell) {
    const int p1 = 73856093, p2 = 19349663, p3 = 83492791;
    int n = (cell.x * p1) ^ (cell.y * p2) ^ (cell.z * p3);
    return uint(n) % ubo.hashSize;
}

vec3 calcGradSpikyPow3Kernel(vec3 r, float h) {
    float dist = length(r);
    if (dist > h || dist < 1e-5) return vec3(0.0);

    float coeff = 45.0 / PI;
    float h6 = h * h * h * h * h * h;
    float diff = h - dist;

    return r * (-diff * diff * (coeff / h6) / dist);
}

// =========================================================================
// MAIN
// =========================================================================
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= ubo.particleCount) return;

    vec3 pos_i = particles[id].pos.xyz;
    vec3 vel_i = particles[id].vel.xyz;
    float h    = ubo.h;

    ivec3 cellCoord = ivec3(floor(pos_i / h));

    if (pass == 0u) {
        // =====================================================================
        // PASS 0: Compute curl  w_i = (1/rho0) * sum_j (v_j - v_i) x gradW_ij
        // =====================================================================
        vec3 curl = vec3(0.0);
        float invRho0 = 1.0 / ubo.rho0;

        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            uint nh = getGridHash(cellCoord + ivec3(dx, dy, dz));
            ivec2 se = offsets[nh];
            if (se.x == -1) continue;
            for (int k = se.x; k < se.y; k++) {
                uint j = hashGrid[k].y;
                if (id == j) continue;

                vec3 rij   = pos_i - particles[j].pos.xyz;
                vec3 gradW = calcGradSpikyPow3Kernel(rij, h);
                vec3 vij   = particles[j].vel.xyz - vel_i;

                curl += cross(vij, gradW);
            }
        }

        // SPH normalization: divide by rest density (mass = 1 assumed)
        solver[id].deltaP_rho.xyz = curl * invRho0;
    }
    else {
        // =====================================================================
        // PASS 1: Compute eta = (1/rho0) * grad|w|, then apply vorticity force
        // =====================================================================
        vec3 eta = vec3(0.0);
        float invRho0 = 1.0 / ubo.rho0;

        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            uint nh = getGridHash(cellCoord + ivec3(dx, dy, dz));
            ivec2 se = offsets[nh];
            if (se.x == -1) continue;
            for (int k = se.x; k < se.y; k++) {
                uint j = hashGrid[k].y;
                if (id == j) continue;

                vec3 rij      = pos_i - particles[j].pos.xyz;
                vec3 gradW    = calcGradSpikyPow3Kernel(rij, h);
                float omegaJ  = length(solver[j].deltaP_rho.xyz);

                eta += omegaJ * gradW;
            }
        }

        // SPH normalization
        eta *= invRho0;

        float etaLen = length(eta);
        if (etaLen < 1e-6) return;

        vec3  N     = eta / etaLen;
        vec3  omega = solver[id].deltaP_rho.xyz;
        float dt    = ubo.gravity_dt.w;

        // f_vorticity = epsilon * (N x omega)
        particles[id].vel.xyz += vorticityEpsilon * cross(N, omega) * dt;
    }
}
