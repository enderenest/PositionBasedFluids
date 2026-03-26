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
// SSBOs (Bindings 1, 2, 3)
// =========================================================================
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
// KERNEL FUNCTIONS
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

vec3 calcGradSpikyPow3Kernel(vec3 r, float h) {
    float dist = length(r);

    if (dist > h || dist < 1e-5) return vec3(0.0);

    float coeff = 45.0 / PI;
    float h2 = h * h;
    float h3 = h2 * h;
    float h6 = h3 * h3;

    float diff = h - dist;
    float diff2 = diff * diff;

    float scalarDerivative = -diff2 * (coeff / h6);
    return r * (scalarDerivative / dist);
}

// =========================================================================
// HASH FUNCTION
// =========================================================================
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

    vec3 xi = solver[id].predPos_lambda.xyz;
    float h = ubo.h;
    
    float rho_i = calcPoly6Kernel(vec3(0.0), h); // 1. Particle's own density contribution
    float sum_grad_Ci_sq = 0.0;
    vec3 grad_Ci_i = vec3(0.0);

    // =========================================================================
    // BOUNDARY HANDLING (GHOST PARTICLES)
    // =========================================================================
    vec3 ghosts[6];
    int numGhosts = 0;

    if (xi.x - ubo.boundsMin.x < h) ghosts[numGhosts++] = vec3(2.0 * (xi.x - ubo.boundsMin.x), 0.0, 0.0);
    if (ubo.boundsMax.x - xi.x < h) ghosts[numGhosts++] = vec3(-2.0 * (ubo.boundsMax.x - xi.x), 0.0, 0.0);
    
    if (xi.y - ubo.boundsMin.y < h) ghosts[numGhosts++] = vec3(0.0, 2.0 * (xi.y - ubo.boundsMin.y), 0.0);
    if (ubo.boundsMax.y - xi.y < h) ghosts[numGhosts++] = vec3(0.0, -2.0 * (ubo.boundsMax.y - xi.y), 0.0);
    
    if (xi.z - ubo.boundsMin.z < h) ghosts[numGhosts++] = vec3(0.0, 0.0, 2.0 * (xi.z - ubo.boundsMin.z));
    if (ubo.boundsMax.z - xi.z < h) ghosts[numGhosts++] = vec3(0.0, 0.0, -2.0 * (ubo.boundsMax.z - xi.z));

    // =========================================================================
    // NEIGHBOR SEARCH & ACCUMULATION
    // =========================================================================
    ivec3 cellCoord = ivec3(floor(xi / h));

    // Real fluid neighbors
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
                        vec3 rij = xi - xj;
                        
                        rho_i += calcPoly6Kernel(rij, h);
                        
                        vec3 gradW = calcGradSpikyPow3Kernel(rij, h);
                        vec3 grad_Ci_j = gradW * (-1.0 / ubo.rho0);
                        sum_grad_Ci_sq += dot(grad_Ci_j, grad_Ci_j);
                        
                        grad_Ci_i += gradW * (1.0 / ubo.rho0);
                    }
                }
            }
        }
    }

    // Ghost particle contributions
    for (int g = 0; g < numGhosts; g++) {
        vec3 rghost = ghosts[g];
        rho_i += calcPoly6Kernel(rghost, h);
        
        vec3 gradW = calcGradSpikyPow3Kernel(rghost, h);
        grad_Ci_i += gradW * (1.0 / ubo.rho0);
    }

    // =========================================================================
    // FINALIZE LAMBDA & RHO
    // =========================================================================
    float C_i = max((rho_i / ubo.rho0) - 1.0, 0.0);
    sum_grad_Ci_sq += dot(grad_Ci_i, grad_Ci_i);
    float lambda_i = -C_i / (sum_grad_Ci_sq + ubo.eps);

    // Save outputs. Storing rho here prevents us from having to recalculate it in CS_ComputeDeltaP
    solver[id].predPos_lambda.w = lambda_i;
    solver[id].deltaP_rho.w = rho_i; 
}