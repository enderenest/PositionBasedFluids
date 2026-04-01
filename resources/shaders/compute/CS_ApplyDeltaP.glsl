#version 430 core

layout(local_size_x = 512) in;

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

    float cohesionStrength;
    float interactionRadius;
    float interactionStrength;
    float padding3;
} ubo;

struct SolverData {
    vec4 predPos_lambda; 
    vec4 deltaP_rho;     
};

layout(std430, binding = 1) buffer SolverBuffer {
    SolverData solver[];
};

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= ubo.particleCount) return;

    vec3 dp = solver[id].deltaP_rho.xyz;

    // Clamp deltaP to half the smoothing radius.
    // No single iteration should move a particle more than this — if it would,
    // it means lambda exploded (isolated particle, near-zero denominator).
    float maxDp = 0.5 * ubo.h;
    float dpLen = length(dp);
    if (dpLen > maxDp) dp *= (maxDp / dpLen);

    solver[id].predPos_lambda.xyz += dp;
}