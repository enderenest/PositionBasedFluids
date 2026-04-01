#ifndef SCENE_H
#define SCENE_H

#include "core/config.h"
#include <vector>

struct Scene {
    const char* name;
    FluidConfig fluid;
    float camYaw;
    float camPitch;
    float camDist;
    float pointSize;
};

inline const std::vector<Scene>& getScenes()
{
    static const std::vector<Scene> scenes = {

        // 0 — Dam Break
        {
            "Dam Break",
            {
                /* dt */                1.0f / 60.0f,
                /* gravity */           { 0.0f, -3.81f, 0.0f },
                /* h */                 0.15f,
                /* rho0 */              1200.0f,
                /* solverIterations */   3,
                /* substepIterations */  2,
                /* eps */               0.0001f,
                /* particleCount */     1 << 15,
                /* spawnMin */          { 2.f, 0.2f, 2.f },
                /* spawnMax */          { 5.f, 5.f, 5.f },
                /* spawnRandom */       false,
                /* spacing */           0.1f,
                /* initialVelocity */   { 0.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 6.0f, 6.0f, 6.0f },
                /* boundDamping */      0.5f,
                /* hashSize */          1 << 17,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.001f,
                /* enableViscosity */   true,
                /* viscosity */         0.1f,
                /* interactionRadius */ 1.5f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   true,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.45f,
            /* camPitch */ 0.35f,
            /* camDist */  4.5f,
            /* pointSize */ 8.0f,
        },

        // 1 — Dripping Faucet
        {
            "Dripping Faucet",
            {
                /* dt */                1.0f / 60.0f,
                /* gravity */           { 0.0f, -6.0f, 0.0f },
                /* h */                 0.12f,
                /* rho0 */              1200.0f,
                /* solverIterations */   3,
                /* substepIterations */  2,
                /* eps */               0.0001f,
                /* particleCount */     1 << 14,
                /* spawnMin */          { 1.3f, 3.5f, 1.3f },
                /* spawnMax */          { 1.7f, 4.5f, 1.7f },
                /* spawnRandom */       true,
                /* spacing */           0.06f,
                /* initialVelocity */   { 0.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 3.0f, 5.0f, 3.0f },
                /* boundDamping */      0.5f,
                /* hashSize */          1 << 16,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.01f,
                /* enableViscosity */   true,
                /* viscosity */         0.05f,
                /* interactionRadius */ 1.5f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   true,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.3f,
            /* camPitch */ 0.25f,
            /* camDist */  5.0f,
            /* pointSize */ 6.0f,
        },

        // 2 — Large Tank
        {
            "Large Tank",
            {
                /* dt */                1.0f / 60.0f,
                /* gravity */           { 0.0f, -5.0f, 0.0f },
                /* h */                 0.15f,
                /* rho0 */              1000.0f,
                /* solverIterations */   4,
                /* substepIterations */  3,
                /* eps */               0.0001f,
                /* particleCount */     1 << 17,
                /* spawnMin */          { 0.2f, 0.2f, 0.2f },
                /* spawnMax */          { 9.8f, 3.0f, 9.8f },
                /* spawnRandom */       false,
                /* spacing */           0.12f,
                /* initialVelocity */   { 0.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 10.0f, 6.0f, 10.0f },
                /* boundDamping */      0.5f,
                /* hashSize */          1 << 19,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.001f,
                /* enableViscosity */   true,
                /* viscosity */         0.1f,
                /* interactionRadius */ 2.0f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   true,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.6f,
            /* camPitch */ 0.5f,
            /* camDist */  12.0f,
            /* pointSize */ 5.0f,
        },

        // 3 — High-Speed Impact
        {
            "High-Speed Impact",
            {
                /* dt */                1.0f / 60.0f,
                /* gravity */           { 0.0f, -2.0f, 0.0f },
                /* h */                 0.15f,
                /* rho0 */              1200.0f,
                /* solverIterations */   3,
                /* substepIterations */  2,
                /* eps */               0.0001f,
                /* particleCount */     1 << 15,
                /* spawnMin */          { 0.3f, 0.3f, 1.5f },
                /* spawnMax */          { 2.5f, 4.0f, 4.5f },
                /* spawnRandom */       false,
                /* spacing */           0.1f,
                /* initialVelocity */   { 8.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 8.0f, 5.0f, 6.0f },
                /* boundDamping */      0.3f,
                /* hashSize */          1 << 17,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.001f,
                /* enableViscosity */   true,
                /* viscosity */         0.1f,
                /* interactionRadius */ 1.5f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   true,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.0f,
            /* camPitch */ 0.4f,
            /* camDist */  8.0f,
            /* pointSize */ 7.0f,
        },

        // 4 — Viscous Goo
        {
            "Viscous Goo",
            {
                /* dt */                1.0f / 60.0f,
                /* gravity */           { 0.0f, -1.5f, 0.0f },
                /* h */                 0.15f,
                /* rho0 */              2000.0f,
                /* solverIterations */   5,
                /* substepIterations */  2,
                /* eps */               0.0001f,
                /* particleCount */     1 << 14,
                /* spawnMin */          { 1.0f, 2.0f, 1.0f },
                /* spawnMax */          { 4.0f, 5.0f, 4.0f },
                /* spawnRandom */       false,
                /* spacing */           0.1f,
                /* initialVelocity */   { 0.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 5.0f, 6.0f, 5.0f },
                /* boundDamping */      0.5f,
                /* hashSize */          1 << 15,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.03f,
                /* enableViscosity */   true,
                /* viscosity */         0.19f,
                /* interactionRadius */ 1.5f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   true,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.5f,
            /* camPitch */ 0.3f,
            /* camDist */  5.5f,
            /* pointSize */ 10.0f,
        },
    };
    return scenes;
}

#endif
