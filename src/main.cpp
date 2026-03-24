// main.cpp
//
// Fixes applied (ONLY problematic parts):
// 1) PBFluids must be constructed / configured with FluidConfig, not Config.
// 2) When pointRadius OR bounds change, we must re-apply BOTH:
//      - collision padding
//      - padded bounds
//      - boundary mesh vertices
//    using ONE helper function, so we never forget order.
// 3) When parameters change, call fluid.setParams(cfg.fluid) (not cfg).
// 4) If pointRadius changes we update Polyscope point radius, and also update bounds padding.
// 5) Keep code minimal; no architecture changes.

#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>

#include "core/Config.h"
#include "fluid/PBFluids.h"
#include "fluid/Particle.h"

#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

#include "imgui.h"

// ---------------------------
// Helpers: random + spawn
// ---------------------------

static float frand01(std::mt19937& rng) {
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}

static std::vector<Particle> spawnParticlesFromConfig(const FluidConfig& c) {
    std::vector<Particle> ps;
    ps.reserve((size_t)std::max(0, c.particleCount));

    std::mt19937 rng(1337);

    if (c.spawnRandom) {
		PVec3 pad = make_pvec3(c.spacing, c.spacing, c.spacing);

        PVec3 minBound = c.spawnMin + pad;
		PVec3 maxBound = c.spawnMax - pad;
		
        for (I32 i = 0; i < c.particleCount; ++i) {
            PVec3 p{
              minBound.x + (maxBound.x - minBound.x) * frand01(rng),
              minBound.y + (maxBound.y - minBound.y) * frand01(rng),
              minBound.z + (maxBound.z - minBound.z) * frand01(rng)
            };

            Particle pt{};
            pt.pos = p;
            pt.predPos = p;
            pt.vel = c.initialVelocity;
            pt.invMass = 1.0f;
            pt.phase = 0;

            ps.push_back(pt);
        }
    }
    else {
        const float dx = std::max(1e-6f, c.spacing);

        for (float z = c.spawnMin.z; z <= c.spawnMax.z && (I32)ps.size() < c.particleCount; z += dx) {
            for (float y = c.spawnMin.y; y <= c.spawnMax.y && (I32)ps.size() < c.particleCount; y += dx) {
                for (float x = c.spawnMin.x; x <= c.spawnMax.x && (I32)ps.size() < c.particleCount; x += dx) {
                    Particle pt{};
                    pt.pos = { x, y, z };
                    pt.predPos = pt.pos;
                    pt.vel = c.initialVelocity;
                    pt.invMass = 1.0f;
                    pt.phase = 0;
                    ps.push_back(pt);
                }
            }
        }
    }

    return ps;
}

static std::vector<std::array<double, 3>> toPSPoints(const std::vector<Particle>& ps) {
    std::vector<std::array<double, 3>> pts;
    pts.resize(ps.size());
    for (size_t i = 0; i < ps.size(); ++i) {
        pts[i] = { (double)ps[i].pos.x, (double)ps[i].pos.y, (double)ps[i].pos.z };
    }
    return pts;
}

// ---------------------------
// Helpers: boundary box mesh
// ---------------------------

static std::vector<std::array<double, 3>> makeBoxVerts(const PVec3& mn, const PVec3& mx) {
    return {
      { (double)mn.x, (double)mn.y, (double)mn.z }, // 0
      { (double)mx.x, (double)mn.y, (double)mn.z }, // 1
      { (double)mx.x, (double)mx.y, (double)mn.z }, // 2
      { (double)mn.x, (double)mx.y, (double)mn.z }, // 3
      { (double)mn.x, (double)mn.y, (double)mx.z }, // 4
      { (double)mx.x, (double)mn.y, (double)mx.z }, // 5
      { (double)mx.x, (double)mx.y, (double)mx.z }, // 6
      { (double)mn.x, (double)mx.y, (double)mx.z }  // 7
    };
}

static std::vector<std::array<size_t, 3>> makeBoxFaces() {
    return {
        {0, 1, 2}, {0, 2, 3},
        {4, 6, 5}, {4, 7, 6},
        {0, 5, 1}, {0, 4, 5},
        {3, 2, 6}, {3, 6, 7},
        {0, 3, 7}, {0, 7, 4},
        {1, 5, 6}, {1, 6, 2}
    };
}

// ---------------------------
// Main
// ---------------------------

int main() {

    Config cfg{};

    PBFluids fluid(cfg.fluid);
    fluid.setParams(cfg.fluid);
    fluid.setParticles(spawnParticlesFromConfig(cfg.fluid));

    // If you implemented padding inside PBFluids, set it once initially.
    fluid.setCollisionPadding(cfg.viewer.pointRadius);

    // Polyscope init
    polyscope::init();
    polyscope::view::setUpDir(polyscope::UpDir::YUp);
    polyscope::options::automaticallyComputeSceneExtents = true;
    polyscope::options::groundPlaneEnabled = false;

    // Point cloud
    auto pts = toPSPoints(fluid.particles());
    polyscope::PointCloud* psCloud = polyscope::registerPointCloud("PBF Particles", pts);
    psCloud->setPointRadius(cfg.viewer.pointRadius, false);

    // Boundary box
    auto boxVerts = makeBoxVerts(cfg.fluid.boundsMin, cfg.fluid.boundsMax);
    auto boxFaces = makeBoxFaces();
    polyscope::SurfaceMesh* psBox = polyscope::registerSurfaceMesh("Boundary Box", boxVerts, boxFaces);
    psBox->setTransparency(0.65);
    psBox->setSurfaceColor({ 0.65, 0.85, 1.0 });
    psBox->setSmoothShade(false);

    // Runtime controls
    static bool paused = false;
    static bool stepOnce = false;
    static bool respawnRequested = false;

    auto applyBoundsAndBox = [&]() {
        // re-apply collision padding (viewer radius)
        fluid.setCollisionPadding(cfg.viewer.pointRadius);

        // setBounds will internally shrink bounds by padding (in your PBFluids implementation)
        fluid.setBounds(cfg.fluid.boundsMin, cfg.fluid.boundsMax, cfg.fluid.boundDamping);

        // update GLASS mesh (visual bounds) to match cfg bounds
        boxVerts = makeBoxVerts(cfg.fluid.boundsMin, cfg.fluid.boundsMax);
        psBox->updateVertexPositions(boxVerts);
        };

    // Apply once so bounds are correct on startup (especially if PBFluids uses padding)
    applyBoundsAndBox();

    polyscope::state::userCallback = [&]() {

        ImGui::PushItemWidth(260.0f);

        ImGui::Text("CPU PBF Prototype");
        ImGui::Separator();

        if (ImGui::Checkbox("Paused", &paused)) {}
        ImGui::SameLine();
        if (ImGui::Button("Step Once")) stepOnce = true;
        ImGui::SameLine();
        if (ImGui::Button("Respawn")) respawnRequested = true;

        // Viewer
        ImGui::Separator();
        ImGui::Text("Viewer");
        bool viewerChanged = false;
        viewerChanged |= ImGui::SliderFloat("Point Radius", &cfg.viewer.pointRadius, 0.001f, 0.08f);

        // Time & Forces
        ImGui::Separator();
        ImGui::Text("Time & Forces");
        bool solverChanged = false;
        solverChanged |= ImGui::SliderFloat("dt", &cfg.fluid.dt, 1.0f / 240.0f, 1.0f / 30.0f);
        solverChanged |= ImGui::SliderFloat3("gravity", &cfg.fluid.gravity.x, -30.0f, 30.0f);

        // PBF Params
        ImGui::Separator();
        ImGui::Text("PBF Params");
        solverChanged |= ImGui::SliderFloat("h (smoothing radius)", &cfg.fluid.h, 0.01f, 0.30f);
        solverChanged |= ImGui::SliderFloat("rho0 (rest density)", &cfg.fluid.rho0, 100.0f, 3000.0f);
        solverChanged |= ImGui::SliderInt("solver iterations", &cfg.fluid.solverIterations, 1, 12);
        solverChanged |= ImGui::SliderInt("substep iterations", &cfg.fluid.substepIterations, 1, 5);
        solverChanged |= ImGui::SliderFloat("eps", &cfg.fluid.eps, 1e-8f, 1e-3f, "%.8f", ImGuiSliderFlags_Logarithmic);

        // Spawn
        ImGui::Separator();
        ImGui::Text("Spawn");
        bool spawnChanged = false;
        spawnChanged |= ImGui::Checkbox("spawnRandom", &cfg.fluid.spawnRandom);
        spawnChanged |= ImGui::SliderFloat("spacing (grid)", &cfg.fluid.spacing, 0.005f, 0.15f);
        spawnChanged |= ImGui::SliderFloat3("spawnMin", &cfg.fluid.spawnMin.x, -2.0f, 2.0f);
        spawnChanged |= ImGui::SliderFloat3("spawnMax", &cfg.fluid.spawnMax.x, -2.0f, 2.0f);
        spawnChanged |= ImGui::SliderFloat3("initialVelocity", &cfg.fluid.initialVelocity.x, -5.0f, 5.0f);

        // Bounds
        ImGui::Separator();
        ImGui::Text("Bounds (AABB)");
        bool boundsChanged = false;
        boundsChanged |= ImGui::SliderFloat3("boundsMin", &cfg.fluid.boundsMin.x, -3.0f, 3.0f);
        boundsChanged |= ImGui::SliderFloat3("boundsMax", &cfg.fluid.boundsMax.x, -3.0f, 3.0f);
        boundsChanged |= ImGui::SliderFloat("boundDamping", &cfg.fluid.boundDamping, 0.0f, 1.0f);

        // Neighbor Search
        ImGui::Separator();
        ImGui::Text("Neighbor Search");
        bool hashChanged = false;
        int hs = (int)cfg.fluid.hashSize;
        hashChanged |= ImGui::InputInt("hashSize", &hs);
        if (hashChanged) {
            if (hs < 1024) hs = 1024;
            cfg.fluid.hashSize = (U32)hs;
            solverChanged = true; // neighbor search depends on this
        }

        // sCorr
        ImGui::Separator();
        ImGui::Text("sCorr (optional)");
        bool scorrChanged = false;
        scorrChanged |= ImGui::Checkbox("enableSCorr", &cfg.fluid.enableSCorr);
        scorrChanged |= ImGui::SliderFloat("kCorr", &cfg.fluid.kCorr, 0.0f, 0.02f);
        scorrChanged |= ImGui::SliderFloat("nCorr", &cfg.fluid.nCorr, 1.0f, 8.0f);
        scorrChanged |= ImGui::SliderFloat("deltaQ", &cfg.fluid.deltaQ, 0.05f, 0.6f);

		// Viscosity
        ImGui::Separator();
        ImGui::Text("Viscosity");
        bool viscChanged = false;
        viscChanged |= ImGui::Checkbox("enableViscosity", &cfg.fluid.enableViscosity);
        viscChanged |= ImGui::SliderFloat("viscosity amount", &cfg.fluid.viscosity, 0.0f, 0.2f); // 0.2f üst limiti, istersen artýrabilirsin
        if (viscChanged) {
            solverChanged = true;
        }

        // ---------------------------
        // APPLY CHANGES
        // ---------------------------

        if (viewerChanged) {
            // visual point size
            psCloud->setPointRadius(cfg.viewer.pointRadius, false);
        }

        // whenever pointRadius OR bounds change, re-apply bounds & box.
        if (viewerChanged || boundsChanged) {
            applyBoundsAndBox();
        }

        if (solverChanged || scorrChanged) {
            fluid.setParams(cfg.fluid);
            applyBoundsAndBox();
        }

        if (spawnChanged) {
            respawnRequested = true;
        }

        if (respawnRequested) {
            auto newParticles = spawnParticlesFromConfig(cfg.fluid);
            fluid.setParticles(newParticles);

            pts = toPSPoints(fluid.particles());
            psCloud->updatePointPositions(pts);

            respawnRequested = false;
        }

        // Run simulation
        if (!paused || stepOnce) {
            fluid.step();
            pts = toPSPoints(fluid.particles());
            psCloud->updatePointPositions(pts);
            stepOnce = false;
        }

        ImGui::PopItemWidth();
        };

    polyscope::show();
    return 0;
}