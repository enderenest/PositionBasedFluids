# Fluid-Structure Interaction (FSI) - Early Prototype

## Overview

This project explores a real-time Fluid-Structure Interaction (FSI) system.

The main idea is to:

- Simulate fluid using **Position-Based Fluids (PBF)** on the GPU  
- Deform a surface mesh using **Laplacian-based deformation** on the CPU  
- Later couple both systems through contact-based interaction  

At this stage, the focus is on building a clean and modular foundation.

---

## Current Status

- GPU particle system setup  
- PBF solver implementation (in progress)  
- CPU mesh deformation module (planned)  
- Fluid-mesh coupling (future phase)

---

## Architecture (Planned)

The system is divided into independent modules:

### Fluid (GPU)
- Position-Based Fluids (PBF)
- OpenGL compute shaders
- SSBO-based particle storage
- Spatial hashing for neighbor search

### Deformation (CPU)
- Surface mesh representation
- Laplacian-based deformation
- Eigen for linear algebra

### Viewer
- Polyscope for visualization
- Parameter control and debugging interface

This separation keeps the simulation modular and easier to extend.

---

## Project Structure
