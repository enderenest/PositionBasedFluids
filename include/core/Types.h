#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Core>
#include <cstdint>
#include <cmath>

// ----------------------------
// Deformation (CPU) — Eigen
// ----------------------------
//
// Keep Eigen types for Laplacian deformation / linear algebra.
//
using EVec3 = Eigen::Matrix<double, 3, 1>;
using EVecX = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using EMatX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

// ----------------------------
// Fluid (GPU) — Plain Vectors
// ----------------------------
//
// Use these for particles / kernels / GPU buffer data to improve performance.
//
using F32 = float;
using I32 = std::int32_t;
using U32 = std::uint32_t;

struct PVec3 {
	F32 x, y, z;
};

struct PVec4 {
	F32 x, y, z, w;
};

// Basic helpers
inline PVec3 make_pvec3(F32 x, F32 y, F32 z) { return { x, y, z }; }

inline PVec3 operator+(PVec3 a, PVec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline PVec3 operator-(PVec3 a, PVec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline PVec3 operator*(PVec3 a, F32 s) { return { a.x * s, a.y * s, a.z * s }; }
inline PVec3 operator*(F32 s, PVec3 a) { return a * s; }

inline PVec3& operator+=(PVec3& a, PVec3 b) { a = a + b; return a; }
inline PVec3& operator-=(PVec3& a, PVec3 b) { a = a - b; return a; }
inline PVec3& operator*=(PVec3& a, F32 s) { a = a * s; return a; }

inline F32 dot(PVec3 a, PVec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline F32 norm2(PVec3 a) { return dot(a, a); }
inline F32 norm(PVec3 a) { return std::sqrt(norm2(a)); }

// Conversions
inline EVec3  toEigen(PVec3 v) { return EVec3(double(v.x), double(v.y), double(v.z)); }
inline PVec3 toPlain(EVec3 v) { return PVec3{ F32(v.x()), F32(v.y()), F32(v.z()) }; }

#endif