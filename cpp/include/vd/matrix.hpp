// Tiny fixed-size matrix utilities for the EKF state (2x2 and 2x1).
//
// Why hand-rolled and not Eigen:
//   - Eigen is excellent but pulls a large header tree, slows compile, and
//     for 2x2 fixed-size matrices the codegen difference is negligible.
//   - This file deliberately uses no heap, no exceptions, and no std::vector,
//     so it cross-compiles cleanly to Cortex-M without modification.
//
// All ops are constexpr-friendly where the compiler allows.
#pragma once

#include <cstddef>
#include <cmath>

namespace vd {

template <std::size_t R, std::size_t C>
struct Mat {
    double d[R * C] = {};
    constexpr double& operator()(std::size_t r, std::size_t c) noexcept { return d[r * C + c]; }
    constexpr double  operator()(std::size_t r, std::size_t c) const noexcept { return d[r * C + c]; }
};

using Mat2 = Mat<2, 2>;
using Vec2 = Mat<2, 1>;

constexpr Mat2 eye2() {
    Mat2 m;
    m(0, 0) = 1.0;
    m(1, 1) = 1.0;
    return m;
}

constexpr Mat2 matmul(const Mat2& a, const Mat2& b) {
    Mat2 r;
    r(0, 0) = a(0, 0) * b(0, 0) + a(0, 1) * b(1, 0);
    r(0, 1) = a(0, 0) * b(0, 1) + a(0, 1) * b(1, 1);
    r(1, 0) = a(1, 0) * b(0, 0) + a(1, 1) * b(1, 0);
    r(1, 1) = a(1, 0) * b(0, 1) + a(1, 1) * b(1, 1);
    return r;
}

constexpr Vec2 matvec(const Mat2& a, const Vec2& x) {
    Vec2 r;
    r(0, 0) = a(0, 0) * x(0, 0) + a(0, 1) * x(1, 0);
    r(1, 0) = a(1, 0) * x(0, 0) + a(1, 1) * x(1, 0);
    return r;
}

constexpr Mat2 transpose(const Mat2& a) {
    Mat2 r;
    r(0, 0) = a(0, 0);
    r(0, 1) = a(1, 0);
    r(1, 0) = a(0, 1);
    r(1, 1) = a(1, 1);
    return r;
}

constexpr Mat2 add(const Mat2& a, const Mat2& b) {
    Mat2 r;
    for (std::size_t i = 0; i < 4; ++i) r.d[i] = a.d[i] + b.d[i];
    return r;
}

constexpr Mat2 sub(const Mat2& a, const Mat2& b) {
    Mat2 r;
    for (std::size_t i = 0; i < 4; ++i) r.d[i] = a.d[i] - b.d[i];
    return r;
}

}  // namespace vd
