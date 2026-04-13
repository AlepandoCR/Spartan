//
// Created by Alepando on 12/4/2026.
//

#include "SpartanSimdOps.h"
#include <stdexcept>

#if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>

namespace org::spartan::internal::simd::implementations {

    #pragma GCC target("neon")
    #pragma clang attribute push(__attribute__((target("neon"))), apply_to=function)

    SimdFloat neon_load(const double* ptr) {
        SimdFloat result;
        float64x2_t v = vld1q_f64(ptr);
        vst1q_f64(result.data, v);
        return result;
    }

    void neon_store(double* ptr, SimdFloat value) {
        float64x2_t v = vld1q_f64(value.data);
        vst1q_f64(ptr, v);
    }

    SimdFloat neon_add(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        float64x2_t va = vld1q_f64(a.data);
        float64x2_t vb = vld1q_f64(b.data);
        vst1q_f64(result.data, vaddq_f64(va, vb));
        return result;
    }

    SimdFloat neon_subtract(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        float64x2_t va = vld1q_f64(a.data);
        float64x2_t vb = vld1q_f64(b.data);
        vst1q_f64(result.data, vsubq_f64(va, vb));
        return result;
    }

    SimdFloat neon_multiply(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        float64x2_t va = vld1q_f64(a.data);
        float64x2_t vb = vld1q_f64(b.data);
        vst1q_f64(result.data, vmulq_f64(va, vb));
        return result;
    }

    SimdFloat neon_divide(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        float64x2_t va = vld1q_f64(a.data);
        float64x2_t vb = vld1q_f64(b.data);
        vst1q_f64(result.data, vdivq_f64(va, vb));
        return result;
    }

    SimdFloat neon_fusedMultiplyAdd(SimdFloat mul1, SimdFloat mul2, SimdFloat add) {
        SimdFloat result;
        float64x2_t vm1 = vld1q_f64(mul1.data);
        float64x2_t vm2 = vld1q_f64(mul2.data);
        float64x2_t va = vld1q_f64(add.data);
        vst1q_f64(result.data, vfmaq_f64(va, vm1, vm2));
        return result;
    }

    SimdFloat neon_maximum(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        float64x2_t va = vld1q_f64(a.data);
        float64x2_t vb = vld1q_f64(b.data);
        vst1q_f64(result.data, vmaxq_f64(va, vb));
        return result;
    }

    SimdFloat neon_minimum(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        float64x2_t va = vld1q_f64(a.data);
        float64x2_t vb = vld1q_f64(b.data);
        vst1q_f64(result.data, vminq_f64(va, vb));
        return result;
    }

    SimdFloat neon_setZero(void) {
        SimdFloat result;
        vst1q_f64(result.data, vdupq_n_f64(0.0));
        return result;
    }

    SimdFloat neon_broadcast(double scalar) {
        SimdFloat result;
        vst1q_f64(result.data, vdupq_n_f64(scalar));
        return result;
    }

    double neon_horizontalSum(SimdFloat value) {
        float64x2_t v = vld1q_f64(value.data);
        return vaddvq_f64(v);
    }

    SimdFloat neon_sqrt(SimdFloat value) {
        SimdFloat result;
        float64x2_t v = vld1q_f64(value.data);
        vst1q_f64(result.data, vsqrtq_f64(v));
        return result;
    }

    SimdFloat neon_abs(SimdFloat value) {
        SimdFloat result;
        float64x2_t v = vld1q_f64(value.data);
        vst1q_f64(result.data, vabsq_f64(v));
        return result;
    }

    SimdFloat neon_compareGreaterThan(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        float64x2_t va = vld1q_f64(a.data);
        float64x2_t vb = vld1q_f64(b.data);
        uint64x2_t mask = vcgtq_f64(va, vb);
        float64x2_t vmask = vreinterpretq_f64_u64(mask);
        vst1q_f64(result.data, vmask);
        return result;
    }

    SimdFloat neon_blend(SimdFloat trueValue, SimdFloat falseValue, SimdFloat mask) {
        SimdFloat result;
        float64x2_t vt = vld1q_f64(trueValue.data);
        float64x2_t vf = vld1q_f64(falseValue.data);
        float64x2_t vm = vld1q_f64(mask.data);
        uint64x2_t umask = vreinterpretq_u64_f64(vm);
        float64x2_t vr = vbslq_f64(umask, vt, vf);
        vst1q_f64(result.data, vr);
        return result;
    }

    SimdOperations createNEONOperations() {
        return SimdOperations{
            .load = neon_load,
            .store = neon_store,
            .add = neon_add,
            .subtract = neon_subtract,
            .multiply = neon_multiply,
            .divide = neon_divide,
            .fusedMultiplyAdd = neon_fusedMultiplyAdd,
            .maximum = neon_maximum,
            .minimum = neon_minimum,
            .setZero = neon_setZero,
            .broadcast = neon_broadcast,
            .horizontalSum = neon_horizontalSum,
            .sqrt = neon_sqrt,
            .abs = neon_abs,
            .compareGreaterThan = neon_compareGreaterThan,
            .blend = neon_blend,
        };
    }

    #pragma clang attribute pop
    #pragma GCC target("default")

}

#else
    // Stub implementation for non-ARM64 platforms
    // This allows the file to compile on x86_64, but createNEONOperations() will throw at runtime if called

namespace org::spartan::internal::simd::implementations {
    SimdOperations createNEONOperations() {
        throw std::runtime_error("NEON operations are not available on this platform (not ARM64)");
    }
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)




