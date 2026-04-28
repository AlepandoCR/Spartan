//
// Created by Alepando on 12/4/2026.
//

#include "SpartanSimdOps.h"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

#include <immintrin.h>
#include <cmath>

namespace org::spartan::internal::simd::implementations {

    #pragma GCC target("avx2")
    #pragma clang attribute push(__attribute__((target("avx2"))), apply_to=function)

    SimdFloat avx2_load(const double* ptr) {
        SimdFloat result{};
        const __m256d v = _mm256_loadu_pd(ptr);
        _mm256_storeu_pd(result.data, v);
        return result;
    }

    void avx2_store(double* ptr, const SimdFloat &value) {
        const __m256d v = _mm256_loadu_pd(value.data);
        _mm256_storeu_pd(ptr, v);
    }

    SimdFloat avx2_add(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m256d va = _mm256_loadu_pd(a.data);
        const __m256d vb = _mm256_loadu_pd(b.data);
        const __m256d vr = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_subtract(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m256d va = _mm256_loadu_pd(a.data);
        const __m256d vb = _mm256_loadu_pd(b.data);
        const __m256d vr = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_multiply(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m256d va = _mm256_loadu_pd(a.data);
        const __m256d vb = _mm256_loadu_pd(b.data);
        const __m256d vr = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_divide(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m256d va = _mm256_loadu_pd(a.data);
        const __m256d vb = _mm256_loadu_pd(b.data);
        const __m256d vr = _mm256_div_pd(va, vb);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_fusedMultiplyAdd(const SimdFloat &mul1, const SimdFloat &mul2, const SimdFloat &add) {
        SimdFloat result{};
        const __m256d vm1 = _mm256_loadu_pd(mul1.data);
        const __m256d vm2 = _mm256_loadu_pd(mul2.data);
        const __m256d va = _mm256_loadu_pd(add.data);
        const __m256d vr = _mm256_fmadd_pd(vm1, vm2, va);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_maximum(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m256d va = _mm256_loadu_pd(a.data);
        const __m256d vb = _mm256_loadu_pd(b.data);
        const __m256d vr = _mm256_max_pd(va, vb);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_minimum(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m256d va = _mm256_loadu_pd(a.data);
        const __m256d vb = _mm256_loadu_pd(b.data);
        const __m256d vr = _mm256_min_pd(va, vb);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_setZero() {
        SimdFloat result{};
        const __m256d vr = _mm256_setzero_pd();
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_broadcast(const double scalar) {
        SimdFloat result{};
        const __m256d vr = _mm256_set1_pd(scalar);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    double avx2_horizontalSum(const SimdFloat &value) {
        const __m256d v = _mm256_loadu_pd(value.data);
        const __m256d h1 = _mm256_hadd_pd(v, v);
        const __m256d h2 = _mm256_permute2f128_pd(h1, h1, 1);
        const __m256d h3 = _mm256_add_pd(h1, h2);
        return _mm256_cvtsd_f64(h3);
    }

    SimdFloat avx2_sqrt(const SimdFloat &value) {
        SimdFloat result{};
        const __m256d v = _mm256_loadu_pd(value.data);
        const __m256d vr = _mm256_sqrt_pd(v);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_abs(const SimdFloat &value) {
        SimdFloat result{};
        const __m256d v = _mm256_loadu_pd(value.data);
        const __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        const __m256d vr = _mm256_and_pd(v, sign_mask);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_compareGreaterThan(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m256d va = _mm256_loadu_pd(a.data);
        const __m256d vb = _mm256_loadu_pd(b.data);
        const __m256d vr = _mm256_cmp_pd(va, vb, _CMP_GT_OQ);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_blend(const SimdFloat &trueValue, const SimdFloat &falseValue, const SimdFloat &mask) {
        SimdFloat result{};
        const __m256d vt = _mm256_loadu_pd(trueValue.data);
        const __m256d vf = _mm256_loadu_pd(falseValue.data);
        const __m256d vm = _mm256_loadu_pd(mask.data);
        const __m256d vr = _mm256_blendv_pd(vf, vt, vm);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdOperations createAVX2Operations() {
        return SimdOperations{
            .load = avx2_load,
            .store = avx2_store,
            .add = avx2_add,
            .subtract = avx2_subtract,
            .multiply = avx2_multiply,
            .divide = avx2_divide,
            .fusedMultiplyAdd = avx2_fusedMultiplyAdd,
            .maximum = avx2_maximum,
            .minimum = avx2_minimum,
            .setZero = avx2_setZero,
            .broadcast = avx2_broadcast,
            .horizontalSum = avx2_horizontalSum,
            .sqrt = avx2_sqrt,
            .abs = avx2_abs,
            .compareGreaterThan = avx2_compareGreaterThan,
            .blend = avx2_blend,
        };
    }

    #pragma clang attribute pop

}

#else

namespace org::spartan::internal::simd::implementations {
    SimdOperations createAVX2Operations() {
        return SimdOperations{};
    }
}

#endif





