//
// Created by Alepando on 12/4/2026.
//

#include "SpartanSimdOps.h"
#include <immintrin.h>
#include <cmath>

namespace org::spartan::internal::simd::implementations {

    // Pragma to enable AVX2 intrinsics for this compilation unit
    #pragma GCC target("avx2")
    #pragma clang attribute push(__attribute__((target("avx2"))), apply_to=function)

    // AVX2: 4 doubles per register (__m256d)

    SimdFloat avx2_load(const double* ptr) {
        SimdFloat result;
        __m256d v = _mm256_loadu_pd(ptr);
        _mm256_storeu_pd(result.data, v);
        return result;
    }

    void avx2_store(double* ptr, SimdFloat value) {
        __m256d v = _mm256_loadu_pd(value.data);
        _mm256_storeu_pd(ptr, v);
    }

    SimdFloat avx2_add(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m256d va = _mm256_loadu_pd(a.data);
        __m256d vb = _mm256_loadu_pd(b.data);
        __m256d vr = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_subtract(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m256d va = _mm256_loadu_pd(a.data);
        __m256d vb = _mm256_loadu_pd(b.data);
        __m256d vr = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_multiply(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m256d va = _mm256_loadu_pd(a.data);
        __m256d vb = _mm256_loadu_pd(b.data);
        __m256d vr = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_divide(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m256d va = _mm256_loadu_pd(a.data);
        __m256d vb = _mm256_loadu_pd(b.data);
        __m256d vr = _mm256_div_pd(va, vb);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_fusedMultiplyAdd(SimdFloat mul1, SimdFloat mul2, SimdFloat add) {
        SimdFloat result;
        __m256d vm1 = _mm256_loadu_pd(mul1.data);
        __m256d vm2 = _mm256_loadu_pd(mul2.data);
        __m256d va = _mm256_loadu_pd(add.data);
        __m256d vr = _mm256_fmadd_pd(vm1, vm2, va);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_maximum(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m256d va = _mm256_loadu_pd(a.data);
        __m256d vb = _mm256_loadu_pd(b.data);
        __m256d vr = _mm256_max_pd(va, vb);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_minimum(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m256d va = _mm256_loadu_pd(a.data);
        __m256d vb = _mm256_loadu_pd(b.data);
        __m256d vr = _mm256_min_pd(va, vb);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_setZero(void) {
        SimdFloat result;
        __m256d vr = _mm256_setzero_pd();
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_broadcast(double scalar) {
        SimdFloat result;
        __m256d vr = _mm256_set1_pd(scalar);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    double avx2_horizontalSum(SimdFloat value) {
        __m256d v = _mm256_loadu_pd(value.data);
        __m256d h1 = _mm256_hadd_pd(v, v);
        __m256d h2 = _mm256_permute2f128_pd(h1, h1, 1);
        __m256d h3 = _mm256_add_pd(h1, h2);
        return _mm256_cvtsd_f64(h3);
    }

    SimdFloat avx2_sqrt(SimdFloat value) {
        SimdFloat result;
        __m256d v = _mm256_loadu_pd(value.data);
        __m256d vr = _mm256_sqrt_pd(v);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_abs(SimdFloat value) {
        SimdFloat result;
        __m256d v = _mm256_loadu_pd(value.data);
        const __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        __m256d vr = _mm256_and_pd(v, sign_mask);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_compareGreaterThan(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m256d va = _mm256_loadu_pd(a.data);
        __m256d vb = _mm256_loadu_pd(b.data);
        __m256d vr = _mm256_cmp_pd(va, vb, _CMP_GT_OQ);
        _mm256_storeu_pd(result.data, vr);
        return result;
    }

    SimdFloat avx2_blend(SimdFloat trueValue, SimdFloat falseValue, SimdFloat mask) {
        SimdFloat result;
        __m256d vt = _mm256_loadu_pd(trueValue.data);
        __m256d vf = _mm256_loadu_pd(falseValue.data);
        __m256d vm = _mm256_loadu_pd(mask.data);
        __m256d vr = _mm256_blendv_pd(vf, vt, vm);
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





