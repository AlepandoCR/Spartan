//
// Created by Alepando on 12/4/2026.
//

#include "SpartanSimdOps.h"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

#include <immintrin.h>

namespace org::spartan::internal::simd::implementations {

    #pragma GCC target("avx512f,avx512dq")
    #pragma clang attribute push(__attribute__((target("avx512f,avx512dq"))), apply_to=function)
    SimdFloat avx512_load(const double* ptr) {
        SimdFloat result{};
        const __m512d v = _mm512_loadu_pd(ptr);
        _mm512_storeu_pd(result.data, v);
        return result;
    }

    void avx512_store(double* ptr, const SimdFloat &value) {
        const __m512d v = _mm512_loadu_pd(value.data);
        _mm512_storeu_pd(ptr, v);
    }

    SimdFloat avx512_add(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m512d va = _mm512_loadu_pd(a.data);
        const __m512d vb = _mm512_loadu_pd(b.data);
        _mm512_storeu_pd(result.data, _mm512_add_pd(va, vb));
        return result;
    }

    SimdFloat avx512_subtract(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m512d va = _mm512_loadu_pd(a.data);
        const __m512d vb = _mm512_loadu_pd(b.data);
        _mm512_storeu_pd(result.data, _mm512_sub_pd(va, vb));
        return result;
    }

    SimdFloat avx512_multiply(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m512d va = _mm512_loadu_pd(a.data);
        const __m512d vb = _mm512_loadu_pd(b.data);
        _mm512_storeu_pd(result.data, _mm512_mul_pd(va, vb));
        return result;
    }

    SimdFloat avx512_divide(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m512d va = _mm512_loadu_pd(a.data);
        const __m512d vb = _mm512_loadu_pd(b.data);
        _mm512_storeu_pd(result.data, _mm512_div_pd(va, vb));
        return result;
    }

    SimdFloat avx512_fusedMultiplyAdd(const SimdFloat &mul1, const SimdFloat &mul2, const SimdFloat &add) {
        SimdFloat result{};
        const __m512d vm1 = _mm512_loadu_pd(mul1.data);
        const __m512d vm2 = _mm512_loadu_pd(mul2.data);
        const __m512d va = _mm512_loadu_pd(add.data);
        _mm512_storeu_pd(result.data, _mm512_fmadd_pd(vm1, vm2, va));
        return result;
    }

    SimdFloat avx512_maximum(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m512d va = _mm512_loadu_pd(a.data);
        const __m512d vb = _mm512_loadu_pd(b.data);
        _mm512_storeu_pd(result.data, _mm512_max_pd(va, vb));
        return result;
    }

    SimdFloat avx512_minimum(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m512d va = _mm512_loadu_pd(a.data);
        const __m512d vb = _mm512_loadu_pd(b.data);
        _mm512_storeu_pd(result.data, _mm512_min_pd(va, vb));
        return result;
    }

    SimdFloat avx512_setZero() {
        SimdFloat result{};
        _mm512_storeu_pd(result.data, _mm512_setzero_pd());
        return result;
    }

    SimdFloat avx512_broadcast(const double scalar) {
        SimdFloat result{};
        _mm512_storeu_pd(result.data, _mm512_set1_pd(scalar));
        return result;
    }

    double avx512_horizontalSum(const SimdFloat &value) {
        const __m512d v = _mm512_loadu_pd(value.data);
        return _mm512_reduce_add_pd(v);
    }

    SimdFloat avx512_sqrt(const SimdFloat &value) {
        SimdFloat result{};
        const __m512d v = _mm512_loadu_pd(value.data);
        _mm512_storeu_pd(result.data, _mm512_sqrt_pd(v));
        return result;
    }

    SimdFloat avx512_abs(const SimdFloat &value) {
        SimdFloat result{};
        const __m512d v = _mm512_loadu_pd(value.data);
        _mm512_storeu_pd(result.data, _mm512_abs_pd(v));
        return result;
    }

    SimdFloat avx512_compareGreaterThan(const SimdFloat &a, const SimdFloat &b) {
        SimdFloat result{};
        const __m512d va = _mm512_loadu_pd(a.data);
        const __m512d vb = _mm512_loadu_pd(b.data);
        const __mmask8 mask = _mm512_cmp_pd_mask(va, vb, _CMP_GT_OQ);
        _mm512_storeu_pd(result.data, _mm512_maskz_mov_pd(mask, _mm512_set1_pd(1.0)));
        return result;
    }

    SimdFloat avx512_blend(const SimdFloat &trueValue, const SimdFloat &falseValue, const SimdFloat &mask) {
        SimdFloat result{};
        const __m512d vt = _mm512_loadu_pd(trueValue.data);
        const __m512d vf = _mm512_loadu_pd(falseValue.data);
        const __m512d vm = _mm512_loadu_pd(mask.data);
        __mmask8 mmask = _mm512_cmp_pd_mask(vm, _mm512_setzero_pd(), _CMP_NEQ_OQ);
        _mm512_storeu_pd(result.data, _mm512_mask_blend_pd(mmask, vf, vt));
        return result;
    }

    SimdOperations createAVX512Operations() {
        return SimdOperations{
            .load = avx512_load,
            .store = avx512_store,
            .add = avx512_add,
            .subtract = avx512_subtract,
            .multiply = avx512_multiply,
            .divide = avx512_divide,
            .fusedMultiplyAdd = avx512_fusedMultiplyAdd,
            .maximum = avx512_maximum,
            .minimum = avx512_minimum,
            .setZero = avx512_setZero,
            .broadcast = avx512_broadcast,
            .horizontalSum = avx512_horizontalSum,
            .sqrt = avx512_sqrt,
            .abs = avx512_abs,
            .compareGreaterThan = avx512_compareGreaterThan,
            .blend = avx512_blend,
        };
    }

    #pragma clang attribute pop

}

#else

namespace org::spartan::internal::simd::implementations {
    SimdOperations createAVX512Operations() {
        return SimdOperations{};
    }
}

#endif





