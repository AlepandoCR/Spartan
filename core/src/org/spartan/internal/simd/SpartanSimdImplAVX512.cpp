//
// Created by Alepando on 12/4/2026.
//

#include "SpartanSimdOps.h"
#include <immintrin.h>

namespace org::spartan::internal::simd::implementations {

    // Pragma to enable AVX-512 intrinsics for this compilation unit
    #pragma GCC target("avx512f,avx512dq")
    #pragma clang attribute push(__attribute__((target("avx512f,avx512dq"))), apply_to=function)

    SimdFloat avx512_load(const double* ptr) {
        SimdFloat result;
        __m512d v = _mm512_loadu_pd(ptr);
        _mm512_storeu_pd(result.data, v);
        return result;
    }

    void avx512_store(double* ptr, SimdFloat value) {
        __m512d v = _mm512_loadu_pd(value.data);
        _mm512_storeu_pd(ptr, v);
    }

    SimdFloat avx512_add(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m512d va = _mm512_loadu_pd(a.data);
        __m512d vb = _mm512_loadu_pd(b.data);
        _mm512_storeu_pd(result.data, _mm512_add_pd(va, vb));
        return result;
    }

    SimdFloat avx512_subtract(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m512d va = _mm512_loadu_pd(a.data);
        __m512d vb = _mm512_loadu_pd(b.data);
        _mm512_storeu_pd(result.data, _mm512_sub_pd(va, vb));
        return result;
    }

    SimdFloat avx512_multiply(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m512d va = _mm512_loadu_pd(a.data);
        __m512d vb = _mm512_loadu_pd(b.data);
        _mm512_storeu_pd(result.data, _mm512_mul_pd(va, vb));
        return result;
    }

    SimdFloat avx512_divide(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m512d va = _mm512_loadu_pd(a.data);
        __m512d vb = _mm512_loadu_pd(b.data);
        _mm512_storeu_pd(result.data, _mm512_div_pd(va, vb));
        return result;
    }

    SimdFloat avx512_fusedMultiplyAdd(SimdFloat mul1, SimdFloat mul2, SimdFloat add) {
        SimdFloat result;
        __m512d vm1 = _mm512_loadu_pd(mul1.data);
        __m512d vm2 = _mm512_loadu_pd(mul2.data);
        __m512d va = _mm512_loadu_pd(add.data);
        _mm512_storeu_pd(result.data, _mm512_fmadd_pd(vm1, vm2, va));
        return result;
    }

    SimdFloat avx512_maximum(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m512d va = _mm512_loadu_pd(a.data);
        __m512d vb = _mm512_loadu_pd(b.data);
        _mm512_storeu_pd(result.data, _mm512_max_pd(va, vb));
        return result;
    }

    SimdFloat avx512_minimum(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m512d va = _mm512_loadu_pd(a.data);
        __m512d vb = _mm512_loadu_pd(b.data);
        _mm512_storeu_pd(result.data, _mm512_min_pd(va, vb));
        return result;
    }

    SimdFloat avx512_setZero(void) {
        SimdFloat result;
        _mm512_storeu_pd(result.data, _mm512_setzero_pd());
        return result;
    }

    SimdFloat avx512_broadcast(double scalar) {
        SimdFloat result;
        _mm512_storeu_pd(result.data, _mm512_set1_pd(scalar));
        return result;
    }

    double avx512_horizontalSum(SimdFloat value) {
        __m512d v = _mm512_loadu_pd(value.data);
        return _mm512_reduce_add_pd(v);
    }

    SimdFloat avx512_sqrt(SimdFloat value) {
        SimdFloat result;
        __m512d v = _mm512_loadu_pd(value.data);
        _mm512_storeu_pd(result.data, _mm512_sqrt_pd(v));
        return result;
    }

    SimdFloat avx512_abs(SimdFloat value) {
        SimdFloat result;
        __m512d v = _mm512_loadu_pd(value.data);
        _mm512_storeu_pd(result.data, _mm512_abs_pd(v));
        return result;
    }

    SimdFloat avx512_compareGreaterThan(SimdFloat a, SimdFloat b) {
        SimdFloat result;
        __m512d va = _mm512_loadu_pd(a.data);
        __m512d vb = _mm512_loadu_pd(b.data);
        __mmask8 mask = _mm512_cmp_pd_mask(va, vb, _CMP_GT_OQ);
        _mm512_storeu_pd(result.data, _mm512_maskz_mov_pd(mask, _mm512_set1_pd(1.0)));
        return result;
    }

    SimdFloat avx512_blend(SimdFloat trueValue, SimdFloat falseValue, SimdFloat mask) {
        SimdFloat result;
        __m512d vt = _mm512_loadu_pd(trueValue.data);
        __m512d vf = _mm512_loadu_pd(falseValue.data);
        __m512d vm = _mm512_loadu_pd(mask.data);
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





