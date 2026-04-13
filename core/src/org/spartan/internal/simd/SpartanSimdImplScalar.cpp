//
// Created by Alepando on 12/4/2026.
//

#include "SpartanSimdOps.h"
#include <cmath>
#include <algorithm>

namespace org::spartan::internal::simd::implementations {

    namespace scalar {


        SimdFloat load(const double* ptr) {
            SimdFloat result;
            result.data[0] = ptr[0];
            return result;
        }

        void store(double* ptr, SimdFloat value) {
            ptr[0] = value.data[0];
        }

        SimdFloat add(SimdFloat a, SimdFloat b) {
            SimdFloat result;
            result.data[0] = a.data[0] + b.data[0];
            return result;
        }

        SimdFloat subtract(SimdFloat a, SimdFloat b) {
            SimdFloat result;
            result.data[0] = a.data[0] - b.data[0];
            return result;
        }

        SimdFloat multiply(SimdFloat a, SimdFloat b) {
            SimdFloat result;
            result.data[0] = a.data[0] * b.data[0];
            return result;
        }

        SimdFloat divide(SimdFloat a, SimdFloat b) {
            SimdFloat result;
            result.data[0] = a.data[0] / b.data[0];
            return result;
        }

        SimdFloat fusedMultiplyAdd(SimdFloat mul1, SimdFloat mul2, SimdFloat add) {
            SimdFloat result;
            result.data[0] = (mul1.data[0] * mul2.data[0]) + add.data[0];
            return result;
        }

        SimdFloat maximum(SimdFloat a, SimdFloat b) {
            SimdFloat result;
            result.data[0] = std::max(a.data[0], b.data[0]);
            return result;
        }

        SimdFloat minimum(SimdFloat a, SimdFloat b) {
            SimdFloat result;
            result.data[0] = std::min(a.data[0], b.data[0]);
            return result;
        }

        SimdFloat setZero(void) {
            SimdFloat result;
            result.data[0] = 0.0;
            return result;
        }

        SimdFloat broadcast(double scalar) {
            SimdFloat result;
            result.data[0] = scalar;
            return result;
        }

        double horizontalSum(SimdFloat value) {
            return value.data[0];
        }

        SimdFloat sqrt_op(SimdFloat value) {
            SimdFloat result;
            result.data[0] = std::sqrt(value.data[0]);
            return result;
        }

        SimdFloat abs_op(SimdFloat value) {
            SimdFloat result;
            result.data[0] = std::abs(value.data[0]);
            return result;
        }

        SimdFloat compareGreaterThan(SimdFloat a, SimdFloat b) {
            SimdFloat result;
            result.data[0] = (a.data[0] > b.data[0]) ? 1.0 : 0.0;
            return result;
        }

        SimdFloat blend(SimdFloat trueValue, SimdFloat falseValue, SimdFloat mask) {
            SimdFloat result;
            result.data[0] = (mask.data[0] > 0.5) ? trueValue.data[0] : falseValue.data[0];
            return result;
        }
    }

    SimdOperations createScalarOperations() {
        return SimdOperations{
            .load = scalar::load,
            .store = scalar::store,
            .add = scalar::add,
            .subtract = scalar::subtract,
            .multiply = scalar::multiply,
            .divide = scalar::divide,
            .fusedMultiplyAdd = scalar::fusedMultiplyAdd,
            .maximum = scalar::maximum,
            .minimum = scalar::minimum,
            .setZero = scalar::setZero,
            .broadcast = scalar::broadcast,
            .horizontalSum = scalar::horizontalSum,
            .sqrt = scalar::sqrt_op,
            .abs = scalar::abs_op,
            .compareGreaterThan = scalar::compareGreaterThan,
            .blend = scalar::blend,
        };
    }

}



