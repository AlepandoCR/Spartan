//
// Created by Alepando on 10/3/2026.
//

#include "SpartanTensorMath.h"
#include "../../simd/SpartanSimd.h"

#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>

namespace org::spartan::internal::math::tensor {

    using namespace org::spartan::internal::math::simd;

    /**
     * Computes a dense (fully connected) forward pass.
     * Uses SIMD to process multiple weight-input products simultaneously.
     * * Optimization: The use of Fused Multiply-Add (FMA) prevents intermediate
     * rounding errors and doubles the throughput on modern x86/ARM hardware.
     */
    void TensorOps::denseForwardPass(
            const std::span<const double> inputs,
            const std::span<const double> weights,
            const std::span<const double> biases,
            const std::span<double> outputs) {

        const size_t inputSize = inputs.size();
        const size_t outputSize = outputs.size();

        if (inputSize == 0 || outputSize == 0) return;

        const double* __restrict inputPtr = inputs.data();
        const double* __restrict weightPtr = weights.data();
        const double* __restrict biasPtr = biases.data();
        double* __restrict outputPtr = outputs.data();

        for (size_t n = 0; n < outputSize; ++n) {
            const double* rowWeightPtr = &weightPtr[n * inputSize];
            SimdFloat acc = simdSetZero();

            size_t i = 0;
            // SIMD block processing based on hardware lane count (8 for AVX-512, 4 for AVX2)
            for (; i + (simdLaneCount - 1) < inputSize; i += simdLaneCount) {
                acc = simdFusedMultiplyAdd(simdLoad(&rowWeightPtr[i]), simdLoad(&inputPtr[i]), acc);
            }

            double sum = simdHorizontalSum(acc);

            // Tail loop for non-aligned elements
            for (; i < inputSize; ++i) {
                sum += rowWeightPtr[i] * inputPtr[i];
            }

            outputPtr[n] = sum + biasPtr[n];
        }
    }

    /**
     * Applies the Rectified Linear Unit (ReLU) activation function.
     * Elements < 0 are set to 0.
     */
    void TensorOps::applyReLU(const std::span<double> tensor) {
        double* ptr = tensor.data();
        const size_t size = tensor.size();
        const SimdFloat zero = simdSetZero();

        size_t i = 0;
        for (; i + (simdLaneCount - 1) < size; i += simdLaneCount) {
            simdStore(&ptr[i], simdMax(simdLoad(&ptr[i]), zero));
        }
        for (; i < size; ++i) ptr[i] = std::max(0.0, ptr[i]);
    }

    /**
     * Applies Leaky ReLU activation.
     * Prevents "dying neurons" by allowing a small gradient (alpha) for negative values.
     */
    void TensorOps::applyLeakyReLU(const std::span<double> tensor, const double alpha) {
        double* ptr = tensor.data();
        const size_t size = tensor.size();
        const SimdFloat sAlpha = simdBroadcast(alpha);
        const SimdFloat sZero = simdSetZero();

        size_t i = 0;
        for (; i + (simdLaneCount - 1) < size; i += simdLaneCount) {
            const SimdFloat val = simdLoad(&ptr[i]);
            // Blend selects (val * alpha) if val < 0, else val
            simdStore(&ptr[i], simdBlend(val, simdMultiply(val, sAlpha), simdCompareGreaterThan(val, sZero)));
        }
        for (; i < size; ++i) ptr[i] = ptr[i] > 0.0 ? ptr[i] : ptr[i] * alpha;
    }

    /**
     * Fast Tanh approximation using a (3/2) Padé approximant.
     * Provides high throughput for policy networks with negligible error.
     */
    void TensorOps::applyTanh(const std::span<double> tensor) {
        double* ptr = tensor.data();
        const size_t size = tensor.size();
        const SimdFloat c27 = simdBroadcast(27.0);
        const SimdFloat c9 = simdBroadcast(9.0);
        const SimdFloat p1 = simdBroadcast(1.0);
        const SimdFloat n1 = simdBroadcast(-1.0);

        size_t i = 0;
        for (; i + (simdLaneCount - 1) < size; i += simdLaneCount) {
            const SimdFloat x = simdLoad(&ptr[i]);
            const SimdFloat x2 = simdMultiply(x, x);
            const SimdFloat num = simdMultiply(x, simdAdd(c27, x2));
            const SimdFloat den = simdFusedMultiplyAdd(c9, x2, c27);
            SimdFloat res = simdDivide(num, den);
            simdStore(&ptr[i], simdMax(n1, simdMin(p1, res)));
        }
        for (; i < size; ++i) ptr[i] = std::tanh(ptr[i]);
    }

    /**
     * Stable Softmax implementation using the Max-Subtraction trick.
     * Prevents exponential overflow in discrete action spaces.
     */
    void TensorOps::applySoftmax(const std::span<double> tensor) {
        if (tensor.empty()) return;
        double* ptr = tensor.data();
        const size_t size = tensor.size();

        double maxVal = ptr[0];
        for (size_t i = 1; i < size; ++i) if (ptr[i] > maxVal) maxVal = ptr[i];

        double totalExp = 0.0;
        for (size_t i = 0; i < size; ++i) {
            ptr[i] = std::exp(ptr[i] - maxVal);
            totalExp += ptr[i];
        }

        // Guard: if all values underflow, fall back to a uniform distribution.
        if (totalExp <= 0.0) {
            const double uniform = 1.0 / static_cast<double>(size);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = uniform;
            }
            return;
        }

        const SimdFloat invTotal = simdBroadcast(1.0 / totalExp);
        size_t i = 0;
        for (; i + (simdLaneCount - 1) < size; i += simdLaneCount) {
            simdStore(&ptr[i], simdMultiply(simdLoad(&ptr[i]), invTotal));
        }
        for (; i < size; ++i) ptr[i] /= totalExp;
    }

    /**
     * Clips gradient norm to prevent exploding gradients in Recurrent models.
     * If the global norm exceeds maxNorm, all gradients are scaled down.
     */
    void TensorOps::clipGradients(std::span<double> gradients, const double maxNorm) {
        const size_t size = gradients.size();
        const double* ptr = gradients.data();
        SimdFloat sumSqAcc = simdSetZero();

        size_t i = 0;
        for (; i + (simdLaneCount - 1) < size; i += simdLaneCount) {
            SimdFloat g = simdLoad(&ptr[i]);
            sumSqAcc = simdFusedMultiplyAdd(g, g, sumSqAcc);
        }

        double totalSumSq = simdHorizontalSum(sumSqAcc);
        for (; i < size; ++i) totalSumSq += ptr[i] * ptr[i];

        double norm = std::sqrt(totalSumSq);
        if (norm > maxNorm) {
            const double scale = maxNorm / (norm + 1e-8);
            const SimdFloat sScale = simdBroadcast(scale);

            i = 0;
            for (; i + (simdLaneCount - 1) < size; i += simdLaneCount) {
                simdStore(&gradients[i], simdMultiply(simdLoad(&gradients[i]), sScale));
            }
            for (; i < size; ++i) gradients[i] *= scale;
        }
    }

    /**
     * Performs a Polyak (Soft) update for target networks.
     * target = tau * online + (1 - tau) * target
     */
    void TensorOps::applyPolyakAveraging(const std::span<const double> online, const std::span<double> target, const double tau) {
        const size_t size = online.size();
        const double* oPtr = online.data();
        double* tPtr = target.data();
        const SimdFloat sTau = simdBroadcast(tau);

        size_t i = 0;
        for (; i + (simdLaneCount - 1) < size; i += simdLaneCount) {
            const SimdFloat oV = simdLoad(&oPtr[i]);
            const SimdFloat tV = simdLoad(&tPtr[i]);
            // (oV - tV) * tau + tV  => Optimized FMA form
            simdStore(&tPtr[i], simdFusedMultiplyAdd(sTau, simdSubtract(oV, tV), tV));
        }
        for (; i < size; ++i) tPtr[i] += tau * (oPtr[i] - tPtr[i]);
    }

    /**
     * Adam Optimizer Update Step.
     * Updates weights using first (m) and second (v) moments of gradients.
     */
    void TensorOps::applyAdamUpdate(
            const std::span<double> w, const std::span<const double> g,
            const std::span<double> m, const std::span<double> v,
            const double lr, const double b1, const double b2,
            const double eps, const int t) {

        const size_t size = w.size();
        const double alphaEff = lr * std::sqrt(1.0 - std::pow(b2, t)) / (1.0 - std::pow(b1, t));

        const SimdFloat sb1 = simdBroadcast(b1), sb1c = simdBroadcast(1.0 - b1);
        const SimdFloat sb2 = simdBroadcast(b2), sb2c = simdBroadcast(1.0 - b2);
        const SimdFloat se = simdBroadcast(eps), sa = simdBroadcast(-alphaEff);

        double* wP = w.data(); const double* gP = g.data();
        double* mP = m.data(); double* vP = v.data();

        size_t i = 0;
        for (; i + (simdLaneCount - 1) < size; i += simdLaneCount) {
            SimdFloat grad = simdLoad(&gP[i]);
            // m = b1 * m + (1 - b1) * g
            SimdFloat nm = simdFusedMultiplyAdd(simdLoad(&mP[i]), sb1, simdMultiply(grad, sb1c));
            // v = b2 * v + (1 - b2) * g^2
            SimdFloat nv = simdFusedMultiplyAdd(simdLoad(&vP[i]), sb2, simdMultiply(simdMultiply(grad, grad), sb2c));

            simdStore(&mP[i], nm);
            simdStore(&vP[i], nv);

            // w = w - alpha * (m / (sqrt(v) + eps))
            SimdFloat update = simdDivide(nm, simdAdd(simdSqrt(nv), se));
            simdStore(&wP[i], simdFusedMultiplyAdd(update, sa, simdLoad(&wP[i])));
        }

        for (; i < size; ++i) {
            mP[i] = b1 * mP[i] + (1.0 - b1) * gP[i];
            vP[i] = b2 * vP[i] + (1.0 - b2) * gP[i] * gP[i];
            wP[i] -= alphaEff * (mP[i] / (std::sqrt(vP[i]) + eps));
        }
    }

    /**
     * Backpropagates gradients through a dense layer.
     * Computes both Weight Gradients (outWG) and Input Gradients (outIG).
     */
    void TensorOps::denseBackwardPass(
            const std::span<const double> in, const std::span<const double> outG,
            const std::span<const double> w, const std::span<double> outWG,
            const std::span<double> outIG) {

        const size_t inS = in.size(), outS = outG.size();
        if (inS == 0 || outS == 0) return;

        std::fill(outIG.begin(), outIG.end(), 0.0);
        const double* inP = in.data(); const double* ogP = outG.data();
        const double* wP = w.data(); double* wgP = outWG.data();
        double* igP = outIG.data();

        for (size_t n = 0; n < outS; ++n) {
            const SimdFloat sGrad = simdBroadcast(ogP[n]);
            const double* rWP = &wP[n * inS];
            double* rWGP = &wgP[n * inS];

            size_t i = 0;
            for (; i + (simdLaneCount - 1) < inS; i += simdLaneCount) {
                // Accumulate weight gradients: wg += input * output_gradient
                simdStore(&rWGP[i], simdFusedMultiplyAdd(sGrad, simdLoad(&inP[i]), simdLoad(&rWGP[i])));
                // Accumulate input gradients: ig += weight * output_gradient
                simdStore(&igP[i], simdFusedMultiplyAdd(sGrad, simdLoad(&rWP[i]), simdLoad(&igP[i])));
            }
            for (; i < inS; ++i) {
                rWGP[i] += ogP[n] * inP[i];
                igP[i] += ogP[n] * rWP[i];
            }
        }
    }

    /**
     * Gaussian Noise with Reparameterization Trick.
     * Essential for Soft Actor-Critic (SAC) to allow backpropagation through sampling.
     */
    void TensorOps::applyGaussianNoise(
            const std::span<const double> m, const std::span<const double> s,
            const std::span<double> out, const uint64_t seed) {

        thread_local std::mt19937_64 gen(seed == 0 ? std::random_device{}() : seed);
        std::normal_distribution dist(0.0, 1.0);
        for (size_t i = 0; i < m.size(); ++i) out[i] = m[i] + s[i] * dist(gen);
    }

    /**
     * Applies the Sigmoid activation function using a fast polynomial approximation.
     * Formula: f(x) = 1 / (1 + e^-x)
     * Uses a rational approximant for speed.
     */
    void TensorOps::applySigmoidFast(const std::span<double> tensor) {
        double* ptr = tensor.data();
        const size_t size = tensor.size();

        // Use rational approximation: sigmoid(x) ≈ 0.5 + 0.125*x / (1 + |x|)
        const SimdFloat p5 = simdBroadcast(0.5);
        const SimdFloat p125 = simdBroadcast(0.125);
        const SimdFloat p1 = simdBroadcast(1.0);

        size_t i = 0;
        for (; i + (simdLaneCount - 1) < size; i += simdLaneCount) {
            const SimdFloat x = simdLoad(&ptr[i]);
            const SimdFloat ax = simdAbs(x);
            const SimdFloat denom = simdAdd(p1, ax);
            const SimdFloat result = simdAdd(p5, simdDivide(simdMultiply(x, p125), denom));
            simdStore(&ptr[i], result);
        }

        // Tail loop for non-SIMD elements
        for (; i < size; ++i) {
            const double x = ptr[i];
            ptr[i] = 0.5 + 0.125 * x / (1.0 + std::abs(x));
        }
    }

    /**
     * Applies a fast polynomial approximation of the Exponential function.
     * Formula: f(x) = e^x
     * Uses Taylor series approximation for speed on embedded hardware.
     */
    void TensorOps::applyExpFast(const std::span<double> tensor) {
        double* ptr = tensor.data();
        const size_t size = tensor.size();

        // Fast exp approximation: exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24 (for |x| < 1)
        // For larger values, clamp to [-10, 10] to prevent overflow
        const SimdFloat p1 = simdBroadcast(1.0);
        const SimdFloat p2 = simdBroadcast(2.0);
        const SimdFloat p6 = simdBroadcast(6.0);
        const SimdFloat p24 = simdBroadcast(24.0);
        const SimdFloat p10 = simdBroadcast(10.0);
        const SimdFloat n10 = simdBroadcast(-10.0);

        size_t i = 0;
        for (; i + (simdLaneCount - 1) < size; i += simdLaneCount) {
            SimdFloat x = simdLoad(&ptr[i]);
            // Clamp to [-10, 10]
            x = simdMax(n10, simdMin(p10, x));

            // Taylor series: 1 + x + x^2/2 + x^3/6 + x^4/24
            const SimdFloat x2 = simdMultiply(x, x);
            const SimdFloat x3 = simdMultiply(x2, x);
            const SimdFloat x4 = simdMultiply(x3, x);

            SimdFloat result = p1;
            result = simdFusedMultiplyAdd(x, p1, result);
            result = simdFusedMultiplyAdd(x2, simdDivide(p1, p2), result);
            result = simdFusedMultiplyAdd(x3, simdDivide(p1, p6), result);
            result = simdFusedMultiplyAdd(x4, simdDivide(p1, p24), result);

            simdStore(&ptr[i], result);
        }

        // Tail loop for non-SIMD elements
        for (; i < size; ++i) {
            double x = ptr[i];
            x = std::max(-10.0, std::min(10.0, x));
            ptr[i] = 1.0 + x + x*x/2.0 + x*x*x/6.0 + x*x*x*x/24.0;
        }
    }

    /**
     * Finds the index of the maximum value in a tensor.
     * Returns the zero-based index of the highest value.
     */
    size_t TensorOps::findArgmax(const std::span<const double> tensor) {
        if (tensor.empty()) return 0;

        const double* ptr = tensor.data();
        size_t maxIdx = 0;
        double maxVal = ptr[0];

        for (size_t i = 1; i < tensor.size(); ++i) {
            if (ptr[i] > maxVal) {
                maxVal = ptr[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    /**
     * Computes the scalar Mean Squared Error loss between two vectors.
     * Formula: Loss = sum((predictions - targets)^2) / N
     */
    double TensorOps::computeMeanSquaredErrorLoss(
            const std::span<const double> predictions,
            const std::span<const double> targets) {

        const size_t size = predictions.size();
        if (size == 0) return 0.0;

        const double* predPtr = predictions.data();
        const double* targPtr = targets.data();

        SimdFloat accSum = simdSetZero();

        size_t i = 0;
        for (; i + (simdLaneCount - 1) < size; i += simdLaneCount) {
            const SimdFloat diff = simdSubtract(simdLoad(&predPtr[i]), simdLoad(&targPtr[i]));
            accSum = simdFusedMultiplyAdd(diff, diff, accSum);
        }

        double sum = simdHorizontalSum(accSum);

        // Tail loop for remaining elements
        for (; i < size; ++i) {
            const double diff = predPtr[i] - targPtr[i];
            sum += diff * diff;
        }

        return sum / size;
    }

    /**
     * Computes the gradient of the Mean Squared Error (MSE) loss function.
     * Formula: Gradient = 2 * (Predictions - Targets) / BatchSize
     */
    void TensorOps::computeMeanSquaredErrorGradient(
            const std::span<const double> predictions,
            const std::span<const double> targets,
            const std::span<double> gradientsOutput) {

        const size_t size = predictions.size();
        if (size == 0) return;

        const double* predPtr = predictions.data();
        const double* targPtr = targets.data();
        double* gradPtr = gradientsOutput.data();

        const double scale = 2.0 / size;
        const SimdFloat sScale = simdBroadcast(scale);

        size_t i = 0;
        for (; i + (simdLaneCount - 1) < size; i += simdLaneCount) {
            const SimdFloat diff = simdSubtract(simdLoad(&predPtr[i]), simdLoad(&targPtr[i]));
            simdStore(&gradPtr[i], simdMultiply(diff, sScale));
        }

        // Tail loop for remaining elements
        for (; i < size; ++i) {
            gradPtr[i] = (predPtr[i] - targPtr[i]) * scale;
        }
    }

    /**
     * Applies the Sigmoid activation function in-place using std::exp.
     * Formula: f(x) = 1 / (1 + e^-x)
     */
    void TensorOps::applySigmoidExact(const std::span<double> tensor) {
        double* ptr = tensor.data();
        const size_t size = tensor.size();
        for (size_t i = 0; i < size; ++i) {
            const double x = ptr[i];
            ptr[i] = 1.0 / (1.0 + std::exp(-x));
        }
    }

    /**
     * Applies the Exponential function in-place using std::exp.
     * Formula: f(x) = e^x
     */
    void TensorOps::applyExpExact(const std::span<double> tensor) {
        double* ptr = tensor.data();
        const size_t size = tensor.size();
        for (size_t i = 0; i < size; ++i) {
            ptr[i] = std::exp(ptr[i]);
        }
    }

    /**
     * Applies the Hyperbolic Tangent (Tanh) activation function in-place using std::tanh.
     * Formula: f(x) = tanh(x)
     */
    void TensorOps::applyTanhExact(const std::span<double> tensor) {
        double* ptr = tensor.data();
        const size_t size = tensor.size();
        for (size_t i = 0; i < size; ++i) {
            ptr[i] = std::tanh(ptr[i]);
        }
    }
}