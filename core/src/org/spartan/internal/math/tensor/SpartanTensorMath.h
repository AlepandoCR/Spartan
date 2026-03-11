//
// Created by Alepando on 10/3/2026.
//

#pragma once

#include <cstdint>
#include <span>

/**
 * @namespace org::spartan::internal::math::tensor
 * @brief Hardware-accelerated tensor operations for Deep Learning inference.
 *
 * Utilizes the platform SIMD abstraction layer to process dense matrix-vector
 * multiplications and activation functions. Automatically selects the optimal
 * instruction set (AVX2 on x86_64, NEON on ARM64, or scalar fallback) at
 * compile time through the SpartanSimd hardware abstraction.
 * Designed for zero-allocation execution over JVM-owned memory segments.
 */
namespace org::spartan::internal::math::tensor {

    /**
     * @class TensorOps
     * @brief High-performance SIMD operations for Neural Network layers.
     */
    class TensorOps {
    public:
        TensorOps() = delete;

        /**
         * @brief Executes the Forward Pass of a fully connected (dense) layer.
         * Formula: Y = (W * X) + B
         *
         * Utilizes the platform Fused Multiply-Add (FMA) abstraction for
         * ultra-low latency execution across supported architectures.
         *
         * @param inputs  Read-only span of the input vector (X). [Size: inputSize]
         * @param weights Read-only span of the flattened weight matrix (W). [Size: outputSize * inputSize]
         * @param biases  Read-only span of the bias vector (B). [Size: outputSize]
         * @param outputs Writable span for the output vector (Y). Mutated in-place. [Size: outputSize]
         */
        static void denseForwardPass(
                std::span<const double> inputs,
                std::span<const double> weights,
                std::span<const double> biases,
                std::span<double> outputs);

        /**
         * @brief Applies the Rectified Linear Unit (ReLU) activation function in-place.
         * Formula: f(x) = max(0, x)
         *
         * @param tensor Writable span of the target vector. Mutated in-place.
         */
        static void applyReLU(std::span<double> tensor);

         /** @brief Applies the Leaky Rectified Linear Unit (Leaky ReLU) activation function in-place.
         * Formula: f(x) = x > 0 ? x : alpha * x
         *
         * @param tensor Writable span of the target vector. Mutated in-place.
         * @param alpha  The small multiplier for negative values (typically 0.01).
         */
        static void applyLeakyReLU(std::span<double> tensor, double alpha);

        /**
         * @brief Applies the Hyperbolic Tangent (Tanh) activation function in-place.
         * Formula: f(x) = tanh(x)
         *
         *
         * @param tensor Writable span of the target vector. Mutated in-place.
         */
        static void applyTanh(std::span<double> tensor);

        /**
         * @brief Applies the Sigmoid activation function in-place using a fast approximation.
         * Formula: f(x) = 1 / (1 + e^-x)
         *
         * @param tensor Writable span of the target vector. Mutated in-place.
         */
        static void applySigmoidFast(std::span<double> tensor);

        /**
         * @brief Applies a fast polynomial approximation of the Exponential function.
         * Formula: f(x) = e^x
         *
         * @param tensor Writable span of the target vector. Mutated in-place.
         */
        static void applyExpFast(std::span<double> tensor);

        /**
         * @brief Performs a Polyak Averaging (Soft Update) from an online network to a target network.
         * Formula: Target = (tau * Online) + ((1 - tau) * Target)
         *
         * @param onlineWeights Read-only span of the active network weights.
         * @param targetWeights Writable span of the target network weights. Mutated in-place.
         * @param tau The smoothing coefficient (typically 0.005).
         */
        static void applyPolyakAveraging(
                std::span<const double> onlineWeights,
                std::span<double> targetWeights,
                double tau);

        /**
         * @brief Finds the index of the maximum value in a tensor.
         *
         * @param tensor Read-only span of the vector to evaluate.
         * @return The zero-based index of the highest value.
         */
        [[nodiscard]] static size_t findArgmax(std::span<const double> tensor);

        /**
         * @brief Computes the scalar Mean Squared Error loss between two vectors.
         * Formula: Loss = sum((predictions - targets)^2) / N
         *
         * Fully SIMD-vectorized using FusedMultiplyAdd accumulators and a single
         * horizontal reduction at the end. Zero allocation, zero branching in the
         * vectorized path.
         *
         * @param predictions Read-only span of the network's current outputs.
         * @param targets     Read-only span of the actual expected values (ground truth).
         * @return The scalar Mean Squared Error loss value.
         */
        [[nodiscard]] static double computeMeanSquaredErrorLoss(
                std::span<const double> predictions,
                std::span<const double> targets);

        /**
         * @brief Computes the gradient of the Mean Squared Error (MSE) loss function.
         * Formula: Gradient = 2 * (Predictions - Targets) / BatchSize
         *
         * @param predictions Read-only span of the network's current outputs.
         * @param targets Read-only span of the actual expected values (ground truth).
         * @param gradientsOutput Writable span where the calculated errors are stored. Mutated in-place.
         */
        static void computeMeanSquaredErrorGradient(
                std::span<const double> predictions,
                std::span<const double> targets,
                std::span<double> gradientsOutput
                );

        /**
         * @brief Updates network weights using the Adam (Adaptive Moment Estimation) optimization algorithm.
         *
         * @param weights Writable span of the current network weights. Mutated in-place.
         * @param gradients Read-only span of the calculated gradients for these weights.
         * @param momentum Writable span tracking the first moment (running average of gradients). Mutated in-place.
         * @param velocity Writable span tracking the second moment (running average of squared gradients). Mutated in-place.
         * @param learningRate The base step size for weight adjustments.
         * @param beta1 The exponential decay rate for the first moment estimates (momentum).
         * @param beta2 The exponential decay rate for the second moment estimates (velocity).
         * @param epsilon A small constant for numerical stability to prevent division by zero.
         * @param timestep The current training step count, used for bias correction.
         */
        static void applyAdamUpdate(
                std::span<double> weights,
                std::span<const double> gradients,
                std::span<double> momentum,
                std::span<double> velocity,
                double learningRate,
                double beta1,
                double beta2,
                double epsilon,
                int timestep
                );


         /** @brief Executes the Backward Pass of a fully connected (dense) layer.
         * * Calculates the gradients for the weights and biases of the current layer,
         * and computes the incoming gradients for the previous layer.
         *
         * @param inputs Read-only span of the original inputs to this layer during the forward pass.
         * @param outputGradients Read-only span of the error gradients coming from the next layer.
         * @param weights Read-only span of the current layer's weights.
         * @param outWeightGradients Writable span where the calculated weight gradients will be accumulated.
         * @param outInputGradients Writable span where the gradients to pass back to the previous layer will be written.
         */
        static void denseBackwardPass(
                std::span<const double> inputs,
                std::span<const double> outputGradients,
                std::span<const double> weights,
                std::span<double> outWeightGradients,
                std::span<double> outInputGradients);

        /**
         * @brief Applies Gaussian noise to a policy output for exploration (Reparameterization Trick).
         * Formula: Action = Mean + (StdDev * N(0, 1))
         *
         * @param means Read-only span of the calculated action means.
         * @param stdDevs Read-only span of the calculated action standard deviations.
         * @param outActions Writable span where the noisy actions will be stored.
         * @param seed Optional seed for deterministic behavior (default 0 uses random device).
         */
        static void applyGaussianNoise(
                std::span<const double> means,
                std::span<const double> stdDevs,
                std::span<double> outActions,
                std::uint64_t seed = 0);

    };
}