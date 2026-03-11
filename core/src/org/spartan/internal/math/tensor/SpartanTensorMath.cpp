//
// Created by Alepando on 10/3/2026.
//

#include "SpartanTensorMath.h"
#include "../../simd/SpartanSimd.h"

#include <cmath>
#include <algorithm>
#include <random>

namespace org::spartan::internal::math::tensor {

    using namespace org::spartan::internal::math::simd;

    /**
     * Computes the mathematical core of a neural network layer.
     *
     * A dense layer connects every input to every output using a weight matrix.
     * It multiplies the incoming signals by their learned weights (which represent
     * their importance) and adds a bias to shift the final result.
     *
     * To avoid processing numbers one by one, this implementation uses the platform
     * SIMD abstraction layer. The CPU multiplies the weight and input, then adds it
     * to a running total using Fused Multiply-Add, processing multiple numbers
     * simultaneously to maximize throughput.
     *
     * @param inputs  The memory span containing the current layer's data.
     * @param weights The flattened matrix of connections between neurons.
     * @param biases  The baseline adjustments for each output neuron.
     * @param outputs The target span where the results will be written directly.
     */
    void TensorOps::denseForwardPass(
            const std::span<const double> inputs,
            const std::span<const double> weights,
            const std::span<const double> biases,
            const std::span<double> outputs) {

        const size_t inputSize = inputs.size();
        const size_t outputSize = outputs.size();

        const double* inputDataPointer = inputs.data();
        const double* weightDataPointer = weights.data();
        const double* biasDataPointer = biases.data();
        double* outputDataPointer = outputs.data();

        for (size_t neuronIndex = 0; neuronIndex < outputSize; ++neuronIndex) {

            const double* weightRowPointer = &weightDataPointer[neuronIndex * inputSize];

            SimdFloat vectorizedAccumulator = simdSetZero();

            size_t elementIndex = 0;
            for (; elementIndex + (simdLaneCount - 1) < inputSize; elementIndex += simdLaneCount) {
                const SimdFloat weightVector = simdLoad(&weightRowPointer[elementIndex]);
                const SimdFloat inputVector = simdLoad(&inputDataPointer[elementIndex]);

                vectorizedAccumulator = simdFusedMultiplyAdd(weightVector, inputVector, vectorizedAccumulator);
            }

            double scalarDotProduct = simdHorizontalSum(vectorizedAccumulator);

            for (; elementIndex < inputSize; ++elementIndex) {
                scalarDotProduct += weightRowPointer[elementIndex] * inputDataPointer[elementIndex];
            }

            outputDataPointer[neuronIndex] = scalarDotProduct + biasDataPointer[neuronIndex];
        }
    }


    /**
     * Applies the Rectified Linear Unit (ReLU) activation function to a tensor.
     *
     * Activation functions act as gates. ReLU lets positive numbers pass through
     * unchanged but completely blocks any negative numbers by turning them into
     * absolute zeros.
     *
     * At the hardware level, this loads a vector of absolute zeros and compares it
     * against the actual data. The processor uses a hardware-level maximum function
     * to keep whichever number is higher, silencing negative values in blocks
     * determined by the current SIMD lane width.
     *
     * @param tensor The memory span containing the layer outputs. Mutated in place.
     */
    void TensorOps::applyReLU(const std::span<double> tensor) {
        double* tensorDataPointer = tensor.data();
        const size_t tensorSize = tensor.size();

        const SimdFloat zeroVector = simdSetZero();

        size_t elementIndex = 0;
        for (; elementIndex + (simdLaneCount - 1) < tensorSize; elementIndex += simdLaneCount) {
            SimdFloat currentValues = simdLoad(&tensorDataPointer[elementIndex]);

            currentValues = simdMax(currentValues, zeroVector);
            simdStore(&tensorDataPointer[elementIndex], currentValues);
        }

        for (; elementIndex < tensorSize; ++elementIndex) {
            tensorDataPointer[elementIndex] = std::max(0.0, tensorDataPointer[elementIndex]);
        }
    }

    /**
     * Applies the Leaky Rectified Linear Unit (Leaky ReLU) activation function to a tensor.
     *
     * Activation functions act as gates. Leaky ReLU solves the "Dying ReLU" problem
     * by allowing a tiny, non-zero gradient (the "leak") when the input is negative.
     * Instead of flatlining at zero, the neuron multiplies the negative value by a
     * very small fraction (the alpha parameter).
     *
     * At the hardware level, this creates a mask of positive values and uses a SIMD
     * blend instruction to seamlessly merge the original positive values with the
     * scaled negative values without interrupting the instruction pipeline.
     *
     * @param tensor The memory span containing the layer outputs. Mutated in place.
     * @param alpha  The small multiplier for negative values. A common default is 0.01.
     */
    void TensorOps::applyLeakyReLU(const std::span<double> tensor, const double alpha) {
        double* tensorDataPointer = tensor.data();
        const size_t tensorSize = tensor.size();

        const SimdFloat simdAlpha = simdBroadcast(alpha);
        const SimdFloat simdZero = simdSetZero();

        size_t elementIndex = 0;
        for (; elementIndex + (simdLaneCount - 1) < tensorSize; elementIndex += simdLaneCount) {
            const SimdFloat inputValues = simdLoad(&tensorDataPointer[elementIndex]);

            // Calculate what the value would be if it is negative (input * alpha)
            const SimdFloat leakedValues = simdMultiply(inputValues, simdAlpha);

            // Create a bitmask where true (all 1s) means the value is strictly greater than zero
            const SimdFloat positiveMask = simdCompareGreaterThan(inputValues, simdZero);

            // Blend the two vectors based on the mask.
            // If the mask is true, it keeps inputValues. If false, it keeps leakedValues.
            const SimdFloat result = simdBlend(inputValues, leakedValues, positiveMask);

            simdStore(&tensorDataPointer[elementIndex], result);
        }

        // Residual loop for remaining elements
        for (; elementIndex < tensorSize; ++elementIndex) {
            const double val = tensorDataPointer[elementIndex];
            tensorDataPointer[elementIndex] = val > 0.0 ? val : val * alpha;
        }
    }


    /**
     * Applies a high-speed approximation of the Hyperbolic Tangent (Tanh) function.
     *
     * Tanh is a smooth curve that compresses any input number into a strict
     * range between -1.0 and 1.0. This is essential for bounding outputs,
     * like generating steering angles or stabilizing internal memory states.
     *
     * Standard math library calls like std::tanh can break hardware vectorization
     * on certain toolchains (like MinGW), forcing the CPU to process numbers
     * one at a time. To bypass this bottleneck, this function uses a Pade
     * rational approximation. It reconstructs the Tanh curve using only basic
     * multiplications and additions. This allows the CPU to use Fused Multiply-Add
     * instructions to process multiple values simultaneously without ever leaving
     * the SIMD registers.
     *
     * @param tensor The memory span to compress. Mutated in place.
     */
    void TensorOps::applyTanh(const std::span<double> tensor) {
        double* tensorDataPointer = tensor.data();
        const size_t tensorSize = tensor.size();

        // We preload the mathematical constants into SIMD registers.
        // These constants define the shape of our rational approximation: x(27 + x^2) / (27 + 9x^2)
        const SimdFloat constantTwentySeven = simdBroadcast(27.0);
        const SimdFloat constantNine        = simdBroadcast(9.0);
        const SimdFloat constantPositiveOne = simdBroadcast(1.0);
        const SimdFloat constantNegativeOne = simdBroadcast(-1.0);

        size_t elementIndex = 0;
        for (; elementIndex + (simdLaneCount - 1) < tensorSize; elementIndex += simdLaneCount) {
            const SimdFloat inputValues = simdLoad(&tensorDataPointer[elementIndex]);

            // Calculate x^2
            const SimdFloat squaredValues = simdMultiply(inputValues, inputValues);

            // Build the numerator: x * (27 + x^2)
            SimdFloat numerator = simdAdd(constantTwentySeven, squaredValues);
            numerator = simdMultiply(inputValues, numerator);

            // Build the denominator: 27 + 9x^2
            const SimdFloat denominator = simdFusedMultiplyAdd(constantNine, squaredValues, constantTwentySeven);

            // Divide to get the approximated tanh curve
            SimdFloat approximationResult = simdDivide(numerator, denominator);

            // The approximation can slightly exceed the bounds at extreme inputs
            // We clamp the final result strictly between -1.0 and 1.0.
            approximationResult = simdMax(constantNegativeOne, simdMin(constantPositiveOne, approximationResult));

            simdStore(&tensorDataPointer[elementIndex], approximationResult);
        }

        for (; elementIndex < tensorSize; ++elementIndex) {
            tensorDataPointer[elementIndex] = std::tanh(tensorDataPointer[elementIndex]);
        }
    }

    /**
     * Applies a high-speed approximation of the Sigmoid function.
     *
     * Sigmoid squashes any input number into a strict probability range between 0.0 and 1.0.
     * It is primarily used inside Recurrent Neural Networks (like the Gated Recurrent Unit)
     * to act as a valve, determining exactly what percentage of memory should be kept or forgotten.
     *
     * Since evaluating true exponentials is computationally expensive and disrupts
     * hardware vectorization, this function leverages the relationship between Sigmoid
     * and the Hyperbolic Tangent: Sigmoid(x) = 0.5 * (1 + Tanh(0.5 * x)). It uses our
     * fast rational Tanh approximation to process multiple values simultaneously.
     *
     * @param tensor The memory span to compress. Mutated in place.
     */
    void TensorOps::applySigmoidFast(const std::span<double> tensor) {
        double* tensorDataPointer = tensor.data();
        const size_t tensorSize = tensor.size();

        const SimdFloat constantHalf = simdBroadcast(0.5);
        const SimdFloat constantOne = simdBroadcast(1.0);

        // Constants for the internal Tanh Padé approximation
        const SimdFloat constantTwentySeven = simdBroadcast(27.0);
        const SimdFloat constantNine = simdBroadcast(9.0);
        const SimdFloat constantNegativeOne = simdBroadcast(-1.0);

        size_t elementIndex = 0;
        for (; elementIndex + (simdLaneCount - 1) < tensorSize; elementIndex += simdLaneCount) {
            SimdFloat inputValues = simdLoad(&tensorDataPointer[elementIndex]);

            // Scale input by 0.5 for the Tanh relationship
            inputValues = simdMultiply(inputValues, constantHalf);

            // -- Begin inline Tanh approximation --
            const SimdFloat squaredValues = simdMultiply(inputValues, inputValues);

            SimdFloat numerator = simdAdd(constantTwentySeven, squaredValues);
            numerator = simdMultiply(inputValues, numerator);

            const SimdFloat denominator = simdFusedMultiplyAdd(constantNine, squaredValues, constantTwentySeven);
            SimdFloat tanhResult = simdDivide(numerator, denominator);
            tanhResult = simdMax(constantNegativeOne, simdMin(constantOne, tanhResult));
            // -- End inline Tanh approximation --

            // Convert Tanh result to Sigmoid: 0.5 * (1.0 + tanhResult)
            SimdFloat sigmoidResult = simdAdd(constantOne, tanhResult);
            sigmoidResult = simdMultiply(constantHalf, sigmoidResult);

            simdStore(&tensorDataPointer[elementIndex], sigmoidResult);
        }

        // Residual loop for remaining elements
        for (; elementIndex < tensorSize; ++elementIndex) {
            tensorDataPointer[elementIndex] = 1.0 / (1.0 + std::exp(-tensorDataPointer[elementIndex]));
        }
    }

    /**
     * Applies a high-speed polynomial approximation of the Exponential function.
     *
     * This is frequently used to convert logarithmic standard deviations (log-std)
     * into true positive standard deviations.
     *
     * The approximation uses a scaled Taylor series expansion designed to evaluate
     * quickly using Fused Multiply-Add hardware instructions. Extreme values are
     * clamped to prevent numerical overflow (infinity) or underflow.
     *
     * @param tensor The memory span to exponentiate. Mutated in place.
     */
    void TensorOps::applyExpFast(const std::span<double> tensor) {
        double* tensorDataPointer = tensor.data();
        const size_t tensorSize = tensor.size();

        // Upper and lower bounds to prevent 64-bit float overflow
        const SimdFloat upperBound = simdBroadcast(88.0);
        const SimdFloat lowerBound = simdBroadcast(-88.0);

        // Taylor series coefficients for e^x
        const SimdFloat coeff2 = simdBroadcast(1.0 / 2.0);
        const SimdFloat coeff3 = simdBroadcast(1.0 / 6.0);
        const SimdFloat coeff4 = simdBroadcast(1.0 / 24.0);
        const SimdFloat constantOne = simdBroadcast(1.0);

        size_t elementIndex = 0;
        for (; elementIndex + (simdLaneCount - 1) < tensorSize; elementIndex += simdLaneCount) {
            SimdFloat inputValues = simdLoad(&tensorDataPointer[elementIndex]);

            // Clamp inputs to safe exponential ranges
            inputValues = simdMax(lowerBound, simdMin(upperBound, inputValues));

            // Approximate e^x using: 1 + x + (x^2)/2 + (x^3)/6 + (x^4)/24
            SimdFloat result = simdFusedMultiplyAdd(coeff4, inputValues, coeff3);
            result = simdFusedMultiplyAdd(result, inputValues, coeff2);
            result = simdFusedMultiplyAdd(result, inputValues, constantOne);
            result = simdFusedMultiplyAdd(result, inputValues, constantOne);

            simdStore(&tensorDataPointer[elementIndex], result);
        }

        for (; elementIndex < tensorSize; ++elementIndex) {
            tensorDataPointer[elementIndex] = std::exp(std::max(-88.0, std::min(88.0, tensorDataPointer[elementIndex])));
        }
    }

    /**
     * Blends the weights of an active learning network into a target network.
     *
     * Reinforcement learning can become highly unstable if the model tries to learn
     * from a moving target. To fix this, algorithms use a "Target Network" that updates
     * very slowly. This function takes a tiny fraction (tau) of the actively learning
     * weights and blends them into the target weights.
     *
     * Mathematically: Target = (tau * Online) + ((1 - tau) * Target).
     * By rearranging the algebra to Target = tau * (Online - Target) + Target,
     * the hardware can execute this entire blend in a single clock cycle per block.
     *
     * @param onlineWeights The memory span containing the active network weights.
     * @param targetWeights The memory span containing the slowly updating target weights.
     * @param tau The smoothing coefficient, dictating how much new information to absorb.
     */
    void TensorOps::applyPolyakAveraging(
            const std::span<const double> onlineWeights,
            const std::span<double> targetWeights,
            const double tau) {

        const size_t tensorSize = onlineWeights.size();
        const double* onlinePointer = onlineWeights.data();
        double* targetPointer = targetWeights.data();

        const SimdFloat simdTau = simdBroadcast(tau);

        size_t elementIndex = 0;
        for (; elementIndex + (simdLaneCount - 1) < tensorSize; elementIndex += simdLaneCount) {
            const SimdFloat onlineVec = simdLoad(&onlinePointer[elementIndex]);
            const SimdFloat targetVec = simdLoad(&targetPointer[elementIndex]);

            // Calculate (Online - Target)
            const SimdFloat differenceVec = simdSubtract(onlineVec, targetVec);

            // Fused Multiply-Add: (tau * difference) + target
            const SimdFloat updatedTarget = simdFusedMultiplyAdd(simdTau, differenceVec, targetVec);

            simdStore(&targetPointer[elementIndex], updatedTarget);
        }

        for (; elementIndex < tensorSize; ++elementIndex) {
            targetPointer[elementIndex] += tau * (onlinePointer[elementIndex] - targetPointer[elementIndex]);
        }
    }

    /**
     * Locates the index of the highest numerical value within a tensor.
     *
     * This is fundamentally used by discrete-action models to make decisions.
     * The model outputs a vector of estimated values for every possible action,
     * and this function scans the vector to find the action with the highest estimated reward.
     *
     * Since action spaces are typically small (e.g., fewer than 20 buttons or directions),
     * a direct scalar iteration avoids the overhead of setting up complex hardware
     * registers and branch-prediction penalties.
     *
     * @param tensor The memory span to evaluate.
     * @return The zero-based index of the maximum value.
     */
    size_t TensorOps::findArgmax(const std::span<const double> tensor) {
        const double* tensorDataPointer = tensor.data();
        const size_t tensorSize = tensor.size();

        if (tensorSize == 0) return 0;

        size_t maxIndex = 0;
        double maxValue = tensorDataPointer[0];

        for (size_t elementIndex = 1; elementIndex < tensorSize; ++elementIndex) {
            if (tensorDataPointer[elementIndex] > maxValue) {
                maxValue = tensorDataPointer[elementIndex];
                maxIndex = elementIndex;
            }
        }

        return maxIndex;
    }

    /**
     * Computes the scalar Mean Squared Error between two vectors using SIMD.
     *
     * Instead of iterating element-by-element, this function loads blocks of values
     * into SIMD registers, computes their differences, squares them via a single
     * multiply instruction, and accumulates the results using Fused Multiply-Add.
     * A single horizontal sum at the end collapses the vector accumulator into a
     * scalar, followed by a division by the element count.
     *
     * This is used by the nested AutoEncoder units to evaluate reconstruction quality
     * without leaving the vectorized execution pipeline.
     *
     * @param predictions The memory span containing the reconstructed values.
     * @param targets The memory span containing the original values.
     * @return The mean of the squared differences.
     */
    double TensorOps::computeMeanSquaredErrorLoss(
            const std::span<const double> predictions,
            const std::span<const double> targets) {

        const size_t tensorSize = predictions.size();
        const double* predictionsPointer = predictions.data();
        const double* targetsPointer = targets.data();

        SimdFloat squaredErrorAccumulator = simdSetZero();

        size_t elementIndex = 0;
        for (; elementIndex + (simdLaneCount - 1) < tensorSize; elementIndex += simdLaneCount) {
            const SimdFloat predictionVector = simdLoad(&predictionsPointer[elementIndex]);
            const SimdFloat targetVector = simdLoad(&targetsPointer[elementIndex]);

            // difference = prediction - target
            const SimdFloat differenceVector = simdSubtract(predictionVector, targetVector);

            // squaredError += difference * difference
            squaredErrorAccumulator = simdFusedMultiplyAdd(
                differenceVector, differenceVector, squaredErrorAccumulator);
        }

        double totalSquaredError = simdHorizontalSum(squaredErrorAccumulator);

        // Residual scalar loop for remaining elements
        for (; elementIndex < tensorSize; ++elementIndex) {
            const double difference = predictionsPointer[elementIndex] - targetsPointer[elementIndex];
            totalSquaredError += difference * difference;
        }

        return totalSquaredError / static_cast<double>(tensorSize);
    }

    /**
     * Calculates the raw error signals needed for backpropagation using Mean Squared Error.
     *
     * When a network makes a prediction, it needs to know exactly how wrong it was
     * and in what direction. This function compares the network's output against
     * the actual target values. By calculating the derivative of the Mean Squared Error,
     * it generates a gradient vector.
     *
     * A positive gradient means the network guessed too high, and a negative gradient
     * means it guessed too low. This vector becomes the raw fuel that the optimizer
     * uses to adjust the internal weights later.
     *
     * @param predictions The memory span containing the network's estimates.
     * @param targets The memory span containing the true values.
     * @param gradientsOutput The memory span where the error derivatives will be written.
     */
    void TensorOps::computeMeanSquaredErrorGradient(
            const std::span<const double> predictions,
            const std::span<const double> targets,
            const std::span<double> gradientsOutput) {

        const size_t tensorSize = predictions.size();
        const double* predictionsPointer = predictions.data();
        const double* targetsPointer = targets.data();
        double* gradientsPointer = gradientsOutput.data();

        // The derivative of MSE is 2 * (prediction - target) / N.
        // For individual batch elements during the backward pass, we focus on the 2 * (prediction - target)
        // component, leaving batch averaging to the upper gradient accumulation logic.
        const SimdFloat multiplierTwo = simdBroadcast(2.0);

        size_t elementIndex = 0;
        for (; elementIndex + (simdLaneCount - 1) < tensorSize; elementIndex += simdLaneCount) {
            const SimdFloat predictionVec = simdLoad(&predictionsPointer[elementIndex]);
            const SimdFloat targetVec = simdLoad(&targetsPointer[elementIndex]);

            // Calculate (Prediction - Target)
            const SimdFloat errorVec = simdSubtract(predictionVec, targetVec);

            // Multiply by 2 to complete the derivative
            const SimdFloat gradientVec = simdMultiply(errorVec, multiplierTwo);

            simdStore(&gradientsPointer[elementIndex], gradientVec);
        }

        for (; elementIndex < tensorSize; ++elementIndex) {
            gradientsPointer[elementIndex] = 2.0 * (predictionsPointer[elementIndex] - targetsPointer[elementIndex]);
        }
    }


    /**
     * Adjusts the neural network weights using the Adaptive Moment Estimation (Adam) algorithm.
     *
     * Traditional gradient descent blindly subtracts the error from the weights, which
     * causes learning to be jerky and unstable. Adam acts like a heavy ball rolling down
     * a hill with friction. It maintains two historical records for every single weight:
     * * 1. Momentum: It remembers the direction it has been moving recently. If the
     * gradients consistently point in one direction, it accelerates.
     * 2. Velocity: It tracks the volatility of the gradients. Weights that experience
     * massive, erratic errors get heavily penalized (slowed down), while stable
     * weights get a learning boost.
     *
     * This implementation mathematically corrects for early-stage bias and applies
     * the updates directly to the memory arrays using SIMD vectorization.
     *
     * @param weights The active neural connections to be updated.
     * @param gradients The raw error signals calculated during the backward pass.
     * @param momentum The historical running average of the gradients.
     * @param velocity The historical running average of the squared gradients.
     * @param learningRate The maximum allowed step size.
     * @param beta1 The decay rate for momentum.
     * @param beta2 The decay rate for velocity.
     * @param epsilon A tiny stabilizer to prevent dividing by zero.
     * @param timestep The current training iteration, used to correct cold-start bias.
     */
    void TensorOps::applyAdamUpdate(
            const std::span<double> weights,
            const std::span<const double> gradients,
            const std::span<double> momentum,
            const std::span<double> velocity,
            const double learningRate,
            const double beta1,
            const double beta2,
            const double epsilon,
            const int timestep) {

        const size_t tensorSize = weights.size();

        // Adam requires bias correction during the first few timesteps because
        // the momentum and velocity arrays start completely empty (zeros).
        const double beta1Time = 1.0 - std::pow(beta1, timestep);
        const double beta2Time = 1.0 - std::pow(beta2, timestep);
        const double effectiveAlpha = learningRate * std::sqrt(beta2Time) / beta1Time;

        // Preload all scalar constants into SIMD broadcast registers
        const SimdFloat simdBeta1 = simdBroadcast(beta1);
        const SimdFloat simdBeta1Complement = simdBroadcast(1.0 - beta1);
        const SimdFloat simdBeta2 = simdBroadcast(beta2);
        const SimdFloat simdBeta2Complement = simdBroadcast(1.0 - beta2);
        const SimdFloat simdEpsilon = simdBroadcast(epsilon);
        const SimdFloat simdNegativeAlpha = simdBroadcast(-effectiveAlpha);

        double* weightsPointer = weights.data();
        const double* gradientsPointer = gradients.data();
        double* momentumPointer = momentum.data();
        double* velocityPointer = velocity.data();

        size_t elementIndex = 0;
        for (; elementIndex + (simdLaneCount - 1) < tensorSize; elementIndex += simdLaneCount) {
            const SimdFloat currentWeights = simdLoad(&weightsPointer[elementIndex]);
            const SimdFloat currentGradients = simdLoad(&gradientsPointer[elementIndex]);
            const SimdFloat currentMomentum = simdLoad(&momentumPointer[elementIndex]);
            const SimdFloat currentVelocity = simdLoad(&velocityPointer[elementIndex]);

            // Update Momentum: m = (beta1 * m) + ((1 - beta1) * gradient)
            SimdFloat newMomentum = simdMultiply(currentGradients, simdBeta1Complement);
            newMomentum = simdFusedMultiplyAdd(currentMomentum, simdBeta1, newMomentum);
            simdStore(&momentumPointer[elementIndex], newMomentum);

            // Update Velocity: v = (beta2 * v) + ((1 - beta2) * (gradient * gradient))
            const SimdFloat squaredGradients = simdMultiply(currentGradients, currentGradients);
            SimdFloat newVelocity = simdMultiply(squaredGradients, simdBeta2Complement);
            newVelocity = simdFusedMultiplyAdd(currentVelocity, simdBeta2, newVelocity);
            simdStore(&velocityPointer[elementIndex], newVelocity);

            // Calculate the actual weight step: momentum / (sqrt(velocity) + epsilon)
            const SimdFloat velocitySqrt = simdSqrt(newVelocity);
            const SimdFloat denominator = simdAdd(velocitySqrt, simdEpsilon);
            const SimdFloat stepSize = simdDivide(newMomentum, denominator);

            // Apply the step to the weights: weights = weights - (alpha * stepSize)
            // Expressed as Fused Multiply-Add: weights = (stepSize * -alpha) + weights
            const SimdFloat updatedWeights = simdFusedMultiplyAdd(stepSize, simdNegativeAlpha, currentWeights);

            simdStore(&weightsPointer[elementIndex], updatedWeights);
        }

        // Residual loop for remaining elements
        for (; elementIndex < tensorSize; ++elementIndex) {
            const double grad = gradientsPointer[elementIndex];

            momentumPointer[elementIndex] = beta1 * momentumPointer[elementIndex] + (1.0 - beta1) * grad;
            velocityPointer[elementIndex] = beta2 * velocityPointer[elementIndex] + (1.0 - beta2) * (grad * grad);

            const double step = momentumPointer[elementIndex] / (std::sqrt(velocityPointer[elementIndex]) + epsilon);
            weightsPointer[elementIndex] -= effectiveAlpha * step;
        }
    }

    /**
     * Executes the Backward Pass of a fully connected (dense) layer.
     *
     * Backpropagation is the core of Deep Learning. When the network makes a mistake,
     * this function calculates two things:
     * 1. How much each weight contributed to the error (Weight Gradients).
     * 2. How much the previous layer's output contributed to the error (Input Gradients).
     *
     * To find the input gradients, we must mathematically reverse the forward pass
     * by multiplying the output gradients by the Transposed weight matrix.
     */
    void TensorOps::denseBackwardPass(
            const std::span<const double> inputs,
            const std::span<const double> outputGradients,
            const std::span<const double> weights,
            const std::span<double> outWeightGradients,
            const std::span<double> outInputGradients) {

        const size_t inputSize = inputs.size();
        const size_t outputSize = outputGradients.size();

        const double* inputDataPointer = inputs.data();
        const double* outputGradientDataPointer = outputGradients.data();
        const double* weightDataPointer = weights.data();
        double* weightGradientDataPointer = outWeightGradients.data();
        double* inputGradientDataPointer = outInputGradients.data();

        // 1. Clear the input gradients accumulator
        std::fill(outInputGradients.begin(), outInputGradients.end(), 0.0);

        // 2. Iterate over each neuron in the output layer
        for (size_t neuronIndex = 0; neuronIndex < outputSize; ++neuronIndex) {
            const double currentOutputGradient = outputGradientDataPointer[neuronIndex];
            const SimdFloat broadcastedGradient = simdBroadcast(currentOutputGradient);

            const double* weightRowPointer = &weightDataPointer[neuronIndex * inputSize];
            double* weightGradientRowPointer = &weightGradientDataPointer[neuronIndex * inputSize];

            size_t elementIndex = 0;
            for (; elementIndex + (simdLaneCount - 1) < inputSize; elementIndex += simdLaneCount) {
                // Calculate Weight Gradients: dW = dW + (dY * X)
                const SimdFloat inputVector = simdLoad(&inputDataPointer[elementIndex]);
                const SimdFloat currentWeightGradient = simdLoad(&weightGradientRowPointer[elementIndex]);
                const SimdFloat updatedWeightGradient = simdFusedMultiplyAdd(broadcastedGradient, inputVector, currentWeightGradient);
                simdStore(&weightGradientRowPointer[elementIndex], updatedWeightGradient);

                // Calculate Input Gradients (Backpropagate to previous layer): dX = dX + (dY * W)
                const SimdFloat weightVector = simdLoad(&weightRowPointer[elementIndex]);
                const SimdFloat currentInputGradient = simdLoad(&inputGradientDataPointer[elementIndex]);
                const SimdFloat updatedInputGradient = simdFusedMultiplyAdd(broadcastedGradient, weightVector, currentInputGradient);
                simdStore(&inputGradientDataPointer[elementIndex], updatedInputGradient);
            }

            // Residual loop
            for (; elementIndex < inputSize; ++elementIndex) {
                weightGradientRowPointer[elementIndex] += currentOutputGradient * inputDataPointer[elementIndex];
                inputGradientDataPointer[elementIndex] += currentOutputGradient * weightRowPointer[elementIndex];
            }
        }
    }

    /**
     * Applies Gaussian noise to a policy output for exploration.
     *
     * In Soft Actor-Critic algorithms, the agent doesn't choose a fixed action.
     * It chooses a probability distribution (a Mean and a Standard Deviation).
     * To actually make a move, we sample from this distribution.
     *
     * This uses a thread-local random number generator to ensure that parallel
     * agent evaluations (tickAll) do not suffer from lock contention or data races.
     */
    void TensorOps::applyGaussianNoise(
            const std::span<const double> means,
            const std::span<const double> stdDevs,
            const std::span<double> outActions,
            const uint64_t seed) {

        // Thread-local RNG guarantees thread-safety during parallel processing without locks
        thread_local std::mt19937_64 generator(seed == 0 ? std::random_device{}() : seed);
        std::normal_distribution standardNormal(0.0, 1.0);

        const size_t size = means.size();
        for (size_t i = 0; i < size; ++i) {
            // Reparameterization Trick: Action = Mean + (StdDev * N(0,1))
            const double noise = standardNormal(generator);
            outActions[i] = means[i] + (stdDevs[i] * noise);
        }
    }
}
