//
// Created by Alepando on 23/2/2026.
//

#pragma once

/**
 * @namespace org::spartan::core::math::fuzzy
 * @brief Provides high-performance Zero-Allocation Fuzzy Logic operations.
 *
 * All operations utilize AVX2 (Advanced Vector Extensions) SIMD instructions
 * to process 4 double-precision floats (256 bits) per clock cycle.
 * Designed to operate directly on raw memory pointers provided by the JVM via FFM.
 */
namespace org::spartan::core::math::fuzzy {

    /**
     * @class FuzzySetOps
     * @brief Core mathematical operations for Fuzzy Sets.
     *
     * Implements standard Zadeh operators for fuzzy logic:
     * Union (Maximum), Intersection (Minimum), and Complement (1.0 - x).
     */
    class FuzzySetOps {
    public:
        FuzzySetOps() = delete;

        /**
         * @brief Computes the Fuzzy Union (MAX) of two sets: A = A U B
         *
         * @param targetSet Pointer to the first set. This array will be mutated to store the result.
         * @param sourceSet Pointer to the second set. Read-only.
         * @param arrayLength The number of valid elements to process.
         */
        static void unionSets(double* targetSet, const double* sourceSet, int arrayLength);

        /**
         * @brief Computes the Fuzzy Intersection (MIN) of two sets: A = A ∩ B
         *
         * @param targetSet Pointer to the first set. This array will be mutated to store the result.
         * @param sourceSet Pointer to the second set. Read-only.
         * @param arrayLength The number of valid elements to process.
         */
        static void intersectSets(double* targetSet, const double* sourceSet, int arrayLength);

        /**
         * @brief Computes the Fuzzy Complement of a set: A = 1.0 - A
         *
         * @param targetSet Pointer to the set to be complemented. Mutated in place.
         * @param arrayLength The number of valid elements to process.
         */
        static void complementSet(double* targetSet, int arrayLength);
    };

    /**
     * @class FuzzyModifiers
     * @brief Linguistic hedges (modifiers) to shift the gravity of fuzzy sets.
     *
     * Operations that modify the intensity of a fuzzy truth value.
     */
    class FuzzyModifiers {
    public:
        FuzzyModifiers() = delete;

        /**
         * @brief Applies "Concentration" (Linguistic "VERY"). Computes A = A^2
         * Reduces the truth value of elements that are not fully 1.0.
         *
         * @param targetSet Pointer to the set to be modified. Mutated in place.
         * @param arrayLength The number of valid elements to process.
         */
        static void applyConcentration(double* targetSet, int arrayLength);

        /**
         * @brief Applies "Dilation" (Linguistic "SOMEWHAT"). Computes A = sqrt(A)
         * Increases the truth value of weak elements.
         *
         * @param targetSet Pointer to the set to be modified. Mutated in place.
         * @param arrayLength The number of valid elements to process.
         */
        static void applyDilation(double* targetSet, int arrayLength);
    };

}
