//
// Created by Alepando on 23/2/2026.
//

#pragma once

/**
 * @namespace org::spartan::core::math::metric
 * @brief Hardware-accelerated distance and similarity metrics for vector spaces.
 */
namespace org::spartan::internal::math::metric {

    /**
     * @class VectorMetrics
     * @brief Computes spatial and logical similarities between raw context arrays.
     * Essential for temporal coherence (tracking entities across ticks without using IDs).
     */
    class VectorMetrics {
    public:
        VectorMetrics() = delete;

        /**
         * @brief Computes the Cosine Similarity between two context vectors.
         * Measures the cosine of the angle between two multidimensional vectors.
         * Useful for variable context elements (SpartanVariableContextElement).
         *
         * @return A value between -1.0 (opposite) and 1.0 (identical).
         */
        static double cosineSimilarity(const double* firstVector, const double* secondVector, int arrayLength);

        /**
         * @brief Computes the Fuzzy Jaccard index: |A ∩ B| / |A U B|
         * Measures the overlap between two fuzzy sets.
         * Excellent for hot-encoded features (SpartanHotContextElement).
         *
         * @return A coherence value between 0.0 (disjoint) and 1.0 (identical).
         */
        static double fuzzyJaccard(const double* firstVector, const double* secondVector, int arrayLength);
    };

}
