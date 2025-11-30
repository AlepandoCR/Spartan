package dev.alepando.spartan.ai.deeplearning.buffer

import dev.alepando.spartan.ai.deeplearning.training.Transition
import java.util.Random
import java.util.ArrayList

/**
 * A highly optimized, thread-safe **Circular Replay Buffer (Ring Buffer)**.
 *
 * **Performance Improvements:**
 * * **Insertion:** O(1) complexity (vs O(N) in standard list removal).
 * * **Sampling:** O(K) complexity where K is batch size (vs O(N) in shuffle).
 * * **Memory:** Pre-allocated capacity prevents resizing overhead.
 *
 * **Mechanism:**
 * Uses a `writePointer` to overwrite the oldest transitions once capacity is reached,
 * eliminating the need for expensive array shifting operations.
 */
class ReplayBuffer(private val capacity: Int = 10000) {
    // Pre-allocate ArrayList to avoid resizing logic during training
    private val buffer = ArrayList<Transition>(capacity)

    /** Points to the index where the next insertion will occur. */
    private var writePointer = 0

    private val random = Random()
    private val lock = Any() // Monitor object for synchronization

    /**
     * Adds a transition to the buffer.
     * If buffer is full, overwrites the oldest entry (Circular Logic).
     *
     * Complexity: **O(1)**
     */
    fun add(t: Transition) {
        synchronized(lock) {
            if (buffer.size < capacity) {
                buffer.add(t)
            } else {
                buffer[writePointer] = t
                writePointer = (writePointer + 1) % capacity
            }
        }
    }

    /**
     * Samples a uniform random batch of transitions without shuffling the whole list.
     *
     * Complexity: **O(batchSize)**
     * @return List of randomly selected transitions.
     */
    fun sample(batchSize: Int): List<Transition> {
        synchronized(lock) {
            val currentSize = buffer.size
            if (currentSize == 0) return emptyList()

            // If requested batch is larger than buffer, return everything
            val effectiveBatchSize = minOf(batchSize, currentSize)

            val batch = ArrayList<Transition>(effectiveBatchSize)
            for (i in 0 until effectiveBatchSize) {
                val randomIndex = random.nextInt(currentSize)
                batch.add(buffer[randomIndex])
            }
            return batch
        }
    }

    /** @return current number of stored transitions. */
    fun size(): Int = synchronized(lock) { buffer.size }

    /**
     * Calculates average reward.
     * Warning: This is an O(N) operation. Use sparingly (e.g., only for logging).
     */
    fun averageReward(): Double {
        synchronized(lock) {
            if (buffer.isEmpty()) return 0.0
            // Avoid creating an iterator if possible, but sumOf is decent in Kotlin
            return buffer.sumOf { it.reward } / buffer.size
        }
    }

    /**
     * Clears the buffer (useful for resetting training episodes).
     */
    fun clear() {
        synchronized(lock) {
            buffer.clear()
            writePointer = 0
        }
    }
}