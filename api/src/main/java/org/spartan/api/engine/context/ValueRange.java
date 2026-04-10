package org.spartan.api.engine.context;

/**
 * Represents a fixed block of memory allocated to a context element.
 * <p>
 * This record is the bridge between logical (user-facing) and physical (memory-level) indexing.
 * Once allocated during context initialization, the physical start index never changes,
 * enabling perfect tensor alignment in C++ neural networks.
 * <p>
 * <b>Example:</b> An element with {@code maxCapacity=100} might have:
 * <ul>
 *   <li>{@code physicalStartIndex=50}: This element's data begins at byte 50*sizeof(double)</li>
 *   <li>{@code maxCapacity=100}: Up to 100 doubles can be stored here</li>
 *   <li>{@code currentValidSize=47}: This tick, only 47 are valid (rest is padding)</li>
 * </ul>
 *
 * @param physicalStartIndex Zero-based offset in doubles where this element's data begins.
 *                          <b>Guaranteed to be stable after context layout is built.</b>
 * @param maxCapacity       Maximum number of doubles this range can hold. Determines allocation size.
 * @param currentValidSize  Number of actually-populated doubles this tick (for variable-length elements).
 */
public record ValueRange(
        int physicalStartIndex,
        int maxCapacity,
        int currentValidSize
) {
    /**
     * Compact validation for ValueRange parameters.
     */
    public ValueRange {

        if (physicalStartIndex < 0) {
            throw new IllegalArgumentException("physicalStartIndex must be >= 0");
        }
        if (maxCapacity <= 0) {
            throw new IllegalArgumentException("maxCapacity must be > 0");
        }
        if (currentValidSize < 0 || currentValidSize > maxCapacity) {
            throw new IllegalArgumentException(
                    "currentValidSize must be between 0 and " + maxCapacity);
        }
    }

    /**
     * Returns the exclusive end index (physicalStartIndex + maxCapacity).
     *
     * @return the physical index just past the end of this range
     */
    public int physicalEndIndex() {
        return physicalStartIndex + maxCapacity;
    }

    /**
     * Checks if a given physical index falls within this range's allocation.
     *
     * @param physicalIndex the index to check
     * @return true if physicalIndex is within [physicalStartIndex, physicalEndIndex)
     */
    public boolean containsPhysicalIndex(int physicalIndex) {
        return physicalIndex >= physicalStartIndex && physicalIndex < physicalEndIndex();
    }

    /**
     * Returns the byte offset for this range's start (multiplied by sizeof(double) = 8).
     *
     * @return byte offset for FFM MemorySegment operations
     */
    public long byteOffset() {
        return (long) physicalStartIndex * 8L;
    }

    @Override
    public String toString() {
        return String.format(
                "ValueRange[physical=%d..%d (capacity=%d), valid=%d]",
                physicalStartIndex, physicalEndIndex(), maxCapacity, currentValidSize);
    }
}
