package org.spartan.internal.util;

import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public final class SpartanMemoryUtil {

    private SpartanMemoryUtil() {} // Utility class

    // Double Operations

    /**
     * Converts a double array to a MemorySegment allocated in the provided Arena.
     * @param data the double array to convert
     * @param arena the Arena to allocate the MemorySegment in
     * @return a MemorySegment containing the data from the double array, or MemorySegment.NULL if the input array is null
     */
    public static MemorySegment toMemorySegment(double[] data, @NotNull Arena arena) {
        if (data == null) return MemorySegment.NULL;

        MemorySegment segment = arena.allocate(
                ValueLayout.JAVA_DOUBLE,
                data.length
        );

        MemorySegment.copy(
                MemorySegment.ofArray(data),
                ValueLayout.JAVA_DOUBLE,
                0,
                segment,
                ValueLayout.JAVA_DOUBLE,
                0,
                data.length
        );

        return segment;
    }

    /**
     * Reads a double value from the MemorySegment at the specified index.
     * @param memorySegment the segment to read from
     * @param index the index (in double units, not bytes)
     * @return the double value at the specified index
     */
    public static double readDouble(@NotNull MemorySegment memorySegment, int index) {
        if (index < 0) {
            throw new IndexOutOfBoundsException("Index " + index + " cannot be negative");
        }
        return memorySegment.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
    }

    /**
     * Writes a double value to the MemorySegment at the specified index.
     * @param memorySegment the segment to write to
     * @param index the index (in double units, not bytes)
     * @param value the value to write
     */
    public static void writeDouble(@NotNull MemorySegment memorySegment, int index, double value) {
        if (index < 0) {
            throw new IndexOutOfBoundsException("Index " + index + " cannot be negative");
        }
        memorySegment.setAtIndex(ValueLayout.JAVA_DOUBLE, index, value);
    }

    /**
     * Reads multiple double values from the MemorySegment.
     * @param memorySegment the segment to read from
     * @param startIndex the starting index (in double units)
     * @param length the number of doubles to read
     * @return an array containing the read values
     */
    public static double @NonNull [] readDoubles(@NotNull MemorySegment memorySegment, int startIndex, int length) {
        if (startIndex < 0) {
            throw new IndexOutOfBoundsException("Start index " + startIndex + " cannot be negative");
        }
        double[] result = new double[length];
        for (int i = 0; i < length; i++) {
            result[i] = memorySegment.getAtIndex(ValueLayout.JAVA_DOUBLE, startIndex + i);
        }
        return result;
    }

    //  Int Operations

    /**
     * Reads an int value from the MemorySegment at the specified index.
     * @param memorySegment the segment to read from
     * @param index the index (in int units, not bytes)
     * @return the int value at the specified index
     */
    public static int readInt(@NotNull MemorySegment memorySegment, int index) {
        if (index < 0) {
            throw new IndexOutOfBoundsException("Index " + index + " cannot be negative");
        }
        return memorySegment.getAtIndex(ValueLayout.JAVA_INT, index);
    }

    /**
     * Writes an int value to the MemorySegment at the specified index.
     * @param memorySegment the segment to write to
     * @param index the index (in int units, not bytes)
     * @param value the value to write
     */
    public static void writeInt(@NotNull MemorySegment memorySegment, int index, int value) {
        if (index < 0) {
            throw new IndexOutOfBoundsException("Index " + index + " cannot be negative");
        }
        memorySegment.setAtIndex(ValueLayout.JAVA_INT, index, value);
    }

    /**
     * Reads multiple int values from the MemorySegment.
     * @param memorySegment the segment to read from
     * @param startIndex the starting index (in int units)
     * @param length the number of ints to read
     * @return an array containing the read values
     */
    public static int @NonNull [] readInts(@NotNull MemorySegment memorySegment, int startIndex, int length) {
        if (startIndex < 0) {
            throw new IndexOutOfBoundsException("Start index " + startIndex + " cannot be negative");
        }
        int[] result = new int[length];
        for (int i = 0; i < length; i++) {
            result[i] = memorySegment.getAtIndex(ValueLayout.JAVA_INT, startIndex + i);
        }
        return result;
    }

    // Long Operations

    /**
     * Reads a long value from the MemorySegment at the specified index.
     * @param memorySegment the segment to read from
     * @param index the index (in long units, not bytes)
     * @return the long value at the specified index
     */
    public static long readLong(@NotNull MemorySegment memorySegment, int index) {
        if (index < 0) {
            throw new IndexOutOfBoundsException("Index " + index + " cannot be negative");
        }
        return memorySegment.getAtIndex(ValueLayout.JAVA_LONG, index);
    }

    /**
     * Writes a long value to the MemorySegment at the specified index.
     * @param memorySegment the segment to write to
     * @param index the index (in long units, not bytes)
     * @param value the value to write
     */
    public static void writeLong(@NotNull MemorySegment memorySegment, int index, long value) {
        if (index < 0) {
            throw new IndexOutOfBoundsException("Index " + index + " cannot be negative");
        }
        memorySegment.setAtIndex(ValueLayout.JAVA_LONG, index, value);
    }

    // Allocation Helpers

    /**
     * Allocates a MemorySegment for a single long value.
     * Useful for passing scalar parameters to native functions.
     * @param arena the Arena to allocate in
     * @param value the initial value
     * @return the allocated segment
     */
    public static @NotNull MemorySegment allocateLong(@NotNull Arena arena, long value) {
        MemorySegment segment = arena.allocate(ValueLayout.JAVA_LONG);
        segment.set(ValueLayout.JAVA_LONG, 0, value);
        return segment;
    }

    /**
     * Allocates a MemorySegment for a single int value.
     * Useful for passing scalar parameters to native functions.
     * @param arena the Arena to allocate in
     * @param value the initial value
     * @return the allocated segment
     */
    public static @NotNull MemorySegment allocateInt(@NotNull Arena arena, int value) {
        MemorySegment segment = arena.allocate(ValueLayout.JAVA_INT);
        segment.set(ValueLayout.JAVA_INT, 0, value);
        return segment;
    }

    /**
     * Allocates a contiguous MemorySegment for doubles with proper alignment.
     * @param arena the Arena to allocate in
     * @param count the number of doubles to allocate
     * @return the allocated segment (zeroed)
     */
    public static MemorySegment allocateDoubles(@NotNull Arena arena, int count) {
        return arena.allocate(ValueLayout.JAVA_DOUBLE, count);
    }

    /**
     * Allocates a contiguous MemorySegment for ints with proper alignment.
     * @param arena the Arena to allocate in
     * @param count the number of ints to allocate
     * @return the allocated segment (zeroed)
     */
    public static MemorySegment allocateInts(@NotNull Arena arena, int count) {
        return arena.allocate(ValueLayout.JAVA_INT, count);
    }
}
