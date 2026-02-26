package org.spartan.internal.util;

import org.jetbrains.annotations.NotNull;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public final class SpartanMemoryUtil {

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

}
