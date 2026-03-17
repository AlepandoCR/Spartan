package org.spartan.api.agent.context.element.variable;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.context.element.SpartanContextElement;

import java.util.Arrays;

/**
 * Base class for variable-length context elements.
 * The backing array can grow, while {@link #getSize()} reports the valid prefix length.
 */
public abstract class SpartanVariableContextElement implements SpartanContextElement {


    private double[] variableCache = new double[16];

    private int filledIndexes = 0;


    @Override
    public final void prepare() {
        filledIndexes = 0;
    }

    /**
     * Appends a single value to the element.
     */
    protected final void add(double value) {
        ensureCapacity(filledIndexes + 1);
        variableCache[filledIndexes++] = value;
    }

    /**
     * Appends a 3D vector to the element.
     */
    protected final void add(double x, double y, double z) {
        ensureCapacity(filledIndexes + 3);
        variableCache[filledIndexes++] = x;
        variableCache[filledIndexes++] = y;
        variableCache[filledIndexes++] = z;
    }


    private void ensureCapacity(int minCapacity) {
        if (minCapacity > variableCache.length) {
            int newCapacity = Math.max(variableCache.length * 2, minCapacity);
            variableCache = Arrays.copyOf(variableCache, newCapacity);
        }
    }

    @Override
    public final double @NotNull [] getData() {
        return variableCache;
    }

    /**
     * @return the number of valid values written via {@link #add(double)}
     */
    @Override
    public final int getSize() {
        return filledIndexes;
    }


}