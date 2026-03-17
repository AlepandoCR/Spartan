package org.spartan.api.agent.context.element;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Range;

/**
 * Convenience base class for a single scalar context element.
 * The value is exposed as a one-element array cached internally.
 */
public abstract class SpartanSingleContextElement implements SpartanContextElement {

    private final double[] dataCache = new double[1];

    /**
     * @return the current scalar value in the range [-1, 1]
     */
    public abstract @Range(from = -1, to = 1) double getValue();

    @Override
    public double @NotNull [] getData() {
        dataCache[0] = getValue();
        return dataCache;
    }

    @Override
    public final void prepare() {
        // no preparation needed, value is calculated on the fly
    }

}
