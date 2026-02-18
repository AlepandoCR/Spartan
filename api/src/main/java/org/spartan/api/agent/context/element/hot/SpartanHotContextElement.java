package org.spartan.api.agent.context.element.hot;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.context.element.SpartanContextElement;

import java.util.Arrays;

public abstract class SpartanHotContextElement implements SpartanContextElement {
    protected final double[] dataCache;
    final int size;

    /**
     * Size is needed for hot encoding, since elements should have a fixed amount of characteristics
     * @param size the size of the data array returned by {@link #getData()}
     */
    protected SpartanHotContextElement(int size) {
        this.size = size;
        this.dataCache = new double[this.size];
    }

    protected void clear() {
        Arrays.fill(dataCache, 0.0);
    }

    @Override
    public double @NotNull [] getData() {
        return dataCache;
    }

    @Override
    public int getSize() {
        return dataCache.length;
    }

    @Override
    public final void prepare() {
        clear();
    }



}
