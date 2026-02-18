package org.spartan.api.agent.context.element;

import org.jetbrains.annotations.NotNull;

public abstract class SpartanSingleContextElement implements SpartanContextElement{

    private final double[] dataCache = new double[1];

    public abstract double getValue();

    @Override
    public double @NotNull [] getData() {
        dataCache[0] = getValue();
        return dataCache;
    }

    @Override
    public final void prepare(){
        // no preparation needed, value is calculated on the fly
    }

}
