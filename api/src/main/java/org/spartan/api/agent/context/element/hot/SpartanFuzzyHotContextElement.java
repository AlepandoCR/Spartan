package org.spartan.api.agent.context.element.hot;

public abstract class SpartanFuzzyHotContextElement extends SpartanHotContextElement {

    /**
     * Fuzzy hot context allows partial activation of characteristics in the array.
     *
     * @param size amount of characteristics in the array returned by {@link #getData()}
     *            hot elements always have a fixed size
     */
    protected SpartanFuzzyHotContextElement(int size) {
       super(size);
    }

    /**
     * Sets a continuous activation value for the given index.
     */
    protected void set(int index, double value) {
        dataCache[index] = value;
    }

}
