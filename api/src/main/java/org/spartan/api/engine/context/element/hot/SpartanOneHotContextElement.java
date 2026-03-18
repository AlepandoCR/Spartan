package org.spartan.api.engine.context.element.hot;

public abstract class SpartanOneHotContextElement extends SpartanHotContextElement {

    /**
     * @param size the size of the data array returned by {@link #getData()}
     */
    protected SpartanOneHotContextElement(int size) {
        super(size);

    }

    /**
     * Sets a single active index (all others become zero).
     */
    protected void set(int index, boolean value) {
        clear(); // one hot encoding means only one value is true at a time
        if (value) {
            dataCache[index] = 1.0;
        }
    }


}