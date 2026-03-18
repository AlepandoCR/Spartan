package org.spartan.api.engine.context.element.hot;

public abstract class SpartanMultiHotContextElement extends SpartanHotContextElement {

    protected SpartanMultiHotContextElement(int size) {
       super(size);
    }

    /**
     * Sets the state of a specific index without clearing other indices.
     */
    protected void set(int index, boolean value) {
        dataCache[index] = value ? 1.0 : 0.0;
    }

}
