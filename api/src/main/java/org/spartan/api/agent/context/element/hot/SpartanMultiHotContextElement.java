package org.spartan.api.agent.context.element.hot;

public abstract class SpartanMultiHotContextElement extends SpartanHotContextElement{

    protected SpartanMultiHotContextElement(int size) {
       super(size);
    }

    protected void set(int index, boolean value) {
        dataCache[index] = value ? 1.0 : 0.0;
    }

}
