package org.spartan.api.agent.context.element;

import org.jetbrains.annotations.NotNull;

public interface SpartanContextElement {

    /**
     * Can be a single value or collection that represents the current element
     * @return the data for this context element
     */
    double @NotNull [] getData();

    /**
     * @return the size of the data returned by {@link #getData()}
     */
    int getSize();

    /**
     * Performs any necessary logic to prepare the element for ticking
     */
    void prepare();

    /**
     * Performs the tick logic where data is set for the element
     */
    void tick();

    /**
     * @return an identifier for this context element
     */
    @NotNull String getIdentifier();

    /**
     * Internal use calls {@link #tick()} and {@link #prepare()}
     */
    default void update(){
        prepare();
        tick();
    }


}
