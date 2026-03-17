package org.spartan.api.agent.context.element;

import org.jetbrains.annotations.NotNull;

/**
 * Base contract for a single context element (sensor) that contributes values to a SpartanContext.
 * Implementations should return a stable array reference and update its contents on each tick.
 */
public interface SpartanContextElement {

    /**
     * Returns the backing data array for this element.
     * Implementations should reuse the same array instance to avoid allocations.
     *
     * @return the data for this context element
     */
    double @NotNull [] getData();

    /**
     * @return the number of valid values in {@link #getData()}
     */
    int getSize();

    /**
     * Prepares the element for ticking (clears caches, resets counters, etc.).
     */
    void prepare();

    /**
     * Updates the element values for the current tick.
     */
    void tick();

    /**
     * @return a stable identifier for this element
     */
    @NotNull String getIdentifier();

    /**
     * Internal lifecycle hook that calls {@link #prepare()} and {@link #tick()} in order.
     */
    default void update() {
        prepare();
        tick();
    }


}
