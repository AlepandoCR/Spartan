package org.spartan.api.agent.action.type;

import org.jetbrains.annotations.NotNull;

public interface SpartanAction {

    /**
     * Identifier for the action, could be a name or unique key
     */
    @NotNull String getIdentifier();

    /**
     * Defines the maximum magnitude for this action
     * which can be used to normalize the action's effect
     * the {@code normalizedMagnitude} parameter in {@link #task(double)} will be a value between
     * {@link #taskMinMagnitude()} and this maximum magnitude from its original 1 to 0 range
     * @return the maximum magnitude for this action
     */
    double taskMaxMagnitude();

    /**
     * Defines the minimum magnitude for this action
     * which can be used to normalize the action's effect
     * the {@code normalizedMagnitude} parameter in {@link #task(double)} will be a value between
     * this minimum magnitude and the maximum magnitude from its original 1 to 0 range
     * @return the minimum magnitude for this action
     */
    double taskMinMagnitude();

    /**
     * Task to by executed by this action
     * @param normalizedMagnitude value to execute the task with, normalized to a range defined by {@link #taskMinMagnitude()} and {@link #taskMaxMagnitude()}
     */
    void task(double normalizedMagnitude);
}
