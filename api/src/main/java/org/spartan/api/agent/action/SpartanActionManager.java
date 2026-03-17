package org.spartan.api.agent.action;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.action.type.SpartanAction;

import java.util.List;


/**
 * Manages the registry of actions available to an agent.
 * <p>
 * <b>Concept:</b> If the Agent is the driver, the ActionManager is the steering wheel and pedals.
 * It translates the neural network's mathematical output (just numbers) into meaningful logic events.
 * You must register every possible move (Jump, Shoot, Move Left) with this manager before starting the agent.
 */
public interface SpartanActionManager {

    /**
     * Registers a new action.
     * The order of registration matters - it corresponds to the output indices of the neural network.
     *
     * @param action the action to register
     * @return this manager (for chaining)
     */
    SpartanActionManager registerAction(SpartanAction action);

    /**
     * Returns all registered actions.
     *
     * @return unmodifiable list of actions
     */
    @NotNull List<SpartanAction> getActions();

    /**
     * Finds and returns all actions of a specific class type.
     * Useful for retrieving a set of related actions (e.g., all MovementActions).
     *
     * @param actionClass the class type to filter by
     * @param <SpartanActionType> the generic type
     * @return list of matching actions
     */
    <SpartanActionType extends SpartanAction> @NotNull List<SpartanActionType> getActionsByType(@NotNull Class<SpartanActionType> actionClass);

    /**
     * Finds and returns all actions matching a string identifier.
     *
     * @param identifier the identifier to search for
     * @return list of matching actions
     */
    @NotNull List<SpartanAction> getActionsByIdentifier(@NotNull String identifier);
}
