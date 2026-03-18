package org.spartan.api.engine.context;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.context.element.SpartanContextElement;
import org.spartan.api.engine.context.element.variable.SpartanVariableContextElement;

import java.util.Collection;

import org.spartan.api.SpartanApi;

/**
 * Represents the agent's perception of the world.
 * <p>
 * <b>Concept:</b> To a neural network, the world is just a long list of numbers (a vector).
 * The SpartanContext is responsible for taking high-level game objects (Players, Items) called {@link SpartanContextElement}s,
 * reading their "numbers" (Health, Position), and flattening them into a highly optimized memory buffer that C++ can read instantly.
 * <p>
 * Elements are written in index order; indices must match the layout used during training.
 * Variable-length elements report a valid prefix length via {@link SpartanVariableContextElement#getSize()}.
 */
public interface SpartanContext {

    /**
     * Creates a new Context via the API.
     *
     * @param api the API instance
     * @param identifier unique name for this context
     * @return a new context
     */
    static SpartanContext build(SpartanApi api, @NotNull String identifier) {
        return api.createContext(identifier);
    }

    /**
     * Returns the total size of the observed world.
     * This is the sum of sizes of all active elements.
     *
     * @return the number of doubles in the observation vector
     */
    int getSize();


    /**
     * Refreshes the agent's view of the world.
     * <p>
     * Call this once per tick before the agent acts.
     * It iterates over all registered elements, asks them for their current values, and writes them to native memory.
     */
    void update();

    /**
     * Registers a new "Sense" or input source.
     *
     * @param element the sensor element (e.g., a "HealthSensor")
     * @param index the position in the vector to write to. IMPORTANT: This must match the index expected by your model's training.
     */
    void addElement(@NotNull SpartanContextElement element, int index);

    /**
     * Removes an element from this context.
     * @param element the element to remove
     */
    void removeElement(@NotNull SpartanContextElement element);

    /**
     * Removes all elements with the specified identifier from this context.
     * @param identifier the identifier of the elements to remove
     */
    void removeElementsByIdentifier(@NotNull String identifier);

    /**
     * Removes all elements of the specified type from this context.
     * @param type the type of elements to remove
     */
    void removeElementsOfType(@NotNull Class<? extends SpartanContextElement> type);

    /**
     * @return the elements that make up this context
     */
    @NotNull Collection<SpartanContextElement> getElements();


    /**
     * Gets the elements of the specified type.
     * @param type the type of elements to retrieve
     * @param <SpartanContextElementType> the type of elements to retrieve
     * @return the elements of the specified type
     */
    <SpartanContextElementType extends SpartanContextElement> @NotNull Collection<SpartanContextElementType> getElementsOfType(@NotNull Class<SpartanContextElementType> type);


    /**
     * Gets the element with the specified identifier. If multiple elements have the same identifier, all will be returned.
     * @param identifier the identifier of the element(s) to retrieve
     * @return the element(s) with the specified identifier
     */
    @NotNull Collection<SpartanContextElement> getElementByIdentifier(@NotNull String identifier);

    /**
     * @return an identifier for this context group
     */
    @NotNull String getIdentifier();

    /**
     * Reads a double value from the shared memory at the specified index.
     * This reflects any changes made by native code.
     *
     * @param index the index of the double to read
     * @return the double value at the specified index
     */
    double readDouble(int index);

    /**
     * Reads multiple double values from the shared memory starting at the specified index.
     * This reflects any changes made by native code.
     *
     * @param startIndex the starting index
     * @param length the number of doubles to read
     * @return an array containing the read values
     */
    double[] readDoubles(int startIndex, int length);

}
