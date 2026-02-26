package org.spartan.api.agent.context;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.context.element.SpartanContextElement;
import org.spartan.api.agent.context.element.variable.SpartanVariableContextElement;

import java.lang.foreign.MemorySegment;
import java.util.Collection;

/**
 * Represents an AI context that holds elements with shared memory backing.
 * The data is stored in a MemorySegment that can be directly accessed by native code (C++).
 * Any modifications made by C++ are instantly visible in Java without copying.
 */
public interface SpartanContext {

    /**
     * Gets the MemorySegment containing the flattened data of all elements.
     * This segment is directly accessible by native code - changes made by C++ are visible instantly.
     * Call {@link #update()} before accessing to ensure data is current.
     *
     * @return the MemorySegment containing the context data
     */
    @NotNull MemorySegment getData();

    /**
     * @return the number of valid doubles in the data segment
     */
    int getSize();


    /**
     * Updates all elements in this context by calling {@link SpartanContextElement#update()} on each element.
     * Copies element data into the shared MemorySegment.
     */
    void update();

    /**
     * Adds an element to this context.
     * @param element the element to add
     * @param index index at which the element's data should be written, this is essential for model learning
     * @apiNote {@link SpartanContext#update()} cleans {@link SpartanVariableContextElement} data so order could be any
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
