package org.spartan.api.agent.context;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.context.element.SpartanContextElement;

import java.util.Collection;

public interface SpartanContext {

    /**
     * flattens the elements in {@link #getElements()} into a single collection of doubles
     * @return the flattened data of the context
     * This data WILL be full of garbage, so it is important to only iterate over {@link #getSize()} elements in the returned array
     * Call {@link #update()} before calling this method to ensure the data is up to date
     */
    double @NotNull [] getData();

    int getSize();

    /**
     * Updates all elements in this context by calling {@link SpartanContextElement#update()} on each element.
     * Updates held data for the context
     */
    void update();

    /**
     * Adds an element to this context.
     * @param element the element to add
     */
    void addElement(SpartanContextElement element);

    /**
     * Removes an element from this context.
     * @param element the element to remove
     */
    void removeElement(SpartanContextElement element);

    /**
     * Removes all elements with the specified identifier from this context.
     * @param identifier the identifier of the elements to remove
     */
    void removeElementsByIdentifier(String identifier);

    /**
     * Removes all elements of the specified type from this context.
     * @param type the type of elements to remove
     */
    void removeElementsOfType(Class<? extends SpartanContextElement> type);

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
    <SpartanContextElementType extends SpartanContextElement> @NotNull Collection<SpartanContextElementType> getElementsOfType(Class<SpartanContextElementType> type);


    /**
     * Gets the element with the specified identifier. If multiple elements have the same identifier, all will be returned.
     * @param identifier the identifier of the element(s) to retrieve
     * @return the element(s) with the specified identifier
     */
    @NotNull Collection<SpartanContextElement> getElementByIdentifier(String identifier);

    /**
     * @return an identifier for this context group
     */
    @NotNull String getIdentifier();

    /**
     * @return the index this context will be placed in the context array passed to the agent
     */
    int getIndex();

}
