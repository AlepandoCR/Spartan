package org.spartan.internal.agent.context;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.context.SpartanContext;
import org.spartan.api.agent.context.element.SpartanContextElement;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class SpartanContextImpl implements SpartanContext {

    private final Map<Class<? extends SpartanContextElement>, Collection<SpartanContextElement>> elementsByType;
    private final Map<String, Collection<SpartanContextElement>> elementsByIdentifier;
    private final Collection<SpartanContextElement> allElements;
    private final String identifier;
    private final int index;
    private double [] masterBuffer = new double[128];
    private int validDataSize = 0;

    public SpartanContextImpl(@NotNull String identifier, int index) {
        this.identifier = identifier;
        this.index = index;
        this.elementsByIdentifier = new ConcurrentHashMap<>();
        this.elementsByType = new ConcurrentHashMap<>();
        this.allElements = ConcurrentHashMap.newKeySet();
    }

    @Override
    public double @NotNull [] getData() {
        return masterBuffer;
    }

    @Override
    public int getSize() {
        return validDataSize;
    }

    @Override
    public void update() {
        int totalSize = 0;
        for (SpartanContextElement element : allElements) {
            element.update();
            totalSize += element.getSize();
        }

        if (totalSize > masterBuffer.length) {
            int newCapacity = Math.max(masterBuffer.length * 2, totalSize);
            masterBuffer = Arrays.copyOf(masterBuffer, newCapacity);
        }

        int offset = 0;
        for (SpartanContextElement element : allElements) {
            double[] elementData = element.getData();
            int size = element.getSize();

            // Copy element data into master buffer, make sure we check the size
            // to avoid copying uninitialized data from variable elements
            System.arraycopy(elementData, 0, masterBuffer, offset, size);
            offset += size;
        }

        validDataSize = totalSize;
    }

    @Override
    public void addElement(@NotNull SpartanContextElement element) {
        allElements.add(element);

        // Register by identifier
        elementsByIdentifier
                .computeIfAbsent(element.getIdentifier(), _ -> ConcurrentHashMap.newKeySet())
                .add(element);

        // Register by all supertypes
        for (Class<? extends SpartanContextElement> type : getElementTypes(element)) {
            elementsByType
                    .computeIfAbsent(type, _ -> ConcurrentHashMap.newKeySet())
                    .add(element);
        }
    }

    @Override
    public void removeElement(@NotNull SpartanContextElement element) {
        allElements.remove(element);

        // Remove from identifier map
        Collection<SpartanContextElement> byIdentifier = elementsByIdentifier.get(element.getIdentifier());
        if (byIdentifier != null) {
            byIdentifier.remove(element);
            if (byIdentifier.isEmpty()) {
                elementsByIdentifier.remove(element.getIdentifier());
            }
        }

        // Remove from all type maps
        removeElementsFromTypeCache(element);
    }

    private void removeElementsFromTypeCache(@NotNull SpartanContextElement element) {
        for (Class<? extends SpartanContextElement> type : getElementTypes(element)) {
            Collection<SpartanContextElement> byType = elementsByType.get(type);
            if (byType != null) {
                byType.remove(element);
                if (byType.isEmpty()) {
                    elementsByType.remove(type);
                }
            }
        }
    }

    @Override
    public void removeElementsByIdentifier(@NotNull String identifier) {
        Collection<SpartanContextElement> elements = elementsByIdentifier.remove(identifier);
        if (elements != null) {
            for (SpartanContextElement element : elements) {
                allElements.remove(element);
                removeElementsFromTypeCache(element);
            }
        }
    }

    @Override
    public void removeElementsOfType(@NotNull Class<? extends SpartanContextElement> type) {
        Collection<SpartanContextElement> elements = elementsByType.remove(type);
        if (elements != null) {
            for (SpartanContextElement element : new ArrayList<>(elements)) {
                removeElement(element);
            }
        }
    }

    @Override
    public @NotNull Collection<SpartanContextElement> getElements() {
        return Collections.unmodifiableCollection(allElements);
    }

    @Override
    public @NotNull <SpartanContextElementType extends SpartanContextElement> Collection<SpartanContextElementType> getElementsOfType(@NotNull Class<SpartanContextElementType> type) {
        Collection<SpartanContextElement> elements = elementsByType.get(type);
        if (elements == null) {
            return Collections.emptyList();
        }

        List<SpartanContextElementType> result = new ArrayList<>(elements.size());
        for (SpartanContextElement element : elements) {
            if (type.isInstance(element)) {
                result.add(type.cast(element));
            }
        }
        return result;
    }

    @Override
    public @NotNull Collection<SpartanContextElement> getElementByIdentifier(@NotNull String identifier) {
        Collection<SpartanContextElement> elements = elementsByIdentifier.get(identifier);
        if (elements == null) {
            return Collections.emptyList();
        }
        return Collections.unmodifiableCollection(elements);
    }

    @Override
    public @NotNull String getIdentifier() {
        return identifier;
    }

    @Override
    public int getIndex() {
        return index;
    }

    /**
     * Extracts all types (class and interfaces) that extend SpartanContextElement from the element's hierarchy.
     *
     * @param element the element to extract types from
     * @return a set of all SpartanContextElement subtypes the element implements
     */
    private @NotNull Set<Class<? extends SpartanContextElement>> getElementTypes(@NotNull SpartanContextElement element) {
        Set<Class<? extends SpartanContextElement>> types = new HashSet<>();
        collectTypes(element.getClass(), types);
        return types;
    }

    private void collectTypes(Class<?> clazz, @NotNull Set<Class<? extends SpartanContextElement>> types) {
        if (clazz == null || clazz == Object.class) {
            return;
        }

        if (SpartanContextElement.class.isAssignableFrom(clazz)) {
            @SuppressWarnings("unchecked")
            Class<? extends SpartanContextElement> elementType = (Class<? extends SpartanContextElement>) clazz;
            types.add(elementType);
        }

        // Traverse superclass
        collectTypes(clazz.getSuperclass(), types);

        // Traverse interfaces
        for (Class<?> iface : clazz.getInterfaces()) {
            collectTypes(iface, types);
        }
    }
}
