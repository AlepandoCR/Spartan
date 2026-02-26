package org.spartan.internal.agent.context;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.context.SpartanContext;
import org.spartan.api.agent.context.element.SpartanContextElement;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Implementation of SpartanContext using MemorySegment for shared memory with native code.
 * The data is stored in a MemorySegment that C++ can directly read/write.
 * Elements are ordered by index to ensure consistent data order for model training.
 */
public class SpartanContextImpl implements SpartanContext {

    private static final int DEFAULT_INITIAL_CAPACITY = 128;
    private static final ValueLayout.OfDouble DOUBLE_LAYOUT = ValueLayout.JAVA_DOUBLE;

    private final Map<Class<? extends SpartanContextElement>, Collection<SpartanContextElement>> elementsByType;
    private final Map<String, Collection<SpartanContextElement>> elementsByIdentifier;
    private final TreeMap<Integer, SpartanContextElement> elementsByIndex;
    private final String identifier;
    private final Arena arena;

    private MemorySegment dataSegment;
    private int capacity;
    private int validDataSize = 0;

    public SpartanContextImpl(@NotNull String identifier, @NotNull Arena arena) {
        this(identifier, arena, DEFAULT_INITIAL_CAPACITY);
    }

    public SpartanContextImpl(@NotNull String identifier, @NotNull Arena arena, int initialCapacity) {
        this.identifier = identifier;
        this.arena = arena;
        this.capacity = initialCapacity;
        this.elementsByIdentifier = new ConcurrentHashMap<>();
        this.elementsByType = new ConcurrentHashMap<>();
        this.elementsByIndex = new TreeMap<>();

        // Allocate initial MemorySegment
        this.dataSegment = arena.allocate(DOUBLE_LAYOUT, capacity);
    }

    @Override
    public @NotNull MemorySegment getData() {
        return dataSegment;
    }

    @Override
    public int getSize() {
        return validDataSize;
    }

    @Override
    public double readDouble(int index) {
        if (index < 0 || index >= validDataSize) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for size " + validDataSize);
        }
        return dataSegment.getAtIndex(DOUBLE_LAYOUT, index);
    }

    @Override
    public double[] readDoubles(int startIndex, int length) {
        if (startIndex < 0 || startIndex + length > validDataSize) {
            throw new IndexOutOfBoundsException("Range [" + startIndex + ", " + (startIndex + length) + ") out of bounds for size " + validDataSize);
        }
        double[] result = new double[length];
        for (int i = 0; i < length; i++) {
            result[i] = dataSegment.getAtIndex(DOUBLE_LAYOUT, startIndex + i);
        }
        return result;
    }

    @Override
    public void update() {
        int totalValidSize = 0;
        // TreeMap guarantees iteration in key (index) order
        for (SpartanContextElement element : elementsByIndex.values()) {
            element.update();
            totalValidSize += element.getSize();
        }

        ensureCapacity(totalValidSize);

        long offsetBytes = 0;
        for (SpartanContextElement element : elementsByIndex.values()) {
            double[] elementData = element.getData();
            int validSize = element.getSize();

            if (validSize > 0) {
                long bytesToCopy = validSize * DOUBLE_LAYOUT.byteSize();
                MemorySegment.copy(
                        MemorySegment.ofArray(elementData), 0,
                        dataSegment, offsetBytes,
                        bytesToCopy
                );
                offsetBytes += bytesToCopy;
            }
        }

        validDataSize = totalValidSize;
    }

    private void ensureCapacity(int requiredCapacity) {
        if (requiredCapacity > capacity) {
            int newCapacity = Math.max(capacity * 2, requiredCapacity);

            // Allocate new segment
            MemorySegment newSegment = arena.allocate(DOUBLE_LAYOUT, newCapacity);

            // Copy existing data if any
            if (validDataSize > 0) {
                MemorySegment.copy(dataSegment, 0, newSegment, 0, validDataSize * DOUBLE_LAYOUT.byteSize());
            }

            dataSegment = newSegment;
            capacity = newCapacity;
        }
    }

    @Override
    public void addElement(@NotNull SpartanContextElement element, int index) {
        if (elementsByIndex.containsKey(index)) {
            throw new IllegalArgumentException("Element already exists at index " + index);
        }

        elementsByIndex.put(index, element);

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
        // Find and remove from index map
        Integer indexToRemove = null;
        for (Map.Entry<Integer, SpartanContextElement> entry : elementsByIndex.entrySet()) {
            if (entry.getValue().equals(element)) {
                indexToRemove = entry.getKey();
                break;
            }
        }
        if (indexToRemove != null) {
            elementsByIndex.remove(indexToRemove);
        }

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
                // Remove from index map
                elementsByIndex.values().remove(element);
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
        return Collections.unmodifiableCollection(elementsByIndex.values());
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
