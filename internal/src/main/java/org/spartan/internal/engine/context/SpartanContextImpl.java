package org.spartan.internal.engine.context;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.engine.context.element.SpartanContextElement;
import org.spartan.api.engine.context.element.variable.SpartanVariableContextElement;
import org.spartan.internal.bridge.SpartanNative;
import org.spartan.internal.engine.model.SpartanModelAllocator;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Implementation of SpartanContext using MemorySegment for shared memory with native code.
 * The data is stored in a MemorySegment that C++ can directly read/write.
 * Elements are ordered by index to ensure consistent data order for model training.
 * <p>
 * Memory Architecture:
 * <ul>
 *   <li>{@code dataSegment}: Contiguous buffer of doubles - flattened element data</li>
 *   <li>{@code cleanSizesSegment}: Int array for variable element sizes (dynamic slicing)</li>
 *   <li>{@code agentIdSegment}: Cached agent identifier for native calls</li>
 *   <li>{@code slotCountSegment}: Cached variable element count for native calls</li>
 * </ul>
 * <p>
 * Zero-GC Hot Path: The {@link #update()} method is designed to run at high frequency
 * without allocating objects. All iteration uses pre-cached arrays indexed by position.
 */
public class SpartanContextImpl implements SpartanContext {

    private static final int DEFAULT_INITIAL_CAPACITY = 128;
    private static final ValueLayout.OfDouble DOUBLE_LAYOUT = ValueLayout.JAVA_DOUBLE;
    private static final ValueLayout.OfInt INT_LAYOUT = ValueLayout.JAVA_INT;


    private final Map<Class<? extends SpartanContextElement>, Collection<SpartanContextElement>> elementsByType;
    private final Map<String, Collection<SpartanContextElement>> elementsByIdentifier;
    private final TreeMap<Integer, SpartanContextElement> elementsByIndex;
    private final String identifier;
    private final Arena arena;

    // Primary data buffer (doubles for context data)
    private MemorySegment dataSegment;
    private int capacity;
    private int validDataSize = 0;

    // Dynamic Slicing Support
    // These fields support variable-length element sizes communicated to C++

    /**
     * Buffer holding the "clean" (valid) size of each variable element.
     * Layout: [size0, size1, size2, ...] where each entry is an int (4 bytes).
     * C++ uses this to know how many valid doubles exist in each variable slice.
     */
    private MemorySegment cleanSizesSegment;

    // ==================== Zero-GC Caches ====================
    // Pre-allocated arrays to avoid iterator/object allocation in update()

    /**
     * Cached array of all elements in index order.
     * Rebuilt only when elements are added/removed (structural modification).
     */
    private SpartanContextElement[] elementsCache;

    /**
     * Cached array of variable elements only (for clean sizes update).
     * Indices correspond to cleanSizesSegment slots.
     */
    private SpartanVariableContextElement[] variableElementsCache;

    /**
     * Number of variable elements (length of variableElementsCache).
     */
    private int variableElementCount = 0;

    /**
     * Flag indicating the element caches need rebuilding.
     */
    private boolean cachesDirty = true;

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

        // Allocate initial data segment with SIMD padding via SpartanModelAllocator
        // This ensures 64-byte alignment and padding to multiple of 8 doubles for AVX-512 safety
        this.dataSegment = SpartanModelAllocator.allocateContextBuffer(arena, capacity);
    }

    public @NotNull MemorySegment getData() {
        return dataSegment;
    }

    /**
     * Internal: Get the Arena used by this context.
     * This arena is shared across all models using this context,
     *
     * @return the Arena used for memory allocation
     */
    public @NotNull Arena getArena() {
        return arena;
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

    /**
     * Updates all elements and flattens their data into the shared dataSegment.
     * <p>
     * Algorithm:
     * <ol>
     *   <li>Rebuild caches if structural modifications occurred</li>
     *   <li>Call update() on each element (computes new data)</li>
     *   <li>Calculate total valid size from all elements</li>
     *   <li>Ensure dataSegment has sufficient capacity</li>
     *   <li>Copy each element's data into dataSegment at correct offset</li>
     *   <li>Write variable element sizes to cleanSizesSegment</li>
     * </ol>
     * <p>
     * Zero-GC: Uses pre-cached arrays with indexed loops. No iterators, no boxing,
     * no intermediate allocations. Safe to call 20+ times per second per agent.
     */
    @Override
    public void update() {
        // Rebuild caches if elements were added/removed
        if (cachesDirty) {
            rebuildCaches();
        }

        // Update all elements and calculate total size
        int totalValidSize = 0;
        final int elementCount = elementsCache.length;
        for (SpartanContextElement element : elementsCache) {
            element.update();
            totalValidSize += element.getSize();
        }

        //Ensure buffer capacity
        ensureCapacity(totalValidSize);

        // Copy element data into dataSegment (contiguous layout)
        // Offset is in BYTES for MemorySegment.copy
        long offsetBytes = 0;
        for (int i = 0; i < elementCount; i++) {
            SpartanContextElement element = elementsCache[i];
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

        // Update clean sizes for variable elements
        // This tells C++ how many valid values each variable slot contains
        if (variableElementCount > 0) {
            for (int i = 0; i < variableElementCount; i++) {
                int size = variableElementsCache[i].getSize();
                cleanSizesSegment.setAtIndex(INT_LAYOUT, i, size);
            }
        }
    }

    /**
     * Synchronizes the clean sizes buffer with the native engine.
     * Call this after {@link #update()} when the agent is registered with C++.
     * <p>
     * This communicates the valid data sizes for each variable element slot,
     * allowing C++ to process only the meaningful portion of variable-length arrays.
     *
     * @param agentIdentifier the unique identifier for this agent in the native registry
     */
    public void syncCleanSizes(long agentIdentifier) {
        if (variableElementCount == 0) {
            return; // No variable elements, nothing to sync
        }

        // Pass primitive values directly - no MemorySegment wrapper needed
        SpartanNative.spartanUpdateCleanSizes(agentIdentifier, cleanSizesSegment, variableElementCount);
    }

    /**
     * Rebuilds the element caches from the TreeMap.
     * Called only when elements are added/removed (not on every update).
     */
    private void rebuildCaches() {
        // Count total elements and variable elements
        int totalCount = elementsByIndex.size();
        int varCount = 0;

        // First pass: count variable elements
        for (SpartanContextElement element : elementsByIndex.values()) {
            if (element instanceof SpartanVariableContextElement) {
                varCount++;
            }
        }

        // Allocate/reallocate caches
        elementsCache = new SpartanContextElement[totalCount];
        variableElementsCache = new SpartanVariableContextElement[varCount];
        variableElementCount = varCount;

        // Second pass: populate caches
        int elementIdx = 0;
        int varIdx = 0;
        for (SpartanContextElement element : elementsByIndex.values()) {
            elementsCache[elementIdx++] = element;
            if (element instanceof SpartanVariableContextElement varElement) {
                variableElementsCache[varIdx++] = varElement;
            }
        }

        // Allocate/reallocate cleanSizesSegment if variable element count changed
        if (varCount > 0) {
            cleanSizesSegment = arena.allocate(INT_LAYOUT, varCount);
        }

        cachesDirty = false;
    }

    private void ensureCapacity(int requiredCapacity) {
        if (requiredCapacity > capacity) {
            int newCapacity = Math.max(capacity * 2, requiredCapacity);

            // Allocate new segment with SIMD padding and alignment via SpartanModelAllocator
            MemorySegment newSegment = SpartanModelAllocator.allocateContextBuffer(arena, newCapacity);

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
        cachesDirty = true; // Mark caches for rebuild

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
            cachesDirty = true; // Mark caches for rebuild
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
