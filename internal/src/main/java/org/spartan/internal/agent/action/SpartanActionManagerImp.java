package org.spartan.internal.agent.action;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.action.type.SpartanAction;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Array;
import java.util.TreeMap;

public class SpartanActionManagerImp implements SpartanActionManager
{

    private MemorySegment predictionDataSegment; // prediction data segment updated each tick by c++ engine
    private final Arena arena;
    private final SpartanAction[] actionArray;

    public SpartanActionManagerImp(@NotNull Arena arena, @NotNull TreeMap<Integer, SpartanAction> actions) {
        this.arena = arena;

        int maxIndex = actions.isEmpty() ? -1 : actions.lastKey();
        int bufferSize = maxIndex + 1;

        this.predictionDataSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, bufferSize);

        this.actionArray = new SpartanAction[bufferSize];
        for (var entry : actions.entrySet()) {
            this.actionArray[entry.getKey()] = entry.getValue();
        }
    }

    @Override
    public @NotNull SpartanAction[] getActions() {
        return actionArray;
    }

    @Override
    @SuppressWarnings("unchecked")
    public @NotNull <SpartanActionType extends SpartanAction> SpartanActionType[] getActionsByType(@NotNull Class<SpartanActionType> actionClass) {
        int count = 0;
        for (SpartanAction action : actionArray) {
            if (actionClass.isInstance(action)) {
                count++;
            }
        }

        SpartanActionType[] result = (SpartanActionType[]) Array.newInstance(actionClass, count);

        int currentIndex = 0;
        for (SpartanAction action : actionArray) {
            if (actionClass.isInstance(action)) {
                result[currentIndex++] = (SpartanActionType) action;
            }
        }

        return result;
    }

    @Override
    public @NotNull SpartanAction[] getActionsByIdentifier(@NotNull String identifier) {
        int count = 0;
        for (SpartanAction action : actionArray) {
            if (action != null && identifier.equals(action.getIdentifier())) {
                count++;
            }
        }

        SpartanAction[] result = new SpartanAction[count];

        int currentIndex = 0;
        for (SpartanAction action : actionArray) {
            if (action != null && identifier.equals(action.getIdentifier())) {
                result[currentIndex++] = action;
            }
        }

        return result;
    }


    /**
     * This method would be called after the model makes a prediction,
     * and would be responsible for running the actions in the action buffer
     */
    public void runActions() {

    }
}
