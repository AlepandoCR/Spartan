package org.spartan.internal.engine.action;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.action.type.SpartanAction;

import java.util.ArrayList;
import java.util.List;

public class SpartanActionManagerImpl implements SpartanActionManager {

    private final List<SpartanAction> actions = new ArrayList<>();

    @Override
    public SpartanActionManager registerAction(SpartanAction action) {
        actions.add(action);
        return this;
    }

    @Override
    public @NotNull List<SpartanAction> getActions() {
        return new ArrayList<>(actions);
    }

    @Override
    public <SpartanActionType extends SpartanAction> @NotNull List<SpartanActionType> getActionsByType(@NotNull Class<SpartanActionType> actionClass) {
        return actions.stream()
                .filter(actionClass::isInstance)
                .map(actionClass::cast)
                .toList();
    }

    @Override
    public @NotNull List<SpartanAction> getActionsByIdentifier(@NotNull String identifier) {
        return actions.stream()
                .filter(a -> a.identifier().equals(identifier))
                .toList();
    }
}
