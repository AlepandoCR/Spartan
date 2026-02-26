package org.spartan.api.agent.action;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.action.type.SpartanAction;

import java.util.Collection;

public interface SpartanActionManager {

    @NotNull SpartanAction[] getActions();

    <SpartanActionType extends SpartanAction> @NotNull SpartanActionType[] getActionsByType(@NotNull Class<SpartanActionType> actionClass);

    @NotNull SpartanAction[] getActionsByIdentifier(@NotNull String identifier);
}
