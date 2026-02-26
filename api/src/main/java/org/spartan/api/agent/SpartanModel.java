package org.spartan.api.agent;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.config.SpartanModelConfig;
import org.spartan.api.agent.context.SpartanContext;

public interface SpartanModel<SpartanModelConfigType extends SpartanModelConfig> {

    @NotNull SpartanContext getSpartanContext();

    @NotNull SpartanModelConfigType getSpartanModelConfig();

    void tick();

}
