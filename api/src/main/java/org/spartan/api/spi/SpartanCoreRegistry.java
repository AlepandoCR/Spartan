package org.spartan.api.spi;

import org.jetbrains.annotations.NotNull;

public final class SpartanCoreRegistry {
    private static SpartanApiProvider provider;

    private SpartanCoreRegistry() {}

    public static void set(@NotNull SpartanApiProvider spartanApiProvider) {
        provider = spartanApiProvider;
    }

    public static SpartanApiProvider get() {
        if (provider == null) {
            throw new IllegalStateException("Spartan Core not initialized. Ensure internal module is loaded.");
        }
        return provider;
    }
}
