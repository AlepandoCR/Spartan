package org.spartan.api.engine.config.spi;

/**
 * Registry to hold the concrete implementation of configuration creation.
 * The internal module must call {@link #set(SpartanConfigFactoryServiceProvider)} during initialization.
 */
public final class SpartanConfigRegistry {
    private static SpartanConfigFactoryServiceProvider provider;

    private SpartanConfigRegistry() {}

    public static void set(SpartanConfigFactoryServiceProvider spartanConfigFactoryServiceProvider) {
        provider = spartanConfigFactoryServiceProvider;
    }

    public static SpartanConfigFactoryServiceProvider get() {
        if (provider == null) {
            throw new IllegalStateException(
                "Spartan Engine implementation not detected! " +
                "Ensure the 'internal' module is present and initialized before creating configurations."
            );
        }
        return provider;
    }
}
