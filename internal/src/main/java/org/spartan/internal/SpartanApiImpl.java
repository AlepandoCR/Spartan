package org.spartan.internal;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.format.NamedTextColor;
import org.bukkit.plugin.java.JavaPlugin;
import org.jetbrains.annotations.NotNull;
import org.spartan.api.SpartanApi;
import org.spartan.internal.bridge.SpartanNative;

import java.lang.foreign.Arena;

public class SpartanApiImpl extends JavaPlugin implements SpartanApi
{
    private Arena sharedArena;

    @Override
    public void onLoad(){
        getComponentLogger().info(Component.text("Spartan Loading...", NamedTextColor.AQUA));
        SpartanNative.spartanInit();
    }

    @Override
    public void onEnable()
    {
        getComponentLogger().info(Component.text("Spartan Enabling...", NamedTextColor.GREEN));
        SpartanNative.spartanLog("Logging from Spartan Java to C++!");

        // Initialize shared Arena for native memory - lives for plugin lifetime
        sharedArena = Arena.ofShared();

    }

    @Override
    public void onDisable(){
        getComponentLogger().info(Component.text("Spartan Disabling...", NamedTextColor.GOLD));

        // Close Arena to free all native memory
        if (sharedArena != null) {
            sharedArena.close();
            sharedArena = null;
        }
    }

    /**
     * Gets the shared Arena used for native memory allocation.
     * All agent contexts should use this Arena so memory persists for the plugin lifetime.
     *
     * @implNote only available in Spartan/internal
     *
     * @return the shared Arena
     */
    public @NotNull Arena getSharedArena() {
        return sharedArena;
    }
}