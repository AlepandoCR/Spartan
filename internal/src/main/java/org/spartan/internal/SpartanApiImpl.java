package org.spartan.internal;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.format.NamedTextColor;
import org.bukkit.plugin.java.JavaPlugin;
import org.spartan.api.SpartanApi;
import org.spartan.core.bridge.SpartanNative;

import java.util.Objects;

public class SpartanApiImpl extends JavaPlugin implements SpartanApi
{

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
    }

    @Override
    public void onDisable(){
        getComponentLogger().info(Component.text("Spartan Disabling...", NamedTextColor.GOLD));
    }
}