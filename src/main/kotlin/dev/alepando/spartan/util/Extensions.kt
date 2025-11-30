package dev.alepando.spartan.util

import dev.alepando.spartan.Spartan
import org.bukkit.Bukkit
import org.bukkit.entity.BlockDisplay
import org.bukkit.plugin.java.JavaPlugin
import org.bukkit.scheduler.BukkitRunnable
import org.bukkit.scheduler.BukkitTask
import org.bukkit.util.Transformation
import org.joml.Quaternionf

private val plugin: JavaPlugin = JavaPlugin.getPlugin(Spartan::class.java)

fun async(handler: () -> Unit) {
    Bukkit.getScheduler().runTaskAsynchronously(plugin, handler)
}


fun sync(handler: () -> Unit) {
    Bukkit.getScheduler().runTask(plugin, handler)
}


fun later(handler: () -> Unit, delay: Long) {
    Bukkit.getScheduler().runTaskLater(plugin, handler,delay)
}

fun timer(handler: () -> Unit, delay: Long, period: Long): BukkitTask {
    val runnable = Bukkit.getScheduler().runTaskTimer(plugin, handler,delay,period)
    return runnable
}

fun BukkitRunnable.timer(delay: Long,period: Long){
    this.runTaskTimer(plugin,delay,period)
}

fun timerRunnable(handler: () -> Unit, delay: Long, period: Long): BukkitRunnable {
    val runnable = object : BukkitRunnable(){
        override fun run(){
            handler.invoke()
        }
    }
    return runnable
}


fun BlockDisplay.rotateX(degrees: Double) {
    val old = this.transformation
    this.transformation = Transformation(
        old.translation,
        Quaternionf(Math.toRadians(degrees).toFloat(), 1f, 0f, 0f),
        old.scale,
        old.rightRotation
    )
}

fun BlockDisplay.rotateY(degrees: Double) {
    val old = this.transformation
    this.transformation = Transformation(
        old.translation,
        Quaternionf(Math.toRadians(degrees).toFloat(), 0f, 1f, 0f),
        old.scale,
        old.rightRotation
    )
}

fun BlockDisplay.rotateZ(degrees: Double) {
    val old = this.transformation
    this.transformation = Transformation(
        old.translation,
        Quaternionf(Math.toRadians(degrees).toFloat(), 0f, 0f, 1f),
        old.scale,
        old.rightRotation
    )
}