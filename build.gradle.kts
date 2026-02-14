plugins {
    id("base")
}

tasks {
    val copyJar by registering(Copy::class) {
        dependsOn(":internal:shadowJar")
        from(project(":internal").tasks.named("shadowJar"))
        into(layout.buildDirectory.dir("libs"))
        rename { "Spartan.jar" }
    }

    named("build") {
        dependsOn(
            ":api:build",
            ":internal:build",
            copyJar
        )
    }
}

