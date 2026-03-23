import com.vanniktech.maven.publish.SonatypeHost

plugins {
    id("base")
    id("com.vanniktech.maven.publish.base") version "0.29.0"
}

mavenPublishing {
    publishToMavenCentral(SonatypeHost.CENTRAL_PORTAL, true)
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