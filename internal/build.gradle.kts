import org.gradle.kotlin.dsl.invoke
import org.gradle.kotlin.dsl.test

plugins {
    id("java")

    id("com.gradleup.shadow") version "8.3.5"
}

group = "org.spartan.internal"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven("https://repo.papermc.io/repository/maven-public/") {
        name = "papermc-repo"
    }
}

dependencies {
    implementation(project(":api"))

    compileOnly("io.papermc.paper:paper-api:1.21.11-R0.1-SNAPSHOT")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}


val generateNativeBindings by tasks.registering(Exec::class) {
    description = "Generates Java FFM bindings from C++ source"
    group = "build"

    workingDir = rootProject.projectDir.resolve("core")
    commandLine("py", rootProject.projectDir.resolve("scripts/generate_ffm_spartan_bridge.py").absolutePath)

    inputs.file(rootProject.projectDir.resolve("core/src/org/spartan/core/SpartanApi.cpp"))
    outputs.file(project.projectDir.resolve("src/main/java/org/spartan/core/bridge/SpartanNative.java"))
}

tasks {

    compileJava {
        dependsOn(generateNativeBindings)
    }

    processResources {
        val props = mapOf("version" to version)
        inputs.properties(props)
        filteringCharset = "UTF-8"
        filesMatching("plugin.yml") {
            expand(props)
        }
    }

    test {
        useJUnitPlatform()
    }

    shadowJar {
        archiveClassifier.set("")
        mergeServiceFiles()
    }

    build {
        dependsOn(shadowJar)
    }
}