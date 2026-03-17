import org.gradle.kotlin.dsl.invoke
import org.gradle.kotlin.dsl.test
import java.util.Properties

plugins {
    id("java")
    id("com.gradleup.shadow") version "8.3.5"
    id("maven-publish")
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
    implementation("org.jetbrains:annotations:26.0.2")

    testImplementation(project(":api"))
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}


val generateNativeBindings by tasks.registering(Exec::class) {
    description = "Generates Java FFM bindings from C++ source"
    group = "build"

    onlyIf { System.getenv("SKIP_NATIVE_BINDINGS") != "1" }

    workingDir = rootProject.projectDir.resolve("internal")
    commandLine("py", rootProject.projectDir.resolve("scripts/generate_ffm_spartan_bridge.py").absolutePath)

    inputs.file(rootProject.projectDir.resolve("core/src/org/spartan/api/SpartanApi.cpp"))
    outputs.file(project.projectDir.resolve("src/main/java/org/spartan/internal/bridge/SpartanNative.java"))
}

tasks {

    compileJava {
        dependsOn(generateNativeBindings)
    }

    withType<JavaExec> {
        jvmArgs("--enable-native-access=ALL-UNNAMED")
    }

    processResources {
        val props = mapOf("version" to version)
        inputs.properties(props)
        filteringCharset = "UTF-8"
        filesMatching("plugin.yml") {
            expand(props)


        }

        val nativeLibDir = providers.gradleProperty("nativeLibDir").orNull
        val coreLibrarySource = if (nativeLibDir.isNullOrBlank()) {
            rootProject.projectDir.resolve("core/cmake-build-debug/bin")
        } else {
            rootProject.projectDir.resolve(nativeLibDir)
        }

        val libraryDestinationPath = "native"
        // copy library files from cmake build for FFM bindings
        from(coreLibrarySource) {
            include("*.dll", "*.so", "*.dylib")
            into(libraryDestinationPath)
        }
    }

    test {
        useJUnitPlatform()
    }

    val nativeClassifier = providers.gradleProperty("nativeClassifier").orNull
    shadowJar {
        archiveClassifier.set(nativeClassifier ?: "")
        mergeServiceFiles()
    }

    build {
        dependsOn(shadowJar)
    }
}

fun loadDotEnv(rootDir: File): Properties {
    val props = Properties()
    val envFile = rootDir.resolve(".env")
    if (envFile.exists()) {
        envFile.inputStream().use { props.load(it) }
    }
    return props
}

val dotEnv = loadDotEnv(rootProject.projectDir)

publishing {
    publications {
        create<MavenPublication>("internal") {
            artifact(tasks.named("shadowJar"))
            groupId = project.group.toString()
            artifactId = "spartan-internal"
            version = project.version.toString()
        }
    }
    repositories {
        val repoUrl = dotEnv.getProperty("MAVEN_URL")
        if (!repoUrl.isNullOrBlank()) {
            maven {
                url = uri(repoUrl)
                credentials {
                    username = dotEnv.getProperty("MAVEN_USERNAME")
                    password = dotEnv.getProperty("MAVEN_PASSWORD")
                }
            }
        }
    }
}
