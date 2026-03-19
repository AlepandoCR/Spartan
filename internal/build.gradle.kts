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

    val rootDir = rootProject.projectDir
    val scriptDir = rootDir.resolve("scripts")
    val cppSourceFile = rootDir.resolve("core/src/org/spartan/api/SpartanApi.cpp")
    val outputDir = project.projectDir.resolve("src/main/java/org/spartan/internal/bridge")
    val outputFile = outputDir.resolve("SpartanNative.java")

    workingDir = scriptDir

    // Detect Python executable - prioritize python3, then py, then python
    val pythonCmd = when {
        System.getProperty("os.name").lowercase().contains("windows") -> {
            // On Windows, try py first (Python launcher), then python
            listOf("py", "python3", "python").firstOrNull { cmd ->
                try {
                    ProcessBuilder(cmd, "--version").redirectError(ProcessBuilder.Redirect.DISCARD).start().waitFor() == 0
                } catch (e: Exception) {
                    false
                }
            } ?: "py"
        }
        else -> {
            // On Unix-like systems, prefer python3
            listOf("python3", "python").firstOrNull { cmd ->
                try {
                    ProcessBuilder(cmd, "--version").redirectError(ProcessBuilder.Redirect.DISCARD).start().waitFor() == 0
                } catch (e: Exception) {
                    false
                }
            } ?: "python3"
        }
    }

    commandLine(pythonCmd, "generate_ffm_spartan_bridge.py",
        "--cpp-source", cppSourceFile.absolutePath,
        "--output", outputDir.absolutePath
    )

    // Track inputs for incremental builds
    inputs.file(cppSourceFile)
    inputs.dir(scriptDir)

    // Track output
    outputs.file(outputFile)

    // Force regeneration on missing output (handles CI where cache might not exist)
    doFirst {
        if (!outputFile.exists()) {
            logger.info("SpartanNative.java not found, forcing regeneration")
            // Ensure output directory exists
            outputDir.mkdirs()
        }
    }
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
val mavenUrl = System.getenv("MAVEN_URL") ?: dotEnv.getProperty("MAVEN_URL")
val mavenUser = System.getenv("MAVEN_USERNAME") ?: dotEnv.getProperty("MAVEN_USERNAME")
val mavenPass = System.getenv("MAVEN_PASSWORD") ?: dotEnv.getProperty("MAVEN_PASSWORD")
val nativeClassifier = providers.gradleProperty("nativeClassifier").orNull
val prebuiltInternalJar = providers.gradleProperty("prebuiltInternalJar").orNull

publishing {
    publications {
        create<MavenPublication>("internal") {
            if (prebuiltInternalJar.isNullOrBlank()) {
                artifact(tasks.named("shadowJar")) {
                    if (!nativeClassifier.isNullOrBlank()) {
                        classifier = nativeClassifier
                    }
                }
            } else {
                artifact(file(prebuiltInternalJar)) {
                    if (!nativeClassifier.isNullOrBlank()) {
                        classifier = nativeClassifier
                    }
                }
            }
            groupId = project.group.toString()
            artifactId = "spartan-internal"
            version = project.version.toString()
        }
    }
    repositories {
        if (!mavenUrl.isNullOrBlank()) {
            maven {
                url = uri(mavenUrl)
                credentials {
                    username = mavenUser
                    password = mavenPass
                }
            }
        }
    }
}
