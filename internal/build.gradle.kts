import org.gradle.kotlin.dsl.invoke
import org.gradle.kotlin.dsl.test
import java.util.Properties

plugins {
    id("java")
    id("com.gradleup.shadow") version "8.3.5"
    id("com.vanniktech.maven.publish")
}

group = "org.spartan.internal"
version = "1.0.23"

java{
    withSourcesJar()
    withJavadocJar()
}

repositories {
    mavenCentral()
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

    val pythonCmd = when {
        System.getProperty("os.name").lowercase().contains("windows") -> {
            listOf("py", "python3", "python").firstOrNull { cmd ->
                try {
                    ProcessBuilder(cmd, "--version").redirectError(ProcessBuilder.Redirect.DISCARD).start().waitFor() == 0
                } catch (e: Exception) {
                    false
                }
            } ?: "py"
        }
        else -> {
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

    inputs.file(cppSourceFile)
    inputs.dir(scriptDir)
    outputs.file(outputFile)

    doFirst {
        if (!outputFile.exists()) {
            logger.info("SpartanNative.java not found, forcing regeneration")
            outputDir.mkdirs()
        }
    }
}

tasks {
    compileJava {
        dependsOn(generateNativeBindings)
    }

    withType<Jar>().configureEach {
        dependsOn(generateNativeBindings)
    }

    withType<JavaExec> {
        jvmArgs("--enable-native-access=ALL-UNNAMED")
    }

    withType<GenerateModuleMetadata>().configureEach {
        enabled = false
    }

    // Avoid duplicate javadoc artifacts in the publication.
    matching { it.name == "mavenPlainJavadocJar" }.configureEach {
        enabled = false
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
val mavenUser = System.getenv("MAVEN_USERNAME") ?: dotEnv.getProperty("MAVEN_USERNAME")
val mavenPass = System.getenv("MAVEN_PASSWORD") ?: dotEnv.getProperty("MAVEN_PASSWORD")

if (mavenUser != null) extra["mavenCentralUsername"] = mavenUser
if (mavenPass != null) extra["mavenCentralPassword"] = mavenPass

val winJar = providers.gradleProperty("winJar").orNull
val linuxJar = providers.gradleProperty("linuxJar").orNull
val macJar = providers.gradleProperty("macJar").orNull
val nativeClassifierProp = providers.gradleProperty("nativeClassifier").orNull
val prebuiltInternalJar = providers.gradleProperty("prebuiltInternalJar").orNull

mavenPublishing {
    coordinates("io.github.alepandocr", "spartan-internal", project.version.toString())

    pom {
        name.set("Spartan Internal")
        description.set("Internal implementation for Spartan")
        url.set("https://github.com/AlepandoCR/Spartan")
        licenses {
            license {
                name.set("GNU Affero General Public License v3.0")
                url.set("https://www.gnu.org/licenses/agpl-3.0.txt")
            }
        }
        developers {
            developer {
                id.set("Alepando")
                name.set("Alepando")
                email.set("aocamporam@gmail.com")
            }
        }
        scm {
            connection.set("scm:git:git://github.com/AlepandoCR/Spartan.git")
            developerConnection.set("scm:git:ssh://github.com/AlepandoCR/Spartan.git")
            url.set("https://github.com/AlepandoCR/Spartan")
        }
    }

    publishToMavenCentral(com.vanniktech.maven.publish.SonatypeHost.CENTRAL_PORTAL, true)
    signAllPublications()
}

publishing {
    publications.withType<MavenPublication>().configureEach {
        if (name == "maven") {
            artifacts.clear()

            if (!winJar.isNullOrBlank() || !linuxJar.isNullOrBlank() || !macJar.isNullOrBlank()) {
                if (!winJar.isNullOrBlank()) artifact(file(winJar)) { classifier = "windows" }
                if (!linuxJar.isNullOrBlank()) artifact(file(linuxJar)) { classifier = "linux" }
                if (!macJar.isNullOrBlank()) artifact(file(macJar)) { classifier = "macos" }
            } else if (!prebuiltInternalJar.isNullOrBlank()) {
                artifact(file(prebuiltInternalJar)) {
                    if (!nativeClassifierProp.isNullOrBlank()) classifier = nativeClassifierProp
                }
            } else {
                artifact(tasks.named("shadowJar")) {
                    if (!nativeClassifierProp.isNullOrBlank()) classifier = nativeClassifierProp
                }
            }

            artifact(tasks.named("sourcesJar"))
            artifact(tasks.named("javadocJar"))
        }
    }
}