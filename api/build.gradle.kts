import java.util.Properties
import org.gradle.api.tasks.compile.JavaCompile

plugins {
    id("java")
    id("com.vanniktech.maven.publish")
}

group = "org.spartan.api"
version = "1.0.28"

java {
    withSourcesJar()
    withJavadocJar()
    sourceCompatibility = JavaVersion.VERSION_25
    targetCompatibility = JavaVersion.VERSION_25
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains:annotations:26.0.2")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks {
    test {
        useJUnitPlatform()
    }

    withType<JavaCompile>().configureEach {
        // Preserve parameter names in compiled bytecode (needed for Javadoc and IDEs)
        options.compilerArgs.addAll(listOf(
            "-encoding", "UTF-8",
            "-parameters"  // Include method parameter names
        ))
    }

    withType<Javadoc>().configureEach {
        // Generate comprehensive Javadoc with parameter documentation
        options {
            encoding = "UTF-8"
            showFromProtected()  // Show protected and public members
            source = "25"  // Match Java version
        }
    }

    withType<GenerateModuleMetadata>().configureEach {
        enabled = false
    }

    // Avoid duplicate javadoc artifacts in the publication.
    matching { it.name == "mavenPlainJavadocJar" }.configureEach {
        enabled = false
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

val prebuiltApiJar = providers.gradleProperty("prebuiltApiJar").orNull
val nativeClassifier = providers.gradleProperty("nativeClassifier").orNull

mavenPublishing {
    coordinates("io.github.alepandocr", "spartan-api", project.version.toString())

    pom {
        name.set("Spartan API")
        description.set("API project for Spartan")
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

            if (!prebuiltApiJar.isNullOrBlank()) {
                artifact(file(prebuiltApiJar)) {
                    if (!nativeClassifier.isNullOrBlank()) classifier = nativeClassifier
                }
            } else {
                artifact(tasks.named("jar")) {
                    if (!nativeClassifier.isNullOrBlank()) classifier = nativeClassifier
                }
            }

            artifact(tasks.named("sourcesJar"))
            artifact(tasks.named("javadocJar"))
        }
    }
}