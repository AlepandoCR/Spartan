import java.util.Properties

plugins {
    id("java")
    id("com.vanniktech.maven.publish")
}

group = "org.spartan.api"
version = "1.0.0"

repositories {
    mavenCentral()
    maven("https://repo.papermc.io/repository/maven-public/") {
        name = "papermc-repo"
    }
}

dependencies {
    compileOnly("io.papermc.paper:paper-api:1.21.11-R0.1-SNAPSHOT")
    implementation("org.jetbrains:annotations:26.0.2")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

val emptyJavadocJar by tasks.registering(Jar::class) {
    archiveClassifier.set("javadoc")
}

val emptySourcesJar by tasks.registering(Jar::class) {
    archiveClassifier.set("sources")
}

tasks {
    test {
        useJUnitPlatform()
    }


    withType<GenerateModuleMetadata>().configureEach {
        dependsOn(emptyJavadocJar, emptySourcesJar)
    }

    withType<GenerateMavenPom>().configureEach {
        dependsOn(emptyJavadocJar, emptySourcesJar)
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
    coordinates(project.group.toString(), "spartan-api", project.version.toString())

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

            if (prebuiltApiJar.isNullOrBlank()) {
                artifact(tasks.named("jar")) {
                    if (!nativeClassifier.isNullOrBlank()) {
                        classifier = nativeClassifier
                    }
                }
            } else {
                artifact(file(prebuiltApiJar))
            }

            artifact(emptyJavadocJar)
            artifact(emptySourcesJar)
        }
    }
}