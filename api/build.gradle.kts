import java.util.Properties

plugins {
    id("java")
    id("maven-publish")
}

group = "org.spartan.api"
version = "1.0-SNAPSHOT"

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

tasks.test {
    useJUnitPlatform()
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
        create<MavenPublication>("api") {
            from(components["java"])
            groupId = project.group.toString()
            artifactId = "spartan-api"
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
