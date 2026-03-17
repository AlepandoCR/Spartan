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
val mavenUrl = System.getenv("MAVEN_URL") ?: dotEnv.getProperty("MAVEN_URL")
val mavenSnapshotUrl = System.getenv("MAVEN_SNAPSHOT_URL") ?: dotEnv.getProperty("MAVEN_SNAPSHOT_URL")
val mavenReleaseUrl = System.getenv("MAVEN_RELEASE_URL") ?: dotEnv.getProperty("MAVEN_RELEASE_URL")
val mavenUser = System.getenv("MAVEN_USERNAME") ?: dotEnv.getProperty("MAVEN_USERNAME")
val mavenPass = System.getenv("MAVEN_PASSWORD") ?: dotEnv.getProperty("MAVEN_PASSWORD")
val isSnapshot = project.version.toString().endsWith("-SNAPSHOT")

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
        val selectedUrl = when {
            isSnapshot && !mavenSnapshotUrl.isNullOrBlank() -> mavenSnapshotUrl
            !isSnapshot && !mavenReleaseUrl.isNullOrBlank() -> mavenReleaseUrl
            else -> mavenUrl
        }
        if (!selectedUrl.isNullOrBlank()) {
            maven {
                url = uri(selectedUrl)
                credentials {
                    username = mavenUser
                    password = mavenPass
                }
            }
        }
    }
}
