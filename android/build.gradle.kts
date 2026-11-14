plugins {
    alias(libs.plugins.android.application) apply false
    // Upgrades AGP 9.0.0 built-in Kotlin compiler classpath to Kotlin 2.4.21 (for LiteRT 2.3.0)
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
