package com.gneissfind.app.ui.theme

import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.graphics.Color

val ObsidianSurface = Color(0xFF0A1118)
val SlateCard = Color(0xEC0C151F)
val SlateCardElevated = Color(0xF5101C28)
val HudBorderSubtle = Color(0x38B0D4E8)
val HudTextPrimary = Color(0xFFF2F7FA)
val HudTextSecondary = Color(0xFFB8C9D4)
val CyanTripwire = Color(0xFF00E5FF)
val CarnelianAmber = Color(0xFFFFB300)
val JasperCrimson = Color(0xFFFF3D57)
val NephriteEmerald = Color(0xFF00E676)
val AlertRedBanner = Color(0xFFE5002B)

private val GneissDarkColorScheme = darkColorScheme(
    primary = CyanTripwire,
    onPrimary = Color(0xFF002026),
    secondary = CarnelianAmber,
    onSecondary = Color(0xFF261900),
    tertiary = NephriteEmerald,
    background = ObsidianSurface,
    surface = ObsidianSurface,
    surfaceContainer = SlateCard,
    surfaceVariant = Color(0xFF1A2836),
    onBackground = HudTextPrimary,
    onSurface = HudTextPrimary,
    onSurfaceVariant = HudTextSecondary,
    outline = HudBorderSubtle,
    error = JasperCrimson
)

@Composable
fun GneissFindTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = GneissDarkColorScheme) {
        CompositionLocalProvider(LocalContentColor provides HudTextPrimary) {
            content()
        }
    }
}
