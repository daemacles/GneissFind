package com.gneissfind.app.ui.theme

import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.PlatformTextStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.gneissfind.app.R

/**
 * Classic 2360s Star Trek TNG LCARS palette (`lcars-ui.md` §2).
 * Every non-camera background pixel and structural 4.dp gutter is pure `#000000` OLED black.
 */
@Immutable
object LcarsColors {
    val CanvasBlack = Color(0xFF000000)
    val GoldenAmber = Color(0xFFFF9900)
    val ApricotPeach = Color(0xFFFFCC99)
    val WarmSand = Color(0xFFFFCC66)
    val Lavender = Color(0xFFCC99CC)
    val Periwinkle = Color(0xFF9999FF)
    val IceBlue = Color(0xFF99CCFF)
    val MutedSlate = Color(0xFF555577)
    val MarsRed = Color(0xFFCC4444)
    val AlertWhite = Color(0xFFFFFFFF)

    // High-contrast anti-camouflage video target overlay colors (backed by 2.dp #000000 keyline)
    val ReticleAgateCyan = Color(0xFF00FFFF)
    val ReticleJasperYellow = Color(0xFFFFFF00)
    val ReticleGreenMint = Color(0xFF00FF88)
    val ReticleWoodGold = Color(0xFFFFAA00)
    val ReticleConfuserDiagnostic = Color(0xFF777777)
}

/**
 * Bundled 100% offline LCARS typefaces:
 * - [LcarsDisplayFontFamily]: `Antonio` (authentic ultra-compressed 2360s LCARS structural chrome).
 * - [LcarsTabularFontFamily]: `Barlow Condensed` (contains full OpenType `tnum` GSUB metrics so
 *   12–30 Hz live numeric telemetry never shifts horizontal advance width by a single pixel).
 */
val LcarsDisplayFontFamily = FontFamily(
    Font(resId = R.font.antonio_regular, weight = FontWeight.Normal),
    Font(resId = R.font.antonio_semibold, weight = FontWeight.SemiBold),
    Font(resId = R.font.antonio_bold, weight = FontWeight.Bold)
)

val LcarsTabularFontFamily = FontFamily(
    Font(resId = R.font.barlow_condensed_semibold, weight = FontWeight.SemiBold),
    Font(resId = R.font.barlow_condensed_bold, weight = FontWeight.Bold)
)

private val TightPlatformStyle = PlatformTextStyle(includeFontPadding = false)

@Immutable
object LcarsTypography {
    val ElbowCode = TextStyle(
        fontFamily = LcarsDisplayFontFamily,
        fontWeight = FontWeight.Bold,
        fontSize = 13.sp,
        lineHeight = 14.sp,
        letterSpacing = 0.06.em,
        fontFeatureSettings = "tnum",
        platformStyle = TightPlatformStyle
    )

    val RailHeader = TextStyle(
        fontFamily = LcarsDisplayFontFamily,
        fontWeight = FontWeight.Bold,
        fontSize = 12.sp,
        lineHeight = 14.sp,
        letterSpacing = 0.06.em,
        fontFeatureSettings = "tnum",
        platformStyle = TightPlatformStyle
    )

    val TelemetryBar = TextStyle(
        fontFamily = LcarsTabularFontFamily,
        fontWeight = FontWeight.Bold,
        fontSize = 13.sp,
        lineHeight = 14.sp,
        letterSpacing = 0.06.em,
        fontFeatureSettings = "tnum",
        platformStyle = TightPlatformStyle
    )

    val TelemetryLarge = TextStyle(
        fontFamily = LcarsTabularFontFamily,
        fontWeight = FontWeight.Bold,
        fontSize = 15.sp,
        lineHeight = 16.sp,
        letterSpacing = 0.06.em,
        fontFeatureSettings = "tnum",
        platformStyle = TightPlatformStyle
    )

    val PillLabel = TextStyle(
        fontFamily = LcarsDisplayFontFamily,
        fontWeight = FontWeight.Bold,
        fontSize = 13.sp,
        lineHeight = 14.sp,
        letterSpacing = 0.06.em,
        fontFeatureSettings = "tnum",
        platformStyle = TightPlatformStyle
    )

    val PillSubLabel = TextStyle(
        fontFamily = LcarsTabularFontFamily,
        fontWeight = FontWeight.Bold,
        fontSize = 11.sp,
        lineHeight = 12.sp,
        letterSpacing = 0.06.em,
        fontFeatureSettings = "tnum",
        platformStyle = TightPlatformStyle
    )

    val MicroCode = TextStyle(
        fontFamily = LcarsTabularFontFamily,
        fontWeight = FontWeight.Bold,
        fontSize = 10.sp,
        lineHeight = 11.sp,
        letterSpacing = 0.06.em,
        fontFeatureSettings = "tnum",
        platformStyle = TightPlatformStyle
    )
}

private val GneissLcarsColorScheme = darkColorScheme(
    primary = LcarsColors.GoldenAmber,
    onPrimary = LcarsColors.CanvasBlack,
    secondary = LcarsColors.Periwinkle,
    onSecondary = LcarsColors.CanvasBlack,
    tertiary = LcarsColors.Lavender,
    onTertiary = LcarsColors.CanvasBlack,
    background = LcarsColors.CanvasBlack,
    surface = LcarsColors.CanvasBlack,
    surfaceContainer = LcarsColors.CanvasBlack,
    surfaceVariant = LcarsColors.MutedSlate,
    onBackground = LcarsColors.ApricotPeach,
    onSurface = LcarsColors.ApricotPeach,
    onSurfaceVariant = LcarsColors.WarmSand,
    outline = LcarsColors.GoldenAmber,
    error = LcarsColors.MarsRed,
    onError = LcarsColors.AlertWhite
)

private val GneissLcarsMaterialTypography = Typography(
    titleMedium = LcarsTypography.ElbowCode,
    bodyMedium = LcarsTypography.TelemetryBar,
    labelLarge = LcarsTypography.PillLabel,
    labelSmall = LcarsTypography.MicroCode
)

@Composable
fun GneissFindTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = GneissLcarsColorScheme,
        typography = GneissLcarsMaterialTypography
    ) {
        CompositionLocalProvider(LocalContentColor provides LcarsColors.ApricotPeach) {
            content()
        }
    }
}
