/*
 * LCARS color taxonomy, parameterized HSV value stepping, and bundled offline typography.
 * Assigns each functional UI element category a distinct hue/saturation family while stepping
 * adjacent peers across three HSV brightness values (base and +-LCARS_ADJACENT_VALUE_STEP).
 */
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
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Tunable HSV Value (brightness) offset fraction applied between adjacent UI elements that belong
 * to the same semantic color family (`0.05f` = `+-5%` around the category base value).
 *
 * Adjust this single constant (for example `0.05f`..`0.12f`) to widen or tighten brightness
 * contrast between neighboring buttons and telemetry bars while keeping hue and saturation fixed.
 */
const val LCARS_ADJACENT_VALUE_STEP: Float = 0.05f

/**
 * Three discrete brightness tiers used to separate adjacent elements of the same semantic category
 * while preserving identical HSV hue and saturation.
 *
 * @property step Signed multiplier (`-1`, `0`, `+1`) applied to [LCARS_ADJACENT_VALUE_STEP].
 */
enum class LcarsValueLevel(val step: Int) {
    MINUS(-1),
    BASE(0),
    PLUS(1)
}

/**
 * Converts an sRGB [Color] into `[hueDegrees (0..360), saturation (0..1), value (0..1)]`.
 */
internal fun colorToHsv(color: Color): FloatArray {
    val r = color.red.coerceIn(0f, 1f)
    val g = color.green.coerceIn(0f, 1f)
    val b = color.blue.coerceIn(0f, 1f)
    val maxC = max(r, max(g, b))
    val minC = min(r, min(g, b))
    val delta = maxC - minC

    val saturation = if (maxC <= 1e-6f) 0f else (delta / maxC).coerceIn(0f, 1f)
    val hue = when {
        delta <= 1e-6f -> 0f
        maxC == r -> 60f * (((g - b) / delta).mod(6f))
        maxC == g -> 60f * (((b - r) / delta) + 2f)
        else -> 60f * (((r - g) / delta) + 4f)
    }
    return floatArrayOf(hue, saturation, maxC)
}

/**
 * Converts HSV components (`hueDegrees in [0, 360)`, `saturation in [0, 1]`, `value in [0, 1]`)
 * back into an sRGB [Color].
 */
internal fun hsvToColor(hueDegrees: Float, saturation: Float, value: Float, alpha: Float = 1f): Color {
    val h = hueDegrees.mod(360f)
    val s = saturation.coerceIn(0f, 1f)
    val v = value.coerceIn(0f, 1f)
    val c = v * s
    val x = c * (1f - abs((h / 60f).mod(2f) - 1f))
    val m = v - c

    val (r1, g1, b1) = when {
        h < 60f -> Triple(c, x, 0f)
        h < 120f -> Triple(x, c, 0f)
        h < 180f -> Triple(0f, c, x)
        h < 240f -> Triple(0f, x, c)
        h < 300f -> Triple(x, 0f, c)
        else -> Triple(c, 0f, x)
    }
    return Color(
        red = (r1 + m).coerceIn(0f, 1f),
        green = (g1 + m).coerceIn(0f, 1f),
        blue = (b1 + m).coerceIn(0f, 1f),
        alpha = alpha.coerceIn(0f, 1f)
    )
}

/**
 * Returns a copy of this [Color] with identical HSV hue and saturation, shifted in HSV Value by
 * `step * stepFraction` (`step in -1..+1`).
 *
 * If the base color's HSV Value lies within `stepFraction` of `0.0` or `1.0`, the center value is
 * nudged inward by only the amount needed so all three steps (`-1`, `0`, `+1`) remain strictly
 * distinct by `stepFraction`.
 */
fun Color.withValueStep(
    step: Int,
    stepFraction: Float = LCARS_ADJACENT_VALUE_STEP
): Color {
    val clampedStep = step.coerceIn(-1, 1)
    val delta = stepFraction.coerceIn(0f, 0.35f)
    if (delta == 0f) return this

    val hsv = colorToHsv(this)
    val centerV = hsv[2].coerceIn(delta, 1f - delta)
    val targetV = (centerV + (clampedStep * delta)).coerceIn(0f, 1f)
    return hsvToColor(
        hueDegrees = hsv[0],
        saturation = hsv[1],
        value = targetV,
        alpha = this.alpha
    )
}

/**
 * Convenience overload of [withValueStep] accepting a named [LcarsValueLevel].
 */
fun Color.withValueLevel(
    level: LcarsValueLevel,
    stepFraction: Float = LCARS_ADJACENT_VALUE_STEP
): Color = withValueStep(step = level.step, stepFraction = stepFraction)

/**
 * Base 2360s LCARS palette (`lcars-ui.md` §2) calibrated at `V ≈ 0.90` on primary fills so
 * `+-LCARS_ADJACENT_VALUE_STEP` produces three crisp, non-clamped brightness levels on OLED panels.
 */
@Immutable
object LcarsColors {
    val CanvasBlack = Color(0xFF000000)
    val GoldenAmber = Color(0xFFE69119)      // H=35°,  S=0.89, V=0.90
    val ApricotPeach = Color(0xFFE6B88A)     // H=30°,  S=0.40, V=0.90
    val WarmSand = Color(0xFFE6BE67)         // H=41°,  S=0.55, V=0.90
    val Lavender = Color(0xFFD98CE6)         // H=291°, S=0.39, V=0.90
    val Periwinkle = Color(0xFF7D8CE6)       // H=231°, S=0.46, V=0.90
    val IceBlue = Color(0xFF7CC0E6)          // H=202°, S=0.46, V=0.90
    val CopperTerracotta = Color(0xFFDB744B) // H=17°,  S=0.66, V=0.86
    val MutedSlate = Color(0xFF3A3A54)       // H=240°, S=0.31, V=0.33 (>6:1 WCAG contrast w/ InactiveLabel)
    val InactiveLabel = Color(0xFFF5D8BA)    // High-contrast warm ivory for text on MutedSlate
    val MarsRed = Color(0xFFD94444)          // H=0°,   S=0.69, V=0.85
    val AlertWhite = Color(0xFFFFFFFF)

    // High-contrast anti-camouflage video target overlay colors (backed by 2.dp #000000 keyline)
    val ReticleAgateCyan = Color(0xFF00FFFF)
    val ReticleJasperYellow = Color(0xFFFFFF00)
    val ReticleGreenMint = Color(0xFF00FF88)
    val ReticleWoodGold = Color(0xFFFFAA00)
    val ReticleConfuserDiagnostic = Color(0xFF777777)
}

/**
 * Semantic UI color taxonomy mapping each functional control or readout class to a dedicated
 * hue/saturation family, with [LcarsValueLevel] stepping for adjacent peers:
 *
 * 1. [toggleButton] — Binary ON/OFF state toggles (mineral classes `01..04`, `2.0X CROP`, `1/500S`,
 *    `EIS LOCK`, `AUDIO`, `OVERSPEED SIM`): [LcarsColors.GoldenAmber] when ON,
 *    [LcarsColors.MutedSlate] when OFF.
 * 2. [multiStateButton] — Buttons switching between active states (`SCAN / LOG` mode,
 *    `5-TILE / 3-TILE` swath): [LcarsColors.Periwinkle].
 * 3. [actionOrDrawerButton] — Momentary calibration triggers, steppers, and drawer selectors
 *    (`SYNC PITCH`, `SOURCE` drawer trigger, `−`/`+` EV steppers): [LcarsColors.Lavender].
 * 4. [realtimeDataLabel] — Passive non-interactive labels showing dynamic live sensor/perf data
 *    (`PITCH`, `VEL`, `FPS`, `Y-PX`, `SWEEP MS`, `PING HZ`): [LcarsColors.IceBlue].
 * 5. [staticStateLabel] — Passive non-interactive labels showing static or derived configuration
 *    state (`RANGE • RESOLUTION`, `ZOOM • SHUTTER • ISO`, `LOG` count, `47-1792`, `EV` bar):
 *    [LcarsColors.ApricotPeach].
 * 6. [structuralChrome] — Non-interactive LCARS swept frame elbows (`GEO-47`, `SYS-01`):
 *    [LcarsColors.CopperTerracotta].
 * 7. [criticalAlert] — Active limits, overspeed warnings, and live disk recording:
 *    [LcarsColors.MarsRed].
 */
@Immutable
object LcarsSemanticColors {
    fun toggleButton(
        enabled: Boolean,
        level: LcarsValueLevel = LcarsValueLevel.BASE,
        stepFraction: Float = LCARS_ADJACENT_VALUE_STEP
    ): Color {
        val base = if (enabled) LcarsColors.GoldenAmber else LcarsColors.MutedSlate
        return base.withValueLevel(level, stepFraction)
    }

    fun multiStateButton(
        level: LcarsValueLevel = LcarsValueLevel.BASE,
        stepFraction: Float = LCARS_ADJACENT_VALUE_STEP
    ): Color = LcarsColors.Periwinkle.withValueLevel(level, stepFraction)

    fun actionOrDrawerButton(
        level: LcarsValueLevel = LcarsValueLevel.BASE,
        stepFraction: Float = LCARS_ADJACENT_VALUE_STEP
    ): Color = LcarsColors.Lavender.withValueLevel(level, stepFraction)

    fun realtimeDataLabel(
        level: LcarsValueLevel = LcarsValueLevel.BASE,
        stepFraction: Float = LCARS_ADJACENT_VALUE_STEP
    ): Color = LcarsColors.IceBlue.withValueLevel(level, stepFraction)

    fun staticStateLabel(
        level: LcarsValueLevel = LcarsValueLevel.BASE,
        stepFraction: Float = LCARS_ADJACENT_VALUE_STEP
    ): Color = LcarsColors.ApricotPeach.withValueLevel(level, stepFraction)

    fun structuralChrome(
        level: LcarsValueLevel = LcarsValueLevel.BASE,
        stepFraction: Float = LCARS_ADJACENT_VALUE_STEP
    ): Color = LcarsColors.CopperTerracotta.withValueLevel(level, stepFraction)

    val criticalAlert: Color = LcarsColors.MarsRed
}

/**
 * Bundled 100% offline LCARS typefaces:
 * - [LcarsDisplayFontFamily]: `Antonio` (ultra-compressed 2360s LCARS structural chrome & labels).
 * - [LcarsTabularFontFamily]: `Barlow Condensed` (contains OpenType `tnum` GSUB metrics so live
 *   numeric telemetry never shifts digit advance width).
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
