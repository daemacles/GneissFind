/*
 * Reusable 2360s LCARS geometric primitives, silent tactile pill buttons, detection-reticle color
 * swatches, and strictly non-interactive telemetry readout blocks.
 *
 * Enforces visual affordance separation (.agents/rules/lcars-ui.md): interactive actuators use
 * rounded pill geometry with an 80 ms silent white tap flash, whereas passive telemetry readouts
 * use flat rectangular geometry with zero click handlers. Consumed by ScannerScreen.kt.
 *
 * Main types:
 * - LcarsElbowOrientation / LcarsElbowMetricsPx / LcarsElbowShape / LcarsElbow: swept 2360s TNG elbow geometry and composable.
 * - LcarsPillEndStyle / LcarsPillButton: interactive pill button with 80 ms white tap flash.
 * - LcarsDetectionReticleSwatch: 24.dp mineral reticle swatch rendered inside class toggle pills.
 * - LcarsHeaderSegment / LcarsTelemetryBlock: passive non-interactive rectangular readout bars.
 * - LcarsSegmentedEvBar: 7-segment canopy exposure bias indicator flanked by -/+ pill steppers.
 */
package com.gneissfind.app.ui.lcars

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Outline
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import com.gneissfind.app.ui.theme.LcarsColors
import com.gneissfind.app.ui.theme.LcarsSemanticColors
import com.gneissfind.app.ui.theme.LcarsTypography
import com.gneissfind.app.ui.theme.LcarsValueLevel
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlinx.coroutines.delay

/** Standard `4.dp` OLED `#000000` gap separating adjacent LCARS rail segments and pill buttons (`.agents/rules/lcars-ui.md` §1). */
val LcarsGutter = 4.dp

/** Default vertical stem width (`68.dp`) of a swept LCARS elbow. */
val LcarsStemWidth = 68.dp

/** Standard horizontal bar height (`22.dp`) of a swept LCARS elbow and [LcarsHeaderSegment]. */
val LcarsBarHeight = 22.dp

/** Default convex outer corner radius (`32.dp`) of a swept LCARS elbow (`.agents/rules/lcars-ui.md` §1). */
val LcarsOuterRadius = 32.dp

/** Default concave inner crook fillet radius (`14.dp`) of a swept LCARS elbow (`.agents/rules/lcars-ui.md` §1). */
val LcarsInnerRadius = 14.dp

/**
 * Quadrant orientation of a swept 2360s TNG LCARS elbow (`LcarsElbow`).
 */
enum class LcarsElbowOrientation {
    TopLeft,
    BottomLeft,
    TopRight,
    BottomRight
}

/**
 * Corner geometry styles separating rounded interactive pills from flat rectangular readout bars.
 */
enum class LcarsPillEndStyle {
    /** Full `50%` semicylindrical capsule ends on both left and right sides. */
    Capsule,

    /** `6.dp` left corners and `999.dp` semicircular right cap (e.g. `+` stepper button). */
    FlatLeftRoundRight,

    /** `999.dp` semicircular left cap and `6.dp` right corners (e.g. `−` stepper button). */
    RoundLeftFlatRight,

    /** `12.dp` rounded corners for compact stem actuators (`54.dp..68.dp` wide). */
    CompactRoundedPill,

    /** Sharp `0.dp` rectangular corners reserved for passive telemetry blocks. */
    FlatBlock;

    /** Converts this [LcarsPillEndStyle] into its Compose [Shape]. */
    fun toShape(): Shape = when (this) {
        Capsule -> RoundedCornerShape(percent = 50)
        FlatLeftRoundRight -> RoundedCornerShape(
            topStart = 6.dp,
            bottomStart = 6.dp,
            topEnd = 999.dp,
            bottomEnd = 999.dp
        )
        RoundLeftFlatRight -> RoundedCornerShape(
            topStart = 999.dp,
            bottomStart = 999.dp,
            topEnd = 6.dp,
            bottomEnd = 6.dp
        )
        CompactRoundedPill -> RoundedCornerShape(12.dp)
        FlatBlock -> RectangleShape
    }
}

/**
 * Measures the exact rendered width of [maxPrototypeText] under [style] plus [horizontalPadding] so
 * live telemetry bars can lock to their high-water-mark width without horizontal layout jitter.
 *
 * @param maxPrototypeText Longest possible string for the telemetry field (e.g. `"SWEEP 000.0MS (00.0HZ)"`).
 * @param style Typography style used to render the telemetry string ([LcarsTypography.TelemetryBar] default).
 * @param horizontalPadding Total horizontal padding added to the measured text width (`22.dp` default).
 * @return High-water-mark [Dp] width sufficient to render [maxPrototypeText] on a single line without clipping.
 */
@Composable
fun rememberHighWaterSegmentWidth(
    maxPrototypeText: String,
    style: TextStyle = LcarsTypography.TelemetryBar,
    horizontalPadding: Dp = 22.dp
): Dp {
    val measurer = rememberTextMeasurer()
    val density = LocalDensity.current
    val upper = remember(maxPrototypeText) { maxPrototypeText.uppercase(Locale.US) }
    val widthPx = remember(measurer, upper, style, density) {
        measurer.measure(text = upper, style = style, maxLines = 1, softWrap = false).size.width
    }
    return with(density) { widthPx.toDp() + horizontalPadding }
}

/**
 * Clamped pixel dimensions of a swept LCARS elbow used for path construction and analytic hit-testing.
 *
 * @property widthPx Total bounding width of the elbow in pixels.
 * @property heightPx Total bounding height of the elbow in pixels.
 * @property stemWidthPx Vertical stem width in pixels (`<= widthPx`).
 * @property barHeightPx Horizontal bar height in pixels (`<= heightPx`).
 * @property outerRadiusPx Convex outer corner radius in pixels (`<= min(widthPx, heightPx)`).
 * @property innerRadiusPx Concave inner crook fillet radius in pixels (`<= min(widthPx - stemWidthPx, heightPx - barHeightPx)`).
 */
@Immutable
data class LcarsElbowMetricsPx(
    val widthPx: Float,
    val heightPx: Float,
    val stemWidthPx: Float,
    val barHeightPx: Float,
    val outerRadiusPx: Float,
    val innerRadiusPx: Float
) {
    /**
     * Analytic test of whether `(x, y)` lies inside the swept L-shaped elbow with outer convex
     * radius [outerRadiusPx] and concave inner crook fillet [innerRadiusPx].
     *
     * @param x Horizontal coordinate in local elbow pixels (`0..widthPx`).
     * @param y Vertical coordinate in local elbow pixels (`0..heightPx`).
     * @param orientation Quadrant orientation of the elbow.
     * @return `true` if `(x, y)` is inside the filled elbow region (including the concave inner fillet), `false` otherwise.
     */
    fun containsPoint(x: Float, y: Float, orientation: LcarsElbowOrientation): Boolean {
        if (x < 0f || x > widthPx || y < 0f || y > heightPx) return false
        val lx = when (orientation) {
            LcarsElbowOrientation.TopLeft, LcarsElbowOrientation.BottomLeft -> x
            LcarsElbowOrientation.TopRight, LcarsElbowOrientation.BottomRight -> widthPx - x
        }
        val ly = when (orientation) {
            LcarsElbowOrientation.TopLeft, LcarsElbowOrientation.TopRight -> y
            LcarsElbowOrientation.BottomLeft, LcarsElbowOrientation.BottomRight -> heightPx - y
        }

        if (lx < outerRadiusPx && ly < outerRadiusPx) {
            val dx = lx - outerRadiusPx
            val dy = ly - outerRadiusPx
            if (dx * dx + dy * dy > outerRadiusPx * outerRadiusPx) return false
        }

        if (lx <= stemWidthPx || ly <= barHeightPx) return true

        if (lx <= stemWidthPx + innerRadiusPx && ly <= barHeightPx + innerRadiusPx) {
            val cx = stemWidthPx + innerRadiusPx
            val cy = barHeightPx + innerRadiusPx
            val dx = lx - cx
            val dy = ly - cy
            return (dx * dx + dy * dy) >= (innerRadiusPx * innerRadiusPx)
        }

        return false
    }
}

/**
 * Clamps raw elbow dimensions and radii into a geometrically valid [LcarsElbowMetricsPx] soneither
 * [outerRadiusPx] nor [innerRadiusPx] can self-intersect on small viewports.
 */
fun resolveLcarsElbowMetricsPx(
    widthPx: Float,
    heightPx: Float,
    stemWidthPx: Float,
    barHeightPx: Float,
    outerRadiusPx: Float,
    innerRadiusPx: Float
): LcarsElbowMetricsPx {
    val w = max(0f, widthPx)
    val h = max(0f, heightPx)
    val stem = stemWidthPx.coerceIn(0f, w)
    val bar = barHeightPx.coerceIn(0f, h)
    val rOut = outerRadiusPx.coerceIn(0f, min(w, h))
    val rIn = innerRadiusPx.coerceIn(0f, min(max(0f, w - stem), max(0f, h - bar)))
    return LcarsElbowMetricsPx(
        widthPx = w,
        heightPx = h,
        stemWidthPx = stem,
        barHeightPx = bar,
        outerRadiusPx = rOut,
        innerRadiusPx = rIn
    )
}

/**
 * Builds the Classic 2360s TNG swept LCARS elbow [Path] with smooth convex outer corner
 * ([outerRadiusPx]) and concave inner crook fillet ([innerRadiusPx]).
 *
 * @param widthPx Total bounding width in pixels.
 * @param heightPx Total bounding height in pixels.
 * @param orientation Quadrant orientation of the elbow.
 * @param stemWidthPx Vertical stem width in pixels.
 * @param barHeightPx Horizontal bar height in pixels.
 * @param outerRadiusPx Convex outer corner radius in pixels.
 * @param innerRadiusPx Concave inner crook fillet radius in pixels.
 * @return Closed [Path] representing the swept L-shaped elbow.
 */
fun buildLcarsElbowPath(
    widthPx: Float,
    heightPx: Float,
    orientation: LcarsElbowOrientation,
    stemWidthPx: Float,
    barHeightPx: Float,
    outerRadiusPx: Float,
    innerRadiusPx: Float
): Path {
    val m = resolveLcarsElbowMetricsPx(
        widthPx = widthPx,
        heightPx = heightPx,
        stemWidthPx = stemWidthPx,
        barHeightPx = barHeightPx,
        outerRadiusPx = outerRadiusPx,
        innerRadiusPx = innerRadiusPx
    )
    val w = m.widthPx
    val h = m.heightPx
    val stem = m.stemWidthPx
    val bar = m.barHeightPx
    val rOut = m.outerRadiusPx
    val rIn = m.innerRadiusPx

    return Path().apply {
        if (w <= 0f || h <= 0f) return@apply

        when (orientation) {
            LcarsElbowOrientation.TopLeft -> {
                moveTo(0f, rOut)
                if (rOut > 0f) {
                    arcTo(
                        rect = Rect(0f, 0f, 2f * rOut, 2f * rOut),
                        startAngleDegrees = 180f,
                        sweepAngleDegrees = 90f,
                        forceMoveTo = false
                    )
                } else {
                    lineTo(0f, 0f)
                }
                lineTo(w, 0f)
                lineTo(w, bar)
                lineTo(stem + rIn, bar)
                if (rIn > 0f) {
                    arcTo(
                        rect = Rect(stem, bar, stem + 2f * rIn, bar + 2f * rIn),
                        startAngleDegrees = 270f,
                        sweepAngleDegrees = -90f,
                        forceMoveTo = false
                    )
                } else {
                    lineTo(stem, bar)
                }
                lineTo(stem, h)
                lineTo(0f, h)
                close()
            }

            LcarsElbowOrientation.BottomLeft -> {
                moveTo(0f, 0f)
                lineTo(stem, 0f)
                lineTo(stem, h - bar - rIn)
                if (rIn > 0f) {
                    arcTo(
                        rect = Rect(stem, h - bar - 2f * rIn, stem + 2f * rIn, h - bar),
                        startAngleDegrees = 180f,
                        sweepAngleDegrees = -90f,
                        forceMoveTo = false
                    )
                } else {
                    lineTo(stem, h - bar)
                }
                lineTo(w, h - bar)
                lineTo(w, h)
                lineTo(rOut, h)
                if (rOut > 0f) {
                    arcTo(
                        rect = Rect(0f, h - 2f * rOut, 2f * rOut, h),
                        startAngleDegrees = 90f,
                        sweepAngleDegrees = 90f,
                        forceMoveTo = false
                    )
                } else {
                    lineTo(0f, h)
                }
                close()
            }

            LcarsElbowOrientation.TopRight -> {
                moveTo(0f, 0f)
                lineTo(w - rOut, 0f)
                if (rOut > 0f) {
                    arcTo(
                        rect = Rect(w - 2f * rOut, 0f, w, 2f * rOut),
                        startAngleDegrees = 270f,
                        sweepAngleDegrees = 90f,
                        forceMoveTo = false
                    )
                } else {
                    lineTo(w, 0f)
                }
                lineTo(w, h)
                lineTo(w - stem, h)
                lineTo(w - stem, bar + rIn)
                if (rIn > 0f) {
                    arcTo(
                        rect = Rect(w - stem - 2f * rIn, bar, w - stem, bar + 2f * rIn),
                        startAngleDegrees = 0f,
                        sweepAngleDegrees = -90f,
                        forceMoveTo = false
                    )
                } else {
                    lineTo(w - stem, bar)
                }
                lineTo(0f, bar)
                close()
            }

            LcarsElbowOrientation.BottomRight -> {
                moveTo(w - stem, 0f)
                lineTo(w, 0f)
                lineTo(w, h - rOut)
                if (rOut > 0f) {
                    arcTo(
                        rect = Rect(w - 2f * rOut, h - 2f * rOut, w, h),
                        startAngleDegrees = 0f,
                        sweepAngleDegrees = 90f,
                        forceMoveTo = false
                    )
                } else {
                    lineTo(w, h)
                }
                lineTo(0f, h)
                lineTo(0f, h - bar)
                lineTo(w - stem - rIn, h - bar)
                if (rIn > 0f) {
                    arcTo(
                        rect = Rect(w - stem - 2f * rIn, h - bar - 2f * rIn, w - stem, h - bar),
                        startAngleDegrees = 90f,
                        sweepAngleDegrees = -90f,
                        forceMoveTo = false
                    )
                } else {
                    lineTo(w - stem, h - bar)
                }
                close()
            }
        }
    }
}

/**
 * Compose [Shape] wrapper around [buildLcarsElbowPath] for clipping or outlining LCARS elbow containers.
 */
class LcarsElbowShape(
    private val orientation: LcarsElbowOrientation,
    private val stemWidth: Dp = LcarsStemWidth,
    private val barHeight: Dp = LcarsBarHeight,
    private val outerRadius: Dp = LcarsOuterRadius,
    private val innerRadius: Dp = LcarsInnerRadius
) : Shape {
    override fun createOutline(
        size: Size,
        layoutDirection: LayoutDirection,
        density: Density
    ): Outline = with(density) {
        Outline.Generic(
            buildLcarsElbowPath(
                widthPx = size.width,
                heightPx = size.height,
                orientation = orientation,
                stemWidthPx = stemWidth.toPx(),
                barHeightPx = barHeight.toPx(),
                outerRadiusPx = outerRadius.toPx(),
                innerRadiusPx = innerRadius.toPx()
            )
        )
    }
}

/**
 * Non-interactive structural 2360s LCARS swept elbow composable with optional [armLabel] and [stemCode].
 */
@Composable
fun LcarsElbow(
    orientation: LcarsElbowOrientation,
    color: Color,
    modifier: Modifier = Modifier,
    stemWidth: Dp = LcarsStemWidth,
    barHeight: Dp = LcarsBarHeight,
    outerRadius: Dp = LcarsOuterRadius,
    innerRadius: Dp = LcarsInnerRadius,
    armLabel: String? = null,
    stemCode: String? = null,
    armTextStyle: TextStyle = LcarsTypography.ElbowCode,
    stemTextStyle: TextStyle = LcarsTypography.MicroCode
) {
    Box(modifier = modifier) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val path = buildLcarsElbowPath(
                widthPx = size.width,
                heightPx = size.height,
                orientation = orientation,
                stemWidthPx = stemWidth.toPx(),
                barHeightPx = barHeight.toPx(),
                outerRadiusPx = outerRadius.toPx(),
                innerRadiusPx = innerRadius.toPx()
            )
            drawPath(path = path, color = color)
        }

        if (!armLabel.isNullOrEmpty()) {
            val armAlignment = when (orientation) {
                LcarsElbowOrientation.TopLeft -> Alignment.TopEnd
                LcarsElbowOrientation.BottomLeft -> Alignment.BottomEnd
                LcarsElbowOrientation.TopRight -> Alignment.TopStart
                LcarsElbowOrientation.BottomRight -> Alignment.BottomStart
            }
            Box(
                modifier = Modifier
                    .align(armAlignment)
                    .height(barHeight)
                    .padding(horizontal = 8.dp),
                contentAlignment = Alignment.CenterEnd
            ) {
                Text(
                    text = armLabel.uppercase(Locale.US),
                    style = armTextStyle,
                    color = LcarsColors.CanvasBlack,
                    maxLines = 1,
                    overflow = TextOverflow.Clip
                )
            }
        }

        if (!stemCode.isNullOrEmpty()) {
            val stemAlignment = when (orientation) {
                LcarsElbowOrientation.TopLeft -> Alignment.BottomStart
                LcarsElbowOrientation.BottomLeft -> Alignment.TopStart
                LcarsElbowOrientation.TopRight -> Alignment.BottomEnd
                LcarsElbowOrientation.BottomRight -> Alignment.TopEnd
            }
            Box(
                modifier = Modifier
                    .align(stemAlignment)
                    .width(stemWidth)
                    .padding(horizontal = 6.dp, vertical = 4.dp),
                contentAlignment = Alignment.CenterEnd
            ) {
                Text(
                    text = stemCode.uppercase(Locale.US),
                    style = stemTextStyle,
                    color = LcarsColors.CanvasBlack,
                    maxLines = 1,
                    overflow = TextOverflow.Clip
                )
            }
        }
    }
}

/**
 * Prominent `24.dp` square swatch drawn inside mineral target pills that mirrors the exact
 * four-corner detection reticle square rendered around stones in the camera viewport.
 */
@Composable
fun LcarsDetectionReticleSwatch(
    reticleColor: Color,
    enabled: Boolean,
    modifier: Modifier = Modifier
) {
    Canvas(
        modifier = modifier
            .size(24.dp)
            .clip(RoundedCornerShape(4.dp))
            .background(LcarsColors.CanvasBlack)
    ) {
        val pad = 3.dp.toPx()
        val l = pad
        val t = pad
        val r = size.width - pad
        val b = size.height - pad
        val arm = 6.dp.toPx()
        val stroke = 2.4.dp.toPx()
        val bracketColor = if (enabled) reticleColor else reticleColor.copy(alpha = 0.48f)

        val corners = arrayOf(
            Offset(l, t) to Offset(l + arm, t),
            Offset(l, t) to Offset(l, t + arm),
            Offset(r, t) to Offset(r - arm, t),
            Offset(r, t) to Offset(r, t + arm),
            Offset(l, b) to Offset(l + arm, b),
            Offset(l, b) to Offset(l, b - arm),
            Offset(r, b) to Offset(r - arm, b),
            Offset(r, b) to Offset(r, b - arm)
        )
        corners.forEach { (start, end) ->
            drawLine(
                color = bracketColor,
                start = start,
                end = end,
                strokeWidth = stroke,
                cap = StrokeCap.Square
            )
        }

        // Bold inner mineral color core so hue association is unmistakable at arm's length
        val coreSize = 9.dp.toPx()
        val coreOffset = (size.width - coreSize) * 0.5f
        drawRect(
            color = if (enabled) reticleColor else reticleColor.copy(alpha = 0.25f),
            topLeft = Offset(coreOffset, coreOffset),
            size = Size(coreSize, coreSize)
        )
    }
}

/**
 * Interactive LCARS pill button with rounded capsule geometry and silent `80 ms` `#FFFFFF` tap
 * flash. Never used for passive sensor/state labels (see [LcarsHeaderSegment] and
 * [LcarsTelemetryBlock]).
 */
@Composable
fun LcarsPillButton(
    label: String,
    active: Boolean,
    activeColor: Color,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    inactiveColor: Color = LcarsColors.MutedSlate,
    prefixCode: String? = null,
    subLabel: String? = null,
    swatchColor: Color? = null,
    endStyle: LcarsPillEndStyle = LcarsPillEndStyle.Capsule,
    compactStemMode: Boolean = false
) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    var flashNonce by remember { mutableIntStateOf(0) }
    var isFlashing by remember { mutableStateOf(false) }

    LaunchedEffect(flashNonce) {
        if (flashNonce > 0) {
            isFlashing = true
            delay(80L)
            isFlashing = false
        }
    }

    val showWhiteFlash = isPressed || isFlashing
    val containerColor = when {
        showWhiteFlash -> LcarsColors.AlertWhite
        active -> activeColor
        else -> inactiveColor
    }
    val contentColor = when {
        showWhiteFlash -> LcarsColors.CanvasBlack
        active && activeColor == LcarsColors.MarsRed -> LcarsColors.AlertWhite
        active -> LcarsColors.CanvasBlack
        else -> LcarsColors.InactiveLabel
    }

    val resolvedEndStyle = if (compactStemMode && endStyle == LcarsPillEndStyle.FlatBlock) {
        LcarsPillEndStyle.CompactRoundedPill
    } else {
        endStyle
    }

    val startPad = when {
        compactStemMode -> 6.dp
        swatchColor != null -> 12.dp
        resolvedEndStyle == LcarsPillEndStyle.Capsule ||
            resolvedEndStyle == LcarsPillEndStyle.RoundLeftFlatRight -> 14.dp
        else -> 10.dp
    }
    val endPad = when {
        compactStemMode -> 7.dp
        resolvedEndStyle == LcarsPillEndStyle.Capsule ||
            resolvedEndStyle == LcarsPillEndStyle.FlatLeftRoundRight -> 15.dp
        else -> 10.dp
    }
    val verticalPad = when {
        compactStemMode -> 3.dp
        swatchColor != null -> 2.dp
        else -> 4.dp
    }

    Box(
        modifier = modifier
            .clip(resolvedEndStyle.toShape())
            .background(containerColor)
            .clickable(
                interactionSource = interactionSource,
                indication = null,
                role = Role.Button,
                onClick = {
                    flashNonce += 1
                    onClick()
                }
            )
            .padding(
                start = startPad,
                end = endPad,
                top = verticalPad,
                bottom = verticalPad
            )
    ) {
        if (compactStemMode) {
            Row(
                modifier = Modifier.fillMaxSize(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = if (!prefixCode.isNullOrEmpty()) {
                    Arrangement.SpaceBetween
                } else {
                    Arrangement.End
                }
            ) {
                if (!prefixCode.isNullOrEmpty()) {
                    Text(
                        text = prefixCode.uppercase(Locale.US),
                        style = LcarsTypography.MicroCode,
                        color = contentColor,
                        maxLines = 1,
                        overflow = TextOverflow.Clip,
                        modifier = Modifier.align(Alignment.Top)
                    )
                }
                Column(
                    horizontalAlignment = Alignment.End,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = label.uppercase(Locale.US),
                        style = LcarsTypography.PillLabel,
                        color = contentColor,
                        textAlign = TextAlign.End,
                        maxLines = 1,
                        overflow = TextOverflow.Clip
                    )
                    if (!subLabel.isNullOrEmpty()) {
                        Text(
                            text = subLabel.uppercase(Locale.US),
                            style = LcarsTypography.MicroCode,
                            color = contentColor,
                            textAlign = TextAlign.End,
                            maxLines = 1,
                            overflow = TextOverflow.Clip
                        )
                    }
                }
            }
        } else {
            Row(
                modifier = Modifier.fillMaxSize(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(7.dp)
                ) {
                    if (swatchColor != null) {
                        LcarsDetectionReticleSwatch(
                            reticleColor = swatchColor,
                            enabled = active
                        )
                    }
                    if (!prefixCode.isNullOrEmpty()) {
                        Text(
                            text = prefixCode.uppercase(Locale.US),
                            style = LcarsTypography.MicroCode,
                            color = contentColor,
                            maxLines = 1,
                            overflow = TextOverflow.Clip
                        )
                    }
                }

                Spacer(modifier = Modifier.width(6.dp))

                Column(
                    horizontalAlignment = Alignment.End,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = label.uppercase(Locale.US),
                        style = LcarsTypography.PillLabel,
                        color = contentColor,
                        textAlign = TextAlign.End,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    if (!subLabel.isNullOrEmpty()) {
                        Text(
                            text = subLabel.uppercase(Locale.US),
                            style = LcarsTypography.PillSubLabel,
                            color = contentColor,
                            textAlign = TextAlign.End,
                            maxLines = 1,
                            overflow = TextOverflow.Clip
                        )
                    }
                }
            }
        }
    }
}

/**
 * Passive non-interactive [LcarsBarHeight] (`22.dp`) horizontal LCARS telemetry rail block with sharp
 * rectangular edges and tabular numerals (`tnum`). Never clickable and never flashes on tap.
 */
@Composable
fun LcarsHeaderSegment(
    text: String,
    color: Color,
    modifier: Modifier = Modifier,
    textColor: Color = LcarsColors.CanvasBlack,
    alignEnd: Boolean = false
) {
    Box(
        modifier = modifier
            .height(LcarsBarHeight)
            .background(color)
            .padding(horizontal = 8.dp),
        contentAlignment = if (alignEnd) Alignment.CenterEnd else Alignment.Center
    ) {
        Text(
            text = text.uppercase(Locale.US),
            style = LcarsTypography.TelemetryBar,
            color = textColor,
            maxLines = 1,
            overflow = TextOverflow.Clip
        )
    }
}

/**
 * Passive non-interactive multi-line rectangular telemetry block used in vertical stems and gauge
 * headers (`FPS`, `LOG` frame count, `Y-PX`). Sharp corners and zero click handling distinguish it
 * visually and behaviorally from rounded [LcarsPillButton] actuators.
 */
@Composable
fun LcarsTelemetryBlock(
    valueText: String,
    color: Color,
    modifier: Modifier = Modifier,
    prefixCode: String? = null,
    subText: String? = null,
    textColor: Color = LcarsColors.CanvasBlack
) {
    Box(
        modifier = modifier
            .background(color)
            .padding(horizontal = 5.dp, vertical = 3.dp)
    ) {
        if (!prefixCode.isNullOrEmpty()) {
            Text(
                text = prefixCode.uppercase(Locale.US),
                style = LcarsTypography.MicroCode,
                color = textColor,
                maxLines = 1,
                overflow = TextOverflow.Clip,
                modifier = Modifier.align(Alignment.TopStart)
            )
        }
        Column(
            modifier = Modifier.align(Alignment.BottomEnd),
            horizontalAlignment = Alignment.End,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = valueText.uppercase(Locale.US),
                style = LcarsTypography.TelemetryBar,
                color = textColor,
                textAlign = TextAlign.End,
                maxLines = 1,
                overflow = TextOverflow.Clip
            )
            if (!subText.isNullOrEmpty()) {
                Text(
                    text = subText.uppercase(Locale.US),
                    style = LcarsTypography.MicroCode,
                    color = textColor,
                    textAlign = TextAlign.End,
                    maxLines = 1,
                    overflow = TextOverflow.Clip
                )
            }
        }
    }
}

/**
 * Segmented LCARS Canopy Exposure Bias controller (`-2.0 .. +1.0 EV` in `0.5 EV` steps).
 * Flanks a strictly passive 7-segment rectangular EV readout bar with dedicated `−` and `+`
 * rounded stepper actuators so each EV adjustment direction has a single button.
 */
@Composable
fun LcarsSegmentedEvBar(
    exposureBiasEv: Float,
    onSelectEv: (Float) -> Unit,
    modifier: Modifier = Modifier
) {
    val steps = remember { listOf(-2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f) }
    val valueCycle = remember {
        listOf(
            LcarsValueLevel.MINUS,
            LcarsValueLevel.BASE,
            LcarsValueLevel.PLUS,
            LcarsValueLevel.MINUS,
            LcarsValueLevel.BASE,
            LcarsValueLevel.PLUS,
            LcarsValueLevel.MINUS
        )
    }

    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(LcarsGutter)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "OPT-EV // CANOPY EXPOSURE BIAS",
                style = LcarsTypography.MicroCode,
                color = LcarsSemanticColors.staticStateLabel(LcarsValueLevel.BASE)
            )
            Text(
                text = String.format(Locale.US, "%+.1f EV", exposureBiasEv),
                style = LcarsTypography.TelemetryBar,
                color = LcarsSemanticColors.staticStateLabel(LcarsValueLevel.PLUS)
            )
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(28.dp),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            LcarsPillButton(
                label = "−",
                active = exposureBiasEv > -1.95f,
                activeColor = LcarsSemanticColors.actionOrDrawerButton(LcarsValueLevel.MINUS),
                inactiveColor = LcarsSemanticColors.toggleButton(false, LcarsValueLevel.MINUS),
                endStyle = LcarsPillEndStyle.RoundLeftFlatRight,
                onClick = {
                    val next = (exposureBiasEv - 0.5f).coerceIn(-2.0f, 1.0f)
                    onSelectEv(next)
                },
                modifier = Modifier
                    .width(36.dp)
                    .fillMaxHeight()
            )

            steps.forEachIndexed { idx, stepEv ->
                val level = valueCycle[idx]
                val isSelected = abs(exposureBiasEv - stepEv) < 0.15f
                val isZeroRef = abs(stepEv) < 0.05f
                val isLit = if (exposureBiasEv >= -0.05f) {
                    stepEv in -0.05f..(exposureBiasEv + 0.05f)
                } else {
                    stepEv in (exposureBiasEv - 0.05f)..0.05f
                }

                val segColor = when {
                    isSelected -> LcarsSemanticColors.staticStateLabel(LcarsValueLevel.PLUS)
                    isLit || isZeroRef -> LcarsSemanticColors.staticStateLabel(level)
                    else -> LcarsSemanticColors.toggleButton(false, level)
                }
                val segTextColor = if (isSelected || isLit || isZeroRef) {
                    LcarsColors.CanvasBlack
                } else {
                    LcarsColors.InactiveLabel
                }
                val segLabel = when {
                    isZeroRef -> "0"
                    stepEv == -2.0f -> "-2"
                    stepEv == -1.0f -> "-1"
                    stepEv == 1.0f -> "+1"
                    else -> ""
                }

                LcarsHeaderSegment(
                    text = segLabel,
                    color = segColor,
                    textColor = segTextColor,
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                )
            }

            LcarsPillButton(
                label = "+",
                active = exposureBiasEv < 0.95f,
                activeColor = LcarsSemanticColors.actionOrDrawerButton(LcarsValueLevel.PLUS),
                inactiveColor = LcarsSemanticColors.toggleButton(false, LcarsValueLevel.PLUS),
                endStyle = LcarsPillEndStyle.FlatLeftRoundRight,
                onClick = {
                    val next = (exposureBiasEv + 0.5f).coerceIn(-2.0f, 1.0f)
                    onSelectEv(next)
                },
                modifier = Modifier
                    .width(36.dp)
                    .fillMaxHeight()
            )
        }
    }
}
