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
import androidx.compose.foundation.shape.CircleShape
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
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Outline
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import com.gneissfind.app.ui.theme.LcarsColors
import com.gneissfind.app.ui.theme.LcarsTypography
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlinx.coroutines.delay

val LcarsGutter = 4.dp
val LcarsStemWidth = 68.dp
val LcarsBarHeight = 22.dp
val LcarsOuterRadius = 32.dp
val LcarsInnerRadius = 14.dp

enum class LcarsElbowOrientation {
    TopLeft,
    BottomLeft,
    TopRight,
    BottomRight
}

enum class LcarsPillEndStyle {
    Capsule,
    FlatLeftRoundRight,
    RoundLeftFlatRight,
    FlatBlock;

    fun toShape(): Shape = when (this) {
        Capsule -> RoundedCornerShape(percent = 50)
        FlatLeftRoundRight -> RoundedCornerShape(
            topStartPercent = 0,
            bottomStartPercent = 0,
            topEndPercent = 50,
            bottomEndPercent = 50
        )
        RoundLeftFlatRight -> RoundedCornerShape(
            topStartPercent = 50,
            bottomStartPercent = 50,
            topEndPercent = 0,
            bottomEndPercent = 0
        )
        FlatBlock -> RectangleShape
    }
}

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
     * radius `outerRadiusPx` and concave inner crook fillet `innerRadiusPx`.
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

        // Check outer top-left convex corner exclusion
        if (lx < outerRadiusPx && ly < outerRadiusPx) {
            val dx = lx - outerRadiusPx
            val dy = ly - outerRadiusPx
            if (dx * dx + dy * dy > outerRadiusPx * outerRadiusPx) return false
        }

        // Inside vertical stem or horizontal top arm
        if (lx <= stemWidthPx || ly <= barHeightPx) return true

        // Check concave inner crook fillet region: [stem, stem + rIn] x [bar, bar + rIn]
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
 * (`outerRadiusPx`) and concave inner crook fillet (`innerRadiusPx`).
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
    stemCode: String? = null
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
                    style = LcarsTypography.ElbowCode,
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
                    style = LcarsTypography.MicroCode,
                    color = LcarsColors.CanvasBlack,
                    maxLines = 1,
                    overflow = TextOverflow.Clip
                )
            }
        }
    }
}

/**
 * Classic 2360s TNG LCARS actuator pill:
 * - Zero UI tap sound (no ripple audio, `indication = null`, root `isSoundEffectsEnabled = false`).
 * - Immediate `80 ms` pure `#FFFFFF` (`AlertWhite`) visual flash upon tap.
 * - Bold `#000000` right-aligned uppercase text on active color fill, or `#FFCC99` (`ApricotPeach`)
 *   on `MutedSlate` (`#555577`) when inactive.
 */
@Composable
fun LcarsPillButton(
    label: String,
    active: Boolean,
    activeColor: Color,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
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
        else -> LcarsColors.MutedSlate
    }
    val contentColor = when {
        showWhiteFlash -> LcarsColors.CanvasBlack
        active && activeColor == LcarsColors.MarsRed -> LcarsColors.AlertWhite
        active -> LcarsColors.CanvasBlack
        else -> LcarsColors.ApricotPeach
    }

    val startPad = when {
        compactStemMode -> 5.dp
        endStyle == LcarsPillEndStyle.Capsule ||
            endStyle == LcarsPillEndStyle.RoundLeftFlatRight -> 15.dp
        else -> 10.dp
    }
    val endPad = when {
        compactStemMode -> 6.dp
        endStyle == LcarsPillEndStyle.Capsule ||
            endStyle == LcarsPillEndStyle.FlatLeftRoundRight -> 16.dp
        else -> 10.dp
    }

    Box(
        modifier = modifier
            .clip(endStyle.toShape())
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
                top = if (compactStemMode) 2.dp else 4.dp,
                bottom = if (compactStemMode) 2.dp else 4.dp
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
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    if (swatchColor != null) {
                        Box(
                            modifier = Modifier
                                .size(11.dp)
                                .clip(CircleShape)
                                .background(LcarsColors.CanvasBlack)
                                .padding(2.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .clip(CircleShape)
                                    .background(swatchColor)
                            )
                        }
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
 * `22.dp` horizontal LCARS telemetry rail block with tabular numerals (`tnum`).
 * Optionally interactive (with silent `80 ms` `#FFFFFF` flash) for inline actions such as `PITCH SYNC`.
 */
@Composable
fun LcarsHeaderSegment(
    text: String,
    color: Color,
    modifier: Modifier = Modifier,
    endStyle: LcarsPillEndStyle = LcarsPillEndStyle.FlatBlock,
    textColor: Color = LcarsColors.CanvasBlack,
    alignEnd: Boolean = false,
    onClick: (() -> Unit)? = null
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

    val flashing = onClick != null && (isPressed || isFlashing)
    val bg = if (flashing) LcarsColors.AlertWhite else color
    val fg = if (flashing) LcarsColors.CanvasBlack else textColor

    val clickMod = if (onClick != null) {
        Modifier.clickable(
            interactionSource = interactionSource,
            indication = null,
            role = Role.Button,
            onClick = {
                flashNonce += 1
                onClick()
            }
        )
    } else {
        Modifier
    }

    Box(
        modifier = modifier
            .height(LcarsBarHeight)
            .clip(endStyle.toShape())
            .background(bg)
            .then(clickMod)
            .padding(horizontal = 8.dp),
        contentAlignment = if (alignEnd) Alignment.CenterEnd else Alignment.Center
    ) {
        Text(
            text = text.uppercase(Locale.US),
            style = LcarsTypography.TelemetryBar,
            color = fg,
            maxLines = 1,
            overflow = TextOverflow.Clip
        )
    }
}

/**
 * Segmented LCARS bar-graph controller for Canopy Exposure Bias (`-2.0 .. +1.0 EV` in `0.5 EV` steps).
 * Uses strictly solid color blocks separated by `4.dp` `#000000` gutters—zero Material Slider tracks.
 */
@Composable
fun LcarsSegmentedEvBar(
    exposureBiasEv: Float,
    onSelectEv: (Float) -> Unit,
    modifier: Modifier = Modifier
) {
    val steps = remember { listOf(-2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f) }

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
                color = LcarsColors.ApricotPeach
            )
            Text(
                text = String.format(Locale.US, "%+.1f EV", exposureBiasEv),
                style = LcarsTypography.TelemetryBar,
                color = if (abs(exposureBiasEv) < 0.05f) LcarsColors.IceBlue else LcarsColors.GoldenAmber
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
                activeColor = LcarsColors.WarmSand,
                endStyle = LcarsPillEndStyle.RoundLeftFlatRight,
                onClick = {
                    val next = (exposureBiasEv - 0.5f).coerceIn(-2.0f, 1.0f)
                    onSelectEv(next)
                },
                modifier = Modifier
                    .width(34.dp)
                    .fillMaxHeight()
            )

            steps.forEach { stepEv ->
                val isSelected = abs(exposureBiasEv - stepEv) < 0.15f
                val isZeroRef = abs(stepEv) < 0.05f
                val isLit = if (exposureBiasEv >= -0.05f) {
                    stepEv in -0.05f..(exposureBiasEv + 0.05f)
                } else {
                    stepEv in (exposureBiasEv - 0.05f)..0.05f
                }

                val segColor = when {
                    isSelected -> LcarsColors.GoldenAmber
                    isZeroRef -> LcarsColors.IceBlue
                    isLit -> LcarsColors.Lavender
                    else -> LcarsColors.MutedSlate
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
                    textColor = if (segColor == LcarsColors.MutedSlate) {
                        LcarsColors.ApricotPeach
                    } else {
                        LcarsColors.CanvasBlack
                    },
                    endStyle = LcarsPillEndStyle.FlatBlock,
                    onClick = { onSelectEv(stepEv) },
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                )
            }

            LcarsPillButton(
                label = "+",
                active = exposureBiasEv < 0.95f,
                activeColor = LcarsColors.WarmSand,
                endStyle = LcarsPillEndStyle.FlatLeftRoundRight,
                onClick = {
                    val next = (exposureBiasEv + 0.5f).coerceIn(-2.0f, 1.0f)
                    onSelectEv(next)
                },
                modifier = Modifier
                    .width(34.dp)
                    .fillMaxHeight()
            )
        }
    }
}
