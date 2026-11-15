package com.gneissfind.app.ui

import android.content.Context
import android.graphics.Outline as AndroidOutline
import android.graphics.Paint
import android.graphics.Typeface
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import android.view.ViewOutlineProvider
import android.widget.FrameLayout
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalInspectionMode
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.gneissfind.app.inference.SyntheticTargetProfile
import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.DetectorBackend
import com.gneissfind.app.model.OperatingMode
import com.gneissfind.app.model.ScanSpeedState
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.SwathTileMode
import com.gneissfind.app.model.WorldPoint3D
import com.gneissfind.app.ui.lcars.LcarsBarHeight
import com.gneissfind.app.ui.lcars.LcarsElbow
import com.gneissfind.app.ui.lcars.LcarsElbowOrientation
import com.gneissfind.app.ui.lcars.LcarsGutter
import com.gneissfind.app.ui.lcars.LcarsHeaderSegment
import com.gneissfind.app.ui.lcars.LcarsPillButton
import com.gneissfind.app.ui.lcars.LcarsPillEndStyle
import com.gneissfind.app.ui.lcars.LcarsSegmentedEvBar
import com.gneissfind.app.ui.lcars.LcarsStemWidth
import com.gneissfind.app.ui.theme.GneissFindTheme
import com.gneissfind.app.ui.theme.LcarsColors
import com.gneissfind.app.ui.theme.LcarsTypography
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private val TopElbowArmWidth = 144.dp
private val TopElbowStemHeight = 50.dp
private val BottomElbowArmWidth = 144.dp
private val BottomElbowStemHeight = 48.dp
private val PitchGaugeWidth = 40.dp

private val ScanSpeedState.isOverspeed: Boolean
    get() = this is ScanSpeedState.SpeedWarning

private val ScanSpeedState.speedMps: Float
    get() = when (this) {
        is ScanSpeedState.Normal -> currentSpeedMps
        is ScanSpeedState.SpeedWarning -> currentSpeedMps
    }

/**
 * Exact optical mapping from `1920x1080` landscape sensor coordinates into the panoramic
 * LCARS viewport slot so `cropY in [0..696]` moves smoothly while keeping the entire `384 px`
 * push-broom strip visible and 1:1 pixel-registered with [PreviewView].
 */
@Immutable
data class PanoramicViewportTransform(
    val scale: Float,
    val frameLeftPx: Float,
    val frameTopPx: Float,
    val renderedFrameWidthPx: Float,
    val renderedFrameHeightPx: Float,
    val stripLeftPx: Float,
    val stripTopPx: Float,
    val stripWidthPx: Float,
    val stripHeightPx: Float
)

fun computePanoramicViewportTransform(
    viewportWidthPx: Float,
    viewportHeightPx: Float,
    cropY: Int,
    swathTileMode: SwathTileMode
): PanoramicViewportTransform = computePanoramicViewportTransform(
    viewportWidthPx = viewportWidthPx,
    viewportHeightPx = viewportHeightPx,
    cropYFloat = cropY.toFloat(),
    swathTileMode = swathTileMode
)

/**
 * Computes continuous sub-pixel optical mapping from `1920x1080` landscape sensor coordinates into
 * the panoramic LCARS viewport slot so 120 Hz GPU `PreviewView.translationY` and the Compose strip
 * overlay stay registered without integer quantization jumps.
 */
fun computePanoramicViewportTransform(
    viewportWidthPx: Float,
    viewportHeightPx: Float,
    cropYFloat: Float,
    swathTileMode: SwathTileMode
): PanoramicViewportTransform {
    val vw = max(1f, viewportWidthPx)
    val vh = max(1f, viewportHeightPx)
    val maxCropY = StripRoiGeometry.MAX_CROP_Y_PX.toFloat()
    val clampedCropY = cropYFloat.coerceIn(0f, maxCropY)

    // Always fill 100% of the panoramic horizontal viewport width (zero side letterbox bars)
    // unless the container is taller than 16:9 (e.g. square desktop window), in which case fit height.
    val scale = if ((vw * StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX / StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX) >= vh) {
        vw / StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX.toFloat()
    } else {
        min(
            vw / StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX.toFloat(),
            vh / StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX.toFloat()
        )
    }
    val renderedW = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX * scale
    val renderedH = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX * scale
    val stripH = StripRoiGeometry.STRIP_HEIGHT_PX * scale
    val frameLeft = (vw - renderedW) * 0.5f

    val frameTop = if (renderedH <= vh) {
        (vh - renderedH) * 0.5f
    } else {
        val travelFraction = clampedCropY / maxCropY
        val desiredStripTop = if (stripH <= vh) {
            travelFraction * (vh - stripH)
        } else {
            // Center the 384 px strip inside compact viewports while panning camera rows smoothly
            (vh - stripH) * 0.5f
        }
        (desiredStripTop - clampedCropY * scale).coerceIn(vh - renderedH, 0f)
    }

    val stripLeft = frameLeft + swathTileMode.canonicalStripLeftPx * scale
    val stripTop = frameTop + clampedCropY * scale
    val stripWidth = swathTileMode.stripWidthPx * scale

    return PanoramicViewportTransform(
        scale = scale,
        frameLeftPx = frameLeft,
        frameTopPx = frameTop,
        renderedFrameWidthPx = renderedW,
        renderedFrameHeightPx = renderedH,
        stripLeftPx = stripLeft,
        stripTopPx = stripTop,
        stripWidthPx = stripWidth,
        stripHeightPx = stripH
    )
}

/**
 * Hardware-clipped Android [FrameLayout] wrapper that confines CameraX [PreviewView]'s underlying
 * [android.view.TextureView] strictly inside the panoramic LCARS viewport bounds (`[0..vw] x [0..vh]`),
 * applying 120 Hz EIS pitch stabilization via GPU RenderNode `previewView.translationY` instead of
 * expensive per-frame View `requestLayout()` passes.
 */
private class ClippingCameraViewportContainer(context: Context) : FrameLayout(context) {
    val previewView: PreviewView = PreviewView(context).apply {
        scaleType = PreviewView.ScaleType.FILL_CENTER
        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        isSoundEffectsEnabled = false
    }

    private var currentCropYFloat: Float = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat()
    private var currentSwathMode: SwathTileMode = SwathTileMode.FIVE_TILE
    private var lastMeasuredViewportW: Int = -1
    private var lastMeasuredViewportH: Int = -1

    init {
        clipChildren = true
        clipToPadding = true
        clipToOutline = true
        isSoundEffectsEnabled = false
        val radiusPx = 8f * resources.displayMetrics.density
        outlineProvider = object : ViewOutlineProvider() {
            override fun getOutline(view: View, outline: AndroidOutline) {
                outline.setRoundRect(0, 0, view.width, view.height, radiusPx)
            }
        }
        addView(
            previewView,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
        )
    }

    fun updateStabilizationState(cropYFloat: Float, swathTileMode: SwathTileMode) {
        currentCropYFloat = cropYFloat
        currentSwathMode = swathTileMode
        if (width > 0 && height > 0) {
            val t = computePanoramicViewportTransform(
                viewportWidthPx = width.toFloat(),
                viewportHeightPx = height.toFloat(),
                cropYFloat = currentCropYFloat,
                swathTileMode = currentSwathMode
            )
            // Pure GPU RenderNode sub-pixel translation — zero measure/layout stalls.
            previewView.translationY = t.frameTopPx
        }
    }

    override fun onLayout(changed: Boolean, left: Int, top: Int, right: Int, bottom: Int) {
        val vwInt = (right - left).coerceAtLeast(1)
        val vhInt = (bottom - top).coerceAtLeast(1)
        val t = computePanoramicViewportTransform(
            viewportWidthPx = vwInt.toFloat(),
            viewportHeightPx = vhInt.toFloat(),
            cropYFloat = currentCropYFloat,
            swathTileMode = currentSwathMode
        )
        if (vwInt != lastMeasuredViewportW || vhInt != lastMeasuredViewportH) {
            lastMeasuredViewportW = vwInt
            lastMeasuredViewportH = vhInt
            val childLeft = t.frameLeftPx.roundToInt()
            val childW = t.renderedFrameWidthPx.roundToInt().coerceAtLeast(1)
            val childH = t.renderedFrameHeightPx.roundToInt().coerceAtLeast(1)
            previewView.measure(
                MeasureSpec.makeMeasureSpec(childW, MeasureSpec.EXACTLY),
                MeasureSpec.makeMeasureSpec(childH, MeasureSpec.EXACTLY)
            )
            previewView.layout(childLeft, 0, childLeft + childW, childH)
        }
        previewView.translationY = t.frameTopPx
    }
}

@Composable
fun ScannerScreen(
    uiState: ScannerUiState,
    hasCameraPermission: Boolean,
    onRequestCameraPermission: () -> Unit,
    onBindPreview: (LifecycleOwner, PreviewView) -> Unit,
    onSelectMode: (OperatingMode) -> Unit,
    onSelectBackend: (DetectorBackend) -> Unit,
    onCycleScenario: () -> Unit,
    onToggleSwathTileMode: () -> Unit,
    onToggleEisLock: (Boolean) -> Unit,
    onSyncPitch: () -> Unit,
    onToggleStoneClass: (StoneClass) -> Unit,
    onToggleShutterClamp: (Boolean) -> Unit,
    onToggleTwoXCrop: (Boolean) -> Unit,
    onChangeExposureBias: (Float) -> Unit,
    onToggleAudioMute: (Boolean) -> Unit,
    onToggleOverspeedTest: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(LcarsColors.CanvasBlack)
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(horizontal = 6.dp, vertical = 4.dp),
        verticalArrangement = Arrangement.spacedBy(LcarsGutter)
    ) {
        // Upper ~48% Height: Geological Viewport (TopLeft GoldenAmber LcarsElbow + Panoramic Camera)
        GeologicalViewportZone(
            uiState = uiState,
            hasCameraPermission = hasCameraPermission,
            onRequestCameraPermission = onRequestCameraPermission,
            onBindPreview = onBindPreview,
            onSelectMode = onSelectMode,
            onSyncPitch = onSyncPitch,
            onToggleAudioMute = onToggleAudioMute,
            modifier = Modifier
                .fillMaxWidth()
                .weight(0.50f)
        )

        // Lower ~50% Height: Tactical Control Deck (BottomLeft Lavender LcarsElbow + 2-Column Matrix)
        TacticalControlDeckZone(
            uiState = uiState,
            onSelectMode = onSelectMode,
            onSelectBackend = onSelectBackend,
            onCycleScenario = onCycleScenario,
            onToggleSwathTileMode = onToggleSwathTileMode,
            onToggleEisLock = onToggleEisLock,
            onSyncPitch = onSyncPitch,
            onToggleStoneClass = onToggleStoneClass,
            onToggleShutterClamp = onToggleShutterClamp,
            onToggleTwoXCrop = onToggleTwoXCrop,
            onChangeExposureBias = onChangeExposureBias,
            onToggleOverspeedTest = onToggleOverspeedTest,
            modifier = Modifier
                .fillMaxWidth()
                .weight(0.50f)
        )
    }
}

@Composable
private fun GeologicalViewportZone(
    uiState: ScannerUiState,
    hasCameraPermission: Boolean,
    onRequestCameraPermission: () -> Unit,
    onBindPreview: (LifecycleOwner, PreviewView) -> Unit,
    onSelectMode: (OperatingMode) -> Unit,
    onSyncPitch: () -> Unit,
    onToggleAudioMute: (Boolean) -> Unit,
    modifier: Modifier = Modifier
) {
    val isLogging = uiState.operatingMode == OperatingMode.LOG_DATA
    val rotCode = if (uiState.displaySurfaceRotation == Surface.ROTATION_270) "R-270" else "R-090"
    val crookStart = LcarsStemWidth + LcarsGutter
    val crookTop = LcarsBarHeight + LcarsGutter

    Box(modifier = modifier.background(LcarsColors.CanvasBlack)) {
        // Top-Left Swept LCARS Elbow ("GEO-47")
        LcarsElbow(
            orientation = LcarsElbowOrientation.TopLeft,
            color = if (isLogging) LcarsColors.MarsRed else LcarsColors.GoldenAmber,
            armLabel = "GEO-47",
            stemCode = rotCode,
            modifier = Modifier
                .align(Alignment.TopStart)
                .width(TopElbowArmWidth)
                .height(TopElbowStemHeight)
        )

        // Top 22.dp Segmented Telemetry Header Rail (Tabular Numerals)
        Row(
            modifier = Modifier
                .align(Alignment.TopStart)
                .fillMaxWidth()
                .padding(start = TopElbowArmWidth + LcarsGutter)
                .height(LcarsBarHeight),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            val pitchColor = when {
                !uiState.eisPitchLockEnabled -> LcarsColors.MutedSlate
                uiState.isPitchClamped -> LcarsColors.MarsRed
                else -> LcarsColors.Periwinkle
            }
            val pitchText = if (!uiState.eisPitchLockEnabled) {
                String.format(Locale.US, "PITCH %02.0f° [EIS OFF]", uiState.imuPitchDeg)
            } else if (uiState.isPitchClamped) {
                String.format(Locale.US, "PITCH LIMIT %02.0f°", uiState.imuPitchDeg)
            } else {
                String.format(
                    Locale.US,
                    "PITCH %02.0f° [%02.0f°]",
                    uiState.imuPitchDeg,
                    uiState.referencePitchDeg
                )
            }
            LcarsHeaderSegment(
                text = pitchText,
                color = pitchColor,
                textColor = when {
                    pitchColor == LcarsColors.MutedSlate -> LcarsColors.ApricotPeach
                    uiState.isPitchClamped -> LcarsColors.AlertWhite
                    else -> LcarsColors.CanvasBlack
                },
                onClick = onSyncPitch
            )

            LcarsHeaderSegment(
                text = String.format(
                    Locale.US,
                    "R %.2fM • %.2f MM/PX",
                    uiState.slantRangeMeters,
                    uiState.gsdMmPerPx
                ),
                color = LcarsColors.ApricotPeach,
                modifier = Modifier.weight(1f)
            )

            val isOverspeed = uiState.speedState.isOverspeed
            val velText = if (isOverspeed) {
                String.format(Locale.US, "SLOW SWEEP %.2f M/S", uiState.speedState.speedMps)
            } else {
                String.format(Locale.US, "VEL %.2f M/S", uiState.speedState.speedMps)
            }
            LcarsHeaderSegment(
                text = velText,
                color = if (isOverspeed) LcarsColors.MarsRed else LcarsColors.IceBlue,
                textColor = if (isOverspeed) LcarsColors.AlertWhite else LcarsColors.CanvasBlack
            )

            val shutterLabel = formatShutterFraction(uiState.exposureTimeNs)
            LcarsHeaderSegment(
                text = String.format(
                    Locale.US,
                    "%.1fX • %s • ISO %d",
                    uiState.activeZoomRatio,
                    shutterLabel,
                    uiState.sensorIso
                ),
                color = LcarsColors.Lavender,
                endStyle = LcarsPillEndStyle.FlatLeftRoundRight,
                alignEnd = true
            )
        }

        // Left 68.dp Stacked Structural Status Stem
        Column(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(top = TopElbowStemHeight + LcarsGutter)
                .width(LcarsStemWidth)
                .fillMaxHeight(),
            verticalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            LcarsPillButton(
                label = if (isLogging) "REC" else "SCAN",
                subLabel = if (isLogging) {
                    String.format(Locale.US, "%04d", uiState.loggedFrameCount)
                } else {
                    "${uiState.swathTileMode.tileCount}-TILE"
                },
                prefixCode = "01",
                active = true,
                activeColor = if (isLogging) LcarsColors.MarsRed else LcarsColors.IceBlue,
                endStyle = LcarsPillEndStyle.FlatBlock,
                compactStemMode = true,
                onClick = {
                    onSelectMode(if (isLogging) OperatingMode.SCAN else OperatingMode.LOG_DATA)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            )

            LcarsPillButton(
                label = String.format(Locale.US, "%.1fHZ", uiState.sweepHz),
                subLabel = String.format(Locale.US, "%.1fMS", uiState.rollingLatencyMs),
                prefixCode = "FPS",
                active = true,
                activeColor = LcarsColors.ApricotPeach,
                endStyle = LcarsPillEndStyle.FlatBlock,
                compactStemMode = true,
                onClick = onSyncPitch,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            )

            LcarsPillButton(
                label = if (uiState.isAudioMuted) "MUTE" else "AUDIO",
                subLabel = if (!uiState.isAudioMuted && uiState.audioPitchHz > 0f) {
                    String.format(Locale.US, "%.0fHZ", uiState.audioPitchHz)
                } else if (uiState.isAudioMuted) {
                    "OFF"
                } else {
                    "ARMED"
                },
                prefixCode = "SND",
                active = !uiState.isAudioMuted,
                activeColor = LcarsColors.WarmSand,
                endStyle = LcarsPillEndStyle.FlatBlock,
                compactStemMode = true,
                onClick = { onToggleAudioMute(!uiState.isAudioMuted) },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            )
        }

        // Inner Crook: Panoramic Camera Viewport + Right-Edge Pitch-Stabilization Travel Gauge
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(start = crookStart, top = crookTop),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            PanoramicCameraViewport(
                uiState = uiState,
                hasCameraPermission = hasCameraPermission,
                onRequestCameraPermission = onRequestCameraPermission,
                onBindPreview = onBindPreview,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .clip(
                        RoundedCornerShape(
                            topStart = 10.dp,
                            topEnd = 4.dp,
                            bottomStart = 4.dp,
                            bottomEnd = 4.dp
                        )
                    )
                    .background(LcarsColors.CanvasBlack)
            )

            PitchTravelGaugeColumn(
                cropY = uiState.cropY,
                eisEnabled = uiState.eisPitchLockEnabled,
                isPitchClamped = uiState.isPitchClamped,
                onSyncPitch = onSyncPitch,
                modifier = Modifier
                    .width(PitchGaugeWidth)
                    .fillMaxHeight()
            )
        }
    }
}

@Composable
private fun PanoramicCameraViewport(
    uiState: ScannerUiState,
    hasCameraPermission: Boolean,
    onRequestCameraPermission: () -> Unit,
    onBindPreview: (LifecycleOwner, PreviewView) -> Unit,
    modifier: Modifier = Modifier
) {
    val isInspection = LocalInspectionMode.current
    val lifecycleOwner = LocalLifecycleOwner.current

    Box(modifier = modifier) {
        if (hasCameraPermission && !isInspection) {
            AndroidView(
                factory = { ctx ->
                    ClippingCameraViewportContainer(ctx).apply {
                        layoutParams = ViewGroup.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.MATCH_PARENT
                        )
                        updateStabilizationState(uiState.displayCropYFloat, uiState.swathTileMode)
                        onBindPreview(lifecycleOwner, previewView)
                    }
                },
                update = { container ->
                    container.updateStabilizationState(uiState.displayCropYFloat, uiState.swathTileMode)
                },
                modifier = Modifier.fillMaxSize()
            )
        } else {
            // Offline / Preview / Pre-permission dark gravel diagnostic substrate
            Canvas(modifier = Modifier.fillMaxSize()) {
                drawRect(color = Color(0xFF111317))
                val gridStep = 24.dp.toPx()
                var gx = gridStep
                while (gx < size.width) {
                    drawLine(
                        color = Color(0xFF1D222B),
                        start = Offset(gx, 0f),
                        end = Offset(gx, size.height),
                        strokeWidth = 1f
                    )
                    gx += gridStep
                }
                var gy = gridStep
                while (gy < size.height) {
                    drawLine(
                        color = Color(0xFF1D222B),
                        start = Offset(0f, gy),
                        end = Offset(size.width, gy),
                        strokeWidth = 1f
                    )
                    gy += gridStep
                }
            }
        }

        // Stabilized Push-Broom Strip Brackets, Tile Seam Ticks & High-Contrast Stone Reticles
        PushBroomTripwireOverlay(
            uiState = uiState,
            modifier = Modifier.fillMaxSize()
        )

        if (!hasCameraPermission && !isInspection) {
            Box(
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(horizontal = 24.dp)
            ) {
                LcarsPillButton(
                    label = "AUTHORIZE OPTICAL SENSOR",
                    subLabel = "GRANT CAMERA FOR 1920X1080 SWATH",
                    prefixCode = "CAM",
                    active = true,
                    activeColor = LcarsColors.GoldenAmber,
                    onClick = onRequestCameraPermission,
                    modifier = Modifier
                        .width(290.dp)
                        .height(40.dp)
                )
            }
        }
    }
}

@Composable
private fun PushBroomTripwireOverlay(
    uiState: ScannerUiState,
    modifier: Modifier = Modifier
) {
    val labelPaint = remember {
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = 24f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        }
    }
    val hudPaint = remember {
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = 21f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        }
    }

    Canvas(modifier = modifier) {
        val transform = computePanoramicViewportTransform(
            viewportWidthPx = size.width,
            viewportHeightPx = size.height,
            cropYFloat = uiState.displayCropYFloat,
            swathTileMode = uiState.swathTileMode
        )

        val stripLeft = transform.stripLeftPx
        val stripTop = transform.stripTopPx
        val stripW = transform.stripWidthPx
        val stripH = transform.stripHeightPx
        val stripRight = stripLeft + stripW
        val stripBottom = stripTop + stripH

        // Subtle dark veil outside the active 384 px push-broom strip
        val veilColor = Color(0x88000000)
        if (stripTop > 0f) {
            drawRect(
                color = veilColor,
                topLeft = Offset(0f, 0f),
                size = Size(size.width, stripTop)
            )
        }
        if (stripBottom < size.height) {
            drawRect(
                color = veilColor,
                topLeft = Offset(0f, stripBottom),
                size = Size(size.width, size.height - stripBottom)
            )
        }
        if (stripLeft > 0f) {
            drawRect(
                color = veilColor,
                topLeft = Offset(0f, max(0f, stripTop)),
                size = Size(stripLeft, min(size.height, stripBottom) - max(0f, stripTop))
            )
        }
        if (stripRight < size.width) {
            drawRect(
                color = veilColor,
                topLeft = Offset(stripRight, max(0f, stripTop)),
                size = Size(size.width - stripRight, min(size.height, stripBottom) - max(0f, stripTop))
            )
        }

        val railColor = when {
            uiState.speedState.isOverspeed -> LcarsColors.MarsRed
            uiState.isPitchClamped -> LcarsColors.MarsRed
            uiState.operatingMode == OperatingMode.LOG_DATA -> LcarsColors.GoldenAmber
            else -> LcarsColors.Periwinkle
        }

        // Top and bottom push-broom tripwire rails (clamped inside visible viewport when stripH >= vh)
        val keylinePx = 2.dp.toPx()
        val railStrokePx = 2.dp.toPx()
        val visibleStripTop = stripTop.coerceIn(railStrokePx, max(railStrokePx, size.height - railStrokePx))
        val visibleStripBottom = stripBottom.coerceIn(railStrokePx, max(railStrokePx, size.height - railStrokePx))
        drawLine(
            color = LcarsColors.CanvasBlack,
            start = Offset(stripLeft, visibleStripTop),
            end = Offset(stripRight, visibleStripTop),
            strokeWidth = railStrokePx + 2f * keylinePx
        )
        drawLine(
            color = LcarsColors.CanvasBlack,
            start = Offset(stripLeft, visibleStripBottom),
            end = Offset(stripRight, visibleStripBottom),
            strokeWidth = railStrokePx + 2f * keylinePx
        )
        drawLine(
            color = railColor,
            start = Offset(stripLeft, visibleStripTop),
            end = Offset(stripRight, visibleStripTop),
            strokeWidth = railStrokePx
        )
        drawLine(
            color = railColor,
            start = Offset(stripLeft, visibleStripBottom),
            end = Offset(stripRight, visibleStripBottom),
            strokeWidth = railStrokePx
        )

        // Left & right vertical strip bracket end-caps
        val endBracketH = min(22.dp.toPx(), (visibleStripBottom - visibleStripTop) * 0.35f)
        listOf(stripLeft, stripRight).forEach { edgeX ->
            drawLine(
                color = LcarsColors.CanvasBlack,
                start = Offset(edgeX, visibleStripTop),
                end = Offset(edgeX, visibleStripBottom),
                strokeWidth = railStrokePx + 2f * keylinePx
            )
            drawLine(
                color = LcarsColors.IceBlue,
                start = Offset(edgeX, visibleStripTop),
                end = Offset(edgeX, visibleStripTop + endBracketH),
                strokeWidth = railStrokePx
            )
            drawLine(
                color = LcarsColors.IceBlue,
                start = Offset(edgeX, visibleStripBottom - endBracketH),
                end = Offset(edgeX, visibleStripBottom),
                strokeWidth = railStrokePx
            )
        }

        // Subtle 1.dp vertical tile seam ticks (x = [64, 416, 768, 1120, 1472, 1856] in 5-tile)
        val seamTickH = 10.dp.toPx()
        val tileOffsets = uiState.swathTileMode.frameXOffsets
        val seamDash = PathEffect.dashPathEffect(floatArrayOf(6f, 8f), 0f)
        tileOffsets.forEachIndexed { idx, tileX ->
            if (idx > 0) {
                val seamStartX = transform.frameLeftPx + tileX * transform.scale
                val seamEndX = transform.frameLeftPx + (tileX + StripRoiGeometry.TILE_OVERLAP_PX) * transform.scale
                drawLine(
                    color = LcarsColors.Periwinkle.copy(alpha = 0.32f),
                    start = Offset(seamStartX, visibleStripTop),
                    end = Offset(seamStartX, visibleStripBottom),
                    strokeWidth = 1.dp.toPx(),
                    pathEffect = seamDash
                )
                listOf(seamStartX, seamEndX).forEach { tickX ->
                    drawLine(
                        color = LcarsColors.IceBlue.copy(alpha = 0.85f),
                        start = Offset(tickX, visibleStripTop),
                        end = Offset(tickX, visibleStripTop + seamTickH),
                        strokeWidth = 1.dp.toPx()
                    )
                    drawLine(
                        color = LcarsColors.IceBlue.copy(alpha = 0.85f),
                        start = Offset(tickX, visibleStripBottom - seamTickH),
                        end = Offset(tickX, visibleStripBottom),
                        strokeWidth = 1.dp.toPx()
                    )
                }
            }
        }

        // Compact strip descriptor tag pinned inside top-left of strip
        val stripTag = String.format(
            Locale.US,
            "%d-TILE • %dX%d • Y=%03d",
            uiState.swathTileMode.tileCount,
            uiState.swathTileMode.stripWidthPx,
            StripRoiGeometry.STRIP_HEIGHT_PX,
            uiState.cropY
        )
        hudPaint.textSize = 10.dp.toPx()
        val tagW = hudPaint.measureText(stripTag) + 12.dp.toPx()
        val tagH = 15.dp.toPx()
        val tagTop = (visibleStripTop + 3.dp.toPx()).coerceAtMost(size.height - tagH)
        drawRect(
            color = LcarsColors.CanvasBlack,
            topLeft = Offset(stripLeft + 4.dp.toPx(), tagTop),
            size = Size(tagW, tagH)
        )
        hudPaint.color = LcarsColors.IceBlue.toArgb()
        drawContext.canvas.nativeCanvas.drawText(
            stripTag,
            stripLeft + 10.dp.toPx(),
            tagTop + tagH - 3.5.dp.toPx(),
            hudPaint
        )

        // Confirmed Target Reticles (SCAN mode only; hard_confuser strictly muted)
        if (uiState.operatingMode == OperatingMode.SCAN) {
            labelPaint.textSize = 11.dp.toPx()
            uiState.confirmedTargets.forEach { target ->
                if (!target.stoneClass.isUserAlertable) return@forEach
                if (!uiState.enabledClasses.contains(target.stoneClass)) return@forEach

                val left = stripLeft + target.stripLeftPx * transform.scale
                val top = stripTop + target.stripTopPx * transform.scale
                val right = stripLeft + target.stripRightPx * transform.scale
                val bottom = stripTop + target.stripBottomPx * transform.scale

                val reticleColor = Color(target.stoneClass.primaryHexColor)
                drawHighContrastCornerBrackets(
                    left = left,
                    top = top,
                    right = right,
                    bottom = bottom,
                    foregroundColor = reticleColor
                )

                val callout = String.format(
                    Locale.US,
                    "%s %s %.0fMM [%d/%d]",
                    target.stoneClass.codePrefix,
                    target.stoneClass.lcarsShortName.substringBefore(" /"),
                    target.estimatedDiameterMm,
                    target.hitCountInWindow,
                    target.windowSize
                )
                val calloutW = labelPaint.measureText(callout) + 10.dp.toPx()
                val calloutH = 16.dp.toPx()
                val calloutLeft = left.coerceIn(
                    stripLeft + 2f,
                    max(stripLeft + 2f, stripRight - calloutW - 2f)
                )
                val calloutTop = if (top - calloutH - 3.dp.toPx() >= 0f) {
                    top - calloutH - 3.dp.toPx()
                } else {
                    min(size.height - calloutH, bottom + 3.dp.toPx())
                }

                drawRect(
                    color = LcarsColors.CanvasBlack,
                    topLeft = Offset(calloutLeft, calloutTop),
                    size = Size(calloutW, calloutH)
                )
                labelPaint.color = reticleColor.toArgb()
                drawContext.canvas.nativeCanvas.drawText(
                    callout,
                    calloutLeft + 5.dp.toPx(),
                    calloutTop + calloutH - 3.5.dp.toPx(),
                    labelPaint
                )
            }
        }

        // MarsRed Overspeed Banner (> 1.10 m/s)
        if (uiState.speedState.isOverspeed) {
            val alertText = String.format(
                Locale.US,
                "SLOW SWEEP — %.2f M/S EXCEEDS 1.10 M/S GATE",
                uiState.speedState.speedMps
            )
            labelPaint.textSize = 12.dp.toPx()
            val bannerW = labelPaint.measureText(alertText) + 24.dp.toPx()
            val bannerH = 22.dp.toPx()
            val bannerLeft = ((size.width - bannerW) * 0.5f).coerceAtLeast(4f)
            val bannerTop = 4.dp.toPx()
            drawRect(
                color = LcarsColors.CanvasBlack,
                topLeft = Offset(bannerLeft - 2.dp.toPx(), bannerTop - 2.dp.toPx()),
                size = Size(bannerW + 4.dp.toPx(), bannerH + 4.dp.toPx())
            )
            drawRect(
                color = LcarsColors.MarsRed,
                topLeft = Offset(bannerLeft, bannerTop),
                size = Size(bannerW, bannerH)
            )
            labelPaint.color = LcarsColors.AlertWhite.toArgb()
            drawContext.canvas.nativeCanvas.drawText(
                alertText,
                bannerLeft + 12.dp.toPx(),
                bannerTop + bannerH - 5.dp.toPx(),
                labelPaint
            )
        }
    }
}

/**
 * Draws anti-camouflage corner reticle brackets backed by a crisp `2.dp` `#000000` outer keyline halo.
 */
private fun DrawScope.drawHighContrastCornerBrackets(
    left: Float,
    top: Float,
    right: Float,
    bottom: Float,
    foregroundColor: Color
) {
    val boxW = max(12.dp.toPx(), right - left)
    val boxH = max(12.dp.toPx(), bottom - top)
    val cx = (left + right) * 0.5f
    val cy = (top + bottom) * 0.5f
    val l = cx - boxW * 0.5f
    val r = cx + boxW * 0.5f
    val t = cy - boxH * 0.5f
    val b = cy + boxH * 0.5f

    val arm = min(14.dp.toPx(), min(boxW, boxH) * 0.36f).coerceAtLeast(5.dp.toPx())
    val innerStroke = 2.2.dp.toPx()
    val haloStroke = innerStroke + 4.dp.toPx() // 2.dp #000000 outer keyline halo on each side

    val segments = arrayOf(
        Offset(l, t) to Offset(l + arm, t),
        Offset(l, t) to Offset(l, t + arm),
        Offset(r, t) to Offset(r - arm, t),
        Offset(r, t) to Offset(r, t + arm),
        Offset(l, b) to Offset(l + arm, b),
        Offset(l, b) to Offset(l, b - arm),
        Offset(r, b) to Offset(r - arm, b),
        Offset(r, b) to Offset(r, b - arm)
    )

    segments.forEach { (start, end) ->
        drawLine(
            color = LcarsColors.CanvasBlack,
            start = start,
            end = end,
            strokeWidth = haloStroke,
            cap = StrokeCap.Square
        )
    }

    segments.forEach { (start, end) ->
        drawLine(
            color = foregroundColor,
            start = start,
            end = end,
            strokeWidth = innerStroke,
            cap = StrokeCap.Square
        )
    }
}

@Composable
private fun PitchTravelGaugeColumn(
    cropY: Int,
    eisEnabled: Boolean,
    isPitchClamped: Boolean,
    onSyncPitch: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(LcarsGutter)
    ) {
        LcarsPillButton(
            label = if (isPitchClamped) "LIM" else String.format(Locale.US, "%03d", cropY),
            subLabel = "Y-PX",
            active = eisEnabled,
            activeColor = if (isPitchClamped) LcarsColors.MarsRed else LcarsColors.Periwinkle,
            endStyle = LcarsPillEndStyle.FlatBlock,
            compactStemMode = true,
            onClick = onSyncPitch,
            modifier = Modifier
                .fillMaxWidth()
                .height(34.dp)
        )

        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .background(LcarsColors.CanvasBlack)
        ) {
            val w = size.width
            val h = size.height
            drawRect(
                color = LcarsColors.MutedSlate,
                topLeft = Offset(0f, 0f),
                size = Size(w, h)
            )
            val inset = 3.dp.toPx()
            drawRect(
                color = LcarsColors.CanvasBlack,
                topLeft = Offset(inset, inset),
                size = Size(max(1f, w - 2f * inset), max(1f, h - 2f * inset))
            )

            // Nominal 348 center reference tick
            val midY = h * 0.5f
            drawLine(
                color = LcarsColors.ApricotPeach,
                start = Offset(0f, midY),
                end = Offset(w, midY),
                strokeWidth = 2.dp.toPx()
            )

            // Active 384/1080 strip window indicator inside [0..696]
            val windowFrac = StripRoiGeometry.STRIP_HEIGHT_PX.toFloat() /
                StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX.toFloat()
            val thumbH = (h * windowFrac).coerceAtLeast(10.dp.toPx())
            val travelFrac = cropY.coerceIn(0, StripRoiGeometry.MAX_CROP_Y_PX).toFloat() /
                StripRoiGeometry.MAX_CROP_Y_PX.toFloat()
            val thumbTop = travelFrac * max(0f, h - thumbH)

            val thumbColor = when {
                !eisEnabled -> LcarsColors.MutedSlate
                isPitchClamped -> LcarsColors.MarsRed
                else -> LcarsColors.Periwinkle
            }
            drawRect(
                color = thumbColor,
                topLeft = Offset(inset + 2.dp.toPx(), thumbTop),
                size = Size(max(2f, w - 2f * (inset + 2.dp.toPx())), thumbH)
            )
        }
    }
}

@Composable
private fun TacticalControlDeckZone(
    uiState: ScannerUiState,
    onSelectMode: (OperatingMode) -> Unit,
    onSelectBackend: (DetectorBackend) -> Unit,
    onCycleScenario: () -> Unit,
    onToggleSwathTileMode: () -> Unit,
    onToggleEisLock: (Boolean) -> Unit,
    onSyncPitch: () -> Unit,
    onToggleStoneClass: (StoneClass) -> Unit,
    onToggleShutterClamp: (Boolean) -> Unit,
    onToggleTwoXCrop: (Boolean) -> Unit,
    onChangeExposureBias: (Float) -> Unit,
    onToggleOverspeedTest: () -> Unit,
    modifier: Modifier = Modifier
) {
    val isScanning = uiState.operatingMode == OperatingMode.SCAN
    val isLogging = uiState.operatingMode == OperatingMode.LOG_DATA
    val isLiveGpu = uiState.detectorBackend == DetectorBackend.RF_DETR_NANO_GPU
    val crookStart = LcarsStemWidth + LcarsGutter
    val crookBottom = LcarsBarHeight + LcarsGutter

    Box(modifier = modifier.background(LcarsColors.CanvasBlack)) {
        // Left 68.dp Primary Mode Stem Above Bottom-Left Elbow
        Column(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(bottom = BottomElbowStemHeight + LcarsGutter)
                .width(LcarsStemWidth)
                .fillMaxHeight(),
            verticalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            LcarsPillButton(
                label = "SCAN",
                subLabel = if (isScanning) "ACTIVE" else "READY",
                prefixCode = "M-01",
                active = isScanning,
                activeColor = LcarsColors.GoldenAmber,
                endStyle = LcarsPillEndStyle.FlatBlock,
                compactStemMode = true,
                onClick = { onSelectMode(OperatingMode.SCAN) },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            )

            LcarsPillButton(
                label = "LOG",
                subLabel = if (isLogging) {
                    String.format(Locale.US, "%04d PNG", uiState.loggedFrameCount)
                } else {
                    "4.0 HZ"
                },
                prefixCode = "M-02",
                active = isLogging,
                activeColor = LcarsColors.MarsRed,
                endStyle = LcarsPillEndStyle.FlatBlock,
                compactStemMode = true,
                onClick = {
                    onSelectMode(if (isLogging) OperatingMode.SCAN else OperatingMode.LOG_DATA)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            )

            LcarsPillButton(
                label = if (isLiveGpu) "GPU" else "SIM",
                subLabel = if (isLiveGpu) "FP16" else "DEMO",
                prefixCode = "M-03",
                active = true,
                activeColor = if (isLiveGpu) LcarsColors.IceBlue else LcarsColors.WarmSand,
                endStyle = LcarsPillEndStyle.FlatBlock,
                compactStemMode = true,
                onClick = {
                    onSelectBackend(
                        if (isLiveGpu) {
                            DetectorBackend.SYNTHETIC_DEMO
                        } else {
                            DetectorBackend.RF_DETR_NANO_GPU
                        }
                    )
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            )
        }

        // Bottom-Left Swept LCARS Elbow ("SYS-01")
        LcarsElbow(
            orientation = LcarsElbowOrientation.BottomLeft,
            color = LcarsColors.Lavender,
            armLabel = "SYS-01",
            stemCode = "M1B",
            modifier = Modifier
                .align(Alignment.BottomStart)
                .width(BottomElbowArmWidth)
                .height(BottomElbowStemHeight)
        )

        // Bottom 22.dp Segmented Telemetry Footer Rail
        Row(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .fillMaxWidth()
                .padding(start = BottomElbowArmWidth + LcarsGutter)
                .height(LcarsBarHeight),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            LcarsHeaderSegment(
                text = "47-${uiState.swathTileMode.stripWidthPx}",
                color = LcarsColors.Periwinkle
            )

            LcarsHeaderSegment(
                text = String.format(
                    Locale.US,
                    "SWEEP %.1f MS (G1 %.1f | TOPK %.1f | G2 %.1f)",
                    uiState.rollingLatencyMs,
                    uiState.part1LatencyMs,
                    uiState.bridgeLatencyMs,
                    uiState.part2LatencyMs
                ),
                color = LcarsColors.ApricotPeach,
                modifier = Modifier.weight(1f)
            )

            val audioTelemetry = when {
                uiState.isAudioMuted -> "AUDIO MUTED"
                uiState.audioPitchHz > 0f -> String.format(
                    Locale.US,
                    "PING %.0f HZ @ %.1f/S",
                    uiState.audioPitchHz,
                    uiState.audioCadenceHz
                )
                else -> "SONIFIER ARMED"
            }
            LcarsHeaderSegment(
                text = audioTelemetry,
                color = if (uiState.audioPitchHz > 0f && !uiState.isAudioMuted) {
                    LcarsColors.GoldenAmber
                } else {
                    LcarsColors.WarmSand
                },
                endStyle = LcarsPillEndStyle.FlatLeftRoundRight,
                alignEnd = true
            )
        }

        // 2-Column Tactical Control Matrix Inside Bottom-Left Crook (Zero Modal Sheets)
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(start = crookStart, bottom = crookBottom),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter * 2)
        ) {
            OpticsAndEisColumn(
                uiState = uiState,
                onToggleTwoXCrop = onToggleTwoXCrop,
                onToggleShutterClamp = onToggleShutterClamp,
                onToggleEisLock = onToggleEisLock,
                onSyncPitch = onSyncPitch,
                onToggleSwathTileMode = onToggleSwathTileMode,
                onChangeExposureBias = onChangeExposureBias,
                modifier = Modifier
                    .weight(0.47f)
                    .fillMaxHeight()
            )

            ProspectMatrixColumn(
                uiState = uiState,
                onToggleStoneClass = onToggleStoneClass,
                onCycleScenario = onCycleScenario,
                onToggleOverspeedTest = onToggleOverspeedTest,
                modifier = Modifier
                    .weight(0.53f)
                    .fillMaxHeight()
            )
        }
    }
}

@Composable
private fun OpticsAndEisColumn(
    uiState: ScannerUiState,
    onToggleTwoXCrop: (Boolean) -> Unit,
    onToggleShutterClamp: (Boolean) -> Unit,
    onToggleEisLock: (Boolean) -> Unit,
    onSyncPitch: () -> Unit,
    onToggleSwathTileMode: () -> Unit,
    onChangeExposureBias: (Float) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(LcarsGutter)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "OPT-2X // OPTICS & EIS STABILIZATION",
                style = LcarsTypography.MicroCode,
                color = LcarsColors.Lavender
            )
            Text(
                text = if (uiState.isPitchClamped) "PITCH LIMIT" else "EIS-SYNC",
                style = LcarsTypography.MicroCode,
                color = if (uiState.isPitchClamped) LcarsColors.MarsRed else LcarsColors.Periwinkle
            )
        }

        // Row 1: 2.0X Sensor Crop [ON] + <= 1/500s Shutter Clamp [ON]
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            LcarsPillButton(
                label = if (uiState.twoXCropEnabled) "2.0X CROP [ON]" else "2.0X CROP [OFF]",
                subLabel = if (uiState.twoXCropEnabled) "LOSSLESS • 0.59MM" else "1.0X WIDE • 1.18MM",
                prefixCode = "OPT",
                active = uiState.twoXCropEnabled,
                activeColor = LcarsColors.Lavender,
                endStyle = LcarsPillEndStyle.FlatLeftRoundRight,
                onClick = { onToggleTwoXCrop(!uiState.twoXCropEnabled) },
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
            )

            LcarsPillButton(
                label = if (uiState.shutterClampEnabled) "1/500S [ON]" else "1/500S [OFF]",
                subLabel = if (uiState.shutterClampEnabled) "<= 2.0MS + ISO" else "AUTO SHUTTER",
                prefixCode = "EXP",
                active = uiState.shutterClampEnabled,
                activeColor = LcarsColors.Lavender,
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = { onToggleShutterClamp(!uiState.shutterClampEnabled) },
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
            )
        }

        // Row 2: EIS Pitch Lock [ON] + SYNC PITCH (45°) + Swath 5-TILE / 3-TILE
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            LcarsPillButton(
                label = if (uiState.eisPitchLockEnabled) {
                    if (uiState.isPitchClamped) "PITCH LIMIT" else "EIS LOCK [ON]"
                } else {
                    "EIS LOCK [OFF]"
                },
                subLabel = String.format(Locale.US, "CROP Y %03d", uiState.cropY),
                prefixCode = "EIS",
                active = uiState.eisPitchLockEnabled,
                activeColor = if (uiState.isPitchClamped) {
                    LcarsColors.MarsRed
                } else {
                    LcarsColors.Periwinkle
                },
                endStyle = LcarsPillEndStyle.FlatLeftRoundRight,
                onClick = { onToggleEisLock(!uiState.eisPitchLockEnabled) },
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
            )

            LcarsPillButton(
                label = String.format(Locale.US, "SYNC (%02.0f°)", uiState.referencePitchDeg),
                subLabel = String.format(Locale.US, "NOW %02.0f°", uiState.imuPitchDeg),
                prefixCode = "CAL",
                active = true,
                activeColor = LcarsColors.WarmSand,
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = onSyncPitch,
                modifier = Modifier
                    .weight(0.88f)
                    .fillMaxHeight()
            )

            LcarsPillButton(
                label = "${uiState.swathTileMode.tileCount}-TILE",
                subLabel = "${uiState.swathTileMode.stripWidthPx}PX",
                prefixCode = "SW",
                active = uiState.swathTileMode == SwathTileMode.FIVE_TILE,
                activeColor = LcarsColors.Periwinkle,
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = onToggleSwathTileMode,
                modifier = Modifier
                    .weight(0.84f)
                    .fillMaxHeight()
            )
        }

        // Row 3: Segmented LCARS Bar-Graph for Canopy Exposure Bias (-2.0 .. +1.0 EV)
        LcarsSegmentedEvBar(
            exposureBiasEv = uiState.exposureBiasEv,
            onSelectEv = onChangeExposureBias,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

@Composable
private fun ProspectMatrixColumn(
    uiState: ScannerUiState,
    onToggleStoneClass: (StoneClass) -> Unit,
    onCycleScenario: () -> Unit,
    onToggleOverspeedTest: () -> Unit,
    modifier: Modifier = Modifier
) {
    val classes = StoneClass.alertableClasses

    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(LcarsGutter)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "PROSPECT MATRIX // MIN 20MM TARGETS",
                style = LcarsTypography.MicroCode,
                color = LcarsColors.GoldenAmber
            )
            Text(
                text = "${uiState.enabledClasses.size}/4 ACTIVE • CONFUSERS MUTED",
                style = LcarsTypography.MicroCode,
                color = LcarsColors.IceBlue
            )
        }

        // 2x2 Grid of Hemispherical Mineral Target Toggle Pills
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .weight(2f),
            verticalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
            ) {
                MineralTargetPill(
                    stoneClass = classes[0],
                    pillFillColor = LcarsColors.GoldenAmber,
                    enabled = uiState.enabledClasses.contains(classes[0]),
                    onClick = { onToggleStoneClass(classes[0]) },
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                )
                MineralTargetPill(
                    stoneClass = classes[1],
                    pillFillColor = LcarsColors.ApricotPeach,
                    enabled = uiState.enabledClasses.contains(classes[1]),
                    onClick = { onToggleStoneClass(classes[1]) },
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                )
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
            ) {
                MineralTargetPill(
                    stoneClass = classes[2],
                    pillFillColor = LcarsColors.IceBlue,
                    enabled = uiState.enabledClasses.contains(classes[2]),
                    onClick = { onToggleStoneClass(classes[2]) },
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                )
                MineralTargetPill(
                    stoneClass = classes[3],
                    pillFillColor = LcarsColors.WarmSand,
                    enabled = uiState.enabledClasses.contains(classes[3]),
                    onClick = { onToggleStoneClass(classes[3]) },
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                )
            }
        }

        // Inline Scenario Cycler + Speed Gate Test Row (Zero Modal Sheets)
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .weight(0.92f),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            val scenarioTitle = if (uiState.detectorBackend == DetectorBackend.RF_DETR_NANO_GPU) {
                "LIVE CAM (GPU FP16)"
            } else {
                when (uiState.syntheticProfile) {
                    SyntheticTargetProfile.MIXED_GRAVEL_WALK -> "MIXED GRAVEL SIM"
                    SyntheticTargetProfile.PEBBLE_20MM_HIGH_PITCH -> "20MM AGATE (780HZ)"
                    SyntheticTargetProfile.COBBLE_55MM_LOW_PITCH -> "55MM JASPER (320HZ)"
                    SyntheticTargetProfile.QUIET_GRAVEL_CONFUSERS -> "CONFUSER MUTE SIM"
                }
            }
            LcarsPillButton(
                label = scenarioTitle,
                subLabel = "TAP TO CYCLE SOURCE",
                prefixCode = "SRC",
                active = true,
                activeColor = if (uiState.detectorBackend == DetectorBackend.RF_DETR_NANO_GPU) {
                    LcarsColors.IceBlue
                } else {
                    LcarsColors.WarmSand
                },
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = onCycleScenario,
                modifier = Modifier
                    .weight(1.45f)
                    .fillMaxHeight()
            )

            val simOverspeedActive = uiState.simulatedSpeedOverrideMps != null
            LcarsPillButton(
                label = if (simOverspeedActive) "OVERSPEED TEST" else "GATE <= 1.10M/S",
                subLabel = if (simOverspeedActive) "1.38 M/S SIM" else "LIVE VIO / IMU",
                prefixCode = "VEL",
                active = simOverspeedActive,
                activeColor = LcarsColors.MarsRed,
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = onToggleOverspeedTest,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
            )
        }
    }
}

@Composable
private fun MineralTargetPill(
    stoneClass: StoneClass,
    pillFillColor: Color,
    enabled: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val stateLabel = if (enabled) "ACTIVE" else "MUTED"
    LcarsPillButton(
        label = stoneClass.lcarsShortName,
        subLabel = "RETICLE • $stateLabel",
        prefixCode = stoneClass.codePrefix,
        swatchColor = Color(stoneClass.primaryHexColor),
        active = enabled,
        activeColor = pillFillColor,
        endStyle = LcarsPillEndStyle.Capsule,
        onClick = onClick,
        modifier = modifier
    )
}

private fun formatShutterFraction(exposureNs: Long): String {
    if (exposureNs <= 0L) return "1/500S"
    val denom = (1_000_000_000.0 / exposureNs.toDouble()).roundToInt().coerceAtLeast(1)
    return "1/${denom}S"
}

@Preview(
    name = "LCARS Geological Tricorder — Active 5-Tile Scan",
    device = "spec:width=915dp,height=411dp,dpi=420,orientation=landscape"
)
@Composable
private fun ScannerScreenActivePreview() {
    GneissFindTheme {
        ScannerScreen(
            uiState = ScannerUiState(
                sweepHz = 18.4f,
                rollingLatencyMs = 14.2f,
                part1LatencyMs = 8.9f,
                bridgeLatencyMs = 0.7f,
                part2LatencyMs = 4.6f,
                imuPitchDeg = 45f,
                referencePitchDeg = 45f,
                cropY = 348,
                speedState = ScanSpeedState.Normal(0.42f),
                sensorIso = 640,
                audioPitchHz = 780f,
                audioCadenceHz = 10.5f,
                confirmedTargets = listOf(
                    ConfirmedTarget(
                        trackId = 1L,
                        stoneClass = StoneClass.CHALCEDONY_AGATE,
                        worldPosition = WorldPoint3D(-0.18f, 0f, -1.05f),
                        stripLeftPx = 370f,
                        stripTopPx = 112f,
                        stripRightPx = 438f,
                        stripBottomPx = 180f,
                        smoothedConfidence = 0.89f,
                        estimatedDiameterMm = 26f,
                        slantRangeMeters = 1.50f,
                        hitCountInWindow = 4,
                        windowSize = 6,
                        lastUpdatedTimestampNs = 1_000_000L
                    ),
                    ConfirmedTarget(
                        trackId = 2L,
                        stoneClass = StoneClass.JASPER_RED_YELLOW,
                        worldPosition = WorldPoint3D(0.24f, 0f, -1.06f),
                        stripLeftPx = 1140f,
                        stripTopPx = 88f,
                        stripRightPx = 1265f,
                        stripBottomPx = 212f,
                        smoothedConfidence = 0.92f,
                        estimatedDiameterMm = 54f,
                        slantRangeMeters = 1.50f,
                        hitCountInWindow = 5,
                        windowSize = 6,
                        lastUpdatedTimestampNs = 1_000_000L
                    )
                )
            ),
            hasCameraPermission = true,
            onRequestCameraPermission = {},
            onBindPreview = { _, _ -> },
            onSelectMode = {},
            onSelectBackend = {},
            onCycleScenario = {},
            onToggleSwathTileMode = {},
            onToggleEisLock = {},
            onSyncPitch = {},
            onToggleStoneClass = {},
            onToggleShutterClamp = {},
            onToggleTwoXCrop = {},
            onChangeExposureBias = {},
            onToggleAudioMute = {},
            onToggleOverspeedTest = {}
        )
    }
}

@Preview(
    name = "LCARS Geological Tricorder — Overspeed & Pitch Limit Alert",
    device = "spec:width=915dp,height=411dp,dpi=420,orientation=landscape"
)
@Composable
private fun ScannerScreenAlertPreview() {
    GneissFindTheme {
        ScannerScreen(
            uiState = ScannerUiState(
                operatingMode = OperatingMode.LOG_DATA,
                loggedFrameCount = 128,
                imuPitchDeg = 54f,
                referencePitchDeg = 45f,
                cropY = 0,
                isPitchClamped = true,
                speedState = ScanSpeedState.SpeedWarning(1.38f),
                simulatedSpeedOverrideMps = 1.38f,
                exposureBiasEv = -0.5f,
                sensorIso = 1250
            ),
            hasCameraPermission = true,
            onRequestCameraPermission = {},
            onBindPreview = { _, _ -> },
            onSelectMode = {},
            onSelectBackend = {},
            onCycleScenario = {},
            onToggleSwathTileMode = {},
            onToggleEisLock = {},
            onSyncPitch = {},
            onToggleStoneClass = {},
            onToggleShutterClamp = {},
            onToggleTwoXCrop = {},
            onChangeExposureBias = {},
            onToggleAudioMute = {},
            onToggleOverspeedTest = {}
        )
    }
}
