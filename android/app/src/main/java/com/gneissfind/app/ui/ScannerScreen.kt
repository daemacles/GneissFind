/*
 * Split-screen landscape LCARS scanner interface (`lcars-ui.md`).
 * Upper half hosts the hardware-clipped panoramic camera viewport, 120 Hz GPU EIS translation,
 * left-side pitch/cropY travel gauge, and high-contrast stone reticles; lower half hosts the
 * non-occluding tactical control deck with semantic color families, deduplicated single-purpose
 * buttons, and an animated scrollable source-selection drawer.
 */
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
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
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
import com.gneissfind.app.model.ConfirmedTarget
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
import com.gneissfind.app.ui.lcars.LcarsTelemetryBlock
import com.gneissfind.app.ui.lcars.rememberHighWaterSegmentWidth
import com.gneissfind.app.ui.theme.GneissFindTheme
import com.gneissfind.app.ui.theme.LcarsColors
import com.gneissfind.app.ui.theme.LcarsSemanticColors
import com.gneissfind.app.ui.theme.LcarsTypography
import com.gneissfind.app.ui.theme.LcarsValueLevel
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private val TopElbowArmWidth = 136.dp
private val TopElbowStemHeight = 50.dp
private val BottomElbowArmWidth = 136.dp
private val BottomElbowStemHeight = 48.dp
private val PitchGaugeWidth = 44.dp

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
        val swathChanged = currentSwathMode != swathTileMode
        currentCropYFloat = cropYFloat
        currentSwathMode = swathTileMode
        if (swathChanged) {
            requestLayout()
        }
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

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val vwInt = MeasureSpec.getSize(widthMeasureSpec).coerceAtLeast(1)
        val vhInt = MeasureSpec.getSize(heightMeasureSpec).coerceAtLeast(1)
        val t = computePanoramicViewportTransform(
            viewportWidthPx = vwInt.toFloat(),
            viewportHeightPx = vhInt.toFloat(),
            cropYFloat = currentCropYFloat,
            swathTileMode = currentSwathMode
        )
        val childW = t.renderedFrameWidthPx.roundToInt().coerceAtLeast(1)
        val childH = t.renderedFrameHeightPx.roundToInt().coerceAtLeast(1)
        // Always measure PreviewView so dynamically attached TextureView children size to non-zero bounds.
        previewView.measure(
            MeasureSpec.makeMeasureSpec(childW, MeasureSpec.EXACTLY),
            MeasureSpec.makeMeasureSpec(childH, MeasureSpec.EXACTLY)
        )
        setMeasuredDimension(vwInt, vhInt)
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
        val childLeft = t.frameLeftPx.roundToInt()
        val childW = t.renderedFrameWidthPx.roundToInt().coerceAtLeast(1)
        val childH = t.renderedFrameHeightPx.roundToInt().coerceAtLeast(1)
        previewView.layout(childLeft, 0, childLeft + childW, childH)
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
    onSelectSource: (DetectorSourceOption) -> Unit,
    onToggleSwathTileMode: () -> Unit,
    onToggleEisLock: (Boolean) -> Unit,
    onSyncPitch: () -> Unit,
    onToggleStoneClass: (StoneClass) -> Unit,
    onToggleShutterClamp: (Boolean) -> Unit,
    onToggleTwoXCrop: (Boolean) -> Unit,
    onChangeExposureBias: (Float) -> Unit,
    onToggleAudioMute: (Boolean) -> Unit,
    onToggleOverspeedTest: () -> Unit,
    initialSourceDrawerOpen: Boolean = false
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(LcarsColors.CanvasBlack)
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(horizontal = 6.dp, vertical = 4.dp),
        verticalArrangement = Arrangement.spacedBy(LcarsGutter)
    ) {
        // Upper ~50% Height: Geological Viewport (Left Pitch Gauge + Panoramic Camera + Passive HUD)
        GeologicalViewportZone(
            uiState = uiState,
            hasCameraPermission = hasCameraPermission,
            onRequestCameraPermission = onRequestCameraPermission,
            onBindPreview = onBindPreview,
            onToggleAudioMute = onToggleAudioMute,
            modifier = Modifier
                .fillMaxWidth()
                .weight(0.50f)
        )

        // Lower ~50% Height: Tactical Control Deck (Single-Action Buttons + Scrollable Source Drawer)
        TacticalControlDeckZone(
            uiState = uiState,
            onSelectMode = onSelectMode,
            onSelectSource = onSelectSource,
            onToggleSwathTileMode = onToggleSwathTileMode,
            onToggleEisLock = onToggleEisLock,
            onSyncPitch = onSyncPitch,
            onToggleStoneClass = onToggleStoneClass,
            onToggleShutterClamp = onToggleShutterClamp,
            onToggleTwoXCrop = onToggleTwoXCrop,
            onChangeExposureBias = onChangeExposureBias,
            onToggleOverspeedTest = onToggleOverspeedTest,
            initialSourceDrawerOpen = initialSourceDrawerOpen,
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
    onToggleAudioMute: (Boolean) -> Unit,
    modifier: Modifier = Modifier
) {
    val isLogging = uiState.operatingMode == OperatingMode.LOG_DATA
    val rotCode = if (uiState.displaySurfaceRotation == Surface.ROTATION_270) "R-270" else "R-090"
    val crookStart = LcarsStemWidth + LcarsGutter
    val crookTop = LcarsBarHeight + LcarsGutter

    val pitchHighWaterWidth = rememberHighWaterSegmentWidth("PITCH -90° [EIS OFF]")
    val velHighWaterWidth = rememberHighWaterSegmentWidth("SLOW SWEEP 9.99 M/S")
    val expHighWaterWidth = rememberHighWaterSegmentWidth("2.0X • 1/2000S • ISO 9999")

    Box(modifier = modifier.background(LcarsColors.CanvasBlack)) {
        // Top-Left Swept LCARS Elbow ("GEO-47" — Structural Chrome family)
        LcarsElbow(
            orientation = LcarsElbowOrientation.TopLeft,
            color = if (isLogging) {
                LcarsSemanticColors.criticalAlert
            } else {
                LcarsSemanticColors.structuralChrome(LcarsValueLevel.BASE)
            },
            armLabel = "GEO-47",
            stemCode = rotCode,
            modifier = Modifier
                .align(Alignment.TopStart)
                .width(TopElbowArmWidth)
                .height(TopElbowStemHeight)
        )

        // Top 22.dp Passive Telemetry Header Rail (Strictly Non-Interactive, High-Water Widths)
        Row(
            modifier = Modifier
                .align(Alignment.TopStart)
                .fillMaxWidth()
                .padding(start = TopElbowArmWidth + LcarsGutter)
                .height(LcarsBarHeight),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            val pitchClampedActive = uiState.eisPitchLockEnabled && uiState.isPitchClamped
            val pitchColor = when {
                !uiState.eisPitchLockEnabled -> LcarsColors.MutedSlate
                pitchClampedActive -> LcarsSemanticColors.criticalAlert
                else -> LcarsSemanticColors.realtimeDataLabel(LcarsValueLevel.MINUS)
            }
            val pitchText = when {
                !uiState.eisPitchLockEnabled ->
                    String.format(Locale.US, "PITCH %02.0f° [EIS OFF]", uiState.imuPitchDeg)
                pitchClampedActive ->
                    String.format(Locale.US, "PITCH LIMIT %02.0f°", uiState.imuPitchDeg)
                else ->
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
                    !uiState.eisPitchLockEnabled -> LcarsColors.InactiveLabel
                    pitchClampedActive -> LcarsColors.AlertWhite
                    else -> LcarsColors.CanvasBlack
                },
                modifier = Modifier.width(pitchHighWaterWidth)
            )

            LcarsHeaderSegment(
                text = String.format(
                    Locale.US,
                    "RANGE %.2f M • RESOLUTION %.2f MM/PX",
                    uiState.slantRangeMeters,
                    uiState.gsdMmPerPx
                ),
                color = LcarsSemanticColors.staticStateLabel(LcarsValueLevel.BASE),
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
                color = if (isOverspeed) {
                    LcarsSemanticColors.criticalAlert
                } else {
                    LcarsSemanticColors.realtimeDataLabel(LcarsValueLevel.PLUS)
                },
                textColor = if (isOverspeed) LcarsColors.AlertWhite else LcarsColors.CanvasBlack,
                modifier = Modifier.width(velHighWaterWidth)
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
                color = LcarsSemanticColors.staticStateLabel(LcarsValueLevel.MINUS),
                alignEnd = true,
                modifier = Modifier.width(expHighWaterWidth)
            )
        }

        // Left 68.dp Stem: Passive FPS Telemetry Block + Sole AUDIO ON/OFF Pill Button
        Column(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(top = TopElbowStemHeight + LcarsGutter)
                .width(LcarsStemWidth)
                .fillMaxHeight(),
            verticalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            LcarsTelemetryBlock(
                prefixCode = "FPS",
                valueText = String.format(Locale.US, "%.1fHZ", uiState.sweepHz),
                subText = String.format(Locale.US, "%.1fMS", uiState.rollingLatencyMs),
                color = LcarsSemanticColors.realtimeDataLabel(LcarsValueLevel.BASE),
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            )

            LcarsPillButton(
                label = if (uiState.isAudioMuted) "MUTE" else "AUDIO",
                subLabel = if (uiState.isAudioMuted) "OFF" else "ON",
                prefixCode = "SND",
                active = !uiState.isAudioMuted,
                activeColor = LcarsSemanticColors.toggleButton(true, LcarsValueLevel.BASE),
                inactiveColor = LcarsSemanticColors.toggleButton(false, LcarsValueLevel.BASE),
                endStyle = LcarsPillEndStyle.CompactRoundedPill,
                compactStemMode = true,
                onClick = { onToggleAudioMute(!uiState.isAudioMuted) },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            )
        }

        // Inner Crook: Left-Side Pitch/CropY Travel Indicator + Panoramic Camera Viewport
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(start = crookStart, top = crookTop),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            PitchTravelGaugeColumn(
                cropY = uiState.cropY,
                displayCropYFloat = uiState.displayCropYFloat,
                eisEnabled = uiState.eisPitchLockEnabled,
                isPitchClamped = uiState.isPitchClamped,
                modifier = Modifier
                    .width(PitchGaugeWidth)
                    .fillMaxHeight()
            )

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
                            topStart = 4.dp,
                            topEnd = 4.dp,
                            bottomStart = 4.dp,
                            bottomEnd = 4.dp
                        )
                    )
                    .background(LcarsColors.CanvasBlack)
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
                    activeColor = LcarsSemanticColors.actionOrDrawerButton(LcarsValueLevel.BASE),
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
            uiState.speedState.isOverspeed -> LcarsSemanticColors.criticalAlert
            uiState.isPitchClamped && uiState.eisPitchLockEnabled -> LcarsSemanticColors.criticalAlert
            uiState.operatingMode == OperatingMode.LOG_DATA -> LcarsColors.GoldenAmber
            else -> LcarsColors.Periwinkle
        }

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
                color = LcarsSemanticColors.criticalAlert,
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
    val haloStroke = innerStroke + 4.dp.toPx()

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

/**
 * Passive non-interactive left-side EIS `cropY` indicator column positioned directly beneath the
 * `PITCH` telemetry bar and directly above the `EIS LOCK` and `SYNC` controls.
 */
@Composable
private fun PitchTravelGaugeColumn(
    cropY: Int,
    displayCropYFloat: Float,
    eisEnabled: Boolean,
    isPitchClamped: Boolean,
    modifier: Modifier = Modifier
) {
    val clampedAlert = eisEnabled && isPitchClamped
    val readoutBg = when {
        !eisEnabled -> LcarsColors.MutedSlate
        clampedAlert -> LcarsSemanticColors.criticalAlert
        else -> LcarsSemanticColors.realtimeDataLabel(LcarsValueLevel.PLUS)
    }
    val readoutFg = when {
        !eisEnabled -> LcarsColors.InactiveLabel
        clampedAlert -> LcarsColors.AlertWhite
        else -> LcarsColors.CanvasBlack
    }

    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(LcarsGutter)
    ) {
        LcarsTelemetryBlock(
            valueText = if (clampedAlert) "LIM" else String.format(Locale.US, "%03d", cropY),
            subText = "Y-PX",
            color = readoutBg,
            textColor = readoutFg,
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

            // Nominal center reference tick (cropY = 348)
            val midY = h * 0.5f
            drawLine(
                color = LcarsSemanticColors.staticStateLabel(LcarsValueLevel.BASE),
                start = Offset(0f, midY),
                end = Offset(w, midY),
                strokeWidth = 2.dp.toPx()
            )

            // Continuous 120 Hz 384/1080 strip window indicator inside [0..696]
            val windowFrac = StripRoiGeometry.STRIP_HEIGHT_PX.toFloat() /
                StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX.toFloat()
            val thumbH = (h * windowFrac).coerceAtLeast(10.dp.toPx())
            val maxCropY = StripRoiGeometry.MAX_CROP_Y_PX.toFloat()
            val travelFrac = displayCropYFloat.coerceIn(0f, maxCropY) / maxCropY
            val thumbTop = travelFrac * max(0f, h - thumbH)

            val thumbColor = when {
                !eisEnabled -> LcarsColors.MutedSlate
                clampedAlert -> LcarsSemanticColors.criticalAlert
                else -> LcarsSemanticColors.realtimeDataLabel(LcarsValueLevel.BASE)
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
    onSelectSource: (DetectorSourceOption) -> Unit,
    onToggleSwathTileMode: () -> Unit,
    onToggleEisLock: (Boolean) -> Unit,
    onSyncPitch: () -> Unit,
    onToggleStoneClass: (StoneClass) -> Unit,
    onToggleShutterClamp: (Boolean) -> Unit,
    onToggleTwoXCrop: (Boolean) -> Unit,
    onChangeExposureBias: (Float) -> Unit,
    onToggleOverspeedTest: () -> Unit,
    modifier: Modifier = Modifier,
    initialSourceDrawerOpen: Boolean = false
) {
    val isLogging = uiState.operatingMode == OperatingMode.LOG_DATA
    val crookStart = LcarsStemWidth + LcarsGutter
    val crookBottom = LcarsBarHeight + LcarsGutter

    val swathCodeHighWaterWidth = rememberHighWaterSegmentWidth("47-1792")
    val pingHighWaterWidth = rememberHighWaterSegmentWidth("PING 999 HZ @ 99.9/S")

    Box(modifier = modifier.background(LcarsColors.CanvasBlack)) {
        // Left 68.dp Stem: Single Combined SCAN/LOG Mode Button + Passive Frame Counter Block
        Column(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(bottom = BottomElbowStemHeight + LcarsGutter)
                .width(LcarsStemWidth)
                .fillMaxHeight(),
            verticalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            LcarsPillButton(
                label = if (isLogging) "LOG" else "SCAN",
                subLabel = if (isLogging) "TO SCAN" else "TO LOG",
                prefixCode = "MOD",
                active = true,
                activeColor = if (isLogging) {
                    LcarsSemanticColors.criticalAlert
                } else {
                    LcarsSemanticColors.multiStateButton(LcarsValueLevel.BASE)
                },
                endStyle = LcarsPillEndStyle.CompactRoundedPill,
                compactStemMode = true,
                onClick = {
                    onSelectMode(if (isLogging) OperatingMode.SCAN else OperatingMode.LOG_DATA)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1.15f)
            )

            LcarsTelemetryBlock(
                prefixCode = "REC",
                valueText = String.format(Locale.US, "%04d", uiState.loggedFrameCount),
                subText = if (isLogging) "4.0 HZ" else "SAVED",
                color = LcarsSemanticColors.staticStateLabel(LcarsValueLevel.MINUS),
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(0.85f)
            )
        }

        // Bottom-Left Swept LCARS Elbow ("SYS-01" — Structural Chrome -5% value step)
        LcarsElbow(
            orientation = LcarsElbowOrientation.BottomLeft,
            color = LcarsSemanticColors.structuralChrome(LcarsValueLevel.MINUS),
            armLabel = "SYS-01",
            stemCode = "M2",
            modifier = Modifier
                .align(Alignment.BottomStart)
                .width(BottomElbowArmWidth)
                .height(BottomElbowStemHeight)
        )

        // Bottom 22.dp Passive Telemetry Footer Rail (High-Water-Mark Ping Width)
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
                color = LcarsSemanticColors.staticStateLabel(LcarsValueLevel.PLUS),
                modifier = Modifier.width(swathCodeHighWaterWidth)
            )

            LcarsHeaderSegment(
                text = String.format(
                    Locale.US,
                    "SWEEP %.1f MS • %s (G1 %.1f | TOPK %.1f | G2 %.1f)",
                    uiState.rollingLatencyMs,
                    uiState.detectorBackend.label,
                    uiState.part1LatencyMs,
                    uiState.bridgeLatencyMs,
                    uiState.part2LatencyMs
                ),
                color = LcarsSemanticColors.realtimeDataLabel(LcarsValueLevel.MINUS),
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
                color = if (uiState.isAudioMuted) {
                    LcarsColors.MutedSlate
                } else {
                    LcarsSemanticColors.realtimeDataLabel(LcarsValueLevel.PLUS)
                },
                textColor = if (uiState.isAudioMuted) {
                    LcarsColors.InactiveLabel
                } else {
                    LcarsColors.CanvasBlack
                },
                alignEnd = true,
                modifier = Modifier.width(pingHighWaterWidth)
            )
        }

        // 2-Column Tactical Control Matrix Inside Bottom-Left Crook
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
                onSelectSource = onSelectSource,
                onToggleOverspeedTest = onToggleOverspeedTest,
                initialSourceDrawerOpen = initialSourceDrawerOpen,
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
                color = LcarsSemanticColors.structuralChrome(LcarsValueLevel.PLUS)
            )
            Text(
                text = if (uiState.isPitchClamped && uiState.eisPitchLockEnabled) {
                    "PITCH LIMIT"
                } else {
                    "EIS-SYNC"
                },
                style = LcarsTypography.MicroCode,
                color = if (uiState.isPitchClamped && uiState.eisPitchLockEnabled) {
                    LcarsSemanticColors.criticalAlert
                } else {
                    LcarsSemanticColors.realtimeDataLabel(LcarsValueLevel.BASE)
                }
            )
        }

        // Row 1: Adjacent ON/OFF Toggles (stepped MINUS vs PLUS in ToggleButton family)
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
                activeColor = LcarsSemanticColors.toggleButton(true, LcarsValueLevel.MINUS),
                inactiveColor = LcarsSemanticColors.toggleButton(false, LcarsValueLevel.MINUS),
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
                activeColor = LcarsSemanticColors.toggleButton(true, LcarsValueLevel.PLUS),
                inactiveColor = LcarsSemanticColors.toggleButton(false, LcarsValueLevel.PLUS),
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = { onToggleShutterClamp(!uiState.shutterClampEnabled) },
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
            )
        }

        // Row 2: ON/OFF Toggle (EIS LOCK) + Action Trigger (SYNC) + Multi-State Switch (5/3-TILE)
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            val pitchLimitActive = uiState.eisPitchLockEnabled && uiState.isPitchClamped
            LcarsPillButton(
                label = when {
                    pitchLimitActive -> "PITCH LIMIT"
                    uiState.eisPitchLockEnabled -> "EIS LOCK [ON]"
                    else -> "EIS LOCK [OFF]"
                },
                subLabel = String.format(Locale.US, "CROP Y %03d", uiState.cropY),
                prefixCode = "EIS",
                active = uiState.eisPitchLockEnabled,
                activeColor = if (pitchLimitActive) {
                    LcarsSemanticColors.criticalAlert
                } else {
                    LcarsSemanticColors.toggleButton(true, LcarsValueLevel.BASE)
                },
                inactiveColor = LcarsSemanticColors.toggleButton(false, LcarsValueLevel.BASE),
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
                activeColor = LcarsSemanticColors.actionOrDrawerButton(LcarsValueLevel.BASE),
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = onSyncPitch,
                modifier = Modifier
                    .weight(0.88f)
                    .fillMaxHeight()
            )

            val isFiveTile = uiState.swathTileMode == SwathTileMode.FIVE_TILE
            LcarsPillButton(
                label = "${uiState.swathTileMode.tileCount}-TILE",
                subLabel = "${uiState.swathTileMode.stripWidthPx}PX",
                prefixCode = "SW",
                active = true,
                activeColor = LcarsSemanticColors.multiStateButton(
                    if (isFiveTile) LcarsValueLevel.PLUS else LcarsValueLevel.MINUS
                ),
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = onToggleSwathTileMode,
                modifier = Modifier
                    .weight(0.84f)
                    .fillMaxHeight()
            )
        }

        // Row 3: Stepper Action Buttons + Passive 7-Segment EV Readout Bar
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
    onSelectSource: (DetectorSourceOption) -> Unit,
    onToggleOverspeedTest: () -> Unit,
    modifier: Modifier = Modifier,
    initialSourceDrawerOpen: Boolean = false
) {
    val classes = StoneClass.alertableClasses
    var isSourceDrawerOpen by rememberSaveable { mutableStateOf(initialSourceDrawerOpen) }
    val activeSource = uiState.activeSourceOption

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
                text = if (isSourceDrawerOpen) {
                    "SRC-SEL // SCROLL & TAP TO SELECT SOURCE"
                } else {
                    "PROSPECT MATRIX // MIN 20MM TARGETS"
                },
                style = LcarsTypography.MicroCode,
                color = if (isSourceDrawerOpen) {
                    LcarsSemanticColors.actionOrDrawerButton(LcarsValueLevel.PLUS)
                } else {
                    LcarsSemanticColors.toggleButton(true, LcarsValueLevel.BASE)
                }
            )
            Text(
                text = "${uiState.enabledClasses.size}/4 ACTIVE • CONFUSERS MUTED",
                style = LcarsTypography.MicroCode,
                color = LcarsSemanticColors.staticStateLabel(LcarsValueLevel.BASE)
            )
        }

        // Upper 2-row slot: slides cleanly between the 2x2 Mineral Matrix and the scrollable
        // SourceSelectionDrawer so underlying mineral pills are never clickable while open.
        AnimatedContent(
            targetState = isSourceDrawerOpen,
            transitionSpec = {
                if (targetState) {
                    (slideInVertically(animationSpec = tween(180)) { fullHeight -> fullHeight } +
                        fadeIn(animationSpec = tween(140))) togetherWith
                        fadeOut(animationSpec = tween(100))
                } else {
                    fadeIn(animationSpec = tween(140)) togetherWith
                        (slideOutVertically(animationSpec = tween(160)) { fullHeight -> fullHeight } +
                            fadeOut(animationSpec = tween(120)))
                }
            },
            label = "SourceSelectionDrawerTransition",
            modifier = Modifier
                .fillMaxWidth()
                .weight(2f)
        ) { drawerOpen ->
            if (drawerOpen) {
                SourceSelectionDrawer(
                    activeOption = activeSource,
                    onSelectOption = { chosen ->
                        onSelectSource(chosen)
                        isSourceDrawerOpen = false
                    },
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Column(
                    modifier = Modifier.fillMaxSize(),
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
                            valueLevel = LcarsValueLevel.PLUS,
                            enabled = uiState.enabledClasses.contains(classes[0]),
                            onClick = { onToggleStoneClass(classes[0]) },
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxHeight()
                        )
                        MineralTargetPill(
                            stoneClass = classes[1],
                            valueLevel = LcarsValueLevel.MINUS,
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
                            valueLevel = LcarsValueLevel.BASE,
                            enabled = uiState.enabledClasses.contains(classes[2]),
                            onClick = { onToggleStoneClass(classes[2]) },
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxHeight()
                        )
                        MineralTargetPill(
                            stoneClass = classes[3],
                            valueLevel = LcarsValueLevel.PLUS,
                            enabled = uiState.enabledClasses.contains(classes[3]),
                            onClick = { onToggleStoneClass(classes[3]) },
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxHeight()
                        )
                    }
                }
            }
        }

        // Bottom Row: SOURCE Drawer Trigger + Speed Gate Simulation Toggle
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .weight(0.92f),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            val drawerArrow = if (isSourceDrawerOpen) "▴" else "▾"
            LcarsPillButton(
                label = "${activeSource.title} $drawerArrow",
                subLabel = if (isSourceDrawerOpen) {
                    "TAP SOURCE IN DRAWER"
                } else {
                    "OPEN SOURCE DRAWER"
                },
                prefixCode = "SRC",
                active = true,
                activeColor = LcarsSemanticColors.actionOrDrawerButton(
                    if (isSourceDrawerOpen) LcarsValueLevel.PLUS else LcarsValueLevel.BASE
                ),
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = { isSourceDrawerOpen = !isSourceDrawerOpen },
                modifier = Modifier
                    .weight(1.45f)
                    .fillMaxHeight()
            )

            val simOverspeedActive = uiState.simulatedSpeedOverrideMps != null
            LcarsPillButton(
                label = if (simOverspeedActive) "SPEED SIM [ON]" else "SPEED SIM [OFF]",
                subLabel = if (simOverspeedActive) "1.38 M/S TEST" else "GATE <= 1.10M/S",
                prefixCode = "SIM",
                active = simOverspeedActive,
                activeColor = LcarsSemanticColors.criticalAlert,
                inactiveColor = LcarsSemanticColors.toggleButton(false, LcarsValueLevel.MINUS),
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = onToggleOverspeedTest,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
            )
        }
    }
}

/**
 * Scrollable LCARS selection drawer presenting all [DetectorSourceOption] items with a live
 * right-edge scroll position thumb. First tap on `SRC` opens this drawer; second tap on any row
 * activates that source and collapses the drawer.
 */
@Composable
private fun SourceSelectionDrawer(
    activeOption: DetectorSourceOption,
    onSelectOption: (DetectorSourceOption) -> Unit,
    modifier: Modifier = Modifier
) {
    val options = DetectorSourceOption.entries
    val listState = rememberLazyListState()
    val rowCycle = remember {
        listOf(
            LcarsValueLevel.MINUS,
            LcarsValueLevel.BASE,
            LcarsValueLevel.PLUS
        )
    }

    LaunchedEffect(activeOption) {
        val idx = options.indexOf(activeOption).coerceAtLeast(0)
        listState.scrollToItem((idx - 1).coerceAtLeast(0))
    }

    Row(
        modifier = modifier
            .clip(RoundedCornerShape(10.dp))
            .background(LcarsColors.CanvasBlack)
            .border(
                width = 2.dp,
                color = LcarsSemanticColors.actionOrDrawerButton(LcarsValueLevel.BASE),
                shape = RoundedCornerShape(10.dp)
            )
            .padding(LcarsGutter),
        horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
    ) {
        LazyColumn(
            state = listState,
            contentPadding = PaddingValues(vertical = 1.dp),
            verticalArrangement = Arrangement.spacedBy(LcarsGutter),
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight()
        ) {
            itemsIndexed(options, key = { _, item -> item.name }) { idx, option ->
                val isSelected = option == activeOption
                val level = rowCycle[idx % rowCycle.size]
                LcarsPillButton(
                    label = if (isSelected) "● ${option.title}" else option.title,
                    subLabel = option.subtitle,
                    prefixCode = option.code,
                    active = isSelected,
                    activeColor = LcarsSemanticColors.actionOrDrawerButton(LcarsValueLevel.PLUS),
                    inactiveColor = LcarsSemanticColors.toggleButton(false, level),
                    endStyle = LcarsPillEndStyle.FlatLeftRoundRight,
                    onClick = { onSelectOption(option) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(40.dp)
                )
            }
        }

        // Visible LCARS scroll-track indicator so scrollable affordance is unmistakable
        Canvas(
            modifier = Modifier
                .width(6.dp)
                .fillMaxHeight()
                .clip(RoundedCornerShape(3.dp))
                .background(LcarsColors.MutedSlate)
        ) {
            val total = options.size.coerceAtLeast(1)
            val visibleCount = listState.layoutInfo.visibleItemsInfo.size.coerceIn(1, total)
            val firstIdx = listState.firstVisibleItemIndex.coerceIn(0, (total - visibleCount).coerceAtLeast(0))
            val thumbFrac = (visibleCount.toFloat() / total.toFloat()).coerceIn(0.25f, 1f)
            val thumbH = size.height * thumbFrac
            val maxScrollItems = (total - visibleCount).coerceAtLeast(1)
            val scrollProgress = if (total <= visibleCount) {
                0f
            } else {
                (firstIdx.toFloat() / maxScrollItems.toFloat()).coerceIn(0f, 1f)
            }
            val thumbTop = scrollProgress * (size.height - thumbH)
            drawRect(
                color = LcarsSemanticColors.actionOrDrawerButton(LcarsValueLevel.PLUS),
                topLeft = Offset(0f, thumbTop),
                size = Size(size.width, thumbH)
            )
        }
    }
}

@Composable
private fun MineralTargetPill(
    stoneClass: StoneClass,
    valueLevel: LcarsValueLevel,
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
        activeColor = LcarsSemanticColors.toggleButton(true, valueLevel),
        inactiveColor = LcarsSemanticColors.toggleButton(false, valueLevel),
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
                displayCropYFloat = 348f,
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
            onSelectSource = {},
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
    name = "LCARS Geological Tricorder — Scrollable Source Drawer Open",
    device = "spec:width=915dp,height=411dp,dpi=420,orientation=landscape"
)
@Composable
private fun ScannerScreenSourceDrawerOpenPreview() {
    GneissFindTheme {
        ScannerScreen(
            uiState = ScannerUiState(
                sweepHz = 14.6f,
                rollingLatencyMs = 8.4f,
                imuPitchDeg = 42f,
                referencePitchDeg = 45f,
                cropY = 412,
                displayCropYFloat = 412f,
                enabledClasses = setOf(
                    StoneClass.CHALCEDONY_AGATE,
                    StoneClass.GREEN_GEM_ROCK,
                    StoneClass.PETRIFIED_WOOD
                )
            ),
            hasCameraPermission = true,
            onRequestCameraPermission = {},
            onBindPreview = { _, _ -> },
            onSelectMode = {},
            onSelectSource = {},
            onToggleSwathTileMode = {},
            onToggleEisLock = {},
            onSyncPitch = {},
            onToggleStoneClass = {},
            onToggleShutterClamp = {},
            onToggleTwoXCrop = {},
            onChangeExposureBias = {},
            onToggleAudioMute = {},
            onToggleOverspeedTest = {},
            initialSourceDrawerOpen = true
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
                displayCropYFloat = 0f,
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
            onSelectSource = {},
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
