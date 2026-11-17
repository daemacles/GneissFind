/*
 * Landscape LCARS Geological Tricorder interface (`.agents/rules/lcars-ui.md`).
 *
 * Consumes immutable [ScannerUiState] snapshots from [ScannerViewModel] and renders an adaptive
 * LCARS layout that preserves the single CameraX [PreviewView] instance across mode transitions:
 * - In `PATROL` (`weight(0.50f)` / `weight(0.50f)` split): [GeologicalViewportZone] on top (120 Hz
 *   GPU RenderNode `translationY` strip stabilization, [PitchTravelGaugeColumn], and
 *   [PushBroomTripwireOverlay]) paired with [TacticalControlDeckZone] on the bottom (`OPT`, `EXP`,
 *   `EYE`, `EIS`, `CAL`, `SW`, 2x2 mineral matrix, central `LOCK NEAREST`, and [SourceSelectionDrawer]).
 * - In `LOCK_ON_NEURAL` & `PINPOINT_CONSTELLATION` (`weight(1f)` full-screen): hides the lower
 *   parameter/toggle deck so the uncropped `16:9` camera feed fills the screen between the slim
 *   `22.dp` top/bottom LCARS telemetry rails, framed by passive LCARS eye-candy readouts and
 *   [LockOnSideConsole] housing the sole interactive control (`BREAK LOCK`).
 *
 * Main types and entry points:
 * - [PanoramicViewportTransform] / [computePanoramicViewportTransform]: optical coordinate mapping
 *   from the canonical `1920x1080` landscape frame into the viewport slot (both `PATROL` strip and
 *   uncropped `16:9` Lock-On / Pinpoint modes).
 * - [ScannerScreen]: top-level composable bound by `MainActivity`.
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
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
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
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalInspectionMode
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.gneissfind.app.BuildConfig
import com.gneissfind.app.model.AcquisitionStage
import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.LockOnHudState
import com.gneissfind.app.model.OperatingMode
import com.gneissfind.app.model.PatrolPowerState
import com.gneissfind.app.model.ScanSpeedState
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.SwathTileMode
import com.gneissfind.app.model.WorldPoint3D
import com.gneissfind.app.ui.lcars.LcarsBarHeight
import com.gneissfind.app.ui.lcars.LcarsDetectionReticleSwatch
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
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

private val TopElbowArmWidth = 156.dp
private val TopElbowStemHeight = 50.dp
private val BottomElbowArmWidth = 192.dp
private val BottomElbowStemHeight = 48.dp
private val PitchGaugeWidth = 44.dp
private val LockOnSideDeckWidth = 168.dp
private val TngLockGold = Color(0xFFFFCC66)
private val EyesUpIdleVeilColor = Color(0xE6000000)

private val ScanSpeedState.isOverspeed: Boolean
    get() = this is ScanSpeedState.SpeedWarning

private val ScanSpeedState.speedMps: Float
    get() = when (this) {
        is ScanSpeedState.Normal -> currentSpeedMps
        is ScanSpeedState.SpeedWarning -> currentSpeedMps
    }

/**
 * Exact optical mapping from canonical [StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX] x
 * [StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX] landscape sensor coordinates into the panoramic
 * LCARS viewport slot.
 *
 * - In `PATROL` (`fullFrameLockOn = false`), `cropY in [0..MAX_CROP_Y_PX]` moves smoothly while
 *   keeping the entire [StripRoiGeometry.STRIP_HEIGHT_PX] push-broom strip visible and 1:1
 *   pixel-registered with [PreviewView].
 * - In `LOCK_ON_NEURAL` / `PINPOINT_CONSTELLATION` (`fullFrameLockOn = true`), fits the entire
 *   uncropped `16:9` (`1920x1080`) frame inside the viewport with `PreviewView.translationY = 0f`.
 *
 * @property scale Uniform scale factor (`viewportPx / sensorPx`) applied to the canonical
 *   landscape frame.
 * @property frameLeftPx Horizontal offset in viewport pixels of the scaled `1920 px` sensor frame.
 * @property frameTopPx Vertical offset in viewport pixels of the scaled `1080 px` sensor frame.
 * @property renderedFrameWidthPx Width in viewport pixels of the scaled `1920 px` sensor frame.
 * @property renderedFrameHeightPx Height in viewport pixels of the scaled `1080 px` sensor frame.
 * @property stripLeftPx Left edge in viewport pixels of the active push-broom strip ROI.
 * @property stripTopPx Top edge in viewport pixels of the active push-broom strip ROI.
 * @property stripWidthPx Width in viewport pixels of the active push-broom strip ROI.
 * @property stripHeightPx Height in viewport pixels of the active push-broom strip ROI.
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

/**
 * Integer-[cropY] overload of [computePanoramicViewportTransform].
 *
 * @param viewportWidthPx Width of the viewport container in screen pixels (clamped to `>= 1f`).
 * @param viewportHeightPx Height of the viewport container in screen pixels (clamped to `>= 1f`).
 * @param cropY Integer vertical top offset of the `384 px` strip in `[0, MAX_CROP_Y_PX]`.
 * @param swathTileMode Active horizontal tile geometry ([SwathTileMode.FIVE_TILE] or [SwathTileMode.THREE_TILE]).
 * @param fullFrameLockOn When `true`, fits the uncropped full `16:9` (`1920x1080`) frame inside the viewport.
 * @return [PanoramicViewportTransform] aligning the camera preview and HUD overlay.
 */
fun computePanoramicViewportTransform(
    viewportWidthPx: Float,
    viewportHeightPx: Float,
    cropY: Int,
    swathTileMode: SwathTileMode,
    fullFrameLockOn: Boolean = false
): PanoramicViewportTransform = computePanoramicViewportTransform(
    viewportWidthPx = viewportWidthPx,
    viewportHeightPx = viewportHeightPx,
    cropYFloat = cropY.toFloat(),
    swathTileMode = swathTileMode,
    fullFrameLockOn = fullFrameLockOn
)

/**
 * Computes continuous sub-pixel optical mapping from canonical `1920x1080` landscape sensor
 * coordinates into the panoramic LCARS viewport slot so 120 Hz GPU `PreviewView.translationY` (in
 * `PATROL`) or uncropped `16:9` homing (`fullFrameLockOn = true`) stays 1:1 registered with the
 * Compose HUD overlay.
 *
 * @param viewportWidthPx Width of the viewport container in screen pixels (clamped to `>= 1f`).
 * @param viewportHeightPx Height of the viewport container in screen pixels (clamped to `>= 1f`).
 * @param cropYFloat Sub-pixel vertical top offset of the `384 px` strip in `[0f, MAX_CROP_Y_PX]`.
 * @param swathTileMode Active horizontal tile geometry ([SwathTileMode.FIVE_TILE] or [SwathTileMode.THREE_TILE]).
 * @param fullFrameLockOn When `true`, fits the entire uncropped `16:9` (`1920x1080`) frame inside `(vw, vh)`.
 * @return [PanoramicViewportTransform] aligning the camera preview and HUD overlay.
 */
fun computePanoramicViewportTransform(
    viewportWidthPx: Float,
    viewportHeightPx: Float,
    cropYFloat: Float,
    swathTileMode: SwathTileMode,
    fullFrameLockOn: Boolean = false
): PanoramicViewportTransform {
    val vw = max(1f, viewportWidthPx)
    val vh = max(1f, viewportHeightPx)
    val maxCropY = StripRoiGeometry.MAX_CROP_Y_PX.toFloat()
    val clampedCropY = cropYFloat.coerceIn(0f, maxCropY)

    if (fullFrameLockOn) {
        val scale = min(
            vw / StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX.toFloat(),
            vh / StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX.toFloat()
        )
        val renderedW = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX * scale
        val renderedH = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX * scale
        val frameLeft = (vw - renderedW) * 0.5f
        val frameTop = (vh - renderedH) * 0.5f
        val stripLeft = frameLeft + swathTileMode.canonicalStripLeftPx * scale
        val stripTop = frameTop + clampedCropY * scale
        val stripWidth = swathTileMode.stripWidthPx * scale
        val stripH = StripRoiGeometry.STRIP_HEIGHT_PX * scale
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
 * applying 120 Hz EIS pitch stabilization via GPU RenderNode `previewView.translationY` in `PATROL`
 * and expanding cleanly to uncropped `16:9` (`previewView.translationY = 0f`) in `LOCK_ON_NEURAL`
 * and `PINPOINT_CONSTELLATION`.
 */
private class ClippingCameraViewportContainer(context: Context) : FrameLayout(context) {
    val previewView: PreviewView = PreviewView(context).apply {
        scaleType = PreviewView.ScaleType.FILL_CENTER
        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        isSoundEffectsEnabled = false
    }

    private var currentCropYFloat: Float = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat()
    private var currentSwathMode: SwathTileMode = SwathTileMode.FIVE_TILE
    private var currentFullFrameLockOn: Boolean = false

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

    /**
     * Updates the sub-pixel vertical crop offset, swath tile mode, and full-frame Lock-On state,
     * applying vertical frame shifts directly to `previewView.translationY` on the UI thread in
     * `PATROL` (`0f` when [fullFrameLockOn] is active) without per-frame layout stalls.
     *
     * @param cropYFloat Continuous 120 Hz stabilized `cropY` in `[0f, MAX_CROP_Y_PX]`.
     * @param swathTileMode Active swath tile mode.
     * @param fullFrameLockOn Whether uncropped `16:9` Lock-On / Pinpoint mode is active.
     */
    fun updateStabilizationState(
        cropYFloat: Float,
        swathTileMode: SwathTileMode,
        fullFrameLockOn: Boolean
    ) {
        val layoutModeChanged =
            currentSwathMode != swathTileMode || currentFullFrameLockOn != fullFrameLockOn
        currentCropYFloat = cropYFloat
        currentSwathMode = swathTileMode
        currentFullFrameLockOn = fullFrameLockOn
        if (layoutModeChanged) {
            requestLayout()
        }
        if (width > 0 && height > 0) {
            if (currentFullFrameLockOn) {
                previewView.translationY = 0f
            } else {
                val t = computePanoramicViewportTransform(
                    viewportWidthPx = width.toFloat(),
                    viewportHeightPx = height.toFloat(),
                    cropYFloat = currentCropYFloat,
                    swathTileMode = currentSwathMode,
                    fullFrameLockOn = false
                )
                // Pure GPU RenderNode sub-pixel translation — zero measure/layout stalls.
                previewView.translationY = t.frameTopPx
            }
        }
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val vwInt = MeasureSpec.getSize(widthMeasureSpec).coerceAtLeast(1)
        val vhInt = MeasureSpec.getSize(heightMeasureSpec).coerceAtLeast(1)
        val t = computePanoramicViewportTransform(
            viewportWidthPx = vwInt.toFloat(),
            viewportHeightPx = vhInt.toFloat(),
            cropYFloat = currentCropYFloat,
            swathTileMode = currentSwathMode,
            fullFrameLockOn = currentFullFrameLockOn
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
            swathTileMode = currentSwathMode,
            fullFrameLockOn = currentFullFrameLockOn
        )
        val childLeft = t.frameLeftPx.roundToInt()
        val childTop = if (currentFullFrameLockOn) t.frameTopPx.roundToInt() else 0
        val childW = t.renderedFrameWidthPx.roundToInt().coerceAtLeast(1)
        val childH = t.renderedFrameHeightPx.roundToInt().coerceAtLeast(1)
        previewView.layout(childLeft, childTop, childLeft + childW, childTop + childH)
        previewView.translationY = if (currentFullFrameLockOn) 0f else t.frameTopPx
    }
}

/**
 * Top-level split-screen landscape LCARS Geological Tricorder screen.
 *
 * Splits the safe-drawing canvas evenly (`0.50f` / `0.50f`) between [GeologicalViewportZone] on top
 * and [TacticalControlDeckZone] on the bottom so interactive buttons never occlude the live camera
 * swath or target reticles (`.agents/rules/lcars-ui.md`).
 *
 * @param uiState Immutable UI and telemetry snapshot emitted by [ScannerViewModel.uiState].
 * @param hasCameraPermission Whether `Manifest.permission.CAMERA` is currently granted.
 * @param onRequestCameraPermission Callback invoked when the operator taps the permission prompt pill.
 * @param onBindPreview Callback invoked once the underlying [PreviewView] is inflated and ready to
 *   bind to CameraX.
 * @param onSelectMode Callback invoked to switch between [OperatingMode.SCAN] and [OperatingMode.LOG_DATA].
 * @param onSelectSource Callback invoked when a [DetectorSourceOption] is selected in the source drawer.
 * @param onToggleSwathTileMode Callback invoked to toggle between 5-tile and 3-tile swath widths.
 * @param onToggleEisLock Callback invoked to enable or disable IMU pitch-lock EIS stabilization.
 * @param onSyncPitch Callback invoked to re-center the reference pitch to the current device pitch.
 * @param onToggleStoneClass Callback invoked to mute or unmute a specific [StoneClass] filter.
 * @param onToggleShutterClamp Callback invoked to toggle the `<= 2.0 ms` (`1/500 s`) manual shutter clamp.
 * @param onToggleTwoXCrop Callback invoked to toggle the `2.0x` lossless sensor crop mode.
 * @param onChangeExposureBias Callback invoked when the operator steps exposure bias in EV.
 * @param onToggleAudioMute Callback invoked to mute or unmute the acoustic sonifier.
 * @param onToggleEyesUpPatrol Callback invoked to toggle the `"EYES-UP"` (`EYE`) OLED-saving idle patrol veil.
 * @param onPatrolViewportTapped Callback invoked when the operator taps `(normalizedStripX, normalizedStripY, minTouchSlopStripPx)` in `PATROL`.
 * @param onToggleCentralLock Callback invoked when the operator taps the central `LOCK NEAREST` / `BREAK LOCK` button.
 * @param initialSourceDrawerOpen Whether the scrollable source drawer starts open (used by Compose Previews).
 */
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
    onToggleEyesUpPatrol: () -> Unit = {},
    onPatrolViewportTapped: (Float, Float, Float) -> Unit = { _, _, _ -> },
    onToggleCentralLock: () -> Unit = {},
    initialSourceDrawerOpen: Boolean = false
) {
    val isLockedOn = uiState.acquisitionStage != AcquisitionStage.PATROL

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(LcarsColors.CanvasBlack)
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(horizontal = 6.dp, vertical = 4.dp),
        verticalArrangement = Arrangement.spacedBy(LcarsGutter)
    ) {
        // Persistent camera host: ~50% height in PATROL, expanding to 100% full-screen in Lock Mode
        // without disposing ClippingCameraViewportContainer or rebinding CameraX PreviewView.
        GeologicalViewportZone(
            uiState = uiState,
            hasCameraPermission = hasCameraPermission,
            onRequestCameraPermission = onRequestCameraPermission,
            onBindPreview = onBindPreview,
            onToggleAudioMute = onToggleAudioMute,
            onPatrolViewportTapped = onPatrolViewportTapped,
            onToggleCentralLock = onToggleCentralLock,
            modifier = Modifier
                .fillMaxWidth()
                .weight(if (isLockedOn) 1f else 0.50f)
        )

        // Lower ~50% Height: Tactical Control Deck (hidden during Lock Mode so video fills the screen)
        if (!isLockedOn) {
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
                onToggleEyesUpPatrol = onToggleEyesUpPatrol,
                onToggleCentralLock = onToggleCentralLock,
                initialSourceDrawerOpen = initialSourceDrawerOpen,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(0.50f)
            )
        }
    }
}

/**
 * Persistent LCARS camera frame (`weight(0.50f)` in `PATROL`, expanding to `weight(1f)` full-screen
 * in `LOCK_ON_NEURAL` and `PINPOINT_CONSTELLATION`) hosting the `GNEISSFIND` top-left elbow,
 * high-water-mark passive telemetry rails, left stem, [PanoramicCameraViewport], and (when locked
 * on) the build-provenance bottom rail plus [LockOnSideConsole] housing the sole `BREAK LOCK` button.
 */
@Composable
private fun GeologicalViewportZone(
    uiState: ScannerUiState,
    hasCameraPermission: Boolean,
    onRequestCameraPermission: () -> Unit,
    onBindPreview: (LifecycleOwner, PreviewView) -> Unit,
    onToggleAudioMute: (Boolean) -> Unit,
    onPatrolViewportTapped: (Float, Float, Float) -> Unit,
    onToggleCentralLock: () -> Unit,
    modifier: Modifier = Modifier
) {
    val isLogging = uiState.operatingMode == OperatingMode.LOG_DATA
    val isLockedOn = uiState.acquisitionStage != AcquisitionStage.PATROL
    val rotCode = if (uiState.displaySurfaceRotation == Surface.ROTATION_270) "R-270" else "R-090"
    val crookStart = LcarsStemWidth + LcarsGutter
    val crookTop = LcarsBarHeight + LcarsGutter
    val crookBottom = if (isLockedOn) LcarsBarHeight + LcarsGutter else 0.dp

    val pitchHighWaterWidth = rememberHighWaterSegmentWidth("PITCH -90° [LOCK 16:9]")
    val velHighWaterWidth = rememberHighWaterSegmentWidth("SLOW SWEEP 9.99 M/S")
    val expHighWaterWidth = rememberHighWaterSegmentWidth("2.0X • 1/2000S • ISO 9999")

    Box(modifier = modifier.background(LcarsColors.CanvasBlack)) {
        // Top-Left Swept LCARS Elbow ("GNEISSFIND" — Structural Chrome family, .agents/rules/lcars-ui.md §2.1)
        LcarsElbow(
            orientation = LcarsElbowOrientation.TopLeft,
            color = if (isLogging) {
                LcarsSemanticColors.criticalAlert
            } else {
                LcarsSemanticColors.structuralChrome(LcarsValueLevel.BASE)
            },
            armLabel = "GNEISSFIND",
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
            val pitchClampedActive = !isLockedOn && uiState.eisPitchLockEnabled && uiState.isPitchClamped
            val pitchColor = when {
                isLockedOn -> LcarsSemanticColors.realtimeDataLabel(LcarsValueLevel.MINUS)
                !uiState.eisPitchLockEnabled -> LcarsColors.MutedSlate
                pitchClampedActive -> LcarsSemanticColors.criticalAlert
                else -> LcarsSemanticColors.realtimeDataLabel(LcarsValueLevel.MINUS)
            }
            val pitchText = when {
                isLockedOn ->
                    String.format(Locale.US, "PITCH %02.0f° [LOCK 16:9]", uiState.imuPitchDeg)
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
                    !isLockedOn && !uiState.eisPitchLockEnabled -> LcarsColors.InactiveLabel
                    pitchClampedActive -> LcarsColors.AlertWhite
                    else -> LcarsColors.CanvasBlack
                },
                modifier = Modifier.width(pitchHighWaterWidth)
            )

            val rangeTelemetryText = if (isLockedOn) {
                val hud = uiState.lockOnHudState
                String.format(
                    Locale.US,
                    "RANGE %.2f M • ±%.0f MM",
                    hud.slantRangeMeters,
                    hud.sigma3dMm
                )
            } else {
                String.format(
                    Locale.US,
                    "RANGE %.2f M • RESOLUTION %.2f MM/PX",
                    uiState.slantRangeMeters,
                    uiState.gsdMmPerPx
                )
            }
            LcarsHeaderSegment(
                text = rangeTelemetryText,
                color = LcarsSemanticColors.staticStateLabel(LcarsValueLevel.BASE),
                modifier = Modifier.weight(1f)
            )

            val isOverspeed = !isLockedOn && uiState.speedState.isOverspeed
            val velText = when {
                isOverspeed ->
                    String.format(Locale.US, "SLOW SWEEP %.2f M/S", uiState.speedState.speedMps)
                uiState.patrolPowerState == PatrolPowerState.PATROL_IDLE && !isLockedOn ->
                    "IDLE 15 FPS"
                else ->
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

        // Left 68.dp Stem: FPS + SND button in PATROL; purely passive LCARS C-frame telemetry in Lock Mode
        Column(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(
                    top = TopElbowStemHeight + LcarsGutter,
                    bottom = if (isLockedOn) BottomElbowStemHeight + LcarsGutter else 0.dp
                )
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

            if (isLockedOn) {
                val hud = uiState.lockOnHudState
                val isHold = hud.isLostTrackGraceActive
                val isPin = uiState.acquisitionStage == AcquisitionStage.PINPOINT_CONSTELLATION
                val isAkazeLocked = isPin && hud.isTargetInFrame && hud.akazeInliers >= 12
                LcarsTelemetryBlock(
                    prefixCode = "TRK",
                    valueText = when {
                        isHold -> "HOLD"
                        isAkazeLocked -> "PIN"
                        isPin -> "SRCH"
                        else -> "LOCK"
                    },
                    subText = when {
                        isHold -> String.format(Locale.US, "%02.0fS", hud.lostTrackRemainingSec)
                        isAkazeLocked -> "AKAZE"
                        else -> "B=1"
                    },
                    color = when {
                        isHold -> LcarsColors.GoldenAmber
                        isPin && !isAkazeLocked -> LcarsColors.IceBlue
                        else -> LcarsSemanticColors.staticStateLabel(LcarsValueLevel.PLUS)
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                )
            } else {
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
        }

        // Bottom-Left Build Provenance Elbow + Bottom 22.dp Telemetry Rail when Full-Screen Locked On
        if (isLockedOn) {
            LcarsBottomProvenanceFooter(
                uiState = uiState,
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .fillMaxWidth()
            )
        }

        // Inner Crook: Persistent Panoramic / Full-16:9 Camera Viewport + Mode-Specific Side Chrome
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(start = crookStart, top = crookTop, bottom = crookBottom),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            if (!isLockedOn) {
                PitchTravelGaugeColumn(
                    cropY = uiState.cropY,
                    displayCropYFloat = uiState.displayCropYFloat,
                    eisEnabled = uiState.eisPitchLockEnabled,
                    isPitchClamped = uiState.isPitchClamped,
                    modifier = Modifier
                        .width(PitchGaugeWidth)
                        .fillMaxHeight()
                )
            }

            PanoramicCameraViewport(
                uiState = uiState,
                hasCameraPermission = hasCameraPermission,
                onRequestCameraPermission = onRequestCameraPermission,
                onBindPreview = onBindPreview,
                onPatrolViewportTapped = onPatrolViewportTapped,
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

            if (isLockedOn) {
                LockOnSideConsole(
                    uiState = uiState,
                    onBreakLock = onToggleCentralLock,
                    modifier = Modifier
                        .width(LockOnSideDeckWidth)
                        .fillMaxHeight()
                )
            }
        }
    }
}

/**
 * Hardware-clipped camera viewport container pairing [ClippingCameraViewportContainer] (when camera
 * permission is granted in a live session) or a dark diagnostic grid (in Compose Previews / before
 * permission grant) with [PushBroomTripwireOverlay] and tap-to-lock reticle selection.
 */
@Composable
private fun PanoramicCameraViewport(
    uiState: ScannerUiState,
    hasCameraPermission: Boolean,
    onRequestCameraPermission: () -> Unit,
    onBindPreview: (LifecycleOwner, PreviewView) -> Unit,
    onPatrolViewportTapped: (Float, Float, Float) -> Unit,
    modifier: Modifier = Modifier
) {
    val isInspection = LocalInspectionMode.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val fullFrameLockOn = uiState.acquisitionStage != AcquisitionStage.PATROL

    val currentUiState by rememberUpdatedState(uiState)
    val currentTapHandler by rememberUpdatedState(onPatrolViewportTapped)

    Box(
        modifier = modifier.pointerInput(Unit) {
            detectTapGestures { offset ->
                val state = currentUiState
                if (state.operatingMode == OperatingMode.SCAN &&
                    state.acquisitionStage == AcquisitionStage.PATROL
                ) {
                    val t = computePanoramicViewportTransform(
                        viewportWidthPx = size.width.toFloat(),
                        viewportHeightPx = size.height.toFloat(),
                        cropYFloat = state.displayCropYFloat,
                        swathTileMode = state.swathTileMode,
                        fullFrameLockOn = false
                    )
                    val normX = (offset.x - t.stripLeftPx) / t.stripWidthPx.coerceAtLeast(1f)
                    val normY = (offset.y - t.stripTopPx) / t.stripHeightPx.coerceAtLeast(1f)
                    // Guarantee >= 48.dp (56.dp diameter) touch slop around every reticle regardless of viewport scale
                    val minSlopStripPx = max(56f, (28.dp.toPx() / t.scale.coerceAtLeast(0.1f)))
                    currentTapHandler(normX, normY, minSlopStripPx)
                }
            }
        }
    ) {
        if (hasCameraPermission && !isInspection) {
            AndroidView(
                factory = { ctx ->
                    ClippingCameraViewportContainer(ctx).apply {
                        layoutParams = ViewGroup.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.MATCH_PARENT
                        )
                        updateStabilizationState(
                            cropYFloat = uiState.displayCropYFloat,
                            swathTileMode = uiState.swathTileMode,
                            fullFrameLockOn = fullFrameLockOn
                        )
                        onBindPreview(lifecycleOwner, previewView)
                    }
                },
                update = { container ->
                    container.updateStabilizationState(
                        cropYFloat = uiState.displayCropYFloat,
                        swathTileMode = uiState.swathTileMode,
                        fullFrameLockOn = fullFrameLockOn
                    )
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

/**
 * Compose [Canvas] overlay drawn directly atop [PanoramicCameraViewport] in 1:1 optical
 * registration with [computePanoramicViewportTransform].
 *
 * - In `PATROL`: renders the semi-transparent letterbox veil outside the `384 px` strip,
 *   `#000000`-keylined strip rails, `32 px` inter-tile overlap ticks, high-contrast stone target
 *   reticles, and the `90%` `#E6000000` `"EYES-UP"` OLED-saving veil when `PATROL_IDLE` is active.
 * - In `LOCK_ON_NEURAL` & `PINPOINT_CONSTELLATION`: hides push-broom strip rails/ticks over the
 *   uncropped `16:9` camera view and renders the high-visibility `320x320` neural detector crop
 *   square, world-locked `1 Hz` neural bounding box, mineral bracket + shrinking dashed
 *   `sigma3dMm` uncertainty circle (`LOCK_ON_NEURAL`) or solid double-ring crosshair
 *   (`PINPOINT_CONSTELLATION`), and pulsing off-screen directional re-acquisition chevron
 *   (`<- 18° [14S]`) with zero flickering floating text over the stone.
 * - Renders the non-blocking inline LCARS banner `COMPILING GPU SHADERS...` during async `SRC` loads.
 */
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

    val isLockedOn = uiState.acquisitionStage != AcquisitionStage.PATROL
    val chevronPulseTransition = rememberInfiniteTransition(label = "OffScreenChevronPulse")
    val chevronPulseAlpha by chevronPulseTransition.animateFloat(
        initialValue = 0.45f,
        targetValue = 1.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 480),
            repeatMode = RepeatMode.Reverse
        ),
        label = "OffScreenChevronAlpha"
    )

    Canvas(modifier = modifier) {
        val transform = computePanoramicViewportTransform(
            viewportWidthPx = size.width,
            viewportHeightPx = size.height,
            cropYFloat = uiState.displayCropYFloat,
            swathTileMode = uiState.swathTileMode,
            fullFrameLockOn = isLockedOn
        )

        if (isLockedOn) {
            drawLockOnFullFrameOverlay(
                hud = uiState.lockOnHudState,
                transform = transform,
                chevronPulseAlpha = chevronPulseAlpha,
                labelPaint = labelPaint,
                hudPaint = hudPaint
            )
        } else {
            drawPatrolStripOverlay(
                uiState = uiState,
                transform = transform,
                labelPaint = labelPaint,
                hudPaint = hudPaint
            )
        }

        if (uiState.isCompilingModel) {
            drawInlineBanner(
                text = "COMPILING GPU SHADERS...",
                backgroundColor = TngLockGold,
                textColor = LcarsColors.CanvasBlack,
                labelPaint = labelPaint,
                topOffsetPx = 6.dp.toPx()
            )
        }
    }
}

private fun DrawScope.drawPatrolStripOverlay(
    uiState: ScannerUiState,
    transform: PanoramicViewportTransform,
    labelPaint: Paint,
    hudPaint: Paint
) {
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
            if (!target.isVisibleInStrip) return@forEach
            if (!target.stoneClass.isUserAlertable) return@forEach
            if (!uiState.enabledClasses.contains(target.stoneClass)) return@forEach

            val rawLeft = stripLeft + target.stripLeftPx * transform.scale
            val rawTop = stripTop + target.stripTopPx * transform.scale
            val rawRight = stripLeft + target.stripRightPx * transform.scale
            val rawBottom = stripTop + target.stripBottomPx * transform.scale

            val left = rawLeft.coerceIn(stripLeft, stripRight)
            val top = rawTop.coerceIn(visibleStripTop, visibleStripBottom)
            val right = rawRight.coerceIn(stripLeft, stripRight)
            val bottom = rawBottom.coerceIn(visibleStripTop, visibleStripBottom)
            if (right - left < 6f || bottom - top < 6f) return@forEach

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
            val calloutW = (labelPaint.measureText(callout) + 10.dp.toPx()).finiteOr(60f).coerceAtLeast(1f)
            val calloutH = 16.dp.toPx().coerceAtLeast(1f)
            val calloutLeft = safeCanvasClamp(
                left,
                stripLeft + 2f,
                max(stripLeft + 2f, stripRight - calloutW - 2f)
            )
            val rawCalloutTop = if (top - calloutH - 3.dp.toPx() >= 0f) {
                top - calloutH - 3.dp.toPx()
            } else {
                min(size.height - calloutH, bottom + 3.dp.toPx())
            }
            val calloutTop = safeCanvasClamp(rawCalloutTop, 0f, max(0f, size.height - calloutH))

            drawRect(
                color = LcarsColors.CanvasBlack,
                topLeft = Offset(calloutLeft, calloutTop),
                size = Size(calloutW.coerceAtLeast(0f), calloutH.coerceAtLeast(0f))
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
        drawInlineBanner(
            text = alertText,
            backgroundColor = LcarsSemanticColors.criticalAlert,
            textColor = LcarsColors.AlertWhite,
            labelPaint = labelPaint,
            topOffsetPx = 4.dp.toPx()
        )
    }

    // "EYES-UP" 90% #E6000000 OLED-saving veil when PATROL_IDLE is active (.agents/rules/lcars-ui.md)
    if (uiState.operatingMode == OperatingMode.SCAN &&
        uiState.patrolPowerState == PatrolPowerState.PATROL_IDLE &&
        uiState.eyesUpPatrolEnabled
    ) {
        drawRect(
            color = EyesUpIdleVeilColor,
            topLeft = Offset.Zero,
            size = size
        )
        val idleTag = "EYES-UP PATROL • 15 FPS • TAP OR TARGET TO WAKE"
        hudPaint.textSize = 11.dp.toPx()
        val idleW = hudPaint.measureText(idleTag)
        hudPaint.color = LcarsColors.IceBlue.copy(alpha = 0.55f).toArgb()
        drawContext.canvas.nativeCanvas.drawText(
            idleTag,
            ((size.width - idleW) * 0.5f).coerceAtLeast(8f),
            size.height * 0.5f + 4.dp.toPx(),
            hudPaint
        )
    }
}

private fun DrawScope.drawLockOnFullFrameOverlay(
    hud: LockOnHudState,
    transform: PanoramicViewportTransform,
    chevronPulseAlpha: Float,
    labelPaint: Paint,
    hudPaint: Paint
) {
    if (size.width <= 1f || size.height <= 1f ||
        transform.renderedFrameWidthPx <= 1f || transform.renderedFrameHeightPx <= 1f ||
        !transform.scale.isFinite() || transform.scale <= 0f
    ) {
        return
    }
    val fLeft = safeCanvasClamp(transform.frameLeftPx, 0f, size.width)
    val fTop = safeCanvasClamp(transform.frameTopPx, 0f, size.height)
    val renderedW = transform.renderedFrameWidthPx.finiteOr(size.width).coerceAtLeast(1f)
    val renderedH = transform.renderedFrameHeightPx.finiteOr(size.height).coerceAtLeast(1f)
    val fRight = safeCanvasClamp(fLeft + renderedW, fLeft, size.width)
    val fBottom = safeCanvasClamp(fTop + renderedH, fTop, size.height)

    // Pure #000000 pillarbox/letterbox outside the uncropped 16:9 (1920x1080) sensor frame
    if (fLeft > 0f) {
        drawRect(
            color = LcarsColors.CanvasBlack,
            topLeft = Offset(0f, 0f),
            size = Size(fLeft.coerceAtLeast(0f), size.height.coerceAtLeast(0f))
        )
    }
    if (fRight < size.width) {
        drawRect(
            color = LcarsColors.CanvasBlack,
            topLeft = Offset(fRight, 0f),
            size = Size((size.width - fRight).coerceAtLeast(0f), size.height.coerceAtLeast(0f))
        )
    }
    if (fTop > 0f) {
        drawRect(
            color = LcarsColors.CanvasBlack,
            topLeft = Offset(0f, 0f),
            size = Size(size.width.coerceAtLeast(0f), fTop.coerceAtLeast(0f))
        )
    }
    if (fBottom < size.height) {
        drawRect(
            color = LcarsColors.CanvasBlack,
            topLeft = Offset(0f, fBottom),
            size = Size(size.width.coerceAtLeast(0f), (size.height - fBottom).coerceAtLeast(0f))
        )
    }

    val isPinpoint = hud.stage == AcquisitionStage.PINPOINT_CONSTELLATION
    val isAkazeLocked = isPinpoint && hud.isTargetInFrame && hud.akazeInliers >= 12
    val frameBorderColor = when {
        hud.isLostTrackGraceActive -> LcarsColors.GoldenAmber
        isAkazeLocked -> Color(hud.stoneClass.primaryHexColor)
        else -> LcarsColors.IceBlue.copy(alpha = 0.78f)
    }
    drawRect(
        color = frameBorderColor,
        topLeft = Offset(fLeft, fTop),
        size = Size((fRight - fLeft).coerceAtLeast(0f), (fBottom - fTop).coerceAtLeast(0f)),
        style = Stroke(width = 1.5.dp.toPx())
    )

    // Mode tag in top-left of 16:9 frame
    val safeLostSec = hud.lostTrackRemainingSec.finiteOr(0f).coerceIn(0f, 99f)
    val modeTag = when {
        !hud.isTargetInFrame -> "OFF-SCREEN HOLD • PAN TOWARD CHEVRON"
        hud.isLostTrackGraceActive -> String.format(
            Locale.US,
            "HOLDING ANCHOR • %.0fS GRACE",
            safeLostSec
        )
        isAkazeLocked -> "PINPOINT 16:9 • 30HZ AKAZE + 1HZ B=1"
        isPinpoint -> "RE-ACQUIRING 16:9 • B=1 WIDE SEARCH"
        else -> String.format(
            Locale.US,
            "LOCK-ON 16:9 • B=1 %dX%d",
            hud.cropSizePx,
            hud.cropSizePx
        )
    }
    hudPaint.textSize = 10.dp.toPx()
    val modeTagW = (hudPaint.measureText(modeTag) + 12.dp.toPx()).finiteOr(80f).coerceAtLeast(1f)
    val modeTagH = 15.dp.toPx().coerceAtLeast(1f)
    drawRect(
        color = LcarsColors.CanvasBlack,
        topLeft = Offset(fLeft + 4.dp.toPx(), fTop + 4.dp.toPx()),
        size = Size(modeTagW, modeTagH)
    )
    hudPaint.color = frameBorderColor.toArgb()
    drawContext.canvas.nativeCanvas.drawText(
        modeTag,
        fLeft + 10.dp.toPx(),
        fTop + 4.dp.toPx() + modeTagH - 3.5.dp.toPx(),
        hudPaint
    )

    // High-visibility adaptive [320..640] neural detector window rendered cleanly without inner text clutter
    if (hud.isTargetInFrame) {
        val cL = safeCanvasClamp(fLeft + hud.cropLeftPx * transform.scale, fLeft, fRight)
        val cT = safeCanvasClamp(fTop + hud.cropTopPx * transform.scale, fTop, fBottom)
        val cR = safeCanvasClamp(fLeft + (hud.cropLeftPx + hud.cropSizePx) * transform.scale, cL, fRight)
        val cB = safeCanvasClamp(fTop + (hud.cropTopPx + hud.cropSizePx) * transform.scale, cT, fBottom)
        val cSize = Size(max(1f, cR - cL), max(1f, cB - cT))
        val cropDash = PathEffect.dashPathEffect(floatArrayOf(10f, 6f), 0f)

        drawRect(
            color = Color(0xB3000000),
            topLeft = Offset(cL, cT),
            size = cSize,
            style = Stroke(width = 3.6.dp.toPx())
        )
        drawRect(
            color = LcarsColors.IceBlue.copy(alpha = 0.86f),
            topLeft = Offset(cL, cT),
            size = cSize,
            style = Stroke(width = 1.8.dp.toPx(), pathEffect = cropDash)
        )
    }

    val mineralColor = Color(hud.stoneClass.primaryHexColor)

    if (hud.isTargetInFrame) {
        // World-locked 1 Hz neural detector stone box inside the detector window during PINPOINT
        if (isPinpoint && hud.hasNeuralDetectorBox) {
            val nLeft = safeCanvasClamp(fLeft + hud.neuralBoxLeftPx * transform.scale, fLeft, fRight)
            val nTop = safeCanvasClamp(fTop + hud.neuralBoxTopPx * transform.scale, fTop, fBottom)
            val nRight = safeCanvasClamp(fLeft + hud.neuralBoxRightPx * transform.scale, nLeft, fRight)
            val nBottom = safeCanvasClamp(fTop + hud.neuralBoxBottomPx * transform.scale, nTop, fBottom)
            if (nRight - nLeft >= 8f && nBottom - nTop >= 8f) {
                val neuralDash = PathEffect.dashPathEffect(floatArrayOf(6f, 4f), 0f)
                drawRect(
                    color = Color(0xCC000000),
                    topLeft = Offset(nLeft, nTop),
                    size = Size((nRight - nLeft).coerceAtLeast(0f), (nBottom - nTop).coerceAtLeast(0f)),
                    style = Stroke(width = 3.2.dp.toPx())
                )
                drawRect(
                    color = LcarsColors.IceBlue,
                    topLeft = Offset(nLeft, nTop),
                    size = Size((nRight - nLeft).coerceAtLeast(0f), (nBottom - nTop).coerceAtLeast(0f)),
                    style = Stroke(width = 1.6.dp.toPx(), pathEffect = neuralDash)
                )
            }
        }

        val left = safeCanvasClamp(fLeft + hud.frameLeftPx * transform.scale, fLeft, fRight)
        val top = safeCanvasClamp(fTop + hud.frameTopPx * transform.scale, fTop, fBottom)
        val right = safeCanvasClamp(fLeft + hud.frameRightPx * transform.scale, left, fRight)
        val bottom = safeCanvasClamp(fTop + hud.frameBottomPx * transform.scale, top, fBottom)
        val cx = (left + right) * 0.5f
        val cy = (top + bottom) * 0.5f

        drawHighContrastCornerBrackets(
            left = left,
            top = top,
            right = right,
            bottom = bottom,
            foregroundColor = mineralColor
        )

        val minSigmaPx = 8.dp.toPx()
        val maxSigmaPx = max(minSigmaPx, min(fRight - fLeft, fBottom - fTop) * 0.44f)
        val sigmaPxScaled = safeCanvasClamp(hud.sigmaRadiusPx * transform.scale, minSigmaPx, maxSigmaPx)

        if (isAkazeLocked) {
            // Solid double-ring crosshair ONLY when AKAZE texture constellation (>= 12 PTS) is actively locked
            val outerR = max(14.dp.toPx(), sigmaPxScaled)
            val innerR = max(6.dp.toPx(), outerR * 0.48f)
            val strokePx = 2.dp.toPx()
            val haloPx = strokePx + 3.dp.toPx()

            drawCircle(
                color = LcarsColors.CanvasBlack,
                radius = outerR,
                center = Offset(cx, cy),
                style = Stroke(width = haloPx)
            )
            drawCircle(
                color = LcarsColors.CanvasBlack,
                radius = innerR,
                center = Offset(cx, cy),
                style = Stroke(width = haloPx)
            )
            drawCircle(
                color = mineralColor,
                radius = outerR,
                center = Offset(cx, cy),
                style = Stroke(width = strokePx)
            )
            drawCircle(
                color = mineralColor,
                radius = innerR,
                center = Offset(cx, cy),
                style = Stroke(width = strokePx)
            )

            val tickLen = 7.dp.toPx()
            val ticks = arrayOf(
                Offset(cx - outerR - tickLen, cy) to Offset(cx - innerR, cy),
                Offset(cx + innerR, cy) to Offset(cx + outerR + tickLen, cy),
                Offset(cx, cy - outerR - tickLen) to Offset(cx, cy - innerR),
                Offset(cx, cy + innerR) to Offset(cx, cy + outerR + tickLen)
            )
            ticks.forEach { (s, e) ->
                drawLine(color = LcarsColors.CanvasBlack, start = s, end = e, strokeWidth = haloPx)
                drawLine(color = mineralColor, start = s, end = e, strokeWidth = strokePx)
            }
        } else {
            // Dashed uncertainty / search circle when converging in 3D or re-acquiring with B=1
            val dashEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 7f), 0f)
            val ringStroke = 2.dp.toPx()
            val ringRadius = if (isPinpoint) max(22.dp.toPx(), sigmaPxScaled) else sigmaPxScaled
            drawCircle(
                color = LcarsColors.CanvasBlack,
                radius = ringRadius,
                center = Offset(cx, cy),
                style = Stroke(width = ringStroke + 3.dp.toPx(), pathEffect = dashEffect)
            )
            drawCircle(
                color = mineralColor,
                radius = ringRadius,
                center = Offset(cx, cy),
                style = Stroke(width = ringStroke, pathEffect = dashEffect)
            )
        }
    } else {
        // Off-screen directional re-acquisition chevron during 15s grace window (.agents/rules/lcars-ui.md §4)
        val chevron = hud.offScreenChevron
        if (chevron != null) {
            val safeEdgeXNorm = safeCanvasClamp(chevron.edgeXNorm, 0.05f, 0.95f)
            val safeEdgeYNorm = safeCanvasClamp(chevron.edgeYNorm, 0.07f, 0.93f)
            val safeDirRad = chevron.directionRad.finiteOr(0f)
            val edgeX = safeCanvasClamp(fLeft + safeEdgeXNorm * (fRight - fLeft), fLeft, fRight)
            val edgeY = safeCanvasClamp(fTop + safeEdgeYNorm * (fBottom - fTop), fTop, fBottom)
            val dirX = cos(safeDirRad).finiteOr(1f)
            val dirY = sin(safeDirRad).finiteOr(0f)
            val perpX = -dirY
            val perpY = dirX

            val tipLen = 12.dp.toPx()
            val wingSpan = 9.dp.toPx()
            val tip = Offset(
                safeCanvasClamp(edgeX + dirX * tipLen * 0.5f, 0f, size.width),
                safeCanvasClamp(edgeY + dirY * tipLen * 0.5f, 0f, size.height)
            )
            val wingA = Offset(
                safeCanvasClamp(edgeX - dirX * tipLen * 0.5f + perpX * wingSpan, 0f, size.width),
                safeCanvasClamp(edgeY - dirY * tipLen * 0.5f + perpY * wingSpan, 0f, size.height)
            )
            val wingB = Offset(
                safeCanvasClamp(edgeX - dirX * tipLen * 0.5f - perpX * wingSpan, 0f, size.width),
                safeCanvasClamp(edgeY - dirY * tipLen * 0.5f - perpY * wingSpan, 0f, size.height)
            )
            val chevPath = Path().apply {
                moveTo(wingA.x, wingA.y)
                lineTo(tip.x, tip.y)
                lineTo(wingB.x, wingB.y)
            }
            val safePulse = safeCanvasClamp(chevronPulseAlpha, 0.1f, 1f)
            val pulsedColor = mineralColor.copy(alpha = safePulse)
            drawPath(
                path = chevPath,
                color = LcarsColors.CanvasBlack,
                style = Stroke(width = 6.dp.toPx(), cap = StrokeCap.Round)
            )
            drawPath(
                path = chevPath,
                color = pulsedColor,
                style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round)
            )

            val degClamped = safeCanvasClamp(chevron.angularOffsetDeg, 0f, 99f)
            val chevLabel = String.format(
                Locale.US,
                "%02.0f° • %02.0fS",
                degClamped,
                safeLostSec
            )
            labelPaint.textSize = 11.dp.toPx()
            val rawBoxW = (labelPaint.measureText("99° • 15S") + 14.dp.toPx()).finiteOr(84f)
            val fixedBoxW = rawBoxW.coerceIn(1f, max(1f, fRight - fLeft))
            val textW = labelPaint.measureText(chevLabel).finiteOr(fixedBoxW * 0.8f)
            val ch = 18.dp.toPx().coerceIn(1f, max(1f, fBottom - fTop))
            val minCl = min(fLeft + 14.dp.toPx(), max(fLeft, fRight - fixedBoxW))
            val maxCl = max(minCl, fRight - fixedBoxW - 14.dp.toPx())
            val cl = safeCanvasClamp(
                edgeX - fixedBoxW * 0.5f - dirX * (fixedBoxW * 0.42f),
                minCl,
                maxCl
            )
            val minCt = min(fTop + 6.dp.toPx(), max(fTop, fBottom - ch))
            val maxCt = max(minCt, fBottom - ch - 6.dp.toPx())
            val ct = safeCanvasClamp(
                edgeY - ch * 0.5f - dirY * 14.dp.toPx(),
                minCt,
                maxCt
            )
            drawRect(
                color = LcarsColors.CanvasBlack,
                topLeft = Offset(cl, ct),
                size = Size(fixedBoxW.coerceAtLeast(0f), ch.coerceAtLeast(0f))
            )
            labelPaint.color = LcarsColors.GoldenAmber.copy(alpha = safePulse).toArgb()
            drawContext.canvas.nativeCanvas.drawText(
                chevLabel,
                cl + (fixedBoxW - textW) * 0.5f,
                ct + ch - 4.dp.toPx(),
                labelPaint
            )
        }
    }
}

private fun DrawScope.drawInlineBanner(
    text: String,
    backgroundColor: Color,
    textColor: Color,
    labelPaint: Paint,
    topOffsetPx: Float
) {
    labelPaint.textSize = 12.dp.toPx()
    val bannerW = labelPaint.measureText(text) + 24.dp.toPx()
    val bannerH = 22.dp.toPx()
    val bannerLeft = ((size.width - bannerW) * 0.5f).coerceAtLeast(4f)
    drawRect(
        color = LcarsColors.CanvasBlack,
        topLeft = Offset(bannerLeft - 2.dp.toPx(), topOffsetPx - 2.dp.toPx()),
        size = Size(bannerW + 4.dp.toPx(), bannerH + 4.dp.toPx())
    )
    drawRect(
        color = backgroundColor,
        topLeft = Offset(bannerLeft, topOffsetPx),
        size = Size(bannerW, bannerH)
    )
    labelPaint.color = textColor.toArgb()
    drawContext.canvas.nativeCanvas.drawText(
        text,
        bannerLeft + 12.dp.toPx(),
        topOffsetPx + bannerH - 5.dp.toPx(),
        labelPaint
    )
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

/**
 * Shared bottom-left swept LCARS build-provenance elbow (`GIT_COMMIT_HASH_6 • BUILD_DATE_ISO`) and
 * `22.dp` passive telemetry footer rail rendered at the bottom of the screen in both `PATROL` and
 * full-screen Lock Mode.
 */
@Composable
private fun LcarsBottomProvenanceFooter(
    uiState: ScannerUiState,
    modifier: Modifier = Modifier
) {
    val swathCodeHighWaterWidth = rememberHighWaterSegmentWidth("47-1792")
    val pingHighWaterWidth = rememberHighWaterSegmentWidth("PING 999 HZ @ 99.9/S")
    val provenanceLabel = remember {
        "${BuildConfig.GIT_COMMIT_HASH_6} • ${BuildConfig.BUILD_DATE_ISO}"
    }

    Box(
        modifier = modifier.height(BottomElbowStemHeight)
    ) {
        LcarsElbow(
            orientation = LcarsElbowOrientation.BottomLeft,
            color = LcarsSemanticColors.staticStateLabel(LcarsValueLevel.BASE),
            armLabel = provenanceLabel,
            stemCode = "M2D",
            armTextStyle = LcarsTypography.TelemetryBar,
            stemTextStyle = LcarsTypography.PillSubLabel,
            modifier = Modifier
                .align(Alignment.BottomStart)
                .width(BottomElbowArmWidth)
                .height(BottomElbowStemHeight)
        )

        Row(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .fillMaxWidth()
                .padding(start = BottomElbowArmWidth + LcarsGutter)
                .height(LcarsBarHeight),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            val statusSegmentCode = when (uiState.acquisitionStage) {
                AcquisitionStage.PATROL -> "47-${uiState.swathTileMode.stripWidthPx}"
                AcquisitionStage.LOCK_ON_NEURAL -> "LCK-320"
                AcquisitionStage.PINPOINT_CONSTELLATION -> "PIN-AKZ"
            }
            LcarsHeaderSegment(
                text = statusSegmentCode,
                color = LcarsSemanticColors.staticStateLabel(LcarsValueLevel.PLUS),
                modifier = Modifier.width(swathCodeHighWaterWidth)
            )

            val sweepFooterText = when (uiState.acquisitionStage) {
                AcquisitionStage.PATROL -> String.format(
                    Locale.US,
                    "SWEEP %.1f MS • %s (G1 %.1f | TOPK %.1f | G2 %.1f)",
                    uiState.rollingLatencyMs,
                    uiState.detectorBackend.label,
                    uiState.part1LatencyMs,
                    uiState.bridgeLatencyMs,
                    uiState.part2LatencyMs
                )
                AcquisitionStage.LOCK_ON_NEURAL -> String.format(
                    Locale.US,
                    "LOCK B=1 %.1f MS • %dX%d W8-DYN (G1 %.1f | G2 %.1f)",
                    uiState.rollingLatencyMs,
                    uiState.lockOnHudState.cropSizePx,
                    uiState.lockOnHudState.cropSizePx,
                    uiState.part1LatencyMs,
                    uiState.part2LatencyMs
                )
                AcquisitionStage.PINPOINT_CONSTELLATION -> {
                    val hud = uiState.lockOnHudState
                    val neuralB1Ms = uiState.part1LatencyMs + uiState.bridgeLatencyMs + uiState.part2LatencyMs
                    if (hud.akazeInliers >= 12) {
                        String.format(
                            Locale.US,
                            "PINPOINT 30HZ AKAZE (%d PTS) • 1HZ B=1 %.1f MS",
                            hud.akazeInliers,
                            if (neuralB1Ms > 0f) neuralB1Ms else uiState.rollingLatencyMs
                        )
                    } else {
                        String.format(
                            Locale.US,
                            "RE-ACQUIRING B=1 %.1f MS • %dX%d WIDE SEARCH",
                            if (neuralB1Ms > 0f) neuralB1Ms else uiState.rollingLatencyMs,
                            hud.cropSizePx,
                            hud.cropSizePx
                        )
                    }
                }
            }
            LcarsHeaderSegment(
                text = sweepFooterText,
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
    }
}

/**
 * Right-hand `168.dp` LCARS tactical pod rendered alongside the full-height `16:9` camera feed in
 * `LOCK_ON_NEURAL` and `PINPOINT_CONSTELLATION`.
 *
 * Occupies the natural `20:9` horizontal pillarbox zone so zero vertical pixels are stolen from the
 * `16:9` video feed, pairing non-redundant, explicitly labeled 1:1 color-correlated LCARS sensor
 * gauges (`MODEL CONF`, `3D ANGLES`, and `TEXTURE PTS` / `HOLD TIMER`) with the **sole interactive
 * actuator on screen**: the prominent Mars Red `BREAK LOCK` button.
 */
@Composable
private fun LockOnSideConsole(
    uiState: ScannerUiState,
    onBreakLock: () -> Unit,
    modifier: Modifier = Modifier
) {
    val hud = uiState.lockOnHudState
    val isPinpoint = uiState.acquisitionStage == AcquisitionStage.PINPOINT_CONSTELLATION
    val isAkazeLocked = isPinpoint && hud.isTargetInFrame && hud.akazeInliers >= 12

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
                text = "SENSOR LOCK",
                style = LcarsTypography.MicroCode,
                color = LcarsSemanticColors.structuralChrome(LcarsValueLevel.PLUS)
            )
            Text(
                text = when {
                    hud.isLostTrackGraceActive -> "HOLDING"
                    isAkazeLocked -> "PINPOINT"
                    isPinpoint -> "SEARCHING"
                    else -> "HOMING"
                },
                style = LcarsTypography.MicroCode,
                color = when {
                    hud.isLostTrackGraceActive -> LcarsColors.GoldenAmber
                    isPinpoint && !isAkazeLocked -> LcarsColors.IceBlue
                    else -> LcarsSemanticColors.staticStateLabel(LcarsValueLevel.BASE)
                }
            )
        }

        // Passive Mineral Signature Block (mineral name + physical diameter, zero top-bar duplication)
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(46.dp)
                .background(LcarsSemanticColors.staticStateLabel(LcarsValueLevel.MINUS))
                .padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            LcarsDetectionReticleSwatch(
                reticleColor = Color(hud.stoneClass.primaryHexColor),
                enabled = true
            )
            Spacer(modifier = Modifier.width(6.dp))
            Column(
                modifier = Modifier.weight(1f),
                horizontalAlignment = Alignment.End,
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    text = hud.stoneClass.lcarsShortName.uppercase(Locale.US),
                    style = LcarsTypography.TelemetryBar,
                    color = LcarsColors.CanvasBlack,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = String.format(
                        Locale.US,
                        "EST. SIZE %.0f MM",
                        hud.estimatedDiameterMm
                    ),
                    style = LcarsTypography.MicroCode,
                    color = LcarsColors.CanvasBlack,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }

        // 3 Explicitly Labeled, 1:1 Color-Correlated 8-Segment LCARS Sensor Gauges
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .border(
                    width = 2.dp,
                    color = LcarsSemanticColors.structuralChrome(LcarsValueLevel.MINUS),
                    shape = RoundedCornerShape(8.dp)
                )
                .padding(horizontal = 8.dp, vertical = 6.dp),
            verticalArrangement = Arrangement.SpaceEvenly
        ) {
            val confPct = (hud.smoothedConfidence * 100f).roundToInt().coerceIn(0, 99)
            LockOnTelemetryGaugeRow(
                label = "MODEL CONF",
                valueText = String.format(Locale.US, "%d%%", confPct),
                fraction = hud.smoothedConfidence.coerceIn(0f, 1f),
                color = LcarsColors.GoldenAmber
            )

            val clampedAngles = hud.parallaxAngleDeg.coerceIn(0f, 45.0f)
            val angleValueText = if (clampedAngles >= 3.0f) {
                String.format(Locale.US, "%.1f° LOCK", clampedAngles)
            } else {
                String.format(Locale.US, "%.1f° / 3.0°", clampedAngles)
            }
            LockOnTelemetryGaugeRow(
                label = "3D ANGLES",
                valueText = angleValueText,
                fraction = (hud.parallaxAngleDeg / 3.0f).coerceIn(0f, 1f),
                color = LcarsColors.IceBlue
            )

            if (hud.isLostTrackGraceActive) {
                LockOnTelemetryGaugeRow(
                    label = "HOLD TIMER",
                    valueText = String.format(Locale.US, "%.1fS / 15S", hud.lostTrackRemainingSec),
                    fraction = (hud.lostTrackRemainingSec / 15.0f).coerceIn(0f, 1f),
                    color = LcarsColors.GoldenAmber
                )
            } else if (isPinpoint) {
                LockOnTelemetryGaugeRow(
                    label = "TEXTURE PTS",
                    valueText = String.format(Locale.US, "%d PTS", hud.akazeInliers),
                    fraction = (hud.akazeInliers / 32f).coerceIn(0f, 1f),
                    color = LcarsColors.Periwinkle
                )
            } else {
                LockOnTelemetryGaugeRow(
                    label = "TEXTURE PTS",
                    valueText = "WAIT 3D",
                    fraction = 0f,
                    color = LcarsColors.InactiveLabel
                )
            }
        }

        // Sole Interactive Control in Lock Mode: Prominent Mars Red BREAK LOCK Actuator
        val breakLabel = if (hud.isLostTrackGraceActive) {
            String.format(Locale.US, "BREAK [%02.0fS]", hud.lostTrackRemainingSec)
        } else {
            "BREAK LOCK"
        }
        val breakSubLabel = if (isPinpoint) {
            "EXIT PINPOINT"
        } else {
            "RETURN TO PATROL"
        }
        LcarsPillButton(
            label = breakLabel,
            subLabel = breakSubLabel,
            prefixCode = "BRK",
            active = true,
            activeColor = LcarsSemanticColors.criticalAlert,
            endStyle = LcarsPillEndStyle.Capsule,
            onClick = onBreakLock,
            modifier = Modifier
                .fillMaxWidth()
                .height(88.dp)
        )
    }
}

/**
 * Single self-explanatory LCARS gauge row inside [LockOnSideConsole] whose left metric title,
 * right numeric readout, and 8-segment horizontal bar share the exact same semantic [color].
 */
@Composable
private fun LockOnTelemetryGaugeRow(
    label: String,
    valueText: String,
    fraction: Float,
    color: Color,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = label,
                style = LcarsTypography.MicroCode,
                color = color,
                maxLines = 1
            )
            Text(
                text = valueText,
                style = LcarsTypography.MicroCode,
                color = color,
                maxLines = 1
            )
        }
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(9.dp)
        ) {
            val segmentCount = 8
            val segGap = 2.dp.toPx()
            val segWidth = max(2f, (size.width - segGap * (segmentCount - 1)) / segmentCount)
            val clampedFrac = fraction.coerceIn(0f, 1f)
            val activeSegments = if (clampedFrac <= 0.01f) {
                0
            } else {
                (clampedFrac * segmentCount).roundToInt().coerceIn(1, segmentCount)
            }
            for (seg in 0 until segmentCount) {
                val x = seg * (segWidth + segGap)
                drawRect(
                    color = if (seg < activeSegments) {
                        color
                    } else {
                        LcarsColors.MutedSlate.copy(alpha = 0.35f)
                    },
                    topLeft = Offset(x, 0f),
                    size = Size(segWidth, size.height)
                )
            }
        }
    }
}

/**
 * Lower ~50% LCARS frame hosting the left mode/recording stem (`SCAN`/`LOG` toggle and `REC` frame
 * counter), the build-provenance bottom-left elbow (`GIT_COMMIT_HASH_6 • BUILD_DATE_ISO`), the
 * bottom telemetry footer rail (`SWEEP` latency breakdown and `PING` sonifier readout), and the
 * two-column control deck ([OpticsAndEisColumn] and [ProspectMatrixColumn]).
 */
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
    onToggleEyesUpPatrol: () -> Unit,
    onToggleCentralLock: () -> Unit,
    modifier: Modifier = Modifier,
    initialSourceDrawerOpen: Boolean = false
) {
    val isLogging = uiState.operatingMode == OperatingMode.LOG_DATA
    val crookStart = LcarsStemWidth + LcarsGutter
    val crookBottom = LcarsBarHeight + LcarsGutter

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

            val recMins = (uiState.loggedDurationSec / 60).coerceIn(0, 99)
            val recSecs = (uiState.loggedDurationSec % 60).coerceIn(0, 59)
            val recValueText = if (isLogging) {
                String.format(Locale.US, "%02d:%02d", recMins, recSecs)
            } else {
                String.format(Locale.US, "%04d", uiState.loggedFrameCount)
            }
            val recSubText = when {
                isLogging -> String.format(Locale.US, "%dF", uiState.loggedFrameCount)
                uiState.loggedDurationSec > 0 -> String.format(Locale.US, "%dS SAVED", uiState.loggedDurationSec)
                else -> "SAVED"
            }
            LcarsTelemetryBlock(
                prefixCode = "REC",
                valueText = recValueText,
                subText = recSubText,
                color = if (isLogging) {
                    LcarsSemanticColors.criticalAlert
                } else {
                    LcarsSemanticColors.staticStateLabel(LcarsValueLevel.MINUS)
                },
                textColor = if (isLogging) {
                    LcarsColors.AlertWhite
                } else {
                    LcarsColors.CanvasBlack
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(0.85f)
            )
        }

        LcarsBottomProvenanceFooter(
            uiState = uiState,
            modifier = Modifier
                .align(Alignment.BottomStart)
                .fillMaxWidth()
        )

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
                onToggleEyesUpPatrol = onToggleEyesUpPatrol,
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
                onToggleCentralLock = onToggleCentralLock,
                onSelectSource = onSelectSource,
                initialSourceDrawerOpen = initialSourceDrawerOpen,
                modifier = Modifier
                    .weight(0.53f)
                    .fillMaxHeight()
            )
        }
    }
}

/**
 * Left column (`weight(0.47f)`) of the lower tactical control deck housing optics, patrol power,
 * and stabilization controls: `2.0X CROP` (`OPT`), `1/500S` (`EXP`), `EYES-UP` (`EYE`), `EIS LOCK`
 * (`EIS`), `SYNC` (`CAL`), `5/3-TILE` (`SW`), and the 7-segment [LcarsSegmentedEvBar].
 */
@Composable
private fun OpticsAndEisColumn(
    uiState: ScannerUiState,
    onToggleTwoXCrop: (Boolean) -> Unit,
    onToggleShutterClamp: (Boolean) -> Unit,
    onToggleEyesUpPatrol: () -> Unit,
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
                text = "OPT-2X // OPTICS, DUTY & EIS",
                style = LcarsTypography.MicroCode,
                color = LcarsSemanticColors.structuralChrome(LcarsValueLevel.PLUS)
            )
            Text(
                text = when {
                    uiState.acquisitionStage != AcquisitionStage.PATROL -> "LOCK 16:9"
                    uiState.isPitchClamped && uiState.eisPitchLockEnabled -> "PITCH LIMIT"
                    uiState.patrolPowerState == PatrolPowerState.PATROL_IDLE -> "IDLE 15FPS"
                    else -> "EIS-SYNC"
                },
                style = LcarsTypography.MicroCode,
                color = if (uiState.isPitchClamped &&
                    uiState.eisPitchLockEnabled &&
                    uiState.acquisitionStage == AcquisitionStage.PATROL
                ) {
                    LcarsSemanticColors.criticalAlert
                } else {
                    LcarsSemanticColors.realtimeDataLabel(LcarsValueLevel.BASE)
                }
            )
        }

        // Row 1: Adjacent ON/OFF Toggles (stepped MINUS / PLUS / BASE in ToggleButton family)
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            LcarsPillButton(
                label = if (uiState.twoXCropEnabled) "2.0X [ON]" else "2.0X [OFF]",
                subLabel = if (uiState.twoXCropEnabled) "0.59 MM/PX" else "1.18 MM/PX",
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
                subLabel = if (uiState.shutterClampEnabled) "<= 2.0MS" else "AUTO EXP",
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

            LcarsPillButton(
                label = if (uiState.eyesUpPatrolEnabled) "EYES-UP [ON]" else "EYES-UP [OFF]",
                subLabel = if (uiState.patrolPowerState == PatrolPowerState.PATROL_IDLE && uiState.eyesUpPatrolEnabled) {
                    "IDLE • 15FPS"
                } else {
                    "5S OLED DIM"
                },
                prefixCode = "EYE",
                active = uiState.eyesUpPatrolEnabled,
                activeColor = LcarsSemanticColors.toggleButton(true, LcarsValueLevel.BASE),
                inactiveColor = LcarsSemanticColors.toggleButton(false, LcarsValueLevel.BASE),
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = onToggleEyesUpPatrol,
                modifier = Modifier
                    .weight(0.95f)
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
            val pitchLimitActive =
                uiState.acquisitionStage == AcquisitionStage.PATROL &&
                    uiState.eisPitchLockEnabled &&
                    uiState.isPitchClamped
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

/**
 * Right column (`weight(0.53f)`) of the lower tactical control deck housing the 2x2 mineral class
 * filter matrix (which smoothly transitions into [SourceSelectionDrawer] when `SRC` is tapped) and
 * the bottom row placing the central stateful **`LOCK NEAREST` / `BREAK LOCK`** button immediately
 * adjacent to the horizontal center divider with **`SRC`** in the outer right slot
 * (`.agents/rules/lcars-ui.md` §3).
 */
@Composable
private fun ProspectMatrixColumn(
    uiState: ScannerUiState,
    onToggleStoneClass: (StoneClass) -> Unit,
    onToggleCentralLock: () -> Unit,
    onSelectSource: (DetectorSourceOption) -> Unit,
    modifier: Modifier = Modifier,
    initialSourceDrawerOpen: Boolean = false
) {
    val classes = StoneClass.alertableClasses
    var isSourceDrawerOpen by rememberSaveable { mutableStateOf(initialSourceDrawerOpen) }
    val activeSource = uiState.activeSourceOption

    val visibleEnabledTargets = remember(uiState.confirmedTargets, uiState.enabledClasses) {
        uiState.confirmedTargets.count {
            it.isVisibleInStrip &&
                it.stoneClass.isUserAlertable &&
                uiState.enabledClasses.contains(it.stoneClass)
        }
    }

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

        // Bottom Row: Central Ambidextrous LOCK NEAREST / BREAK LOCK + Outer Right SRC Selector
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .weight(0.92f),
            horizontalArrangement = Arrangement.spacedBy(LcarsGutter)
        ) {
            val isLocked = uiState.acquisitionStage != AcquisitionStage.PATROL
            val lockHud = uiState.lockOnHudState
            val lockButtonActive = isLocked || visibleEnabledTargets > 0
            val lockButtonLabel = when {
                isLocked && lockHud.isLostTrackGraceActive ->
                    String.format(Locale.US, "BREAK LOCK [%02.0fS]", lockHud.lostTrackRemainingSec)
                isLocked -> "BREAK LOCK"
                else -> "LOCK NEAREST"
            }
            val lockButtonSubLabel = when {
                uiState.acquisitionStage == AcquisitionStage.PINPOINT_CONSTELLATION ->
                    "PINPOINT • TAP TO EXIT"
                isLocked -> "16:9 LOCK • TAP TO EXIT"
                visibleEnabledTargets > 0 -> "$visibleEnabledTargets IN SWATH • TAP LOCK"
                else -> "NO TARGET IN SWATH"
            }
            val lockButtonColor = if (isLocked) {
                LcarsSemanticColors.criticalAlert
            } else {
                TngLockGold
            }

            LcarsPillButton(
                label = lockButtonLabel,
                subLabel = lockButtonSubLabel,
                prefixCode = if (isLocked) "BRK" else "LCK",
                active = lockButtonActive,
                activeColor = lockButtonColor,
                inactiveColor = LcarsSemanticColors.toggleButton(false, LcarsValueLevel.BASE),
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = onToggleCentralLock,
                modifier = Modifier
                    .weight(1.18f)
                    .fillMaxHeight()
            )

            val drawerArrow = if (isSourceDrawerOpen) "▴" else "▾"
            LcarsPillButton(
                label = "${activeSource.title} $drawerArrow",
                subLabel = when {
                    uiState.isCompilingModel -> "COMPILING SHADERS..."
                    isSourceDrawerOpen -> "TAP SOURCE IN DRAWER"
                    else -> "MODEL SELECTOR"
                },
                prefixCode = "SRC",
                active = true,
                activeColor = LcarsSemanticColors.actionOrDrawerButton(
                    if (isSourceDrawerOpen) LcarsValueLevel.PLUS else LcarsValueLevel.BASE
                ),
                endStyle = LcarsPillEndStyle.Capsule,
                onClick = { isSourceDrawerOpen = !isSourceDrawerOpen },
                modifier = Modifier
                    .weight(1.27f)
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

/**
 * Capsule LCARS toggle pill for an individual alertable [StoneClass] in the 2x2 prospect matrix,
 * displaying the class [StoneClass.codePrefix], mineral color swatch, and `ACTIVE`/`MUTED` reticle
 * status.
 */
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

/**
 * Formats a Camera2 sensor exposure duration in nanoseconds as a human-readable reciprocal shutter
 * fraction string (e.g., `2_000_000L -> "1/500S"`).
 *
 * @param exposureNs Sensor exposure duration in nanoseconds; values `<= 0L` default to `"1/500S"`.
 * @return Reciprocal shutter speed string in `"1/<denom>S"` format.
 */
private fun formatShutterFraction(exposureNs: Long): String {
    if (exposureNs <= 0L) return "1/500S"
    val denom = (1_000_000_000.0 / exposureNs.toDouble()).roundToInt().coerceAtLeast(1)
    return "1/${denom}S"
}

private fun Float.finiteOr(fallback: Float): Float =
    if (isFinite()) this else fallback

private fun safeCanvasClamp(value: Float, minVal: Float, maxVal: Float): Float {
    val lo = minVal.finiteOr(0f)
    val hi = max(lo, maxVal.finiteOr(lo))
    return value.finiteOr(lo).coerceIn(lo, hi)
}

@Preview(
    name = "LCARS Geological Tricorder — Active 5-Tile Patrol",
    device = "spec:width=915dp,height=411dp,dpi=420,orientation=landscape"
)
@Composable
private fun ScannerScreenActivePreview() {
    GneissFindTheme {
        ScannerScreen(
            uiState = ScannerUiState(
                sweepHz = 7.4f,
                rollingLatencyMs = 135.2f,
                part1LatencyMs = 112.9f,
                bridgeLatencyMs = 0.7f,
                part2LatencyMs = 21.6f,
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
                        slantRangeMeters = 1.32f,
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
            onToggleAudioMute = {}
        )
    }
}

@Preview(
    name = "LCARS Geological Tricorder — Full-Screen 16:9 Pinpoint Homing",
    device = "spec:width=915dp,height=411dp,dpi=420,orientation=landscape"
)
@Composable
private fun ScannerScreenLockOnPinpointPreview() {
    GneissFindTheme {
        ScannerScreen(
            uiState = ScannerUiState(
                acquisitionStage = AcquisitionStage.PINPOINT_CONSTELLATION,
                lockOnHudState = LockOnHudState(
                    stage = AcquisitionStage.PINPOINT_CONSTELLATION,
                    lockedTrackId = 1L,
                    stoneClass = StoneClass.CHALCEDONY_AGATE,
                    smoothedConfidence = 0.96f,
                    estimatedDiameterMm = 34f,
                    slantRangeMeters = 0.64f,
                    sigma3dMm = 8f,
                    sigmaRadiusPx = 42f,
                    parallaxAngleDeg = 8.4f,
                    triangulatedViews = 6,
                    akazeInliers = 29,
                    frameLeftPx = 890f,
                    frameTopPx = 470f,
                    frameRightPx = 1030f,
                    frameBottomPx = 610f,
                    cropLeftPx = 800,
                    cropTopPx = 380,
                    cropSizePx = 320,
                    hasNeuralDetectorBox = true,
                    neuralBoxLeftPx = 898f,
                    neuralBoxTopPx = 476f,
                    neuralBoxRightPx = 1022f,
                    neuralBoxBottomPx = 604f,
                    isTargetInFrame = true,
                    trackingSourceBadge = "AKAZE"
                ),
                sweepHz = 29.8f,
                rollingLatencyMs = 2.8f,
                part1LatencyMs = 46.2f,
                bridgeLatencyMs = 0.6f,
                part2LatencyMs = 15.4f,
                imuPitchDeg = 52f,
                referencePitchDeg = 45f,
                sensorIso = 520,
                audioPitchHz = 910f,
                audioCadenceHz = 13.4f
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
            onToggleAudioMute = {}
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
                sweepHz = 7.2f,
                rollingLatencyMs = 138.4f,
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
                isAudioMuted = false
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
            onToggleAudioMute = {}
        )
    }
}
