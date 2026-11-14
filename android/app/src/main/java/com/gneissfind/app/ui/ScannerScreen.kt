package com.gneissfind.app.ui

import android.view.ViewGroup
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.VolumeOff
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.FiberManualRecord
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInRoot
import androidx.compose.ui.platform.LocalInspectionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Devices
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.gneissfind.app.inference.SyntheticTargetProfile
import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.DetectorBackend
import com.gneissfind.app.model.OperatingMode
import com.gneissfind.app.model.ScanSpeedState
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.WorldPoint3D
import com.gneissfind.app.ui.theme.AlertRedBanner
import com.gneissfind.app.ui.theme.CarnelianAmber
import com.gneissfind.app.ui.theme.CyanTripwire
import com.gneissfind.app.ui.theme.GneissFindTheme
import com.gneissfind.app.ui.theme.HudBorderSubtle
import com.gneissfind.app.ui.theme.HudTextPrimary
import com.gneissfind.app.ui.theme.HudTextSecondary
import com.gneissfind.app.ui.theme.JasperCrimson
import com.gneissfind.app.ui.theme.NephriteEmerald
import com.gneissfind.app.ui.theme.ObsidianSurface
import com.gneissfind.app.ui.theme.SlateCard
import com.gneissfind.app.ui.theme.SlateCardElevated
import java.io.File
import java.util.Locale
import kotlin.math.roundToInt

enum class ControlsSheetTab(val title: String) {
    OPTICS_AND_SPEED("Optics & Speed"),
    MODEL_AND_SIM("Model & Sim")
}

@Composable
fun ScannerScreen(
    uiState: ScannerUiState,
    hasCameraPermission: Boolean,
    onRequestCameraPermission: () -> Unit,
    onBindPreview: (androidx.lifecycle.LifecycleOwner, PreviewView) -> Unit,
    onSelectMode: (OperatingMode) -> Unit,
    onSelectBackend: (DetectorBackend) -> Unit,
    onSelectSyntheticProfile: (SyntheticTargetProfile) -> Unit,
    onToggleStoneClass: (StoneClass) -> Unit,
    onToggleShutterClamp: (Boolean) -> Unit,
    onToggleTwoXCrop: (Boolean) -> Unit,
    onChangeExposureBias: (Float) -> Unit,
    onToggleAudioMute: (Boolean) -> Unit,
    onChangeSimSpeed: (Float?) -> Unit,
    modifier: Modifier = Modifier,
    initialActiveSheetTab: ControlsSheetTab? = null
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val isInspection = LocalInspectionMode.current
    var activeSheetTab by remember { mutableStateOf(initialActiveSheetTab) }
    var topHudBottomPx by remember { mutableFloatStateOf(0f) }

    Surface(
        modifier = modifier.fillMaxSize(),
        color = ObsidianSurface,
        contentColor = HudTextPrimary
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            if (hasCameraPermission) {
                if (isInspection) {
                    // Simulated gravel-bar camera feed for Compose Previews
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(
                                Brush.verticalGradient(
                                    colors = listOf(
                                        Color(0xFF3A352F),
                                        Color(0xFF4E473E),
                                        Color(0xFF2B2927)
                                    )
                                )
                            )
                    )
                } else {
                    AndroidView(
                        modifier = Modifier.fillMaxSize(),
                        factory = { ctx ->
                            PreviewView(ctx).apply {
                                layoutParams = ViewGroup.LayoutParams(
                                    ViewGroup.LayoutParams.MATCH_PARENT,
                                    ViewGroup.LayoutParams.MATCH_PARENT
                                )
                                scaleType = PreviewView.ScaleType.FILL_CENTER
                                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                                onBindPreview(lifecycleOwner, this)
                            }
                        }
                    )
                }
            } else {
                PermissionFallbackCard(
                    onRequestCameraPermission = onRequestCameraPermission,
                    modifier = Modifier.align(Alignment.Center)
                )
            }

            // Push-Broom 1088x384 Strip Tripwire & Confirmed Target Reticle Canvas Overlay
            PushBroomTripwireOverlay(
                stripGeometry = uiState.stripGeometry,
                confirmedTargets = uiState.confirmedTargets,
                operatingMode = uiState.operatingMode,
                topHudBottomPx = topHudBottomPx,
                modifier = Modifier.fillMaxSize()
            )

            // Foreground HUD chrome respecting Android 16 edge-to-edge safe drawing insets
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .windowInsetsPadding(WindowInsets.safeDrawing)
                    .padding(horizontal = 12.dp, vertical = 6.dp),
                verticalArrangement = Arrangement.SpaceBetween
            ) {
                // COMPACT TOP HUD SECTION (kept tight so the upper-middle 1088x384 tripwire stays unobstructed)
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .onGloballyPositioned { coords ->
                            topHudBottomPx = coords.positionInRoot().y + coords.size.height
                        },
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    TopModeAndActionsBar(
                        operatingMode = uiState.operatingMode,
                        isControlsOpen = activeSheetTab != null,
                        isAudioMuted = uiState.isAudioMuted,
                        onSelectMode = onSelectMode,
                        onToggleControls = {
                            activeSheetTab = if (activeSheetTab != null) {
                                null
                            } else {
                                ControlsSheetTab.OPTICS_AND_SPEED
                            }
                        },
                        onToggleAudioMute = { onToggleAudioMute(!uiState.isAudioMuted) }
                    )

                    CompactTelemetryStrip(
                        uiState = uiState,
                        isModelSheetOpen = activeSheetTab == ControlsSheetTab.MODEL_AND_SIM,
                        onOpenModelSheet = {
                            activeSheetTab = if (activeSheetTab == ControlsSheetTab.MODEL_AND_SIM) {
                                null
                            } else {
                                ControlsSheetTab.MODEL_AND_SIM
                            }
                        }
                    )

                    // High-contrast SLOW SWEEP warning banner (> 1.1 m/s)
                    val speedWarning = uiState.speedState as? ScanSpeedState.SpeedWarning
                    AnimatedVisibility(
                        visible = speedWarning != null,
                        enter = fadeIn() + expandVertically(),
                        exit = fadeOut() + shrinkVertically()
                    ) {
                        SlowSweepBanner(
                            speedMps = speedWarning?.currentSpeedMps ?: 0f
                        )
                    }

                    AnimatedVisibility(
                        visible = uiState.operatingMode == OperatingMode.LOG_DATA,
                        enter = fadeIn() + expandVertically(),
                        exit = fadeOut() + shrinkVertically()
                    ) {
                        FieldLoggerBanner(
                            savedFrames = uiState.loggedFrameCount,
                            sessionDir = uiState.fieldLogDirectory
                        )
                    }
                }

                // BOTTOM HUD SECTION: Collapsible Tabbed Controls Sheet + 2x2 Target Class Grid
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    AnimatedVisibility(
                        visible = activeSheetTab != null,
                        enter = fadeIn() + expandVertically(expandFrom = Alignment.Bottom),
                        exit = fadeOut() + shrinkVertically(shrinkTowards = Alignment.Bottom)
                    ) {
                        val currentTab = activeSheetTab ?: ControlsSheetTab.OPTICS_AND_SPEED
                        ScannerControlsSheet(
                            selectedTab = currentTab,
                            uiState = uiState,
                            onSelectTab = { activeSheetTab = it },
                            onDismiss = { activeSheetTab = null },
                            onToggleShutterClamp = onToggleShutterClamp,
                            onToggleTwoXCrop = onToggleTwoXCrop,
                            onChangeExposureBias = onChangeExposureBias,
                            onChangeSimSpeed = onChangeSimSpeed,
                            onSelectBackend = onSelectBackend,
                            onSelectSyntheticProfile = onSelectSyntheticProfile
                        )
                    }

                    TargetClassGridCard(
                        enabledClasses = uiState.enabledClasses,
                        onToggleStoneClass = onToggleStoneClass
                    )
                }
            }
        }
    }
}

@Composable
private fun HudPanel(
    modifier: Modifier = Modifier,
    backgroundColor: Color = SlateCard,
    borderColor: Color = HudBorderSubtle,
    cornerRadius: Dp = 14.dp,
    content: @Composable () -> Unit
) {
    Surface(
        modifier = modifier,
        color = backgroundColor,
        contentColor = HudTextPrimary,
        shape = RoundedCornerShape(cornerRadius),
        border = BorderStroke(1.dp, borderColor),
        content = content
    )
}

@Composable
private fun TopModeAndActionsBar(
    operatingMode: OperatingMode,
    isControlsOpen: Boolean,
    isAudioMuted: Boolean,
    onSelectMode: (OperatingMode) -> Unit,
    onToggleControls: () -> Unit,
    onToggleAudioMute: () -> Unit
) {
    HudPanel(
        modifier = Modifier.fillMaxWidth(),
        cornerRadius = 14.dp
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // High-affordance SCAN / LOG DATA pill toggle
            Row(
                modifier = Modifier
                    .weight(1f)
                    .height(42.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(Color(0xFF070D14))
                    .border(1.dp, HudBorderSubtle, RoundedCornerShape(10.dp))
                    .padding(3.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                val scanSelected = operatingMode == OperatingMode.SCAN
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxSize()
                        .clip(RoundedCornerShape(8.dp))
                        .background(
                            if (scanSelected) CyanTripwire.copy(alpha = 0.24f) else Color.Transparent
                        )
                        .then(
                            if (scanSelected) {
                                Modifier.border(1.dp, CyanTripwire, RoundedCornerShape(8.dp))
                            } else {
                                Modifier
                            }
                        )
                        .clickable(role = Role.Tab) { onSelectMode(OperatingMode.SCAN) },
                    contentAlignment = Alignment.Center
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        if (scanSelected) {
                            Icon(
                                imageVector = Icons.Default.Check,
                                contentDescription = null,
                                tint = CyanTripwire,
                                modifier = Modifier.size(15.dp)
                            )
                            Spacer(Modifier.width(4.dp))
                        }
                        Text(
                            text = "SCAN",
                            color = if (scanSelected) HudTextPrimary else HudTextSecondary,
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp,
                            maxLines = 1
                        )
                    }
                }

                val logSelected = operatingMode == OperatingMode.LOG_DATA
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxSize()
                        .clip(RoundedCornerShape(8.dp))
                        .background(
                            if (logSelected) NephriteEmerald.copy(alpha = 0.22f) else Color.Transparent
                        )
                        .then(
                            if (logSelected) {
                                Modifier.border(1.dp, NephriteEmerald, RoundedCornerShape(8.dp))
                            } else {
                                Modifier
                            }
                        )
                        .clickable(role = Role.Tab) { onSelectMode(OperatingMode.LOG_DATA) },
                    contentAlignment = Alignment.Center
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.FiberManualRecord,
                            contentDescription = null,
                            tint = if (logSelected) AlertRedBanner else HudTextSecondary,
                            modifier = Modifier.size(12.dp)
                        )
                        Spacer(Modifier.width(5.dp))
                        Text(
                            text = "LOG DATA",
                            color = if (logSelected) HudTextPrimary else HudTextSecondary,
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp,
                            maxLines = 1
                        )
                    }
                }
            }

            // Afforded Controls Drawer button (Optics, Speed Gate, Backend & Sim Presets)
            Surface(
                onClick = onToggleControls,
                color = if (isControlsOpen) CyanTripwire.copy(alpha = 0.24f) else Color(0xFF14212E),
                contentColor = if (isControlsOpen) CyanTripwire else HudTextPrimary,
                shape = RoundedCornerShape(10.dp),
                border = BorderStroke(
                    width = 1.dp,
                    color = if (isControlsOpen) CyanTripwire else HudBorderSubtle
                ),
                modifier = Modifier.height(42.dp)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Tune,
                        contentDescription = "Toggle scanner controls",
                        modifier = Modifier.size(18.dp)
                    )
                    Text(
                        text = "Setup",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.SemiBold,
                        maxLines = 1
                    )
                }
            }

            // Afforded Audio Mute / Unmute button
            Surface(
                onClick = onToggleAudioMute,
                color = if (isAudioMuted) JasperCrimson.copy(alpha = 0.18f) else CarnelianAmber.copy(alpha = 0.16f),
                contentColor = if (isAudioMuted) JasperCrimson else CarnelianAmber,
                shape = RoundedCornerShape(10.dp),
                border = BorderStroke(
                    width = 1.dp,
                    color = if (isAudioMuted) JasperCrimson.copy(alpha = 0.7f) else CarnelianAmber.copy(alpha = 0.65f)
                ),
                modifier = Modifier.size(42.dp)
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(
                        imageVector = if (isAudioMuted) {
                            Icons.AutoMirrored.Filled.VolumeOff
                        } else {
                            Icons.AutoMirrored.Filled.VolumeUp
                        },
                        contentDescription = if (isAudioMuted) "Unmute sonifier" else "Mute sonifier",
                        modifier = Modifier.size(20.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun CompactTelemetryStrip(
    uiState: ScannerUiState,
    isModelSheetOpen: Boolean,
    onOpenModelSheet: () -> Unit
) {
    val speedMps = when (val s = uiState.speedState) {
        is ScanSpeedState.Normal -> s.currentSpeedMps
        is ScanSpeedState.SpeedWarning -> s.currentSpeedMps
    }
    val shutterFraction = if (uiState.exposureTimeNs > 0L) {
        "1/${(1_000_000_000L / uiState.exposureTimeNs).coerceAtLeast(1)}s"
    } else {
        "≤1/500s"
    }

    val modelBadgeLabel = when (uiState.detectorBackend) {
        DetectorBackend.SYNTHETIC_DEMO -> "SIM: ${uiState.syntheticProfile.label}"
        DetectorBackend.RF_DETR_NANO_GPU -> String.format(
            Locale.US,
            "GPU FP16 • %.1f ms",
            uiState.rollingLatencyMs
        )
    }

    HudPanel(
        modifier = Modifier.fillMaxWidth(),
        cornerRadius = 12.dp
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 6.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            // Row 1: Tappable Active Backend/Preset pill (left) + Non-wrapping Optics readout (right)
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Row(
                    modifier = Modifier.weight(1f),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Surface(
                        onClick = onOpenModelSheet,
                        color = if (isModelSheetOpen) {
                            CyanTripwire.copy(alpha = 0.24f)
                        } else {
                            CyanTripwire.copy(alpha = 0.12f)
                        },
                        contentColor = CyanTripwire,
                        shape = RoundedCornerShape(6.dp),
                        border = BorderStroke(1.dp, CyanTripwire.copy(alpha = 0.45f))
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 7.dp, vertical = 2.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                imageVector = Icons.Default.Memory,
                                contentDescription = null,
                                modifier = Modifier.size(12.dp)
                            )
                            Spacer(Modifier.width(4.dp))
                            Text(
                                text = modelBadgeLabel,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.SemiBold,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.weight(1f, fill = false)
                            )
                            Icon(
                                imageVector = Icons.Default.KeyboardArrowDown,
                                contentDescription = "Change model or simulation preset",
                                modifier = Modifier.size(14.dp)
                            )
                        }
                    }
                }

                Text(
                    text = "${uiState.activeZoomRatio.roundToInt()}× • $shutterFraction • ISO ${uiState.sensorIso}",
                    fontFamily = FontFamily.Monospace,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Medium,
                    color = HudTextPrimary,
                    maxLines = 1,
                    softWrap = false
                )
            }

            // Row 2: Pose/Geometry metrics (left) + Live Sonifier Pitch/Cadence badge (right)
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = String.format(
                        Locale.US,
                        "Pitch %.0f° • R %.2fm • %.2fmm/px • %.2fm/s",
                        uiState.imuPitchDeg,
                        uiState.slantRangeMeters,
                        uiState.gsdMmPerPx,
                        speedMps
                    ),
                    fontFamily = FontFamily.Monospace,
                    fontSize = 11.sp,
                    color = HudTextSecondary,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f)
                )

                val audioStatusText = when {
                    uiState.isAudioMuted -> "MUTED"
                    uiState.audioPitchHz > 0f -> String.format(
                        Locale.US,
                        "%.0fHz • %.1f/s",
                        uiState.audioPitchHz,
                        uiState.audioCadenceHz
                    )
                    else -> "IDLE"
                }
                val audioStatusColor = when {
                    uiState.isAudioMuted -> JasperCrimson
                    uiState.audioPitchHz > 0f -> CarnelianAmber
                    else -> HudTextSecondary
                }

                Text(
                    text = audioStatusText,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = audioStatusColor,
                    maxLines = 1,
                    softWrap = false
                )
            }
        }
    }
}

@Composable
private fun SlowSweepBanner(speedMps: Float) {
    Surface(
        color = AlertRedBanner,
        contentColor = Color.White,
        shape = RoundedCornerShape(12.dp),
        border = BorderStroke(1.5.dp, Color.White),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .padding(horizontal = 12.dp, vertical = 8.dp)
                .semantics(mergeDescendants = true) {},
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center
        ) {
            Icon(
                imageVector = Icons.Default.Warning,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(20.dp)
            )
            Spacer(Modifier.width(8.dp))
            Text(
                text = String.format(Locale.US, "SLOW SWEEP — %.2f m/s (> 1.10 m/s)", speedMps),
                color = Color.White,
                fontWeight = FontWeight.ExtraBold,
                fontSize = 14.sp,
                maxLines = 1
            )
        }
    }
}

@Composable
private fun FieldLoggerBanner(
    savedFrames: Int,
    sessionDir: String
) {
    val shortSessionName = remember(sessionDir) {
        sessionDir.trimEnd('/').substringAfterLast(File.separatorChar, sessionDir)
    }
    Surface(
        color = Color(0xEB0E2419),
        contentColor = HudTextPrimary,
        shape = RoundedCornerShape(10.dp),
        border = BorderStroke(1.dp, NephriteEmerald),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = "REC 1088×384 PNG+JSONL • $savedFrames saved",
                color = Color(0xFF69F0AE),
                fontWeight = FontWeight.Bold,
                fontSize = 12.sp,
                maxLines = 1
            )
            Spacer(Modifier.width(8.dp))
            Text(
                text = shortSessionName,
                color = HudTextSecondary,
                fontFamily = FontFamily.Monospace,
                fontSize = 10.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
private fun PushBroomTripwireOverlay(
    stripGeometry: StripRoiGeometry,
    confirmedTargets: List<ConfirmedTarget>,
    operatingMode: OperatingMode,
    topHudBottomPx: Float,
    modifier: Modifier = Modifier
) {
    val textMeasurer = rememberTextMeasurer()
    val borderColor = if (operatingMode == OperatingMode.LOG_DATA) {
        NephriteEmerald
    } else {
        CyanTripwire
    }

    Canvas(modifier = modifier) {
        val canvasW = size.width
        val canvasH = size.height
        if (canvasW <= 0f || canvasH <= 0f) return@Canvas

        // Map upright camera frame coordinates onto FILL_CENTER viewfinder canvas
        val frameAspect = stripGeometry.uprightFrameWidth.toFloat() /
            stripGeometry.uprightFrameHeight.coerceAtLeast(1).toFloat()
        val canvasAspect = canvasW / canvasH

        val renderedW: Float
        val renderedH: Float
        val offsetX: Float
        val offsetY: Float

        if (canvasAspect > frameAspect) {
            renderedW = canvasW
            renderedH = canvasW / frameAspect
            offsetX = 0f
            offsetY = (canvasH - renderedH) * 0.5f
        } else {
            renderedH = canvasH
            renderedW = canvasH * frameAspect
            offsetX = (canvasW - renderedW) * 0.5f
            offsetY = 0f
        }

        val boxLeft = (offsetX + stripGeometry.normalizedLeft * renderedW).coerceIn(16f, canvasW - 32f)
        val boxWidth = (stripGeometry.normalizedWidth * renderedW).coerceAtMost(canvasW - boxLeft - 16f)
        val boxHeight = (boxWidth * (StripRoiGeometry.STRIP_HEIGHT_PX.toFloat() / stripGeometry.stripWidthPx.coerceAtLeast(1)))
            .coerceAtLeast(76f)

        // Ensure tripwire header & box never collide with the measured Top HUD bottom edge
        val minClearTopPx = if (topHudBottomPx > 0f) {
            (topHudBottomPx + 30.dp.toPx()).coerceAtMost(canvasH * 0.52f)
        } else {
            canvasH * 0.24f
        }
        val rawBoxTop = offsetY + stripGeometry.normalizedTop * renderedH
        val boxTop = rawBoxTop.coerceIn(minClearTopPx, (canvasH * 0.62f).coerceAtLeast(minClearTopPx))

        // Subtle darkened mask outside the active 1088x384 push-broom strip
        drawRect(
            color = Color(0x40000000),
            topLeft = Offset(0f, 0f),
            size = Size(canvasW, boxTop)
        )
        drawRect(
            color = Color(0x40000000),
            topLeft = Offset(0f, boxTop + boxHeight),
            size = Size(canvasW, (canvasH - (boxTop + boxHeight)).coerceAtLeast(0f))
        )

        // Outer dark contrast halo + crisp Push-Broom Tripwire Box
        drawRoundRect(
            color = Color(0x88000000),
            topLeft = Offset(boxLeft, boxTop),
            size = Size(boxWidth, boxHeight),
            cornerRadius = CornerRadius(10.dp.toPx(), 10.dp.toPx()),
            style = Stroke(width = 6f)
        )
        drawRoundRect(
            color = borderColor,
            topLeft = Offset(boxLeft, boxTop),
            size = Size(boxWidth, boxHeight),
            cornerRadius = CornerRadius(10.dp.toPx(), 10.dp.toPx()),
            style = Stroke(width = 3.5f)
        )

        // Draw internal 3-tile overlap seam ticks (x = 352, 384, 704, 736 in 1088px space)
        val seamDash = PathEffect.dashPathEffect(floatArrayOf(8f, 8f), 0f)
        val seamPositionsPx = floatArrayOf(352f, 384f, 704f, 736f)
        for (seamPx in seamPositionsPx) {
            val sx = boxLeft + (seamPx / stripGeometry.stripWidthPx.coerceAtLeast(1)) * boxWidth
            drawLine(
                color = borderColor.copy(alpha = 0.38f),
                start = Offset(sx, boxTop),
                end = Offset(sx, boxTop + boxHeight),
                strokeWidth = 1.5f,
                pathEffect = seamDash
            )
        }

        // High-contrast Tripwire Header Pill anchored cleanly above the box
        val tripwireLabel = textMeasurer.measure(
            text = "PUSH-BROOM 1088×384 • 3×384 TILES",
            style = TextStyle(
                color = borderColor,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace
            )
        )
        val headerPadX = 6.dp.toPx()
        val headerPadY = 2.dp.toPx()
        val headerPillW = tripwireLabel.size.width + headerPadX * 2
        val headerPillH = tripwireLabel.size.height + headerPadY * 2
        val headerTop = (boxTop - headerPillH - 5.dp.toPx()).coerceAtLeast(4f)

        drawRoundRect(
            color = Color(0xDE081018),
            topLeft = Offset(boxLeft, headerTop),
            size = Size(headerPillW, headerPillH),
            cornerRadius = CornerRadius(6.dp.toPx(), 6.dp.toPx())
        )
        drawRoundRect(
            color = borderColor.copy(alpha = 0.55f),
            topLeft = Offset(boxLeft, headerTop),
            size = Size(headerPillW, headerPillH),
            cornerRadius = CornerRadius(6.dp.toPx(), 6.dp.toPx()),
            style = Stroke(width = 1.5f)
        )
        drawText(
            textLayoutResult = tripwireLabel,
            topLeft = Offset(boxLeft + headerPadX, headerTop + headerPadY)
        )

        // Render world-locked ConfirmedTarget reticles (strictly excluding Class 4 hard_confuser)
        val scaleX = boxWidth / stripGeometry.stripWidthPx.coerceAtLeast(1).toFloat()
        val scaleY = boxHeight / StripRoiGeometry.STRIP_HEIGHT_PX.toFloat()

        for (target in confirmedTargets) {
            if (!target.stoneClass.isUserAlertable) continue

            val primaryColor = Color(target.stoneClass.primaryHexColor)
            val accentColor = Color(target.stoneClass.accentHexColor)

            val rx = boxLeft + target.stripLeftPx * scaleX
            val ry = boxTop + target.stripTopPx * scaleY
            val rw = ((target.stripRightPx - target.stripLeftPx) * scaleX).coerceAtLeast(24f)
            val rh = ((target.stripBottomPx - target.stripTopPx) * scaleY).coerceAtLeast(24f)

            // Dark outer halo + vibrant class reticle
            drawRoundRect(
                color = Color(0xAA000000),
                topLeft = Offset(rx, ry),
                size = Size(rw, rh),
                cornerRadius = CornerRadius(6.dp.toPx(), 6.dp.toPx()),
                style = Stroke(width = 6.5f)
            )
            drawRoundRect(
                color = primaryColor,
                topLeft = Offset(rx, ry),
                size = Size(rw, rh),
                cornerRadius = CornerRadius(6.dp.toPx(), 6.dp.toPx()),
                style = Stroke(width = 3.5f)
            )

            // Center reticle crosshair
            val cx = rx + rw * 0.5f
            val cy = ry + rh * 0.5f
            drawCircle(
                color = Color(0xAA000000),
                radius = 6f,
                center = Offset(cx, cy)
            )
            drawCircle(
                color = accentColor,
                radius = 4f,
                center = Offset(cx, cy)
            )

            // High-contrast dark obsidian badge with bright white text + colored border (never dark-on-dark!)
            val tagText = String.format(
                Locale.US,
                "%s %.0fmm [%d/%d]",
                target.stoneClass.displayName,
                target.estimatedDiameterMm,
                target.hitCountInWindow,
                target.windowSize
            )
            val measuredTag = textMeasurer.measure(
                text = tagText,
                style = TextStyle(
                    color = HudTextPrimary,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold
                )
            )
            val dotSpace = 14.dp.toPx()
            val padX = 7.dp.toPx()
            val padY = 3.dp.toPx()
            val pillW = dotSpace + measuredTag.size.width + padX * 2
            val pillH = measuredTag.size.height + padY * 2

            val tagLeft = rx.coerceIn(
                boxLeft + 4f,
                (boxLeft + boxWidth - pillW - 4f).coerceAtLeast(boxLeft + 4f)
            )
            val tagTop = if (ry - pillH - 6f >= boxTop + 4f) {
                ry - pillH - 6f
            } else {
                (ry + rh + 6f).coerceAtMost(boxTop + boxHeight - pillH - 4f)
            }

            drawRoundRect(
                color = Color(0xF00A121B),
                topLeft = Offset(tagLeft, tagTop),
                size = Size(pillW, pillH),
                cornerRadius = CornerRadius(6.dp.toPx(), 6.dp.toPx())
            )
            drawRoundRect(
                color = primaryColor,
                topLeft = Offset(tagLeft, tagTop),
                size = Size(pillW, pillH),
                cornerRadius = CornerRadius(6.dp.toPx(), 6.dp.toPx()),
                style = Stroke(width = 2f)
            )
            drawCircle(
                color = primaryColor,
                radius = 4.dp.toPx(),
                center = Offset(tagLeft + padX + 4.dp.toPx(), tagTop + pillH * 0.5f)
            )
            drawText(
                textLayoutResult = measuredTag,
                topLeft = Offset(tagLeft + padX + dotSpace, tagTop + padY)
            )
        }
    }
}

@Composable
private fun ScannerControlsSheet(
    selectedTab: ControlsSheetTab,
    uiState: ScannerUiState,
    onSelectTab: (ControlsSheetTab) -> Unit,
    onDismiss: () -> Unit,
    onToggleShutterClamp: (Boolean) -> Unit,
    onToggleTwoXCrop: (Boolean) -> Unit,
    onChangeExposureBias: (Float) -> Unit,
    onChangeSimSpeed: (Float?) -> Unit,
    onSelectBackend: (DetectorBackend) -> Unit,
    onSelectSyntheticProfile: (SyntheticTargetProfile) -> Unit
) {
    HudPanel(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(max = 345.dp),
        backgroundColor = SlateCardElevated,
        borderColor = CyanTripwire.copy(alpha = 0.38f),
        cornerRadius = 16.dp
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Sheet Header: Tab Switcher + Explicit Close Button
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Row(
                    modifier = Modifier
                        .weight(1f)
                        .height(40.dp)
                        .clip(RoundedCornerShape(10.dp))
                        .background(Color(0xFF081018))
                        .border(1.dp, HudBorderSubtle, RoundedCornerShape(10.dp))
                        .padding(3.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    ControlsSheetTab.entries.forEach { tab ->
                        val selected = tab == selectedTab
                        Surface(
                            onClick = { onSelectTab(tab) },
                            color = if (selected) CyanTripwire.copy(alpha = 0.22f) else Color.Transparent,
                            contentColor = if (selected) CyanTripwire else HudTextSecondary,
                            shape = RoundedCornerShape(7.dp),
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxSize()
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Text(
                                    text = tab.title,
                                    color = if (selected) CyanTripwire else HudTextSecondary,
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.SemiBold,
                                    maxLines = 1
                                )
                            }
                        }
                    }
                }

                IconButton(
                    onClick = onDismiss,
                    modifier = Modifier
                        .size(40.dp)
                        .clip(RoundedCornerShape(10.dp))
                        .background(Color(0xFF172635))
                        .border(1.dp, HudBorderSubtle, RoundedCornerShape(10.dp))
                ) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "Close scanner controls",
                        tint = HudTextPrimary,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 4.dp, vertical = 2.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                when (selectedTab) {
                    ControlsSheetTab.OPTICS_AND_SPEED -> {
                        OpticsAndSpeedTabContent(
                            shutterClampEnabled = uiState.shutterClampEnabled,
                            twoXCropEnabled = uiState.twoXCropEnabled,
                            exposureBiasEv = uiState.exposureBiasEv,
                            simulatedSpeedMps = uiState.simulatedSpeedOverrideMps,
                            onToggleShutterClamp = onToggleShutterClamp,
                            onToggleTwoXCrop = onToggleTwoXCrop,
                            onChangeExposureBias = onChangeExposureBias,
                            onChangeSimSpeed = onChangeSimSpeed
                        )
                    }

                    ControlsSheetTab.MODEL_AND_SIM -> {
                        ModelAndSimTabContent(
                            uiState = uiState,
                            onSelectBackend = onSelectBackend,
                            onSelectSyntheticProfile = onSelectSyntheticProfile
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun OpticsAndSpeedTabContent(
    shutterClampEnabled: Boolean,
    twoXCropEnabled: Boolean,
    exposureBiasEv: Float,
    simulatedSpeedMps: Float?,
    onToggleShutterClamp: (Boolean) -> Unit,
    onToggleTwoXCrop: (Boolean) -> Unit,
    onChangeExposureBias: (Float) -> Unit,
    onChangeSimSpeed: (Float?) -> Unit
) {
    val switchColors = SwitchDefaults.colors(
        checkedThumbColor = ObsidianSurface,
        checkedTrackColor = CyanTripwire,
        uncheckedThumbColor = HudTextSecondary,
        uncheckedTrackColor = Color(0xFF1B2B3A),
        uncheckedBorderColor = HudBorderSubtle
    )

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .toggleable(
                value = shutterClampEnabled,
                role = Role.Switch,
                onValueChange = onToggleShutterClamp
            )
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f).padding(end = 10.dp)) {
            Text(
                text = "Shutter Clamp (≤ 1/500 s)",
                color = HudTextPrimary,
                fontWeight = FontWeight.SemiBold,
                fontSize = 13.sp
            )
            Text(
                text = "Locks ≤ 2,000,000 ns exposure with floating ISO",
                color = HudTextSecondary,
                fontSize = 11.sp
            )
        }
        Switch(
            checked = shutterClampEnabled,
            onCheckedChange = null,
            colors = switchColors
        )
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 40.dp)
            .clip(RoundedCornerShape(8.dp))
            .toggleable(
                value = twoXCropEnabled,
                role = Role.Switch,
                onValueChange = onToggleTwoXCrop
            )
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "2.0× Lossless Center Sensor Crop",
            color = HudTextPrimary,
            fontWeight = FontWeight.SemiBold,
            fontSize = 13.sp,
            modifier = Modifier.weight(1f).padding(end = 10.dp)
        )
        Switch(
            checked = twoXCropEnabled,
            onCheckedChange = null,
            colors = switchColors
        )
    }

    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Canopy Exposure Bias",
                color = HudTextPrimary,
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium
            )
            Text(
                text = String.format(Locale.US, "%+.1f EV", exposureBiasEv),
                color = CyanTripwire,
                fontFamily = FontFamily.Monospace,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold
            )
        }
        Slider(
            value = exposureBiasEv,
            onValueChange = onChangeExposureBias,
            valueRange = -2.0f..2.0f,
            steps = 7,
            modifier = Modifier.padding(horizontal = 4.dp),
            colors = SliderDefaults.colors(
                thumbColor = CyanTripwire,
                activeTrackColor = CyanTripwire,
                inactiveTrackColor = Color(0xFF223446)
            )
        )
    }

    // Desk-test Walk Speed Override for verifying the 1.1 m/s SLOW SWEEP banner & buzz indoors
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.weight(1f).padding(end = 8.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Speed,
                    contentDescription = null,
                    tint = CyanTripwire,
                    modifier = Modifier.size(16.dp)
                )
                Spacer(Modifier.width(6.dp))
                Text(
                    text = if (simulatedSpeedMps == null) {
                        "Speed Gate: Live IMU / ARCore"
                    } else {
                        String.format(Locale.US, "Desk Test Speed: %.2f m/s", simulatedSpeedMps)
                    },
                    color = HudTextPrimary,
                    fontSize = 12.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }

            Surface(
                onClick = { onChangeSimSpeed(if (simulatedSpeedMps == null) 1.35f else null) },
                color = if (simulatedSpeedMps != null) {
                    CarnelianAmber.copy(alpha = 0.20f)
                } else {
                    CyanTripwire.copy(alpha = 0.14f)
                },
                contentColor = if (simulatedSpeedMps != null) CarnelianAmber else CyanTripwire,
                shape = RoundedCornerShape(8.dp),
                border = BorderStroke(
                    1.dp,
                    if (simulatedSpeedMps != null) CarnelianAmber else CyanTripwire.copy(alpha = 0.6f)
                )
            ) {
                Text(
                    text = if (simulatedSpeedMps == null) "Test Gate" else "Use Live IMU",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                )
            }
        }
        if (simulatedSpeedMps != null) {
            Slider(
                value = simulatedSpeedMps,
                onValueChange = { onChangeSimSpeed(it) },
                valueRange = 0.0f..2.0f,
                modifier = Modifier.padding(horizontal = 4.dp),
                colors = SliderDefaults.colors(
                    thumbColor = CarnelianAmber,
                    activeTrackColor = CarnelianAmber,
                    inactiveTrackColor = Color(0xFF223446)
                )
            )
        }
    }
}

@Composable
private fun ModelAndSimTabContent(
    uiState: ScannerUiState,
    onSelectBackend: (DetectorBackend) -> Unit,
    onSelectSyntheticProfile: (SyntheticTargetProfile) -> Unit
) {
    Text(
        text = "On-Device StoneDetector Engine",
        color = HudTextSecondary,
        fontSize = 11.sp,
        fontWeight = FontWeight.SemiBold
    )

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        DetectorBackend.entries.forEach { backend ->
            val selected = backend == uiState.detectorBackend
            Surface(
                onClick = { onSelectBackend(backend) },
                color = if (selected) CyanTripwire.copy(alpha = 0.20f) else Color(0xFF13202D),
                contentColor = if (selected) HudTextPrimary else HudTextSecondary,
                shape = RoundedCornerShape(10.dp),
                border = BorderStroke(
                    width = if (selected) 1.5.dp else 1.dp,
                    color = if (selected) CyanTripwire else HudBorderSubtle
                ),
                modifier = Modifier
                    .weight(1f)
                    .heightIn(min = 42.dp)
            ) {
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 8.dp)
                ) {
                    Text(
                        text = backend.label,
                        fontSize = 12.sp,
                        fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }
    }

    Text(
        text = String.format(
            Locale.US,
            "%s • Sweep %.1f ms (P1 %.1f | TopK %.1f | P2 %.1f)",
            uiState.detectorStatus,
            uiState.rollingLatencyMs,
            uiState.part1LatencyMs,
            uiState.bridgeLatencyMs,
            uiState.part2LatencyMs
        ),
        color = HudTextSecondary,
        fontFamily = FontFamily.Monospace,
        fontSize = 10.sp,
        maxLines = 2
    )

    if (uiState.detectorBackend == DetectorBackend.SYNTHETIC_DEMO) {
        Spacer(Modifier.height(2.dp))
        Text(
            text = "Simulation Target Preset (All 4 Visible)",
            color = HudTextSecondary,
            fontSize = 11.sp,
            fontWeight = FontWeight.SemiBold
        )

        val presets = SyntheticTargetProfile.entries
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            presets.chunked(2).forEach { rowItems ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    rowItems.forEach { preset ->
                        val selected = preset == uiState.syntheticProfile
                        Surface(
                            onClick = { onSelectSyntheticProfile(preset) },
                            color = if (selected) {
                                CarnelianAmber.copy(alpha = 0.22f)
                            } else {
                                Color(0xFF13202D)
                            },
                            contentColor = if (selected) HudTextPrimary else HudTextSecondary,
                            shape = RoundedCornerShape(9.dp),
                            border = BorderStroke(
                                width = if (selected) 1.5.dp else 1.dp,
                                color = if (selected) CarnelianAmber else HudBorderSubtle
                            ),
                            modifier = Modifier
                                .weight(1f)
                                .heightIn(min = 42.dp)
                        ) {
                            Box(
                                contentAlignment = Alignment.CenterStart,
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp)
                            ) {
                                Text(
                                    text = preset.label,
                                    color = if (selected) CarnelianAmber else HudTextPrimary,
                                    fontSize = 11.sp,
                                    fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun TargetClassGridCard(
    enabledClasses: Set<StoneClass>,
    onToggleStoneClass: (StoneClass) -> Unit
) {
    HudPanel(
        modifier = Modifier.fillMaxWidth(),
        cornerRadius = 14.dp
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            StoneClass.alertableClasses.chunked(2).forEach { rowClasses ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    rowClasses.forEach { stoneClass ->
                        val enabled = stoneClass in enabledClasses
                        val swatchColor = Color(stoneClass.primaryHexColor)

                        Surface(
                            onClick = { onToggleStoneClass(stoneClass) },
                            color = if (enabled) {
                                swatchColor.copy(alpha = 0.18f)
                            } else {
                                Color(0xFF0E1822)
                            },
                            contentColor = if (enabled) HudTextPrimary else HudTextSecondary,
                            shape = RoundedCornerShape(10.dp),
                            border = BorderStroke(
                                width = if (enabled) 1.5.dp else 1.dp,
                                color = if (enabled) swatchColor.copy(alpha = 0.85f) else HudBorderSubtle
                            ),
                            modifier = Modifier
                                .weight(1f)
                                .heightIn(min = 44.dp)
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 10.dp, vertical = 10.dp)
                                    .semantics(mergeDescendants = true) {},
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(10.dp)
                                        .clip(CircleShape)
                                        .then(
                                            if (enabled) {
                                                Modifier.background(swatchColor)
                                            } else {
                                                Modifier.border(1.5.dp, HudTextSecondary, CircleShape)
                                            }
                                        )
                                )
                                Spacer(Modifier.width(7.dp))
                                Text(
                                    text = stoneClass.displayName,
                                    color = if (enabled) HudTextPrimary else HudTextSecondary.copy(alpha = 0.72f),
                                    fontSize = 12.sp,
                                    fontWeight = if (enabled) FontWeight.SemiBold else FontWeight.Normal,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                    modifier = Modifier.weight(1f)
                                )
                                if (enabled) {
                                    Icon(
                                        imageVector = Icons.Default.Check,
                                        contentDescription = null,
                                        tint = swatchColor,
                                        modifier = Modifier.size(14.dp)
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PermissionFallbackCard(
    onRequestCameraPermission: () -> Unit,
    modifier: Modifier = Modifier
) {
    HudPanel(
        modifier = modifier.padding(24.dp),
        backgroundColor = SlateCardElevated,
        cornerRadius = 16.dp
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Icon(
                imageVector = Icons.Default.CameraAlt,
                contentDescription = null,
                tint = CyanTripwire,
                modifier = Modifier.size(44.dp)
            )
            Text(
                text = "GneissFind Optical Push-Broom Scanner",
                color = HudTextPrimary,
                fontWeight = FontWeight.Bold,
                fontSize = 17.sp
            )
            Text(
                text = "Grant Rear 50MP Camera access to run the 4K 2× shutter-clamped (≤1/500s) rockhounding scanner and offline field logger.",
                color = HudTextSecondary,
                fontSize = 13.sp
            )
            Button(onClick = onRequestCameraPermission) {
                Text("Enable Rear Camera")
            }
        }
    }
}

private fun samplePreviewUiState(): ScannerUiState {
    return ScannerUiState(
        operatingMode = OperatingMode.SCAN,
        detectorBackend = DetectorBackend.SYNTHETIC_DEMO,
        syntheticProfile = SyntheticTargetProfile.MIXED_GRAVEL_WALK,
        imuPitchDeg = 45f,
        slantRangeMeters = 1.50f,
        gsdMmPerPx = 0.85f,
        exposureTimeNs = 1_992_000L,
        sensorIso = 1250,
        activeZoomRatio = 2.0f,
        audioPitchHz = 320f,
        audioCadenceHz = 13.0f,
        confirmedTargets = listOf(
            ConfirmedTarget(
                trackId = 1L,
                stoneClass = StoneClass.JASPER_RED_YELLOW,
                worldPosition = WorldPoint3D(0.05f, 0f, 1.06f),
                stripLeftPx = 460f,
                stripTopPx = 150f,
                stripRightPx = 530f,
                stripBottomPx = 220f,
                smoothedConfidence = 0.92f,
                estimatedDiameterMm = 54f,
                slantRangeMeters = 1.50f,
                hitCountInWindow = 6,
                windowSize = 6,
                lastUpdatedTimestampNs = 1_000_000L
            )
        )
    )
}

@Preview(name = "Scanner HUD (Default Scan)", device = Devices.PIXEL_7_PRO, showSystemUi = true)
@Composable
private fun ScannerScreenDefaultPreview() {
    GneissFindTheme {
        ScannerScreen(
            uiState = samplePreviewUiState(),
            hasCameraPermission = true,
            onRequestCameraPermission = {},
            onBindPreview = { _, _ -> },
            onSelectMode = {},
            onSelectBackend = {},
            onSelectSyntheticProfile = {},
            onToggleStoneClass = {},
            onToggleShutterClamp = {},
            onToggleTwoXCrop = {},
            onChangeExposureBias = {},
            onToggleAudioMute = {},
            onChangeSimSpeed = {}
        )
    }
}

@Preview(name = "Scanner HUD (Setup Sheet Open)", device = Devices.PIXEL_7_PRO, showSystemUi = true)
@Composable
private fun ScannerScreenOpticsOpenPreview() {
    GneissFindTheme {
        ScannerScreen(
            uiState = samplePreviewUiState(),
            hasCameraPermission = true,
            onRequestCameraPermission = {},
            onBindPreview = { _, _ -> },
            onSelectMode = {},
            onSelectBackend = {},
            onSelectSyntheticProfile = {},
            onToggleStoneClass = {},
            onToggleShutterClamp = {},
            onToggleTwoXCrop = {},
            onChangeExposureBias = {},
            onToggleAudioMute = {},
            onChangeSimSpeed = {},
            initialActiveSheetTab = ControlsSheetTab.OPTICS_AND_SPEED
        )
    }
}
