/*
 * Single-activity Android entry point locked to reversible sensorLandscape orientation.
 *
 * Synchronizes display rotation across CameraX, IMU coordinate remapping, and ARCore, routes
 * sonification volume to STREAM_MUSIC, silences UI tap effects, manages CAMERA runtime permissions,
 * and hosts ScannerScreen bound to ScannerViewModel.
 *
 * Main types:
 * - MainActivity: root ComponentActivity hosting the GneissFind Compose UI and lifecycle observers.
 */
package com.gneissfind.app

import android.Manifest
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Color as AndroidColor
import android.hardware.display.DisplayManager
import android.media.AudioManager
import android.os.Bundle
import android.view.MotionEvent
import android.view.Surface
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gneissfind.app.ui.ScannerScreen
import com.gneissfind.app.ui.ScannerViewModel
import com.gneissfind.app.ui.theme.GneissFindTheme

/**
 * Single-activity entry point for the GneissFind geological tricorder.
 *
 * Responsibilities:
 * - Enforces [ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE] and registers a [DisplayManager.DisplayListener]
 *   so 180-degree flips between [Surface.ROTATION_90] and [Surface.ROTATION_270] update [ScannerViewModel]
 *   even when Android suppresses Activity recreation.
 * - Keeps the display awake (`FLAG_KEEP_SCREEN_ON`), routes hardware volume keys to [AudioManager.STREAM_MUSIC],
 *   and disables view tap sound effects (`isSoundEffectsEnabled = false`) so the audio channel stays
 *   strictly reserved for `MetalDetectorSonifier` stone pings and overspeed warnings (`.agents/rules/lcars-ui.md` §5).
 * - Starts and pauses IMU sensor listeners and the `AudioTrack` sonifier on `ON_RESUME` / `ON_PAUSE`.
 */
class MainActivity : ComponentActivity() {

    private val viewModel: ScannerViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setShowWhenLocked(true)
        setTurnScreenOn(true)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(AndroidColor.BLACK),
            navigationBarStyle = SystemBarStyle.dark(AndroidColor.BLACK)
        )
        volumeControlStream = AudioManager.STREAM_MUSIC
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.decorView.isSoundEffectsEnabled = false

        syncCurrentDisplayGeometry()
        if (savedInstanceState == null) {
            handleDebugReplayIntent(intent)
        }

        setContent {
            val hostView = LocalView.current
            LaunchedEffect(hostView) {
                hostView.isSoundEffectsEnabled = false
            }

            // Track 180-degree sensorLandscape flips (ROTATION_90 <-> ROTATION_270) even when
            // Android skips configuration recreation between opposite landscape orientations.
            DisposableEffect(Unit) {
                val displayManager = getSystemService(DISPLAY_SERVICE) as? DisplayManager
                val listener = object : DisplayManager.DisplayListener {
                    override fun onDisplayAdded(displayId: Int) = Unit
                    override fun onDisplayRemoved(displayId: Int) = Unit
                    override fun onDisplayChanged(displayId: Int) {
                        syncCurrentDisplayGeometry()
                    }
                }
                displayManager?.registerDisplayListener(listener, null)
                onDispose {
                    displayManager?.unregisterDisplayListener(listener)
                }
            }

            GneissFindTheme {
                var hasCameraPermission by remember {
                    mutableStateOf(
                        ContextCompat.checkSelfPermission(
                            this,
                            Manifest.permission.CAMERA
                        ) == PackageManager.PERMISSION_GRANTED
                    )
                }

                val permissionLauncher = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.RequestPermission()
                ) { granted ->
                    hasCameraPermission = granted
                }

                LaunchedEffect(Unit) {
                    if (!hasCameraPermission) {
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }

                val lifecycleOwner = LocalLifecycleOwner.current
                DisposableEffect(lifecycleOwner) {
                    val observer = LifecycleEventObserver { _, event ->
                        when (event) {
                            Lifecycle.Event.ON_RESUME -> {
                                syncCurrentDisplayGeometry()
                                viewModel.onResumeSensorsAndAudio()
                            }
                            Lifecycle.Event.ON_PAUSE -> viewModel.onPauseSensorsAndAudio()
                            else -> Unit
                        }
                    }
                    lifecycleOwner.lifecycle.addObserver(observer)
                    onDispose {
                        lifecycleOwner.lifecycle.removeObserver(observer)
                    }
                }

                val uiState by viewModel.uiState.collectAsStateWithLifecycle()

                LaunchedEffect(uiState.patrolPowerState) {
                    val targetHz = uiState.patrolPowerState.preferredRefreshRateHz
                    val params = window.attributes
                    if (params.preferredRefreshRate != targetHz) {
                        params.preferredRefreshRate = targetHz
                        window.attributes = params
                    }
                }

                ScannerScreen(
                    uiState = uiState,
                    hasCameraPermission = hasCameraPermission,
                    onRequestCameraPermission = {
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    },
                    onBindPreview = { owner, previewView ->
                        viewModel.bindCamera(owner, previewView, currentSurfaceRotation())
                    },
                    onSelectMode = viewModel::setOperatingMode,
                    onSelectSource = viewModel::selectSourceOption,
                    onToggleSwathTileMode = viewModel::toggleSwathTileMode,
                    onToggleEisLock = viewModel::setEisPitchLockEnabled,
                    onSyncPitch = viewModel::syncReferencePitch,
                    onToggleStoneClass = viewModel::toggleStoneClass,
                    onToggleShutterClamp = viewModel::setShutterClampEnabled,
                    onToggleTwoXCrop = viewModel::setTwoXCropEnabled,
                    onChangeExposureBias = viewModel::setExposureBiasEv,
                    onToggleAudioMute = viewModel::setAudioMuted,
                    onToggleEyesUpPatrol = viewModel::toggleEyesUpPatrol,
                    onPatrolViewportTapped = viewModel::onPatrolViewportTapped,
                    onToggleCentralLock = viewModel::toggleCentralLockButton,
                    onSelectReplaySession = { item -> viewModel.selectReplaySession(item) },
                    onOpenSourceDrawer = viewModel::refreshAvailableReplaySessions
                )
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleDebugReplayIntent(intent)
    }

    private fun handleDebugReplayIntent(intent: Intent?) {
        if (intent == null) return
        val recordSeconds = intent.getIntExtra(EXTRA_DEBUG_RECORD_SECONDS, 0)
        if (recordSeconds > 0) {
            intent.removeExtra(EXTRA_DEBUG_RECORD_SECONDS)
            viewModel.startDebugRecordSessionFromIntent(recordSeconds)
            return
        }
        val replaySession = intent.getStringExtra(EXTRA_DEBUG_REPLAY_SESSION)
            ?.trim()
            ?.takeIf { it.isNotEmpty() }
            ?: return
        val autoLockFrame = intent.getIntExtra(EXTRA_DEBUG_AUTO_LOCK_FRAME, -1)
        val loopReplay = intent.getBooleanExtra(EXTRA_DEBUG_LOOP_REPLAY, true)
        intent.removeExtra(EXTRA_DEBUG_REPLAY_SESSION)
        viewModel.startDebugReplaySessionFromIntent(
            sessionNameOrPath = replaySession,
            autoLockFrameIdx = autoLockFrame,
            loopReplay = loopReplay
        )
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (ev.actionMasked == MotionEvent.ACTION_DOWN) {
            viewModel.onUserTouchInteraction()
        }
        return super.dispatchTouchEvent(ev)
    }

    /** Returns the current display [Surface] rotation, defaulting to [Surface.ROTATION_90] if unavailable. */
    private fun currentSurfaceRotation(): Int {
        return display?.rotation ?: Surface.ROTATION_90
    }

    /** Queries current window bounds and display rotation and forwards them to [ScannerViewModel.onDisplayGeometryChanged]. */
    private fun syncCurrentDisplayGeometry() {
        val bounds = windowManager.currentWindowMetrics.bounds
        viewModel.onDisplayGeometryChanged(
            surfaceRotation = currentSurfaceRotation(),
            viewportWidthPx = bounds.width(),
            viewportHeightPx = bounds.height()
        )
    }

    companion object {
        const val EXTRA_DEBUG_REPLAY_SESSION = "debug_replay_session"
        const val EXTRA_DEBUG_AUTO_LOCK_FRAME = "debug_auto_lock_frame"
        const val EXTRA_DEBUG_LOOP_REPLAY = "debug_loop_replay"
        const val EXTRA_DEBUG_RECORD_SECONDS = "debug_record_seconds"
    }
}
