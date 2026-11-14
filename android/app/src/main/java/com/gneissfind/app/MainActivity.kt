package com.gneissfind.app

import android.Manifest
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Color as AndroidColor
import android.hardware.display.DisplayManager
import android.media.AudioManager
import android.os.Bundle
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

class MainActivity : ComponentActivity() {

    private val viewModel: ScannerViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(AndroidColor.BLACK),
            navigationBarStyle = SystemBarStyle.dark(AndroidColor.BLACK)
        )
        volumeControlStream = AudioManager.STREAM_MUSIC
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.decorView.isSoundEffectsEnabled = false

        syncCurrentDisplayGeometry()

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
                    onSelectBackend = viewModel::setDetectorBackend,
                    onCycleScenario = viewModel::cycleScenarioPreset,
                    onToggleSwathTileMode = viewModel::toggleSwathTileMode,
                    onToggleEisLock = viewModel::setEisPitchLockEnabled,
                    onSyncPitch = viewModel::syncReferencePitch,
                    onToggleStoneClass = viewModel::toggleStoneClass,
                    onToggleShutterClamp = viewModel::setShutterClampEnabled,
                    onToggleTwoXCrop = viewModel::setTwoXCropEnabled,
                    onChangeExposureBias = viewModel::setExposureBiasEv,
                    onToggleAudioMute = viewModel::setAudioMuted,
                    onToggleOverspeedTest = viewModel::toggleOverspeedTest
                )
            }
        }
    }

    private fun currentSurfaceRotation(): Int {
        return display?.rotation ?: Surface.ROTATION_90
    }

    private fun syncCurrentDisplayGeometry() {
        val bounds = windowManager.currentWindowMetrics.bounds
        viewModel.onDisplayGeometryChanged(
            surfaceRotation = currentSurfaceRotation(),
            viewportWidthPx = bounds.width(),
            viewportHeightPx = bounds.height()
        )
    }
}
