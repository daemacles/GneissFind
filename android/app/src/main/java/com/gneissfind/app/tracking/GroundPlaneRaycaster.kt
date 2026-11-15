/*
 * IMU pitch stabilization, reversible-landscape axis remapping, and ground-plane raycasting.
 * Consumes magnetometer-free TYPE_GAME_ROTATION_VECTOR and TYPE_LINEAR_ACCELERATION sensor events
 * (plus optional ARCore horizontal planes) to emit 120 Hz continuous EIS cropY offsets for display,
 * timestamp-matched integer cropY offsets for YUV strip extraction, and 3D ground coordinates for
 * TemporalTargetTracker.
 */
package com.gneissfind.app.tracking

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.view.Surface
import com.gneissfind.app.camera.PushBroomCameraController
import com.gneissfind.app.model.StripDetection
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.WorldPoint3D
import com.gneissfind.app.vision.PushBroomCropper
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Result of raycasting a 2D strip detection onto the horizontal ground plane (`y = 0 m`).
 *
 * @property worldPoint Projected 3D ground coordinates in meters (`yMeters = 0`).
 * @property slantRangeMeters Line-of-sight distance (m) from the camera optical center to [worldPoint].
 * @property gsdMmPerPx Ground sample distance (mm/px) at [slantRangeMeters] for the active zoom crop.
 * @property usedArCorePlane `true` when hit-tested against a tracked ARCore horizontal plane; `false`
 *   when computed trigonometrically from waist height and IMU pitch.
 */
data class RaycastResult(
    val worldPoint: WorldPoint3D,
    val slantRangeMeters: Float,
    val gsdMmPerPx: Float,
    val usedArCorePlane: Boolean
)

/**
 * Sub-pixel continuous vertical strip position (in 1080p stream rows) used to drive 120 Hz GPU
 * `PreviewView.translationY` without integer quantization steps.
 *
 * @property rawCropYFloat Unclamped top row offset (px) before `[0, MAX_CROP_Y_PX]` clamping.
 * @property clampedCropYFloat Clamped top row offset (px) in `[0f, MAX_CROP_Y_PX]`.
 * @property isPitchClamped `true` when [rawCropYFloat] exceeded `[0f, MAX_CROP_Y_PX]`.
 * @property matchedPitchDeg Downward optical pitch (degrees, `0` = horizon, `90` = nadir) used.
 * @property referencePitchDeg Active EIS pitch setpoint (degrees).
 * @property verticalFocalLengthPx Stream-scaled vertical focal length (px) used for `fy * tan(delta)`.
 */
data class PitchStabilizationFloatState(
    val rawCropYFloat: Float,
    val clampedCropYFloat: Float,
    val isPitchClamped: Boolean,
    val matchedPitchDeg: Float,
    val referencePitchDeg: Float,
    val verticalFocalLengthPx: Float
)

/**
 * Integer-quantized IMU-stabilized vertical strip position inside the landscape camera frame for
 * YUV row slicing in [PushBroomCropper].
 */
data class PitchStabilizationState(
    val rawCropY: Int,
    val clampedCropY: Int,
    val isPitchClamped: Boolean,
    val matchedPitchDeg: Float,
    val referencePitchDeg: Float,
    val verticalFocalLengthPx: Float
)

/**
 * Snapshot of current sensor/VIO pose, remapped landscape pitch/roll, stabilized `cropY`,
 * slant range `R`, and ground speed.
 */
data class MotionPoseSnapshot(
    val cameraWorldPosition: WorldPoint3D,
    val forwardPitchDeg: Float,
    val rawPitchDeg: Float,
    val rollDeg: Float,
    val referencePitchDeg: Float,
    val cropY: Int,
    val isPitchClamped: Boolean,
    val centerSlantRangeMeters: Float,
    val horizontalSpeedMps: Float,
    val arPlaneLocked: Boolean,
    val timestampNs: Long
)

/**
 * Reversible-landscape IMU pitch stabilizer (`Sensor.TYPE_GAME_ROTATION_VECTOR`), ARCore display
 * geometry synchronizer, and ground-plane raycaster:
 *  - Uses magnetometer-free `Sensor.TYPE_GAME_ROTATION_VECTOR` (`SENSOR_DELAY_GAME`) so magnetite
 *    in Pacific Northwest basalt cobbles never causes yaw/pitch drift.
 *  - Remaps IMU axes for `Surface.ROTATION_90` (`AXIS_Y, AXIS_MINUS_X`) and `Surface.ROTATION_270`
 *    (`AXIS_MINUS_Y, AXIS_X`) so downward pitch (`0 deg` = horizon, `45 deg` = forward slant,
 *    `90 deg` = straight down nadir) and roll stay identical across 180-degree landscape flips.
 *  - Stabilizes `cropY` in `[0..696]` around `referencePitchDeg` (`default 45.0 deg`, updatable
 *    via `SYNC PITCH`) matched to `imageProxy.imageInfo.timestamp`, illuminating `isPitchClamped`
 *    when clamped at `0` or `696`.
 */
class GroundPlaneRaycaster(
    context: Context? = null,
    private val waistHeightMeters: Float = DEFAULT_WAIST_HEIGHT_METERS
) : SensorEventListener, AutoCloseable {

    private val sensorManager: SensorManager? =
        context?.applicationContext?.getSystemService(Context.SENSOR_SERVICE) as? SensorManager

    // Strictly prefer magnetometer-free TYPE_GAME_ROTATION_VECTOR per android-camera-ar.md
    private val gameRotationSensor: Sensor? =
        sensorManager?.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
    private val linearAccelSensor: Sensor? =
        sensorManager?.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

    private val rawRotationMatrix = FloatArray(9)
    private val remappedRotationMatrix = FloatArray(9)

    // Timestamp-indexed IMU ring buffer for matching frame capture timestamps
    private val imuRingLock = Any()
    private val ringTimestampsNs = LongArray(IMU_RING_CAPACITY)
    private val ringPitchDeg = FloatArray(IMU_RING_CAPACITY)
    private val ringRollDeg = FloatArray(IMU_RING_CAPACITY)
    private var ringWriteIndex = 0
    private var ringCount = 0

    @Volatile
    var displaySurfaceRotation: Int = Surface.ROTATION_90
        private set

    @Volatile
    var currentForwardPitchDeg: Float = DEFAULT_PITCH_DEGREES
        private set

    @Volatile
    var latestRawPitchDeg: Float = DEFAULT_PITCH_DEGREES
        private set

    @Volatile
    var currentRollDeg: Float = 0.0f
        private set

    @Volatile
    var referencePitchDeg: Float = DEFAULT_PITCH_DEGREES
        private set

    @Volatile
    var eisPitchLockEnabled: Boolean = true

    @Volatile
    var imuHorizontalSpeedMps: Float = 0.0f
        private set

    @Volatile
    var simulatedWalkSpeedOverrideMps: Float? = null

    @Volatile
    var simulatedPitchOverrideDeg: Float? = null

    /**
     * Optional high-rate (~120 Hz) callback invoked on the main sensor looper thread whenever a new
     * smoothed landscape pitch sample is ready so the UI and GPU `PreviewView.translationY` update
     * smoothly at display refresh rate instead of waiting for ~10–15 Hz `ImageAnalysis` sweeps.
     */
    @Volatile
    var onRealtimeImuPitchListener: ((smoothedPitchDeg: Float, rollDeg: Float, referencePitchDeg: Float) -> Unit)? = null

    @Volatile
    var hasLiveImuStream: Boolean = false
        private set

    private var hasInitializedDisplayPitch = false
    private var lastGameRotationTimestampNs = 0L
    private var lastRealtimeCallbackNs = 0L
    private var integratedX = 0f
    private var integratedZ = 0f
    private var vx = 0f
    private var vz = 0f
    private var lastLinearAccelNs = 0L

    private var lastArPoseX = 0f
    private var lastArPoseZ = 0f
    private var lastArPoseTimestampNs = 0L
    private var smoothedArSpeedMps = 0f

    fun startImuListening() {
        val sm = sensorManager ?: return
        gameRotationSensor?.let { sensor ->
            // Request 8,000 us (125 Hz, safely within Android 12's <= 200 Hz unprivileged rate cap)
            // so EIS translation matches 120 Hz ProMotion displays; fall back to SENSOR_DELAY_GAME.
            val registered = runCatching {
                sm.registerListener(this, sensor, TARGET_IMU_SAMPLING_PERIOD_US)
            }.getOrDefault(false)
            if (!registered) {
                sm.registerListener(this, sensor, SensorManager.SENSOR_DELAY_GAME)
            }
        }
        linearAccelSensor?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    fun stopImuListening() {
        sensorManager?.unregisterListener(this)
        hasLiveImuStream = false
        lastGameRotationTimestampNs = 0L
    }

    override fun close() {
        stopImuListening()
    }

    /**
     * Synchronizes reversible-landscape display rotation (`ROTATION_90` vs `ROTATION_270`) for both
     * IMU coordinate remapping and any active ARCore [Session].
     */
    fun updateDisplayGeometry(
        surfaceRotation: Int,
        viewportWidthPx: Int = 0,
        viewportHeightPx: Int = 0,
        arSession: Session? = null
    ) {
        val normalized = if (surfaceRotation == Surface.ROTATION_270) {
            Surface.ROTATION_270
        } else {
            Surface.ROTATION_90
        }
        displaySurfaceRotation = normalized
        if (arSession != null && viewportWidthPx > 0 && viewportHeightPx > 0) {
            runCatching {
                arSession.setDisplayGeometry(normalized, viewportWidthPx, viewportHeightPx)
            }
        }
    }

    /**
     * Captures the user's current handheld downward tilt as the active EIS `thetaRef` setpoint
     * (`SYNC PITCH` actuator) and snaps the smoothed display pitch to avoid transient slew.
     */
    fun syncReferencePitch(): Float {
        val synced = (simulatedPitchOverrideDeg ?: latestRawPitchDeg).coerceIn(12.0f, 85.0f)
        referencePitchDeg = synced
        currentForwardPitchDeg = synced
        return synced
    }

    fun setReferencePitchDegrees(pitchDeg: Float) {
        referencePitchDeg = pitchDeg.coerceIn(12.0f, 85.0f)
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                SensorManager.getRotationMatrixFromVector(rawRotationMatrix, event.values)
                remapLandscapeRotationMatrix(
                    rawR = rawRotationMatrix,
                    surfaceRotation = displaySurfaceRotation,
                    outRemappedR = remappedRotationMatrix
                )
                val (rawPitch, rawRoll) = computePitchAndRollDegrees(remappedRotationMatrix)
                latestRawPitchDeg = rawPitch
                hasLiveImuStream = true

                val dtSec = if (lastGameRotationTimestampNs != 0L && event.timestamp > lastGameRotationTimestampNs) {
                    ((event.timestamp - lastGameRotationTimestampNs) * 1e-9f).coerceIn(0.002f, 0.1f)
                } else {
                    0.008f
                }
                lastGameRotationTimestampNs = event.timestamp

                if (!hasInitializedDisplayPitch) {
                    currentForwardPitchDeg = rawPitch
                    currentRollDeg = rawRoll
                    hasInitializedDisplayPitch = true
                } else {
                    // Adaptive first-order low-pass filter: damp <0.35 deg stationary hand tremor
                    // (tau = 45 ms) while tracking deliberate wrist pitch changes with near-zero lag
                    // (tau = 14 ms) so 120 Hz GPU preview translation feels locked to the ground.
                    val pitchDelta = abs(rawPitch - currentForwardPitchDeg)
                    val motionBlend = ((pitchDelta - 0.15f) / 1.10f).coerceIn(0f, 1f)
                    val tauSec = 0.045f * (1f - motionBlend) + 0.014f * motionBlend
                    val alpha = (dtSec / (tauSec + dtSec)).coerceIn(0.08f, 0.78f)
                    currentForwardPitchDeg += (rawPitch - currentForwardPitchDeg) * alpha
                    currentRollDeg += (rawRoll - currentRollDeg) * alpha
                }

                synchronized(imuRingLock) {
                    ringTimestampsNs[ringWriteIndex] = event.timestamp
                    ringPitchDeg[ringWriteIndex] = rawPitch
                    ringRollDeg[ringWriteIndex] = rawRoll
                    ringWriteIndex = (ringWriteIndex + 1) % IMU_RING_CAPACITY
                    if (ringCount < IMU_RING_CAPACITY) ringCount++
                }

                val listener = onRealtimeImuPitchListener
                if (listener != null &&
                    (lastRealtimeCallbackNs == 0L || event.timestamp - lastRealtimeCallbackNs >= MIN_DISPLAY_UPDATE_INTERVAL_NS)
                ) {
                    lastRealtimeCallbackNs = event.timestamp
                    val displayPitch = simulatedPitchOverrideDeg ?: currentForwardPitchDeg
                    listener.invoke(displayPitch, currentRollDeg, referencePitchDeg)
                }
            }

            Sensor.TYPE_LINEAR_ACCELERATION -> {
                val nowNs = event.timestamp
                if (lastLinearAccelNs != 0L) {
                    val dt = ((nowNs - lastLinearAccelNs) * 1e-9f).coerceIn(0f, 0.1f)
                    val ax = event.values[0]
                    val ay = event.values[1]
                    val deadbandAx = if (abs(ax) < 0.18f) 0f else ax
                    val deadbandAy = if (abs(ay) < 0.18f) 0f else ay
                    vx = (vx + deadbandAx * dt) * 0.92f
                    vz = (vz + deadbandAy * dt) * 0.92f
                    val speed = sqrt(vx * vx + vz * vz)
                    imuHorizontalSpeedMps = imuHorizontalSpeedMps * 0.85f + speed * 0.15f
                    integratedX += vx * dt
                    integratedZ += vz * dt
                }
                lastLinearAccelNs = nowNs
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    /**
     * Looks up the raw IMU pitch sample closest in timestamp to [targetTimestampNs] for zero-lag
     * software EIS `cropY` computation.
     */
    fun lookupPitchAtTimestampNs(targetTimestampNs: Long): Float {
        simulatedPitchOverrideDeg?.let { return it }
        synchronized(imuRingLock) {
            if (ringCount == 0) return latestRawPitchDeg
            var bestPitch = latestRawPitchDeg
            var bestDelta = Long.MAX_VALUE
            for (i in 0 until ringCount) {
                val delta = abs(ringTimestampsNs[i] - targetTimestampNs)
                if (delta < bestDelta) {
                    bestDelta = delta
                    bestPitch = ringPitchDeg[i]
                }
            }
            return bestPitch
        }
    }

    /**
     * Captures the synchronized pose, timestamp-matched `cropY` stabilization state, slant range `R`,
     * and horizontal ground velocity `v_xy`.
     */
    fun capturePoseSnapshot(
        arFrame: Frame? = null,
        fallbackTimestampNs: Long = System.nanoTime(),
        verticalFocalLengthPx: Float = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX
    ): MotionPoseSnapshot {
        val frameTs = arFrame?.timestamp ?: fallbackTimestampNs
        val matchedRawPitch = lookupPitchAtTimestampNs(frameTs)
        val displayPitch = simulatedPitchOverrideDeg ?: currentForwardPitchDeg
        val refPitch = referencePitchDeg

        val stabilization = computeStabilizedCropY(
            currentPitchDeg = matchedRawPitch,
            referencePitchDeg = refPitch,
            verticalFocalLengthPx = verticalFocalLengthPx,
            eisEnabled = eisPitchLockEnabled
        )

        // When EIS locks cropY onto thetaRef (unclamped), the strip looks along thetaRef; if EIS is
        // off or clamped, compute effective look depression angle of the strip center.
        val stripCenterOffsetPx = stabilization.clampedCropY - StripRoiGeometry.NOMINAL_CROP_Y_PX
        val stripCenterLookPitchDeg =
            matchedRawPitch + Math.toDegrees(atan2(stripCenterOffsetPx.toDouble(), verticalFocalLengthPx.toDouble())).toFloat()
        val slantR = computeSlantRangeForPitch(waistHeightMeters, stripCenterLookPitchDeg)

        val overrideSpeed = simulatedWalkSpeedOverrideMps
        if (arFrame != null && arFrame.camera.trackingState == TrackingState.TRACKING) {
            val pose = arFrame.camera.pose
            val px = pose.tx()
            val py = pose.ty()
            val pz = pose.tz()
            val ts = arFrame.timestamp

            if (lastArPoseTimestampNs != 0L && ts > lastArPoseTimestampNs) {
                val dt = (ts - lastArPoseTimestampNs) * 1e-9f
                if (dt in 0.002f..0.5f) {
                    val dx = px - lastArPoseX
                    val dz = pz - lastArPoseZ
                    val instantVxy = sqrt(dx * dx + dz * dz) / dt
                    smoothedArSpeedMps = smoothedArSpeedMps * 0.70f + instantVxy * 0.30f
                }
            }
            lastArPoseX = px
            lastArPoseZ = pz
            lastArPoseTimestampNs = ts

            val hasHorizontalPlane = arFrame.getUpdatedTrackables(Plane::class.java).any {
                it.type == Plane.Type.HORIZONTAL_UPWARD_FACING && it.trackingState == TrackingState.TRACKING
            }

            return MotionPoseSnapshot(
                cameraWorldPosition = WorldPoint3D(px, py, pz),
                forwardPitchDeg = displayPitch,
                rawPitchDeg = matchedRawPitch,
                rollDeg = currentRollDeg,
                referencePitchDeg = refPitch,
                cropY = stabilization.clampedCropY,
                isPitchClamped = stabilization.isPitchClamped,
                centerSlantRangeMeters = slantR,
                horizontalSpeedMps = overrideSpeed ?: smoothedArSpeedMps,
                arPlaneLocked = hasHorizontalPlane,
                timestampNs = ts
            )
        }

        val activeSpeed = overrideSpeed ?: imuHorizontalSpeedMps
        if (overrideSpeed != null && lastLinearAccelNs == 0L) {
            integratedZ += overrideSpeed * (1f / 15f)
        }

        return MotionPoseSnapshot(
            cameraWorldPosition = WorldPoint3D(integratedX, waistHeightMeters, integratedZ),
            forwardPitchDeg = displayPitch,
            rawPitchDeg = matchedRawPitch,
            rollDeg = currentRollDeg,
            referencePitchDeg = refPitch,
            cropY = stabilization.clampedCropY,
            isPitchClamped = stabilization.isPitchClamped,
            centerSlantRangeMeters = slantR,
            horizontalSpeedMps = activeSpeed,
            arPlaneLocked = false,
            timestampNs = frameTs
        )
    }

    /**
     * Raycasts a [StripDetection] onto the horizontal ground plane using ARCore `Frame.hitTest`
     * when a tracked horizontal plane is hit, or trigonometric IMU ray-plane projection otherwise.
     */
    fun raycastStripDetection(
        detection: StripDetection,
        geometry: StripRoiGeometry,
        poseSnapshot: MotionPoseSnapshot,
        zoomRatio: Float = PushBroomCropper.DEFAULT_ZOOM_RATIO,
        arFrame: Frame? = null
    ): RaycastResult {
        val screenX = geometry.stripLeftPx + detection.centerXPx
        val screenY = geometry.stripTopPx + detection.centerYPx

        if (arFrame != null && arFrame.camera.trackingState == TrackingState.TRACKING) {
            val hits = arFrame.hitTest(screenX, screenY)
            for (hit in hits) {
                val trackable = hit.trackable
                if (trackable is Plane &&
                    trackable.type == Plane.Type.HORIZONTAL_UPWARD_FACING &&
                    trackable.isPoseInPolygon(hit.hitPose)
                ) {
                    val hp = hit.hitPose
                    val worldPt = WorldPoint3D(hp.tx(), hp.ty(), hp.tz())
                    val slantR = hit.distance.coerceIn(0.40f, 4.0f)
                    val gsd = PushBroomCropper.computeGsdMmPerPx(slantR, zoomRatio)
                    return RaycastResult(
                        worldPoint = worldPt,
                        slantRangeMeters = slantR,
                        gsdMmPerPx = gsd,
                        usedArCorePlane = true
                    )
                }
            }
        }

        val effectivePitch = if (eisPitchLockEnabled && !poseSnapshot.isPitchClamped) {
            poseSnapshot.referencePitchDeg
        } else {
            poseSnapshot.forwardPitchDeg
        }

        return raycastTrigonometric(
            stripCenterXPx = detection.centerXPx,
            stripCenterYPx = detection.centerYPx,
            stripWidthPx = geometry.stripWidthPx,
            cameraWorldPos = poseSnapshot.cameraWorldPosition,
            forwardPitchDeg = effectivePitch,
            waistHeightMeters = waistHeightMeters,
            zoomRatio = zoomRatio
        )
    }

    companion object {
        const val DEFAULT_WAIST_HEIGHT_METERS = 1.06f
        const val DEFAULT_PITCH_DEGREES = 45.0f
        private const val TARGET_IMU_SAMPLING_PERIOD_US = 8_000 // 125 Hz (<= 200 Hz Android 12 unprivileged cap)
        private const val MIN_DISPLAY_UPDATE_INTERVAL_NS = 8_000_000L // ~125 Hz max UI/GPU translation cadence
        private const val IMU_RING_CAPACITY = 64
        // Vertical angular span of the 384 px tall 2x-cropped push-broom strip (~7.4 deg on 1080p)
        private const val STRIP_VERTICAL_FOV_2X_DEG = 7.36f

        /**
         * Remaps raw Android device rotation matrix [rawR] into landscape screen coordinates:
         *  - `Surface.ROTATION_90`: `(AXIS_Y, AXIS_MINUS_X)`
         *  - `Surface.ROTATION_270`: `(AXIS_MINUS_Y, AXIS_X)`
         * Includes exact orthogonal column-permutation fallback when `SensorManager.remapCoordinateSystem`
         * is stubbed in host-side JVM unit tests.
         */
        fun remapLandscapeRotationMatrix(
            rawR: FloatArray,
            surfaceRotation: Int,
            outRemappedR: FloatArray
        ) {
            val isRot270 = surfaceRotation == Surface.ROTATION_270
            val axisX = if (isRot270) SensorManager.AXIS_MINUS_Y else SensorManager.AXIS_Y
            val axisY = if (isRot270) SensorManager.AXIS_X else SensorManager.AXIS_MINUS_X

            val remappedByFramework = SensorManager.remapCoordinateSystem(rawR, axisX, axisY, outRemappedR)
            val nonZero = outRemappedR.any { it != 0f }
            if (remappedByFramework && nonZero) return

            // Exact analytic 3x3 column permutation matching SensorManager.remapCoordinateSystem
            for (row in 0..2) {
                val base = row * 3
                val devX = rawR[base + 0]
                val devY = rawR[base + 1]
                val devZ = rawR[base + 2]
                if (isRot270) {
                    outRemappedR[base + 0] = -devY
                    outRemappedR[base + 1] = devX
                    outRemappedR[base + 2] = devZ
                } else {
                    outRemappedR[base + 0] = devY
                    outRemappedR[base + 1] = -devX
                    outRemappedR[base + 2] = devZ
                }
            }
        }

        /**
         * Extracts downward optical pitch (`0 deg` = horizon, `45 deg` = forward rockhounding slant,
         * `90 deg` = straight down nadir) and screen roll (`deg`) from [remappedR].
         */
        fun computePitchAndRollDegrees(remappedR: FloatArray): Pair<Float, Float> {
            // World ENU Up vector is Row 2 of remappedR: (R[6], R[7], R[8]) = (Up·X_scr, Up·Y_scr, Up·Z_scr)
            val upDotScreenRight = remappedR[6].toDouble()
            val upDotScreenUp = remappedR[7].toDouble()
            val upDotScreenOut = remappedR[8].toDouble()

            val pitchDeg = Math.toDegrees(atan2(upDotScreenOut, upDotScreenUp)).toFloat()
            val rollDeg = Math.toDegrees(
                atan2(upDotScreenRight, hypot(upDotScreenUp, upDotScreenOut).coerceAtLeast(1e-6))
            ).toFloat()
            return pitchDeg to rollDeg
        }

        /**
         * Computes continuous sub-pixel stabilized `cropY` inside `[0f..MAX_CROP_Y_PX]` on the
         * `1920x1080` landscape stream for 120 Hz GPU preview translation (`PreviewView.translationY`).
         */
        fun computeStabilizedCropYFloat(
            currentPitchDeg: Float,
            referencePitchDeg: Float = DEFAULT_PITCH_DEGREES,
            verticalFocalLengthPx: Float = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX,
            eisEnabled: Boolean = true
        ): PitchStabilizationFloatState {
            if (!eisEnabled) {
                val nominal = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat()
                return PitchStabilizationFloatState(
                    rawCropYFloat = nominal,
                    clampedCropYFloat = nominal,
                    isPitchClamped = false,
                    matchedPitchDeg = currentPitchDeg,
                    referencePitchDeg = referencePitchDeg,
                    verticalFocalLengthPx = verticalFocalLengthPx
                )
            }

            val deltaPitchRad = Math.toRadians(
                (currentPitchDeg - referencePitchDeg).coerceIn(-60f, 60f).toDouble()
            )
            val deltaYPx = (verticalFocalLengthPx * tan(deltaPitchRad)).toFloat()
            val rawCropY = StripRoiGeometry.NOMINAL_CROP_Y_PX.toFloat() - deltaYPx
            val maxCropY = StripRoiGeometry.MAX_CROP_Y_PX.toFloat()
            val clampedCropY = rawCropY.coerceIn(0f, maxCropY)
            val isClamped = rawCropY < 0f || rawCropY > maxCropY

            return PitchStabilizationFloatState(
                rawCropYFloat = rawCropY,
                clampedCropYFloat = clampedCropY,
                isPitchClamped = isClamped,
                matchedPitchDeg = currentPitchDeg,
                referencePitchDeg = referencePitchDeg,
                verticalFocalLengthPx = verticalFocalLengthPx
            )
        }

        /**
         * Computes integer-quantized stabilized `cropY` inside `[0..MAX_CROP_Y_PX]` on the `1920x1080`
         * stream for YUV strip slicing:
         *  - When user pitches DOWN (`currentPitchDeg > referencePitchDeg`), the ground patch moves
         *    UP toward the top of the sensor frame (`cropY` decreases toward `0`).
         *  - When user pitches UP (`currentPitchDeg < referencePitchDeg`), the ground patch moves
         *    DOWN toward the bottom of the sensor frame (`cropY` increases toward `696`).
         */
        fun computeStabilizedCropY(
            currentPitchDeg: Float,
            referencePitchDeg: Float = DEFAULT_PITCH_DEGREES,
            verticalFocalLengthPx: Float = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX,
            eisEnabled: Boolean = true
        ): PitchStabilizationState {
            val floatState = computeStabilizedCropYFloat(
                currentPitchDeg = currentPitchDeg,
                referencePitchDeg = referencePitchDeg,
                verticalFocalLengthPx = verticalFocalLengthPx,
                eisEnabled = eisEnabled
            )
            val rawCropY = floatState.rawCropYFloat.roundToInt()
            val clampedCropY = rawCropY.coerceIn(0, StripRoiGeometry.MAX_CROP_Y_PX)
            val isClamped = rawCropY < 0 || rawCropY > StripRoiGeometry.MAX_CROP_Y_PX

            return PitchStabilizationState(
                rawCropY = rawCropY,
                clampedCropY = clampedCropY,
                isPitchClamped = isClamped,
                matchedPitchDeg = currentPitchDeg,
                referencePitchDeg = referencePitchDeg,
                verticalFocalLengthPx = verticalFocalLengthPx
            )
        }

        fun computeSlantRangeForPitch(
            waistHeightMeters: Float = DEFAULT_WAIST_HEIGHT_METERS,
            forwardPitchDeg: Float = DEFAULT_PITCH_DEGREES
        ): Float {
            val clampedPitchRad = Math.toRadians(forwardPitchDeg.coerceIn(10.0f, 88.0f).toDouble())
            return (waistHeightMeters / sin(clampedPitchRad)).toFloat().coerceIn(0.40f, 6.50f)
        }

        /**
         * Pure trigonometric ground-plane projection from waist height `h` at depression angle
         * `forwardPitchDeg` (`45 deg` -> `R = 1.50 m`, `Z_ahead = 1.06 m`, `GSD = 0.59 mm/px` @ `2.0x`).
         */
        fun raycastTrigonometric(
            stripCenterXPx: Float,
            stripCenterYPx: Float,
            stripWidthPx: Int = StripRoiGeometry.STRIP_WIDTH_5_TILE_PX,
            cameraWorldPos: WorldPoint3D = WorldPoint3D(0f, DEFAULT_WAIST_HEIGHT_METERS, 0f),
            forwardPitchDeg: Float = DEFAULT_PITCH_DEGREES,
            waistHeightMeters: Float = DEFAULT_WAIST_HEIGHT_METERS,
            zoomRatio: Float = PushBroomCropper.DEFAULT_ZOOM_RATIO
        ): RaycastResult {
            val stripVFov = STRIP_VERTICAL_FOV_2X_DEG * (2.0f / zoomRatio.coerceAtLeast(1.0f))
            val normYFromCenter = (stripCenterYPx - StripRoiGeometry.STRIP_HEIGHT_PX * 0.5f) /
                StripRoiGeometry.STRIP_HEIGHT_PX.toFloat()
            val rayPitchDeg = (forwardPitchDeg + normYFromCenter * stripVFov).coerceIn(10.0f, 88.0f)
            val rayPitchRad = Math.toRadians(rayPitchDeg.toDouble())

            val slantRangeM = (waistHeightMeters / sin(rayPitchRad)).toFloat().coerceIn(0.40f, 6.50f)
            val forwardGroundDistM = (waistHeightMeters / tan(rayPitchRad)).toFloat()

            val gsdMmPerPx = PushBroomCropper.computeGsdMmPerPx(slantRangeM, zoomRatio)
            val gsdMetersPerPx = gsdMmPerPx * 0.001f
            val crossTrackOffsetM = (stripCenterXPx - stripWidthPx * 0.5f) * gsdMetersPerPx

            return RaycastResult(
                worldPoint = WorldPoint3D(
                    xMeters = cameraWorldPos.xMeters + crossTrackOffsetM,
                    yMeters = 0.0f,
                    zMeters = cameraWorldPos.zMeters + forwardGroundDistM
                ),
                slantRangeMeters = slantRangeM,
                gsdMmPerPx = gsdMmPerPx,
                usedArCorePlane = false
            )
        }
    }
}
