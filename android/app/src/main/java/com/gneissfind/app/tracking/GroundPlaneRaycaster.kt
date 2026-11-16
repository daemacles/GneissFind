/*
 * IMU pitch stabilization, reversible-landscape axis remapping, 3D world-ray anchoring, and
 * ground-plane raycasting.
 *
 * Consumes magnetometer-free TYPE_GAME_ROTATION_VECTOR and TYPE_LINEAR_ACCELERATION sensor events
 * (plus optional ARCore horizontal planes) to emit 120 Hz continuous EIS cropY offsets and 3x3
 * landscape rotation matrices for display reprojection, timestamp-matched integer cropY offsets for
 * YUV strip extraction, and rotation-invariant 3D world rays and ground coordinates for
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
import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.StripDetection
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.WorldPoint3D
import com.gneissfind.app.model.WorldRay3D
import com.gneissfind.app.vision.PushBroomCropper
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Result of raycasting a 2D strip detection onto the horizontal ground plane (`y = 0 m`) and
 * anchoring its unit line-of-sight bearing ray in the IMU world frame.
 *
 * @property worldPoint Projected 3D ground coordinates in meters (`yMeters = 0`).
 * @property slantRangeMeters Line-of-sight distance (m) from the camera optical center to [worldPoint].
 * @property gsdMmPerPx Ground sample distance (mm/px) at [slantRangeMeters] for the active zoom crop.
 * @property usedArCorePlane `true` when hit-tested against a tracked ARCore horizontal plane; `false`
 *   when intersected with the waist-height horizontal ground plane using the 3D IMU world ray.
 * @property worldRay Unit 3D bearing vector in the `TYPE_GAME_ROTATION_VECTOR` world frame.
 * @property angularWidthRad Horizontal angular width (`widthPx / focalLengthPx`, in radians).
 * @property angularHeightRad Vertical angular height (`heightPx / focalLengthPx`, in radians).
 */
data class RaycastResult(
    val worldPoint: WorldPoint3D,
    val slantRangeMeters: Float,
    val gsdMmPerPx: Float,
    val usedArCorePlane: Boolean,
    val worldRay: WorldRay3D = WorldRay3D(0f, 0.70710677f, -0.70710677f),
    val angularWidthRad: Float = 0f,
    val angularHeightRad: Float = 0f
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
 * Snapshot of current sensor/VIO pose, remapped landscape pitch/roll, 3x3 landscape rotation matrix,
 * stabilized `cropY`, calibrated vertical focal length, slant range `R`, and ground speed.
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
    val timestampNs: Long,
    val verticalFocalLengthPx: Float = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX,
    val rotationMatrix: FloatArray = GroundPlaneRaycaster.buildLandscapeRotationMatrix(
        pitchDeg = rawPitchDeg,
        rollDeg = rollDeg,
        yawDeg = 0f
    )
)

/**
 * Result of reprojecting a world-anchored target ray into the current camera frame and strip ROI.
 *
 * @property stripLeftPx Reprojected left X coordinate in unified strip pixels.
 * @property stripTopPx Reprojected top Y coordinate in unified strip pixels.
 * @property stripRightPx Reprojected right X coordinate in unified strip pixels.
 * @property stripBottomPx Reprojected bottom Y coordinate in unified strip pixels.
 * @property isVisibleInStrip True when the target is in front of the camera and its reprojected
 *   box intersects the active `[0..stripWidthPx] x [0..stripHeightPx]` push-broom strip ROI.
 */
data class ReprojectedStripBox(
    val stripLeftPx: Float,
    val stripTopPx: Float,
    val stripRightPx: Float,
    val stripBottomPx: Float,
    val isVisibleInStrip: Boolean
)

/**
 * Reversible-landscape IMU pitch stabilizer (`Sensor.TYPE_GAME_ROTATION_VECTOR`), ARCore display
 * geometry synchronizer, 3D world-ray projector, and ground-plane raycaster:
 *  - Uses magnetometer-free `Sensor.TYPE_GAME_ROTATION_VECTOR` (`SENSOR_DELAY_GAME`) so magnetite
 *    in Pacific Northwest basalt cobbles never causes yaw/pitch drift.
 *  - Remaps IMU axes for `Surface.ROTATION_90` (`AXIS_Y, AXIS_MINUS_X`) and `Surface.ROTATION_270`
 *    (`AXIS_MINUS_Y, AXIS_X`) so downward pitch (`0 deg` = horizon, `45 deg` = forward slant,
 *    `90 deg` = straight down nadir), roll, and 3x3 world rotation matrices stay identical across
 *    180-degree landscape flips.
 *  - Maintains a 128-entry timestamped ring buffer (~1.0 s at 125 Hz) of full 3x3 landscape
 *    rotation matrices so detections from a ~390 ms GPU sweep are back-projected using the exact
 *    camera orientation at `imageProxy.imageInfo.timestamp` and forward-reprojected at 120 Hz.
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
    private val latestRemappedRotationMatrix = buildLandscapeRotationMatrix(
        pitchDeg = DEFAULT_PITCH_DEGREES,
        rollDeg = 0f,
        yawDeg = 0f
    )

    // Timestamp-indexed IMU ring buffer for matching frame capture timestamps (~1.0 s at 125 Hz)
    private val imuRingLock = Any()
    private val ringTimestampsNs = LongArray(IMU_RING_CAPACITY)
    private val ringPitchDeg = FloatArray(IMU_RING_CAPACITY)
    private val ringRollDeg = FloatArray(IMU_RING_CAPACITY)
    private val ringRotationMatrices = FloatArray(IMU_RING_CAPACITY * 9)
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
     * smoothed landscape pitch sample is ready so the UI, GPU `PreviewView.translationY`, and
     * world-anchored target reticles update smoothly at display refresh rate.
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
            // so EIS translation and reticle reprojection match 120 Hz ProMotion displays.
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

    /**
     * Records a synthetic 3D orientation sample `(pitchDeg, rollDeg, yawDeg)` at [timestampNs] into
     * the IMU ring buffer and updates the live orientation state. Used by deterministic JVM unit
     * tests and synthetic simulation harnesses.
     */
    fun recordSyntheticOrientationSample(
        timestampNs: Long,
        pitchDeg: Float,
        rollDeg: Float = 0f,
        yawDeg: Float = 0f
    ) {
        val mat = buildLandscapeRotationMatrix(
            pitchDeg = pitchDeg,
            rollDeg = rollDeg,
            yawDeg = yawDeg
        )
        latestRawPitchDeg = pitchDeg
        currentForwardPitchDeg = pitchDeg
        currentRollDeg = rollDeg
        hasInitializedDisplayPitch = true
        synchronized(imuRingLock) {
            System.arraycopy(mat, 0, latestRemappedRotationMatrix, 0, 9)
            ringTimestampsNs[ringWriteIndex] = timestampNs
            ringPitchDeg[ringWriteIndex] = pitchDeg
            ringRollDeg[ringWriteIndex] = rollDeg
            System.arraycopy(mat, 0, ringRotationMatrices, ringWriteIndex * 9, 9)
            ringWriteIndex = (ringWriteIndex + 1) % IMU_RING_CAPACITY
            if (ringCount < IMU_RING_CAPACITY) ringCount++
        }
    }

    /**
     * Copies the most recent 3x3 landscape rotation matrix (`R_world_from_scr`) for 120 Hz display
     * reticle reprojection.
     */
    fun copyLatestRotationMatrix(outMatrix: FloatArray = FloatArray(9)): FloatArray {
        val overridePitch = simulatedPitchOverrideDeg
        if (overridePitch != null) {
            return buildLandscapeRotationMatrix(
                pitchDeg = overridePitch,
                rollDeg = currentRollDeg,
                yawDeg = 0f,
                outMatrix = outMatrix
            )
        }
        synchronized(imuRingLock) {
            System.arraycopy(latestRemappedRotationMatrix, 0, outMatrix, 0, 9)
        }
        return outMatrix
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
                    System.arraycopy(remappedRotationMatrix, 0, latestRemappedRotationMatrix, 0, 9)
                    ringTimestampsNs[ringWriteIndex] = event.timestamp
                    ringPitchDeg[ringWriteIndex] = rawPitch
                    ringRollDeg[ringWriteIndex] = rawRoll
                    System.arraycopy(remappedRotationMatrix, 0, ringRotationMatrices, ringWriteIndex * 9, 9)
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
     * Looks up the 3x3 landscape rotation matrix (`R_world_from_scr`) closest in timestamp to
     * [targetTimestampNs] so a detection from a camera frame is back-projected using the exact
     * orientation at frame exposure time.
     */
    fun lookupRotationMatrixAtTimestampNs(
        targetTimestampNs: Long,
        outMatrix: FloatArray = FloatArray(9)
    ): FloatArray {
        simulatedPitchOverrideDeg?.let { overridePitch ->
            return buildLandscapeRotationMatrix(
                pitchDeg = overridePitch,
                rollDeg = currentRollDeg,
                yawDeg = 0f,
                outMatrix = outMatrix
            )
        }
        synchronized(imuRingLock) {
            if (ringCount == 0) {
                System.arraycopy(latestRemappedRotationMatrix, 0, outMatrix, 0, 9)
                return outMatrix
            }
            var bestIdx = 0
            var bestDelta = Long.MAX_VALUE
            for (i in 0 until ringCount) {
                val delta = abs(ringTimestampsNs[i] - targetTimestampNs)
                if (delta < bestDelta) {
                    bestDelta = delta
                    bestIdx = i
                }
            }
            System.arraycopy(ringRotationMatrices, bestIdx * 9, outMatrix, 0, 9)
            return outMatrix
        }
    }

    /**
     * Captures the synchronized pose, timestamp-matched `cropY` stabilization state, 3x3 rotation
     * matrix at [fallbackTimestampNs], slant range `R`, and horizontal ground velocity `v_xy`.
     */
    fun capturePoseSnapshot(
        arFrame: Frame? = null,
        fallbackTimestampNs: Long = System.nanoTime(),
        verticalFocalLengthPx: Float = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX
    ): MotionPoseSnapshot {
        val frameTs = arFrame?.timestamp ?: fallbackTimestampNs
        val matchedRawPitch = lookupPitchAtTimestampNs(frameTs)
        val matchedRotationMatrix = lookupRotationMatrixAtTimestampNs(frameTs)
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
                timestampNs = ts,
                verticalFocalLengthPx = verticalFocalLengthPx,
                rotationMatrix = matchedRotationMatrix
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
            timestampNs = frameTs,
            verticalFocalLengthPx = verticalFocalLengthPx,
            rotationMatrix = matchedRotationMatrix
        )
    }

    /**
     * Raycasts a [StripDetection] onto the horizontal ground plane and computes its 3D unit bearing
     * ray in the `TYPE_GAME_ROTATION_VECTOR` world frame using the capture-timestamp rotation matrix
     * and calibrated focal length.
     */
    fun raycastStripDetection(
        detection: StripDetection,
        geometry: StripRoiGeometry,
        poseSnapshot: MotionPoseSnapshot,
        zoomRatio: Float = PushBroomCropper.DEFAULT_ZOOM_RATIO,
        arFrame: Frame? = null
    ): RaycastResult {
        val frameX = geometry.stripLeftPx + detection.centerXPx
        val frameY = poseSnapshot.cropY + detection.centerYPx
        val focalPx = poseSnapshot.verticalFocalLengthPx.coerceAtLeast(100f)

        val worldRay = pixelToWorldRay(
            frameXPx = frameX,
            frameYPx = frameY,
            focalLengthPx = focalPx,
            rotationMatrix = poseSnapshot.rotationMatrix,
            frameWidthPx = geometry.uprightFrameWidth,
            frameHeightPx = geometry.uprightFrameHeight
        )
        val angW = (detection.widthPx / focalPx).coerceAtLeast(0.004f)
        val angH = (detection.heightPx / focalPx).coerceAtLeast(0.004f)

        if (arFrame != null && arFrame.camera.trackingState == TrackingState.TRACKING) {
            val hits = arFrame.hitTest(frameX, frameY)
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
                        usedArCorePlane = true,
                        worldRay = worldRay,
                        angularWidthRad = angW,
                        angularHeightRad = angH
                    )
                }
            }
        }

        // Intersect the 3D world ray (eastX, northY, upZ) from (camX, waistHeight, camZ) with y = 0 m
        val downwardSin = (-worldRay.upZ).coerceIn(MIN_DOWNWARD_SIN, MAX_DOWNWARD_SIN)
        val slantRangeM = (waistHeightMeters / downwardSin).coerceIn(0.40f, 6.50f)
        val gsdMmPerPx = PushBroomCropper.computeGsdMmPerPx(slantRangeM, zoomRatio)
        val camPos = poseSnapshot.cameraWorldPosition

        return RaycastResult(
            worldPoint = WorldPoint3D(
                xMeters = camPos.xMeters + slantRangeM * worldRay.eastX,
                yMeters = 0.0f,
                zMeters = camPos.zMeters + slantRangeM * worldRay.northY
            ),
            slantRangeMeters = slantRangeM,
            gsdMmPerPx = gsdMmPerPx,
            usedArCorePlane = false,
            worldRay = worldRay,
            angularWidthRad = angW,
            angularHeightRad = angH
        )
    }

    /**
     * Reprojects a list of [ConfirmedTarget]s into the current camera orientation and strip `cropYFloat`
     * at 120 Hz so reticles stay locked to physical objects in the live viewport.
     */
    fun reprojectConfirmedTargetsToCurrentPose(
        targets: List<ConfirmedTarget>,
        stripGeometry: StripRoiGeometry,
        currentCropYFloat: Float,
        currentFocalLengthPx: Float,
        currentRotationMatrix: FloatArray = copyLatestRotationMatrix()
    ): List<ConfirmedTarget> {
        if (targets.isEmpty()) return targets
        return reprojectTargetsWithMatrix(
            targets = targets,
            stripGeometry = stripGeometry,
            currentCropYFloat = currentCropYFloat,
            currentFocalLengthPx = currentFocalLengthPx,
            currentRotationMatrix = currentRotationMatrix
        )
    }

    companion object {
        const val DEFAULT_WAIST_HEIGHT_METERS = 1.06f
        const val DEFAULT_PITCH_DEGREES = 45.0f
        private const val TARGET_IMU_SAMPLING_PERIOD_US = 8_000 // 125 Hz (<= 200 Hz Android 12 unprivileged cap)
        private const val MIN_DISPLAY_UPDATE_INTERVAL_NS = 8_000_000L // ~125 Hz max UI/GPU translation cadence
        private const val IMU_RING_CAPACITY = 128 // ~1.02 s history at 125 Hz
        // Vertical angular span of the 384 px tall 2x-cropped push-broom strip (~7.4 deg on 1080p)
        private const val STRIP_VERTICAL_FOV_2X_DEG = 7.36f
        private val MIN_DOWNWARD_SIN = sin(Math.toRadians(10.0)).toFloat()
        private val MAX_DOWNWARD_SIN = sin(Math.toRadians(88.0)).toFloat()

        /**
         * Builds an orthonormal 3x3 landscape rotation matrix (`R_world_from_scr`, row-major) from
         * downward optical pitch [pitchDeg] (`0` = horizon, `90` = nadir), screen roll [rollDeg],
         * and clockwise horizontal azimuth [yawDeg] (`0` = North `+Y_w`, `90` = East `+X_w`).
         *
         * Columns of `outMatrix` represent Screen Right (`+x_scr`), Screen Up (`+y_scr`), and
         * Screen Out (`+z_scr`, opposite the camera optical axis `-z_scr`) in ENU world coordinates.
         */
        fun buildLandscapeRotationMatrix(
            pitchDeg: Float,
            rollDeg: Float = 0f,
            yawDeg: Float = 0f,
            outMatrix: FloatArray = FloatArray(9)
        ): FloatArray {
            val pitchRad = Math.toRadians(pitchDeg.toDouble())
            val rollRad = Math.toRadians(rollDeg.toDouble())
            val yawRad = Math.toRadians(yawDeg.toDouble())

            val sinP = sin(pitchRad).toFloat()
            val cosP = cos(pitchRad).toFloat()
            val sinR = sin(rollRad).toFloat()
            val cosR = cos(rollRad).toFloat()
            val sinY = sin(yawRad).toFloat()
            val cosY = cos(yawRad).toFloat()

            // Exact SO(3) composition: pitch around East (+X_w), then roll around the optical axis (+z_scr):
            // (e_x, e_y, e_z) is orthonormal, and (x0, y0) rotates (e_x, e_y) in the plane orthogonal to e_z.
            val x0E = cosR
            val x0N = sinR * sinP
            val x0U = sinR * cosP

            val y0E = -sinR
            val y0N = cosR * sinP
            val y0U = cosR * cosP

            val z0E = 0f
            val z0N = -cosP
            val z0U = sinP

            // Rotate horizontal (East, North) components clockwise by yawDeg
            outMatrix[0] = cosY * x0E + sinY * x0N
            outMatrix[1] = cosY * y0E + sinY * y0N
            outMatrix[2] = cosY * z0E + sinY * z0N

            outMatrix[3] = -sinY * x0E + cosY * x0N
            outMatrix[4] = -sinY * y0E + cosY * y0N
            outMatrix[5] = -sinY * z0E + cosY * z0N

            outMatrix[6] = x0U
            outMatrix[7] = y0U
            outMatrix[8] = z0U
            return outMatrix
        }

        /**
         * Back-projects a pixel `(frameXPx, frameYPx)` in the upright `1920x1080` landscape camera
         * frame into a unit 3D ray in the `TYPE_GAME_ROTATION_VECTOR` ENU world frame using
         * [rotationMatrix] (`R_world_from_scr`) and [focalLengthPx].
         */
        fun pixelToWorldRay(
            frameXPx: Float,
            frameYPx: Float,
            focalLengthPx: Float,
            rotationMatrix: FloatArray,
            frameWidthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
            frameHeightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX
        ): WorldRay3D {
            val f = focalLengthPx.coerceAtLeast(100f)
            val cx = frameWidthPx * 0.5f
            val cy = frameHeightPx * 0.5f

            // Screen basis: +x_scr = right (+u), +y_scr = up (-v), -z_scr = optical axis into scene
            val xScr = (frameXPx - cx) / f
            val yScr = (cy - frameYPx) / f
            val zScr = -1.0f
            val invNorm = 1.0f / sqrt(xScr * xScr + yScr * yScr + zScr * zScr)
            val ux = xScr * invNorm
            val uy = yScr * invNorm
            val uz = zScr * invNorm

            val eastX = rotationMatrix[0] * ux + rotationMatrix[1] * uy + rotationMatrix[2] * uz
            val northY = rotationMatrix[3] * ux + rotationMatrix[4] * uy + rotationMatrix[5] * uz
            val upZ = rotationMatrix[6] * ux + rotationMatrix[7] * uy + rotationMatrix[8] * uz
            return WorldRay3D(eastX = eastX, northY = northY, upZ = upZ).normalized()
        }

        /**
         * Forward-projects a world-anchored unit ray [worldRay] and angular box size
         * `(angularWidthRad, angularHeightRad)` through the transpose of [currentRotationMatrix]
         * (`R_scr_from_world`) into strip-local coordinates relative to [stripGeometry] and
         * [currentCropYFloat].
         */
        fun reprojectWorldRayToStripBox(
            worldRay: WorldRay3D,
            angularWidthRad: Float,
            angularHeightRad: Float,
            currentRotationMatrix: FloatArray,
            currentFocalLengthPx: Float,
            currentCropYFloat: Float,
            stripGeometry: StripRoiGeometry = StripRoiGeometry()
        ): ReprojectedStripBox {
            val f = currentFocalLengthPx.coerceAtLeast(100f)
            val cx = stripGeometry.uprightFrameWidth * 0.5f
            val cy = stripGeometry.uprightFrameHeight * 0.5f

            // Multiply by R_world_from_scr^T: column j of R dotted with worldRay
            val xScr =
                currentRotationMatrix[0] * worldRay.eastX +
                    currentRotationMatrix[3] * worldRay.northY +
                    currentRotationMatrix[6] * worldRay.upZ
            val yScr =
                currentRotationMatrix[1] * worldRay.eastX +
                    currentRotationMatrix[4] * worldRay.northY +
                    currentRotationMatrix[7] * worldRay.upZ
            val zScr =
                currentRotationMatrix[2] * worldRay.eastX +
                    currentRotationMatrix[5] * worldRay.northY +
                    currentRotationMatrix[8] * worldRay.upZ

            val forwardDepth = -zScr
            if (forwardDepth <= 0.05f) {
                return ReprojectedStripBox(
                    stripLeftPx = -1000f,
                    stripTopPx = -1000f,
                    stripRightPx = -990f,
                    stripBottomPx = -990f,
                    isVisibleInStrip = false
                )
            }

            val frameX = cx + f * (xScr / forwardDepth)
            val frameY = cy - f * (yScr / forwardDepth)

            val stripCenterX = frameX - stripGeometry.stripLeftPx
            val stripCenterY = frameY - currentCropYFloat

            val halfW = max(6f, angularWidthRad * f * 0.5f)
            val halfH = max(6f, angularHeightRad * f * 0.5f)

            val leftPx = stripCenterX - halfW
            val rightPx = stripCenterX + halfW
            val topPx = stripCenterY - halfH
            val bottomPx = stripCenterY + halfH

            val stripW = stripGeometry.stripWidthPx.toFloat()
            val stripH = stripGeometry.stripHeightPx.toFloat()

            // Consider visible when the box center is inside the strip (with a small margin for edge targets)
            val marginX = min(24f, halfW * 0.5f)
            val marginY = min(24f, halfH * 0.5f)
            val isVisible =
                stripCenterX in -marginX..(stripW + marginX) &&
                    stripCenterY in -marginY..(stripH + marginY) &&
                    rightPx > 0f &&
                    leftPx < stripW &&
                    bottomPx > 0f &&
                    topPx < stripH

            return ReprojectedStripBox(
                stripLeftPx = leftPx,
                stripTopPx = topPx,
                stripRightPx = rightPx,
                stripBottomPx = bottomPx,
                isVisibleInStrip = isVisible
            )
        }

        /**
         * Reprojects every [ConfirmedTarget] that carries a [ConfirmedTarget.worldRay] into the
         * current camera orientation [currentRotationMatrix] and [currentCropYFloat]. Targets
         * without a [worldRay] (e.g., synthetic tests) are returned unchanged.
         */
        fun reprojectTargetsWithMatrix(
            targets: List<ConfirmedTarget>,
            stripGeometry: StripRoiGeometry,
            currentCropYFloat: Float,
            currentFocalLengthPx: Float,
            currentRotationMatrix: FloatArray
        ): List<ConfirmedTarget> {
            if (targets.isEmpty()) return targets
            return targets.map { target ->
                val ray = target.worldRay ?: return@map target
                val box = reprojectWorldRayToStripBox(
                    worldRay = ray,
                    angularWidthRad = target.angularWidthRad,
                    angularHeightRad = target.angularHeightRad,
                    currentRotationMatrix = currentRotationMatrix,
                    currentFocalLengthPx = currentFocalLengthPx,
                    currentCropYFloat = currentCropYFloat,
                    stripGeometry = stripGeometry
                )
                target.copy(
                    stripLeftPx = box.stripLeftPx,
                    stripTopPx = box.stripTopPx,
                    stripRightPx = box.stripRightPx,
                    stripBottomPx = box.stripBottomPx,
                    isVisibleInStrip = box.isVisibleInStrip
                )
            }
        }

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
