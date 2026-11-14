package com.gneissfind.app.tracking

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.gneissfind.app.model.StripDetection
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.WorldPoint3D
import com.gneissfind.app.vision.PushBroomCropper
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.TrackingState
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Result of raycasting a 2D strip detection onto the horizontal gravel-bar ground plane.
 */
data class RaycastResult(
    val worldPoint: WorldPoint3D,
    val slantRangeMeters: Float,
    val gsdMmPerPx: Float,
    val usedArCorePlane: Boolean
)

/**
 * Snapshot of current sensor/VIO pose, pitch angle, slant range, and ground speed.
 */
data class MotionPoseSnapshot(
    val cameraWorldPosition: WorldPoint3D,
    val forwardPitchDeg: Float,
    val centerSlantRangeMeters: Float,
    val horizontalSpeedMps: Float,
    val arPlaneLocked: Boolean,
    val timestampNs: Long
)

/**
 * Projects 2D detections inside the 1088x384 push-broom strip onto 3D ground coordinates `(X, Y, Z)`
 * and computes slant range `R`:
 *  1. Primary path (when an active ARCore [Frame] has horizontal plane lock): performs
 *     `Frame.hitTest(screenX, screenY)` against `Plane.Type.HORIZONTAL_UPWARD_FACING` and reads
 *     exact `Camera.pose` 6-DoF translation.
 *  2. Continuous fallback path (when ARCore plane lock is initializing or CameraX holds exclusive
 *     4K shutter-clamped camera control): uses fused IMU rotation-vector pitch + linear acceleration
 *     damped velocity integration from nominal waist height `h = 1.06 m` (`~45 deg` -> `R ~ 1.50 m`).
 */
class GroundPlaneRaycaster(
    context: Context? = null,
    private val waistHeightMeters: Float = DEFAULT_WAIST_HEIGHT_METERS
) : SensorEventListener, AutoCloseable {

    private val sensorManager: SensorManager? =
        context?.applicationContext?.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
    private val rotationSensor: Sensor? =
        sensorManager?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    private val linearAccelSensor: Sensor? =
        sensorManager?.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

    private val rotationMatrix = FloatArray(9)

    @Volatile
    var currentForwardPitchDeg: Float = DEFAULT_PITCH_DEGREES
        private set

    @Volatile
    var imuHorizontalSpeedMps: Float = 0.0f
        private set

    @Volatile
    var simulatedWalkSpeedOverrideMps: Float? = null

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
        rotationSensor?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        linearAccelSensor?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    fun stopImuListening() {
        sensorManager?.unregisterListener(this)
    }

    override fun close() {
        stopImuListening()
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ROTATION_VECTOR -> {
                SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                // Device optical axis (-Z in sensor coords) dot world gravity/up axis (row 2 of R):
                // R[6], R[7], R[8] represent device Z axis in world ENU coordinates (where +Z_world is Up).
                // Camera optical look vector is -Z_device => world Z component of look vector is -R[8].
                // Depression angle below horizon: sin(pitchDown) = +R[8] when back of phone faces sky/downward.
                val lookDownSin = rotationMatrix[8].coerceIn(-1f, 1f)
                val horizMag = sqrt((1f - lookDownSin * lookDownSin).coerceAtLeast(1e-6f))
                val depressionDeg = Math.toDegrees(atan2(lookDownSin.toDouble(), horizMag.toDouble())).toFloat()
                // Clamp to practical scanning depression range [18 deg .. 82 deg] for stable raycasting
                if (depressionDeg > 10f) {
                    currentForwardPitchDeg = currentForwardPitchDeg * 0.80f + depressionDeg.coerceIn(18f, 82f) * 0.20f
                }
            }

            Sensor.TYPE_LINEAR_ACCELERATION -> {
                val nowNs = event.timestamp
                if (lastLinearAccelNs != 0L) {
                    val dt = ((nowNs - lastLinearAccelNs) * 1e-9f).coerceIn(0f, 0.1f)
                    val ax = event.values[0]
                    val ay = event.values[1]
                    // Apply deadband and leaky integrator damping for handheld walking cadence estimation
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
     * Updates pose and ground speed from an ARCore [Frame] when available, or returns the live
     * IMU-assisted pose snapshot.
     */
    fun capturePoseSnapshot(arFrame: Frame? = null, fallbackTimestampNs: Long = System.nanoTime()): MotionPoseSnapshot {
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

            val pitchDeg = currentForwardPitchDeg
            val slantR = computeSlantRangeForPitch(waistHeightMeters, pitchDeg)

            return MotionPoseSnapshot(
                cameraWorldPosition = WorldPoint3D(px, py, pz),
                forwardPitchDeg = pitchDeg,
                centerSlantRangeMeters = slantR,
                horizontalSpeedMps = overrideSpeed ?: smoothedArSpeedMps,
                arPlaneLocked = hasHorizontalPlane,
                timestampNs = ts
            )
        }

        val activeSpeed = overrideSpeed ?: imuHorizontalSpeedMps
        if (overrideSpeed != null && lastLinearAccelNs == 0L) {
            // Advance virtual walk Z position when desk-testing simulated walk speed
            integratedZ += overrideSpeed * (1f / 15f)
        }
        val pitchDeg = currentForwardPitchDeg
        val slantR = computeSlantRangeForPitch(waistHeightMeters, pitchDeg)

        return MotionPoseSnapshot(
            cameraWorldPosition = WorldPoint3D(integratedX, waistHeightMeters, integratedZ),
            forwardPitchDeg = pitchDeg,
            centerSlantRangeMeters = slantR,
            horizontalSpeedMps = activeSpeed,
            arPlaneLocked = false,
            timestampNs = fallbackTimestampNs
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
                    val gsd = PushBroomCropper.computeGsdMmPerPx(slantR)
                    return RaycastResult(
                        worldPoint = worldPt,
                        slantRangeMeters = slantR,
                        gsdMmPerPx = gsd,
                        usedArCorePlane = true
                    )
                }
            }
        }

        return raycastTrigonometric(
            stripCenterXPx = detection.centerXPx,
            stripCenterYPx = detection.centerYPx,
            cameraWorldPos = poseSnapshot.cameraWorldPosition,
            forwardPitchDeg = poseSnapshot.forwardPitchDeg,
            waistHeightMeters = waistHeightMeters
        )
    }

    companion object {
        const val DEFAULT_WAIST_HEIGHT_METERS = 1.06f
        const val DEFAULT_PITCH_DEGREES = 45.0f
        // Vertical angular span of the 384 px tall 2x-cropped push-broom strip (~11.8 deg)
        private const val STRIP_VERTICAL_FOV_DEG = 11.8f

        fun computeSlantRangeForPitch(
            waistHeightMeters: Float = DEFAULT_WAIST_HEIGHT_METERS,
            forwardPitchDeg: Float = DEFAULT_PITCH_DEGREES
        ): Float {
            val clampedPitchRad = Math.toRadians(forwardPitchDeg.coerceIn(18.0f, 85.0f).toDouble())
            return (waistHeightMeters / sin(clampedPitchRad)).toFloat().coerceIn(0.40f, 4.00f)
        }

        /**
         * Pure trigonometric ground-plane projection from waist height `h` at depression angle
         * `forwardPitchDeg` (`45 deg` -> `R = 1.50 m`, `Z_ahead = 1.06 m`).
         */
        fun raycastTrigonometric(
            stripCenterXPx: Float,
            stripCenterYPx: Float,
            cameraWorldPos: WorldPoint3D = WorldPoint3D(0f, DEFAULT_WAIST_HEIGHT_METERS, 0f),
            forwardPitchDeg: Float = DEFAULT_PITCH_DEGREES,
            waistHeightMeters: Float = DEFAULT_WAIST_HEIGHT_METERS
        ): RaycastResult {
            // Top of strip (y = 0) looks slightly shallower (further ahead); bottom (y = 384) looks steeper (closer)
            val normYFromCenter = (stripCenterYPx - StripRoiGeometry.STRIP_HEIGHT_PX * 0.5f) /
                StripRoiGeometry.STRIP_HEIGHT_PX.toFloat()
            val rayPitchDeg = (forwardPitchDeg + normYFromCenter * STRIP_VERTICAL_FOV_DEG).coerceIn(18.0f, 85.0f)
            val rayPitchRad = Math.toRadians(rayPitchDeg.toDouble())

            val slantRangeM = (waistHeightMeters / sin(rayPitchRad)).toFloat().coerceIn(0.40f, 4.00f)
            val forwardGroundDistM = (waistHeightMeters / tan(rayPitchRad)).toFloat()

            val gsdMmPerPx = PushBroomCropper.computeGsdMmPerPx(slantRangeM)
            val gsdMetersPerPx = gsdMmPerPx * 0.001f
            val crossTrackOffsetM =
                (stripCenterXPx - StripRoiGeometry.STRIP_WIDTH_PX * 0.5f) * gsdMetersPerPx

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
