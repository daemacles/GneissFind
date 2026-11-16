/*
 * Asynchronous field dataset recorder for LOG DATA Mode.
 *
 * Consumes shutter-clamped, IMU-pitch-stabilized 1792x384 (or 1088x384) ARGB strip pixels
 * and per-frame camera/IMU telemetry from ScannerViewModel, and writes lossless PNG strip
 * images plus metadata.jsonl records to app-specific external media storage for offline
 * ingestion by /ml-pipeline via adb pull.
 *
 * Main types:
 * - FieldStripMetadata: per-frame camera, IMU, and geometry telemetry record.
 * - FieldDataLogger: single-threaded background PNG + JSONL session writer with non-blocking drop-on-busy backpressure.
 */
package com.gneissfind.app.logging

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.gneissfind.app.model.StripRoiGeometry
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Metadata record persisted per captured stabilized `1792x384` (or `1088x384`) strip frame in `metadata.jsonl`.
 *
 * @property timestampNs Camera sensor timestamp (`SENSOR_TIMESTAMP` / `ImageProxy.imageInfo.timestamp`) in nanoseconds.
 * @property wallTimeMs System wall-clock epoch timestamp (`System.currentTimeMillis()`) in milliseconds.
 * @property stripWidthPx Width of the recorded push-broom strip in pixels (`1792` for 5-tile or `1088` for 3-tile).
 * @property stripHeightPx Height of the recorded push-broom strip in pixels ([StripRoiGeometry.STRIP_HEIGHT_PX]).
 * @property tileCount Number of active `384x384` tiles in the strip (`5` or `3`).
 * @property zoomRatio Active Camera2 digital crop zoom ratio (`2.0f` nominal).
 * @property exposureTimeNs Sensor exposure time (`SENSOR_EXPOSURE_TIME`) in nanoseconds.
 * @property sensorIso Sensor sensitivity (`SENSOR_SENSITIVITY`) in ISO units.
 * @property shutterClamped `true` when `SENSOR_EXPOSURE_TIME` is within the `<= 1/2000s` (`500,000 ns`) motion-blur ceiling.
 * @property exposureBiasEv Active AE exposure compensation bias in EV stops (`-1.0f` when clamping is active).
 * @property imuPitchDeg Current IMU inclination below horizontal in degrees (`45.0 deg` nominal).
 * @property referencePitchDeg Reference pitch angle in degrees (`45.0 deg`) around which vertical ROI stabilization is centered.
 * @property cropY Top Y offset in `1920x1080` sensor coordinates (`0..696 px`) of the extracted `384 px` strip.
 * @property isPitchClamped `true` when [cropY] hit the `0` or [StripRoiGeometry.MAX_CROP_Y_PX] sensor boundary (`PITCH LIMIT`).
 * @property slantRangeMeters Estimated optical slant range $R$ from lens to gravel bar along the strip center in meters.
 * @property gsdMmPerPx Estimated ground sample distance in millimeters per pixel (`0.59 mm/px` nominal at `1.2 m` height).
 * @property horizontalSpeedMps Estimated horizontal walking speed over ground in meters per second.
 */
data class FieldStripMetadata(
    val timestampNs: Long,
    val wallTimeMs: Long,
    val stripWidthPx: Int = StripRoiGeometry.STRIP_WIDTH_5_TILE_PX,
    val stripHeightPx: Int = StripRoiGeometry.STRIP_HEIGHT_PX,
    val tileCount: Int = StripRoiGeometry.MAX_TILES,
    val zoomRatio: Float = 2.0f,
    val exposureTimeNs: Long,
    val sensorIso: Int,
    val shutterClamped: Boolean,
    val exposureBiasEv: Float,
    val imuPitchDeg: Float,
    val referencePitchDeg: Float = 45.0f,
    val cropY: Int = StripRoiGeometry.NOMINAL_CROP_Y_PX,
    val isPitchClamped: Boolean = false,
    val slantRangeMeters: Float,
    val gsdMmPerPx: Float,
    val horizontalSpeedMps: Float
)

/**
 * Asynchronous field dataset recorder for `LOG DATA` Mode.
 * Saves shutter-clamped, IMU-pitch-stabilized `1792x384` (or `1088x384`) lossless PNG strip frames
 * at [DEFAULT_CAPTURE_HZ] (`2–5 Hz` spec cadence per `.agents/rules/android-camera-ar.md` §4) alongside `metadata.jsonl` inside:
 * `Android/media/com.gneissfind.app/sessions/<timestamp>/`
 * for zero-permission `adb pull` ingestion by `/ml-pipeline` on the Dell T7920.
 *
 * State & Memory:
 * - Pre-allocates a single [workerPixelCopy] `IntArray` and a single mutable [reusableStripBitmap] owned
 *   exclusively by [ioExecutor] so PNG compression performs zero per-frame full-bitmap heap allocations.
 *
 * Lifecycle & Threading:
 * - [startSession] and [stopSession] are `@Synchronized`.
 * - [maybeRecordStripFrame] is called on the camera analysis thread; if [minFrameIntervalNs] has elapsed and
 *   [isWriterBusy] is `false`, it copies pixels into [workerPixelCopy] before dispatching disk I/O to [ioExecutor].
 * - Call [close] when the owning ViewModel is cleared to flush `metadata.jsonl` and shut down [ioExecutor].
 *
 * @param appContext Application [Context] used to resolve external media directories.
 * @param targetCaptureHz Target strip logging cadence in Hz (clamped to `1.0..15.0 Hz`, defaults to [DEFAULT_CAPTURE_HZ]).
 */
class FieldDataLogger(
    private val appContext: Context,
    targetCaptureHz: Float = DEFAULT_CAPTURE_HZ
) : AutoCloseable {

    /** Minimum elapsed sensor timestamp interval in nanoseconds between consecutive logged frames. */
    private val minFrameIntervalNs: Long =
        (1_000_000_000.0 / targetCaptureHz.coerceIn(1.0f, 15.0f)).toLong()

    /** Single-threaded daemon executor serializing PNG compression and `metadata.jsonl` appends. */
    private val ioExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "GneissFind-FieldLogger").apply { isDaemon = true }
    }

    /** Non-blocking backpressure gate ensuring at most one frame is queued/writing on [ioExecutor] at any instant. */
    private val isWriterBusy = AtomicBoolean(false)

    /** Count of PNG strip frames written in the current (or most recent) logging session. */
    private val savedFramesCounter = AtomicInteger(0)

    /** Directory of the currently active `LOG DATA` session, or `null` when logging is inactive. */
    @Volatile
    private var activeSessionDir: File? = null

    /** Buffered writer appending to `<activeSessionDir>/metadata.jsonl`, owned by [ioExecutor] after creation. */
    @Volatile
    private var metadataWriter: BufferedWriter? = null

    /** Timestamp string (`yyyyMMdd_HHmmss`) identifying the active session. */
    @Volatile
    private var activeSessionId: String = ""

    /** Sensor timestamp (`ns`) of the most recently accepted frame for cadence throttling. */
    private var lastQueuedTimestampNs: Long = 0L

    /** Single reusable mutable strip [Bitmap] owned exclusively by [ioExecutor] (zero per-frame bitmap allocation). */
    private var reusableStripBitmap: Bitmap? = null

    /** Pre-allocated pixel staging buffer copied on the caller thread while [isWriterBusy] is held and read by [ioExecutor]. */
    private val workerPixelCopy = IntArray(
        StripRoiGeometry.MAX_STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX
    )

    /** Number of PNG strip frames persisted so far in the current session. */
    val savedFrameCount: Int
        get() = savedFramesCounter.get()

    /** Absolute filesystem path of the active session directory, or the base `sessions/` parent directory when idle. */
    val currentSessionPath: String
        get() = activeSessionDir?.absolutePath ?: resolveBaseSessionsDir(appContext).absolutePath

    /**
     * Opens a new timestamped session directory under `Android/media/com.gneissfind.app/sessions/<yyyyMMdd_HHmmss>/`
     * and initializes its `metadata.jsonl` writer, or returns the existing directory if a session is already active.
     *
     * @return The active session [File] directory.
     */
    @Synchronized
    fun startSession(): File {
        activeSessionDir?.let { return it }

        val formatter = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        val sessionId = formatter.format(Date())
        val baseDir = resolveBaseSessionsDir(appContext)
        val sessionDir = File(baseDir, sessionId).apply { mkdirs() }

        val metaFile = File(sessionDir, "metadata.jsonl")
        metadataWriter = BufferedWriter(FileWriter(metaFile, true))
        activeSessionId = sessionId
        activeSessionDir = sessionDir
        savedFramesCounter.set(0)
        lastQueuedTimestampNs = 0L

        Log.i(TAG, "Started LOG DATA session at ${sessionDir.absolutePath}")
        return sessionDir
    }

    /**
     * Marks the current logging session inactive and schedules flushing and closing of `metadata.jsonl` on [ioExecutor].
     * No-op if no session is currently active.
     */
    @Synchronized
    fun stopSession() {
        val dir = activeSessionDir ?: return
        activeSessionDir = null
        ioExecutor.execute {
            runCatching { metadataWriter?.flush() }
            runCatching { metadataWriter?.close() }
            metadataWriter = null
            Log.i(TAG, "Closed LOG DATA session ${dir.absolutePath} (${savedFramesCounter.get()} strips)")
        }
    }

    /**
     * Submits a shutter-clamped, pitch-stabilized ARGB strip for non-blocking disk write if a session
     * is active, the [minFrameIntervalNs] cadence interval has elapsed, and the background writer is idle.
     *
     * @param stripArgbPixels Packed `0xAARRGGBB` pixel array of at least `metadata.stripWidthPx * metadata.stripHeightPx` elements.
     * @param metadata Per-frame camera, IMU, and geometry telemetry to serialize alongside the PNG.
     * @return `true` if the frame was accepted and queued to [ioExecutor], or `false` if dropped due to inactive session, cadence throttling, or busy writer.
     */
    fun maybeRecordStripFrame(
        stripArgbPixels: IntArray,
        metadata: FieldStripMetadata
    ): Boolean {
        val sessionDir = activeSessionDir ?: return false
        if (lastQueuedTimestampNs != 0L &&
            (metadata.timestampNs - lastQueuedTimestampNs) < minFrameIntervalNs
        ) {
            return false
        }
        if (!isWriterBusy.compareAndSet(false, true)) {
            return false
        }

        val stripW = metadata.stripWidthPx.coerceIn(1, StripRoiGeometry.MAX_STRIP_WIDTH_PX)
        val stripH = metadata.stripHeightPx.coerceIn(1, StripRoiGeometry.STRIP_HEIGHT_PX)
        val pixelCount = (stripW * stripH).coerceAtMost(workerPixelCopy.size)

        lastQueuedTimestampNs = metadata.timestampNs
        System.arraycopy(stripArgbPixels, 0, workerPixelCopy, 0, pixelCount)
        val sessionIdSnapshot = activeSessionId

        ioExecutor.execute {
            try {
                val frameNumber = savedFramesCounter.incrementAndGet()
                val fileName = String.format(Locale.US, "strip_%06d.png", frameNumber)
                val pngFile = File(sessionDir, fileName)

                val existingBmp = reusableStripBitmap
                val bmp = if (existingBmp != null && existingBmp.width == stripW && existingBmp.height == stripH) {
                    existingBmp
                } else {
                    existingBmp?.recycle()
                    Bitmap.createBitmap(stripW, stripH, Bitmap.Config.ARGB_8888).also {
                        reusableStripBitmap = it
                    }
                }

                bmp.setPixels(workerPixelCopy, 0, stripW, 0, 0, stripW, stripH)

                BufferedOutputStream(FileOutputStream(pngFile), 64 * 1024).use { out ->
                    bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
                }

                val jsonLine = buildMetadataJsonLine(
                    sessionId = sessionIdSnapshot,
                    frameIndex = frameNumber,
                    fileName = fileName,
                    metadata = metadata
                )

                metadataWriter?.apply {
                    write(jsonLine)
                    newLine()
                    flush()
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Failed writing field strip log", t)
            } finally {
                isWriterBusy.set(false)
            }
        }
        return true
    }

    override fun close() {
        stopSession()
        ioExecutor.shutdown()
    }

    companion object {
        private const val TAG = "FieldDataLogger"

        /** Default strip logging rate (`4.0 Hz`), inside the locked `2–5 Hz` v2 strip logger cadence (`.agents/rules/android-camera-ar.md` §4). */
        const val DEFAULT_CAPTURE_HZ = 4.0f

        /**
         * Resolves and creates if necessary the base `sessions/` directory under
         * `/storage/emulated/0/Android/media/com.gneissfind.app/sessions/` (falling back to external files or internal files if unavailable).
         */
        @Suppress("DEPRECATION")
        fun resolveBaseSessionsDir(context: Context): File {
            // Maps directly to /storage/emulated/0/Android/media/com.gneissfind.app/sessions/
            val mediaRoot = context.externalMediaDirs?.firstOrNull()
                ?: context.getExternalFilesDir(null)
                ?: context.filesDir
            return File(mediaRoot, "sessions").apply { mkdirs() }
        }

        /**
         * Serializes a single compact JSON line for `metadata.jsonl`.
         *
         * Emits both short and canonical key aliases (`exposure_ns`/`exposure_time_ns`, `iso`/`sensor_iso`,
         * `pitch_deg`/`imu_pitch_deg`, `gsd_mm_px`/`gsd_mm_per_px`, `ground_speed_mps`/`walk_speed_mps`)
         * so downstream Python ingestion scripts in `/ml-pipeline` can query either schema convention.
         *
         * @param sessionId Timestamp identifier (`yyyyMMdd_HHmmss`) of the recording session.
         * @param frameIndex 1-based sequential index of the saved strip within the session.
         * @param fileName Relative PNG filename (e.g. `strip_000001.png`).
         * @param metadata Captured [FieldStripMetadata] telemetry for the strip.
         * @return Single-line JSON string (without trailing newline).
         */
        fun buildMetadataJsonLine(
            sessionId: String,
            frameIndex: Int,
            fileName: String,
            metadata: FieldStripMetadata
        ): String {
            return JSONObject().apply {
                put("frame_id", frameIndex)
                put("session_id", sessionId)
                put("file_name", fileName)
                put("timestamp_ns", metadata.timestampNs)
                put("wall_time_ms", metadata.wallTimeMs)
                put("strip_width", metadata.stripWidthPx)
                put("strip_height", metadata.stripHeightPx)
                put("tile_count", metadata.tileCount)
                put("zoom_ratio", round3(metadata.zoomRatio))
                put("exposure_ns", metadata.exposureTimeNs)
                put("exposure_time_ns", metadata.exposureTimeNs)
                put("iso", metadata.sensorIso)
                put("sensor_iso", metadata.sensorIso)
                put("shutter_clamped", metadata.shutterClamped)
                put("exposure_bias_ev", round3(metadata.exposureBiasEv))
                put("pitch_deg", round3(metadata.imuPitchDeg))
                put("imu_pitch_deg", round3(metadata.imuPitchDeg))
                put("ref_pitch_deg", round3(metadata.referencePitchDeg))
                put("crop_y", metadata.cropY)
                put("is_pitch_clamped", metadata.isPitchClamped)
                put("slant_range_m", round3(metadata.slantRangeMeters))
                put("gsd_mm_px", round3(metadata.gsdMmPerPx))
                put("gsd_mm_per_px", round3(metadata.gsdMmPerPx))
                put("ground_speed_mps", round3(metadata.horizontalSpeedMps))
                put("walk_speed_mps", round3(metadata.horizontalSpeedMps))
            }.toString()
        }

        /** Rounds [value] to 3 decimal places for compact JSONL serialization. */
        private fun round3(value: Float): Double =
            kotlin.math.round(value * 1000.0) / 1000.0
    }
}
