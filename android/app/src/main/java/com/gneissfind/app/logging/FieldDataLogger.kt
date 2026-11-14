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
 * Metadata record persisted per captured 1088x384 strip frame in `metadata.jsonl`.
 */
data class FieldStripMetadata(
    val timestampNs: Long,
    val wallTimeMs: Long,
    val imuPitchDeg: Float,
    val slantRangeMeters: Float,
    val gsdMmPerPx: Float,
    val horizontalSpeedMps: Float,
    val exposureTimeNs: Long,
    val sensorIso: Int,
    val shutterClamped: Boolean,
    val exposureBiasEv: Float
)

/**
 * Asynchronous field dataset recorder for `LOG DATA` Mode.
 * Saves shutter-clamped `1088x384` lossless PNG strip frames at `5–10 Hz` (default `6 Hz`)
 * alongside a companion `metadata.jsonl` inside:
 * `Android/media/com.gneissfind.app/field_logs/<session_id>/`
 * for zero-permission `adb pull` ingestion by `/ml-pipeline` on the Dell T7920.
 */
class FieldDataLogger(
    private val appContext: Context,
    targetCaptureHz: Float = DEFAULT_CAPTURE_HZ
) : AutoCloseable {

    private val minFrameIntervalNs: Long =
        (1_000_000_000.0 / targetCaptureHz.coerceIn(1.0f, 15.0f)).toLong()

    private val ioExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "GneissFind-FieldLogger").apply { isDaemon = true }
    }

    private val isWriterBusy = AtomicBoolean(false)
    private val savedFramesCounter = AtomicInteger(0)

    @Volatile
    private var activeSessionDir: File? = null

    @Volatile
    private var metadataWriter: BufferedWriter? = null

    @Volatile
    private var activeSessionId: String = ""

    private var lastQueuedTimestampNs: Long = 0L

    // Single reusable 1088x384 mutable Bitmap owned exclusively by ioExecutor (zero full-frame 4K allocation)
    private var reusableStripBitmap: Bitmap? = null
    private val workerPixelCopy = IntArray(StripRoiGeometry.STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX)

    val savedFrameCount: Int
        get() = savedFramesCounter.get()

    val currentSessionPath: String
        get() = activeSessionDir?.absolutePath ?: resolveBaseFieldLogsDir(appContext).absolutePath

    @Synchronized
    fun startSession(): File {
        activeSessionDir?.let { return it }

        val formatter = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        val sessionId = "session_${formatter.format(Date())}"
        val baseDir = resolveBaseFieldLogsDir(appContext)
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
     * Submits a shutter-clamped 1088x384 ARGB strip for non-blocking disk write if the 5–10 Hz
     * cadence interval has elapsed and the background writer is idle.
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

        lastQueuedTimestampNs = metadata.timestampNs
        System.arraycopy(stripArgbPixels, 0, workerPixelCopy, 0, workerPixelCopy.size)
        val sessionIdSnapshot = activeSessionId

        ioExecutor.execute {
            try {
                val frameNumber = savedFramesCounter.incrementAndGet()
                val fileName = String.format(Locale.US, "strip_%06d.png", frameNumber)
                val pngFile = File(sessionDir, fileName)

                val bmp = reusableStripBitmap ?: Bitmap.createBitmap(
                    StripRoiGeometry.STRIP_WIDTH_PX,
                    StripRoiGeometry.STRIP_HEIGHT_PX,
                    Bitmap.Config.ARGB_8888
                ).also { reusableStripBitmap = it }

                bmp.setPixels(
                    workerPixelCopy,
                    0,
                    StripRoiGeometry.STRIP_WIDTH_PX,
                    0,
                    0,
                    StripRoiGeometry.STRIP_WIDTH_PX,
                    StripRoiGeometry.STRIP_HEIGHT_PX
                )

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
        const val DEFAULT_CAPTURE_HZ = 6.0f // Inside locked 5–10 Hz target cadence

        @Suppress("DEPRECATION")
        fun resolveBaseFieldLogsDir(context: Context): File {
            // Maps directly to /storage/emulated/0/Android/media/com.gneissfind.app/field_logs/
            val mediaRoot = context.externalMediaDirs?.firstOrNull()
                ?: context.getExternalFilesDir(null)
                ?: context.filesDir
            return File(mediaRoot, "field_logs").apply { mkdirs() }
        }

        fun buildMetadataJsonLine(
            sessionId: String,
            frameIndex: Int,
            fileName: String,
            metadata: FieldStripMetadata
        ): String {
            return JSONObject().apply {
                put("session_id", sessionId)
                put("frame_index", frameIndex)
                put("file_name", fileName)
                put("timestamp_ns", metadata.timestampNs)
                put("wall_time_ms", metadata.wallTimeMs)
                put("strip_width", StripRoiGeometry.STRIP_WIDTH_PX)
                put("strip_height", StripRoiGeometry.STRIP_HEIGHT_PX)
                put("imu_pitch_deg", round3(metadata.imuPitchDeg))
                put("slant_range_m", round3(metadata.slantRangeMeters))
                put("gsd_mm_per_px", round3(metadata.gsdMmPerPx))
                put("walk_speed_mps", round3(metadata.horizontalSpeedMps))
                put("exposure_time_ns", metadata.exposureTimeNs)
                put("sensor_iso", metadata.sensorIso)
                put("shutter_clamped", metadata.shutterClamped)
                put("exposure_bias_ev", round3(metadata.exposureBiasEv))
            }.toString()
        }

        private fun round3(value: Float): Double =
            kotlin.math.round(value * 1000.0) / 1000.0
    }
}
