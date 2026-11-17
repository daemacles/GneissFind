/*
 * Hardware MediaCodec 30 FPS 1080p H.264 MP4 + 120 Hz IMU JSONL trace recorder for MOD: LOG.
 *
 * Consumes upright 1920x1080 YUV_420_888 frames directly from CameraX ImageAnalysis (< 1.5 ms
 * copy into a bounded 4-slot NV12 pool—never binding a 3rd CameraX VideoCapture surface) alongside
 * 120 Hz remapped IMU rotation matrices from GroundPlaneRaycaster, encoding clip.mp4 (video/avc,
 * 1920x1080, 30 FPS, 14 Mbps, 1s keyframe interval) via MediaCodec + MediaMuxer paired with
 * trace.jsonl inside Android/media/com.gneissfind.app/replays/<YYYYMMDD_HHMMSS>/.
 *
 * Main types:
 * - ReplayTraceHeader / ReplayImuRecord / ReplayFrameRecord / ReplayTraceBundle: trace.jsonl schema types.
 * - FieldFrameTelemetry: per-frame Camera2 optics and monotonic timestamp snapshot.
 * - FieldDataLogger: non-blocking background MediaCodec + MediaMuxer + trace.jsonl session recorder.
 */
package com.gneissfind.app.logging

import android.content.Context
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.util.Log
import android.view.Surface
import com.gneissfind.app.BuildConfig
import com.gneissfind.app.camera.PushBroomCameraController
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.vision.PushBroomCropper
import com.gneissfind.app.vision.Yuv420Frame
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.File
import java.io.FileReader
import java.io.FileWriter
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.min

/**
 * Line-0 `"type": "header"` record of `trace.jsonl` describing stream geometry, optics, and git provenance.
 */
data class ReplayTraceHeader(
    val schemaVersion: Int = 1,
    val width: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
    val height: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX,
    val fps: Int = FieldDataLogger.DEFAULT_VIDEO_FPS,
    val zoomRatio: Float = PushBroomCropper.DEFAULT_ZOOM_RATIO,
    val focalLengthPx: Float = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX,
    val refPitchDeg: Float = 45.0f,
    val displayRotation: Int = Surface.ROTATION_90,
    val gitHash: String = BuildConfig.GIT_COMMIT_HASH_6
)

/**
 * Continuous `120 Hz` `"type": "imu"` record in `trace.jsonl` on the aligned `System.nanoTime()` timeline.
 */
class ReplayImuRecord(
    val timestampNs: Long,
    val rotationMatrix: FloatArray,
    val pitchDeg: Float
)

/**
 * Per-video-frame `"type": "frame"` record in `trace.jsonl` linking `clip.mp4` `pts_us` to `t_ns` and Camera2 optics.
 */
data class ReplayFrameRecord(
    val frameIdx: Int,
    val ptsUs: Long,
    val timestampNs: Long,
    val exposureNs: Long,
    val iso: Int,
    val zoomRatio: Float,
    val shutterClamp: Boolean,
    val evBias: Float
)

/**
 * Parsed contents of a `<session>/trace.jsonl` sidecar file consumed by `ReplayCameraSource`.
 */
data class ReplayTraceBundle(
    val header: ReplayTraceHeader,
    val imuRecords: List<ReplayImuRecord>,
    val frameRecords: List<ReplayFrameRecord>
) {
    /** Total clip duration in seconds derived from `pts_us` or frame count at `header.fps`. */
    val durationSec: Int
        get() {
            if (frameRecords.size >= 2) {
                val spanUs = (frameRecords.last().ptsUs - frameRecords.first().ptsUs).coerceAtLeast(0L)
                return ((spanUs + 500_000L) / 1_000_000L).toInt().coerceAtLeast(1)
            }
            if (frameRecords.size == 1) return 1
            return 0
        }
}

/**
 * Per-frame Camera2 exposure, optics, and monotonic timestamp metadata submitted with each `1080p` frame.
 */
data class FieldFrameTelemetry(
    val timestampMonoNs: Long,
    val exposureTimeNs: Long,
    val sensorIso: Int,
    val zoomRatio: Float = PushBroomCropper.DEFAULT_ZOOM_RATIO,
    val focalLengthPx: Float = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX,
    val referencePitchDeg: Float = 45.0f,
    val displayRotation: Int = Surface.ROTATION_90,
    val shutterClamped: Boolean = true,
    val exposureBiasEv: Float = 0.0f
)

/**
 * Asynchronous hardware `MediaCodec` (`video/avc`, `1920x1080`, `30 FPS`, `14 Mbps`) + `MediaMuxer`
 * MP4 and `120 Hz` IMU `trace.jsonl` recorder for `MOD: LOG` (`.agents/rules/android-camera-ar.md §2`).
 *
 * Zero-CameraX-Rebind Guarantee:
 * - Never binds a `VideoCapture` surface; copies upright `NV12` bytes directly from the live
 *   `ImageAnalysis` `YUV_420_888` frame in `< 1.5 ms` into a bounded 4-slot pool before `imageProxy.close()`.
 */
class FieldDataLogger(
    private val appContext: Context,
    private val videoWidthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
    private val videoHeightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX,
    private val videoFps: Int = DEFAULT_VIDEO_FPS,
    private val videoBitrateBps: Int = DEFAULT_VIDEO_BITRATE_BPS
) : AutoCloseable {

    private class PooledNv12Slot(frameBytes: Int) {
        val nv12Bytes = ByteArray(frameBytes)
        val inUse = AtomicBoolean(false)
        var telemetry: FieldFrameTelemetry = FieldFrameTelemetry(
            timestampMonoNs = 0L,
            exposureTimeNs = PushBroomCameraController.MAX_SHUTTER_EXPOSURE_NS,
            sensorIso = 400
        )
    }

    private class PooledImuSlot {
        var timestampMonoNs: Long = 0L
        val rotMat = FloatArray(9)
        var pitchDeg: Float = 45.0f
    }

    private val loggerExecutor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "GneissFind-MediaLogger").apply {
            isDaemon = true
            priority = Thread.NORM_PRIORITY + 1
        }
    }

    private val isRecordingActive = AtomicBoolean(false)
    private val drainScheduled = AtomicBoolean(false)
    private val savedFramesCounter = AtomicInteger(0)
    private val sessionStartMonoNs = AtomicLong(0L)

    @Volatile
    private var framePool: Array<PooledNv12Slot>? = null
    private val imuFreePool = ArrayBlockingQueue<PooledImuSlot>(IMU_POOL_CAPACITY)
    private val pendingImuQueue = ConcurrentLinkedQueue<PooledImuSlot>()
    private val pendingFrameQueue = ConcurrentLinkedQueue<PooledNv12Slot>()

    @Volatile
    private var activeSessionDir: File? = null

    // Confined strictly to loggerExecutor:
    private var traceWriter: BufferedWriter? = null
    private var headerWritten: Boolean = false
    private var pendingHeaderTemplate: ReplayTraceHeader = ReplayTraceHeader()
    private var encoder: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var muxerTrackIndex: Int = -1
    private var muxerStarted: Boolean = false
    private var muxedSampleCount: Int = 0
    private var encodedInputFrameIdx: Int = 0
    private var firstFrameMonoNs: Long = 0L
    private var lastQueuedPtsUs: Long = -1L
    private val codecBufferInfo = MediaCodec.BufferInfo()
    private val jsonScratch = StringBuilder(384)

    /** Number of `1080p` video frames encoded in the active (or most recently completed) `LOG` session. */
    val savedFrameCount: Int
        get() = savedFramesCounter.get()

    /** Elapsed recording duration in whole seconds while `MOD == LOG_DATA`, or last clip duration when stopped. */
    val recordedDurationSec: Int
        get() {
            val frames = savedFramesCounter.get()
            if (!isRecordingActive.get()) {
                return if (frames == 0) 0 else ((frames + (videoFps / 2)) / videoFps).coerceAtLeast(1)
            }
            val startNs = sessionStartMonoNs.get()
            if (startNs == 0L) return 0
            val wallSec = ((System.nanoTime() - startNs).coerceAtLeast(0L) / 1_000_000_000L).toInt()
            val frameSec = frames / videoFps.coerceAtLeast(1)
            return kotlin.math.max(wallSec, frameSec)
        }

    /** True while an active `MOD: LOG` session is accepting video and `120 Hz` IMU samples. */
    val isRecording: Boolean
        get() = isRecordingActive.get()

    /** Absolute path of the active `<session>` directory, or the base `replays/` root when idle. */
    val currentSessionPath: String
        get() = activeSessionDir?.absolutePath ?: resolveBaseReplaysDir(appContext).absolutePath

    /**
     * Starts a new `<YYYYMMDD_HHMMSS>` recording session under `Android/media/com.gneissfind.app/replays/`
     * containing `clip.mp4` + `trace.jsonl`.
     */
    @Synchronized
    fun startSession(
        initialZoomRatio: Float = PushBroomCropper.DEFAULT_ZOOM_RATIO,
        initialFocalLengthPx: Float = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX,
        initialRefPitchDeg: Float = 45.0f,
        initialDisplayRotation: Int = Surface.ROTATION_90,
        sessionDirOverride: File? = null
    ): File {
        activeSessionDir?.let { existing ->
            if (isRecordingActive.get()) return existing
        }

        ensurePoolsAllocated()
        val sessionDir = sessionDirOverride?.apply { mkdirs() } ?: allocateUniqueSessionDir(appContext)
        val header = ReplayTraceHeader(
            schemaVersion = 1,
            width = videoWidthPx and -2,
            height = videoHeightPx and -2,
            fps = videoFps,
            zoomRatio = initialZoomRatio,
            focalLengthPx = initialFocalLengthPx,
            refPitchDeg = initialRefPitchDeg,
            displayRotation = initialDisplayRotation,
            gitHash = BuildConfig.GIT_COMMIT_HASH_6
        )

        savedFramesCounter.set(0)
        sessionStartMonoNs.set(System.nanoTime())
        activeSessionDir = sessionDir
        isRecordingActive.set(true)

        loggerExecutor.execute {
            openSessionResourcesOnLoggerThread(sessionDir, header)
        }
        Log.i(TAG, "Started MOD:LOG 1080p30 + 120Hz IMU session at ${sessionDir.absolutePath}")
        return sessionDir
    }

    /**
     * Asynchronously finalizes the active `clip.mp4` (`END_OF_STREAM` + `moov` atom flush) and closes `trace.jsonl`.
     */
    @Synchronized
    fun stopSession() {
        if (!isRecordingActive.compareAndSet(true, false)) return
        val closingDir = activeSessionDir
        activeSessionDir = null
        loggerExecutor.execute {
            finalizeSessionOnLoggerThread(closingDir)
        }
    }

    /**
     * Synchronously stops and flushes the active session (used by deterministic instrumentation tests
     * and ViewModel shutdown so `clip.mp4` and `trace.jsonl` are guaranteed closed on return).
     */
    fun stopSessionBlocking(timeoutMs: Long = 5_000L): File? {
        val dirSnapshot = activeSessionDir
        val wasActive = isRecordingActive.getAndSet(false)
        if (wasActive) {
            activeSessionDir = null
        }
        val latch = CountDownLatch(1)
        runCatching {
            loggerExecutor.execute {
                try {
                    if (wasActive || encoder != null || traceWriter != null) {
                        finalizeSessionOnLoggerThread(dirSnapshot)
                    }
                } finally {
                    latch.countDown()
                }
            }
            latch.await(timeoutMs, TimeUnit.MILLISECONDS)
        }
        return dirSnapshot
    }

    /**
     * Records a `120 Hz` IMU orientation matrix sample onto the lock-free trace queue while `MOD: LOG` is active.
     */
    fun recordImuSample(
        timestampMonoNs: Long,
        rotationMatrix9: FloatArray,
        pitchDeg: Float
    ) {
        if (!isRecordingActive.get() || rotationMatrix9.size < 9) return
        val slot = imuFreePool.poll() ?: PooledImuSlot()
        slot.timestampMonoNs = timestampMonoNs
        System.arraycopy(rotationMatrix9, 0, slot.rotMat, 0, 9)
        slot.pitchDeg = pitchDeg
        pendingImuQueue.offer(slot)
        scheduleDrainOnLoggerThread()
    }

    /**
     * Copies an upright `1920x1080` `NV12` frame from [frame] in `< 1.5 ms` into a free pool slot
     * (so `ImageProxy.close()` runs immediately) and schedules hardware H.264 encoding + `trace.jsonl` append.
     */
    fun recordCameraFrame(
        frame: Yuv420Frame,
        cropper: PushBroomCropper,
        telemetry: FieldFrameTelemetry
    ): Boolean {
        if (!isRecordingActive.get()) return false
        val pool = framePool ?: return false
        val freeSlot = pool.firstOrNull { it.inUse.compareAndSet(false, true) } ?: return false

        val evenW = videoWidthPx and -2
        val evenH = videoHeightPx and -2
        val copied = try {
            cropper.extractUprightNv12Frame(
                frame = frame,
                outNv12Bytes = freeSlot.nv12Bytes,
                dstWidthPx = evenW,
                dstHeightPx = evenH
            )
        } catch (t: Throwable) {
            freeSlot.inUse.set(false)
            Log.w(TAG, "Dropped LOG frame during NV12 copy: ${t.message}")
            return false
        }
        if (!copied) {
            freeSlot.inUse.set(false)
            return false
        }

        freeSlot.telemetry = telemetry
        pendingFrameQueue.offer(freeSlot)
        scheduleDrainOnLoggerThread()
        return true
    }

    override fun close() {
        stopSessionBlocking(timeoutMs = 3_000L)
        loggerExecutor.shutdown()
    }

    private fun ensurePoolsAllocated() {
        val evenW = videoWidthPx and -2
        val evenH = videoHeightPx and -2
        val neededBytes = evenW * evenH * 3 / 2
        val existing = framePool
        if (existing == null || existing.firstOrNull()?.nv12Bytes?.size != neededBytes) {
            framePool = Array(FRAME_POOL_SLOTS) { PooledNv12Slot(neededBytes) }
        }
        while (imuFreePool.remainingCapacity() > 0) {
            if (!imuFreePool.offer(PooledImuSlot())) break
        }
    }

    private fun scheduleDrainOnLoggerThread() {
        if (drainScheduled.compareAndSet(false, true)) {
            try {
                loggerExecutor.execute(::drainQueuesOnLoggerThread)
            } catch (_: Throwable) {
                drainScheduled.set(false)
            }
        }
    }

    private fun openSessionResourcesOnLoggerThread(
        sessionDir: File,
        header: ReplayTraceHeader
    ) {
        // Clean up any leftover resources before opening a new session
        if (encoder != null || traceWriter != null) {
            finalizeSessionOnLoggerThread(null)
        }
        discardStaleQueues()

        try {
            sessionDir.mkdirs()
            val traceFile = File(sessionDir, TRACE_FILE_NAME)
            traceWriter = BufferedWriter(FileWriter(traceFile, false), 32 * 1024)
            pendingHeaderTemplate = header
            headerWritten = false
            encodedInputFrameIdx = 0
            muxedSampleCount = 0
            firstFrameMonoNs = 0L
            lastQueuedPtsUs = -1L

            val clipFile = File(sessionDir, CLIP_FILE_NAME)
            if (clipFile.exists()) clipFile.delete()

            val evenW = header.width
            val evenH = header.height
            val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            val colorFormat = selectSupportedYuvColorFormat(codec)
            val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, evenW, evenH).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, colorFormat)
                setInteger(MediaFormat.KEY_BIT_RATE, videoBitrateBps)
                setInteger(MediaFormat.KEY_FRAME_RATE, videoFps)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL_SEC)
            }
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            codec.start()
            encoder = codec
            muxer = MediaMuxer(clipFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxerTrackIndex = -1
            muxerStarted = false
        } catch (t: Throwable) {
            Log.e(TAG, "Failed initializing MediaCodec/MediaMuxer for ${sessionDir.absolutePath}", t)
            isRecordingActive.set(false)
            finalizeSessionOnLoggerThread(sessionDir)
        }
    }

    private fun drainQueuesOnLoggerThread() {
        try {
            while (true) {
                var didWork = false

                while (true) {
                    val imu = pendingImuQueue.poll() ?: break
                    didWork = true
                    writeImuRecordLocked(imu)
                    imuFreePool.offer(imu)
                }

                val frameSlot = pendingFrameQueue.poll()
                if (frameSlot != null) {
                    didWork = true
                    try {
                        encodeSingleFrameOnLoggerThread(frameSlot)
                    } finally {
                        frameSlot.inUse.set(false)
                    }
                }

                if (!didWork) break
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Error draining MediaCodec/trace queues", t)
        } finally {
            drainScheduled.set(false)
            if (isRecordingActive.get() &&
                (pendingImuQueue.isNotEmpty() || pendingFrameQueue.isNotEmpty())
            ) {
                scheduleDrainOnLoggerThread()
            }
        }
    }

    private fun ensureHeaderWrittenOnLoggerThread(firstTelemetry: FieldFrameTelemetry?) {
        if (headerWritten) return
        val writer = traceWriter ?: return
        val effectiveHeader = if (firstTelemetry != null) {
            pendingHeaderTemplate.copy(
                zoomRatio = firstTelemetry.zoomRatio,
                focalLengthPx = firstTelemetry.focalLengthPx,
                refPitchDeg = firstTelemetry.referencePitchDeg,
                displayRotation = firstTelemetry.displayRotation
            )
        } else {
            pendingHeaderTemplate
        }
        writer.write(buildHeaderJsonLine(effectiveHeader, jsonScratch))
        writer.newLine()
        headerWritten = true
    }

    private fun writeImuRecordLocked(imu: PooledImuSlot) {
        val writer = traceWriter ?: return
        ensureHeaderWrittenOnLoggerThread(null)
        writer.write(buildImuJsonLine(imu.timestampMonoNs, imu.rotMat, imu.pitchDeg, jsonScratch))
        writer.newLine()
    }

    private fun encodeSingleFrameOnLoggerThread(slot: PooledNv12Slot) {
        val codec = encoder ?: return
        val writer = traceWriter ?: return
        val tel = slot.telemetry
        ensureHeaderWrittenOnLoggerThread(tel)

        val evenW = pendingHeaderTemplate.width
        val evenH = pendingHeaderTemplate.height
        if (firstFrameMonoNs == 0L) {
            firstFrameMonoNs = tel.timestampMonoNs
        }
        val rawPtsUs = ((tel.timestampMonoNs - firstFrameMonoNs).coerceAtLeast(0L)) / 1_000L
        val ptsUs = if (lastQueuedPtsUs < 0L) {
            0L
        } else {
            kotlin.math.max(lastQueuedPtsUs + 1_000L, rawPtsUs)
        }

        // Drain any pending output buffers first so encoder input slots never back up
        drainEncoderOutputOnLoggerThread(endOfStream = false)

        val inputIdx = codec.dequeueInputBuffer(CODEC_TIMEOUT_US)
        if (inputIdx < 0) {
            drainEncoderOutputOnLoggerThread(endOfStream = false)
            return
        }

        val inputImg = runCatching { codec.getInputImage(inputIdx) }.getOrNull()
        val queuedBytes: Int
        if (inputImg != null) {
            try {
                queuedBytes = copyNv12IntoCodecInputImage(slot.nv12Bytes, evenW, evenH, inputImg)
            } finally {
                inputImg.close()
            }
        } else {
            val inBuf = codec.getInputBuffer(inputIdx)
            if (inBuf == null) {
                codec.queueInputBuffer(inputIdx, 0, 0, ptsUs, 0)
                return
            }
            inBuf.clear()
            val toWrite = min(inBuf.remaining(), slot.nv12Bytes.size)
            inBuf.put(slot.nv12Bytes, 0, toWrite)
            queuedBytes = toWrite
        }

        codec.queueInputBuffer(inputIdx, 0, queuedBytes, ptsUs, 0)
        lastQueuedPtsUs = ptsUs
        val frameIdx = encodedInputFrameIdx++
        savedFramesCounter.set(encodedInputFrameIdx)

        val frameRecord = ReplayFrameRecord(
            frameIdx = frameIdx,
            ptsUs = ptsUs,
            timestampNs = tel.timestampMonoNs,
            exposureNs = tel.exposureTimeNs,
            iso = tel.sensorIso,
            zoomRatio = tel.zoomRatio,
            shutterClamp = tel.shutterClamped,
            evBias = tel.exposureBiasEv
        )
        writer.write(buildFrameJsonLine(frameRecord, jsonScratch))
        writer.newLine()

        drainEncoderOutputOnLoggerThread(endOfStream = false)
    }

    private fun drainEncoderOutputOnLoggerThread(endOfStream: Boolean) {
        val codec = encoder ?: return
        val activeMuxer = muxer ?: return

        if (endOfStream) {
            val eosInIdx = codec.dequeueInputBuffer(CODEC_EOS_TIMEOUT_US)
            if (eosInIdx >= 0) {
                val eosPts = (lastQueuedPtsUs.coerceAtLeast(0L)) + (1_000_000L / videoFps.coerceAtLeast(1))
                codec.queueInputBuffer(
                    eosInIdx,
                    0,
                    0,
                    eosPts,
                    MediaCodec.BUFFER_FLAG_END_OF_STREAM
                )
            }
        }

        var eosSpinBudget = if (endOfStream) 60 else 0
        while (true) {
            val timeoutUs = if (endOfStream) 10_000L else 0L
            val outIdx = codec.dequeueOutputBuffer(codecBufferInfo, timeoutUs)
            when {
                outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream || eosSpinBudget-- <= 0) break
                }
                outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (!muxerStarted) {
                        muxerTrackIndex = activeMuxer.addTrack(codec.outputFormat)
                        activeMuxer.start()
                        muxerStarted = true
                    }
                }
                outIdx >= 0 -> {
                    val outBuf = codec.getOutputBuffer(outIdx)
                    val isConfig = (codecBufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0
                    if (isConfig) {
                        codecBufferInfo.size = 0
                    }
                    if (outBuf != null && codecBufferInfo.size > 0 && muxerStarted && muxerTrackIndex >= 0) {
                        outBuf.position(codecBufferInfo.offset)
                        outBuf.limit(codecBufferInfo.offset + codecBufferInfo.size)
                        activeMuxer.writeSampleData(muxerTrackIndex, outBuf, codecBufferInfo)
                        muxedSampleCount++
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                    if ((codecBufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        break
                    }
                }
                else -> break
            }
        }
    }

    private fun finalizeSessionOnLoggerThread(closingDir: File?) {
        // Flush any remaining queued frames & IMU records before EOS
        drainQueuesOnLoggerThread()

        if (encodedInputFrameIdx > 0) {
            runCatching { drainEncoderOutputOnLoggerThread(endOfStream = true) }
                .onFailure { Log.w(TAG, "MediaCodec EOS drain warning: ${it.message}") }
        }

        runCatching {
            if (muxerStarted && muxedSampleCount > 0) {
                muxer?.stop()
            }
        }.onFailure { Log.w(TAG, "MediaMuxer stop warning: ${it.message}") }
        runCatching { muxer?.release() }
        muxer = null
        muxerStarted = false
        muxerTrackIndex = -1

        runCatching { encoder?.stop() }
        runCatching { encoder?.release() }
        encoder = null

        runCatching {
            if (!headerWritten && traceWriter != null) {
                ensureHeaderWrittenOnLoggerThread(null)
            }
            traceWriter?.flush()
            traceWriter?.close()
        }
        traceWriter = null

        if (closingDir != null) {
            if (encodedInputFrameIdx == 0) {
                // Remove empty placeholder clip if user toggled LOG off before 1 frame arrived
                runCatching { File(closingDir, CLIP_FILE_NAME).delete() }
                runCatching { File(closingDir, TRACE_FILE_NAME).delete() }
                runCatching { closingDir.delete() }
            } else {
                Log.i(
                    TAG,
                    "Finalized MOD:LOG session ${closingDir.absolutePath} " +
                        "($encodedInputFrameIdx frames, $muxedSampleCount muxed samples)"
                )
            }
        }
    }

    private fun discardStaleQueues() {
        while (true) {
            val imu = pendingImuQueue.poll() ?: break
            imuFreePool.offer(imu)
        }
        while (true) {
            val frame = pendingFrameQueue.poll() ?: break
            frame.inUse.set(false)
        }
    }

    companion object {
        private const val TAG = "FieldDataLogger"

        const val CLIP_FILE_NAME = "clip.mp4"
        const val TRACE_FILE_NAME = "trace.jsonl"
        const val DEFAULT_VIDEO_FPS = 30
        const val DEFAULT_VIDEO_BITRATE_BPS = 14_000_000
        private const val I_FRAME_INTERVAL_SEC = 1
        private const val FRAME_POOL_SLOTS = 4
        private const val IMU_POOL_CAPACITY = 256
        private const val CODEC_TIMEOUT_US = 12_000L
        private const val CODEC_EOS_TIMEOUT_US = 25_000L

        /**
         * Primary external media `replays/` root (`Android/media/com.gneissfind.app/replays/`),
         * accessible via zero-permission `adb pull` / `adb push`.
         */
        @Suppress("DEPRECATION")
        fun resolveBaseReplaysDir(context: Context): File {
            val mediaRoot = context.externalMediaDirs?.firstOrNull()
                ?: context.getExternalFilesDir(null)
                ?: context.filesDir
            return File(mediaRoot, "replays").apply { mkdirs() }
        }

        /** Alias retained for backward compatibility with existing callers. */
        fun resolveBaseSessionsDir(context: Context): File = resolveBaseReplaysDir(context)

        /** External app-sandbox `Android/data/com.gneissfind.app/files/replays/` root (unrestricted for `adb push` on Android 17 Scoped Storage). */
        fun resolveExternalSandboxReplaysDir(context: Context): File {
            val extRoot = context.getExternalFilesDir(null) ?: context.filesDir
            return File(extRoot, "replays").apply { mkdirs() }
        }

        /** Fallback internal `filesDir/replays/` root scanned alongside external media for `adb` test pushes. */
        fun resolveFallbackInternalReplaysDir(context: Context): File =
            File(context.filesDir, "replays").apply { mkdirs() }

        /**
         * Discovers all valid, readable `<session>` directories containing non-empty `clip.mp4` and
         * `trace.jsonl` across `Android/media/`, `Android/data/`, and `filesDir/`, sorted newest first.
         */
        fun discoverValidReplaySessionDirs(context: Context): List<File> {
            val roots = linkedSetOf(
                resolveBaseReplaysDir(context).canonicalFile,
                resolveExternalSandboxReplaysDir(context).canonicalFile,
                resolveFallbackInternalReplaysDir(context).canonicalFile
            )
            val sessions = ArrayList<File>()
            val seenNames = HashSet<String>()
            for (root in roots) {
                val children = root.listFiles()?.filter { it.isDirectory } ?: continue
                for (dir in children.sortedByDescending { it.name }) {
                    val clip = File(dir, CLIP_FILE_NAME)
                    val trace = File(dir, TRACE_FILE_NAME)
                    if (isReadableNonEmptyFile(clip, 256L) &&
                        isReadableNonEmptyFile(trace, 16L) &&
                        seenNames.add(dir.name)
                    ) {
                        sessions.add(dir)
                    }
                }
            }
            return sessions.sortedByDescending { it.name }
        }

        internal fun isReadableNonEmptyFile(file: File, minBytes: Long): Boolean {
            if (!file.isFile || file.length() <= minBytes) return false
            return runCatching {
                FileReader(file).use { it.read() }
                true
            }.getOrDefault(false)
        }

        private fun allocateUniqueSessionDir(context: Context): File {
            val baseDir = resolveBaseReplaysDir(context)
            val stem = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            var candidate = File(baseDir, stem)
            var suffix = 1
            while (candidate.exists()) {
                candidate = File(baseDir, String.format(Locale.US, "%s_%02d", stem, suffix++))
            }
            candidate.mkdirs()
            return candidate
        }

        @Suppress("DEPRECATION")
        private fun selectSupportedYuvColorFormat(codec: MediaCodec): Int {
            val caps = runCatching {
                codec.codecInfo.getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_AVC)
            }.getOrNull() ?: return MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
            val formats = caps.colorFormats
            return when {
                formats.contains(MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible) ->
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
                formats.contains(MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar) ->
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar
                formats.contains(MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar) ->
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar
                else -> MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
            }
        }

        private fun copyNv12IntoCodecInputImage(
            nv12Bytes: ByteArray,
            width: Int,
            height: Int,
            dstImage: Image
        ): Int {
            val planes = dstImage.planes
            val yPlane = planes[0]
            val uPlane = planes[1]
            val vPlane = planes[2]

            val yBuf: ByteBuffer = yPlane.buffer
            val yRowStride = yPlane.rowStride
            val yPixStride = yPlane.pixelStride
            val lumaSize = width * height

            if (yPixStride == 1 && yRowStride == width && yBuf.remaining() >= lumaSize) {
                yBuf.position(0)
                yBuf.put(nv12Bytes, 0, lumaSize)
            } else {
                for (row in 0 until height) {
                    val dstRowOffset = row * yRowStride
                    if (yPixStride == 1) {
                        val maxCopy = min(width, (yBuf.limit() - dstRowOffset).coerceAtLeast(0))
                        if (maxCopy > 0) {
                            yBuf.position(dstRowOffset)
                            yBuf.put(nv12Bytes, row * width, maxCopy)
                        }
                    } else {
                        val srcRowOffset = row * width
                        for (col in 0 until width) {
                            val idx = dstRowOffset + col * yPixStride
                            if (idx in 0 until yBuf.limit()) {
                                yBuf.put(idx, nv12Bytes[srcRowOffset + col])
                            }
                        }
                    }
                }
            }

            val uvCols = width shr 1
            val uvRows = height shr 1
            val uBuf: ByteBuffer = uPlane.buffer
            val vBuf: ByteBuffer = vPlane.buffer
            val uRowStride = uPlane.rowStride
            val uPixStride = uPlane.pixelStride
            val vRowStride = vPlane.rowStride
            val vPixStride = vPlane.pixelStride
            val uLimit = uBuf.limit()
            val vLimit = vBuf.limit()

            var srcUv = lumaSize
            for (uvY in 0 until uvRows) {
                val uRowBase = uvY * uRowStride
                val vRowBase = uvY * vRowStride
                for (uvX in 0 until uvCols) {
                    val uByte = nv12Bytes[srcUv++]
                    val vByte = nv12Bytes[srcUv++]
                    val uIdx = uRowBase + uvX * uPixStride
                    val vIdx = vRowBase + uvX * vPixStride
                    if (uIdx in 0 until uLimit) uBuf.put(uIdx, uByte)
                    if (vIdx in 0 until vLimit) vBuf.put(vIdx, vByte)
                }
            }
            return nv12Bytes.size
        }

        /** Formats Line 0 (`"type": "header"`) of `trace.jsonl`. */
        fun buildHeaderJsonLine(
            header: ReplayTraceHeader,
            sb: StringBuilder = StringBuilder(256)
        ): String {
            sb.setLength(0)
            sb.append("{\"type\":\"header\",\"schema_version\":").append(header.schemaVersion)
                .append(",\"width\":").append(header.width)
                .append(",\"height\":").append(header.height)
                .append(",\"fps\":").append(header.fps)
                .append(",\"zoom_ratio\":").append(String.format(Locale.US, "%.3f", header.zoomRatio))
                .append(",\"focal_length_px\":").append(String.format(Locale.US, "%.2f", header.focalLengthPx))
                .append(",\"ref_pitch_deg\":").append(String.format(Locale.US, "%.3f", header.refPitchDeg))
                .append(",\"display_rotation\":").append(header.displayRotation)
                .append(",\"git_hash\":\"").append(header.gitHash.replace("\"", "")).append("\"}")
            return sb.toString()
        }

        /** Formats a `120 Hz` `"type": "imu"` JSONL line with zero reflection or heap map churn. */
        fun buildImuJsonLine(
            timestampNs: Long,
            rotMat: FloatArray,
            pitchDeg: Float,
            sb: StringBuilder = StringBuilder(256)
        ): String {
            sb.setLength(0)
            sb.append("{\"type\":\"imu\",\"t_ns\":").append(timestampNs).append(",\"rot_mat\":[")
            for (i in 0 until 9) {
                if (i > 0) sb.append(',')
                sb.append(String.format(Locale.US, "%.6f", rotMat[i]))
            }
            sb.append("],\"pitch_deg\":").append(String.format(Locale.US, "%.3f", pitchDeg)).append('}')
            return sb.toString()
        }

        /** Formats a per-frame `"type": "frame"` JSONL line. */
        fun buildFrameJsonLine(
            frame: ReplayFrameRecord,
            sb: StringBuilder = StringBuilder(256)
        ): String {
            sb.setLength(0)
            sb.append("{\"type\":\"frame\",\"frame_idx\":").append(frame.frameIdx)
                .append(",\"pts_us\":").append(frame.ptsUs)
                .append(",\"t_ns\":").append(frame.timestampNs)
                .append(",\"exposure_ns\":").append(frame.exposureNs)
                .append(",\"iso\":").append(frame.iso)
                .append(",\"zoom_ratio\":").append(String.format(Locale.US, "%.3f", frame.zoomRatio))
                .append(",\"shutter_clamp\":").append(frame.shutterClamp)
                .append(",\"ev_bias\":").append(String.format(Locale.US, "%.2f", frame.evBias))
                .append('}')
            return sb.toString()
        }

        /**
         * Parses a `trace.jsonl` file (`header` + `imu` + `frame` lines) using pure Kotlin parsing
         * so it executes identically in both Android runtime and pure JVM unit tests without Android
         * `org.json` stub dependencies.
         */
        fun parseReplayTraceFile(traceFile: File): ReplayTraceBundle {
            var header = ReplayTraceHeader()
            val imuList = ArrayList<ReplayImuRecord>(2048)
            val frameList = ArrayList<ReplayFrameRecord>(512)

            BufferedReader(FileReader(traceFile), 32 * 1024).use { reader ->
                while (true) {
                    val rawLine = reader.readLine() ?: break
                    val line = rawLine.trim()
                    if (line.isEmpty()) continue
                    when (extractJsonString(line, "type")) {
                        "header" -> {
                            header = ReplayTraceHeader(
                                schemaVersion = extractJsonInt(line, "schema_version", 1),
                                width = extractJsonInt(line, "width", StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX),
                                height = extractJsonInt(line, "height", StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX),
                                fps = extractJsonInt(line, "fps", DEFAULT_VIDEO_FPS),
                                zoomRatio = extractJsonFloat(line, "zoom_ratio", PushBroomCropper.DEFAULT_ZOOM_RATIO),
                                focalLengthPx = extractJsonFloat(
                                    line,
                                    "focal_length_px",
                                    PushBroomCameraController.DEFAULT_FY_2X_1080P_PX
                                ),
                                refPitchDeg = extractJsonFloat(line, "ref_pitch_deg", 45.0f),
                                displayRotation = extractJsonInt(line, "display_rotation", Surface.ROTATION_90),
                                gitHash = extractJsonString(line, "git_hash") ?: "UNKNWN"
                            )
                        }
                        "imu" -> {
                            val tNs = extractJsonLong(line, "t_ns", 0L)
                            val rot = extractJsonFloatArray9(line, "rot_mat")
                            val pitch = extractJsonFloat(line, "pitch_deg", 45.0f)
                            if (rot != null) {
                                imuList.add(ReplayImuRecord(timestampNs = tNs, rotationMatrix = rot, pitchDeg = pitch))
                            }
                        }
                        "frame" -> {
                            frameList.add(
                                ReplayFrameRecord(
                                    frameIdx = extractJsonInt(line, "frame_idx", frameList.size),
                                    ptsUs = extractJsonLong(line, "pts_us", 0L),
                                    timestampNs = extractJsonLong(line, "t_ns", 0L),
                                    exposureNs = extractJsonLong(
                                        line,
                                        "exposure_ns",
                                        PushBroomCameraController.MAX_SHUTTER_EXPOSURE_NS
                                    ),
                                    iso = extractJsonInt(line, "iso", 400),
                                    zoomRatio = extractJsonFloat(line, "zoom_ratio", header.zoomRatio),
                                    shutterClamp = extractJsonBoolean(line, "shutter_clamp", true),
                                    evBias = extractJsonFloat(line, "ev_bias", 0.0f)
                                )
                            )
                        }
                    }
                }
            }
            imuList.sortBy { it.timestampNs }
            frameList.sortBy { it.frameIdx }
            return ReplayTraceBundle(
                header = header,
                imuRecords = imuList,
                frameRecords = frameList
            )
        }

        /** Quick lightweight duration reader (`[XXS]`) for populating `SRC` drawer labels without decoding video. */
        fun readReplayDurationSecondsQuick(sessionDir: File): Int {
            val traceFile = File(sessionDir, TRACE_FILE_NAME)
            if (!traceFile.isFile) return 0
            return runCatching {
                var fps = DEFAULT_VIDEO_FPS
                var frameCount = 0
                var firstPtsUs = -1L
                var lastPtsUs = -1L
                BufferedReader(FileReader(traceFile), 16 * 1024).use { reader ->
                    while (true) {
                        val line = reader.readLine() ?: break
                        if (line.contains("\"header\"")) {
                            fps = extractJsonInt(line, "fps", DEFAULT_VIDEO_FPS).coerceAtLeast(1)
                        } else if (line.contains("\"frame\"")) {
                            frameCount++
                            val pts = extractJsonLong(line, "pts_us", -1L)
                            if (pts >= 0L) {
                                if (firstPtsUs < 0L) firstPtsUs = pts
                                lastPtsUs = pts
                            }
                        }
                    }
                }
                if (firstPtsUs >= 0L && lastPtsUs > firstPtsUs) {
                    (((lastPtsUs - firstPtsUs) + 500_000L) / 1_000_000L).toInt().coerceAtLeast(1)
                } else if (frameCount > 0) {
                    ((frameCount + fps / 2) / fps).coerceAtLeast(1)
                } else {
                    0
                }
            }.getOrDefault(0)
        }

        private fun findValueSlice(json: String, key: String): Int {
            val token = "\"$key\""
            val keyIdx = json.indexOf(token)
            if (keyIdx < 0) return -1
            val colonIdx = json.indexOf(':', keyIdx + token.length)
            if (colonIdx < 0) return -1
            var i = colonIdx + 1
            while (i < json.length && json[i].isWhitespace()) i++
            return if (i < json.length) i else -1
        }

        private fun extractJsonString(json: String, key: String): String? {
            val start = findValueSlice(json, key)
            if (start < 0 || json[start] != '"') return null
            val end = json.indexOf('"', start + 1)
            return if (end > start) json.substring(start + 1, end) else null
        }

        private fun extractJsonScalarToken(json: String, key: String): String? {
            val start = findValueSlice(json, key)
            if (start < 0) return null
            var end = start
            while (end < json.length) {
                val c = json[end]
                if (c == ',' || c == '}' || c == ']' || c.isWhitespace()) break
                end++
            }
            return if (end > start) json.substring(start, end) else null
        }

        private fun extractJsonInt(json: String, key: String, fallback: Int): Int =
            extractJsonScalarToken(json, key)?.toIntOrNull() ?: fallback

        private fun extractJsonLong(json: String, key: String, fallback: Long): Long =
            extractJsonScalarToken(json, key)?.toLongOrNull() ?: fallback

        private fun extractJsonFloat(json: String, key: String, fallback: Float): Float =
            extractJsonScalarToken(json, key)?.toFloatOrNull() ?: fallback

        private fun extractJsonBoolean(json: String, key: String, fallback: Boolean): Boolean {
            val tok = extractJsonScalarToken(json, key) ?: return fallback
            return when (tok) {
                "true" -> true
                "false" -> false
                else -> fallback
            }
        }

        private fun extractJsonFloatArray9(json: String, key: String): FloatArray? {
            val start = findValueSlice(json, key)
            if (start < 0 || json[start] != '[') return null
            val end = json.indexOf(']', start + 1)
            if (end <= start) return null
            val parts = json.substring(start + 1, end).split(',')
            if (parts.size < 9) return null
            val out = FloatArray(9)
            for (i in 0 until 9) {
                out[i] = parts[i].trim().toFloatOrNull() ?: return null
            }
            return out
        }
    }
}
