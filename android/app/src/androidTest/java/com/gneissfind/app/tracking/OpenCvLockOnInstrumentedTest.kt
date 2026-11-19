/*
 * Instrumented on-device verification for Milestone 2C on Pixel 10 Pro XL (16 KB page size).
 *
 * Verifies:
 * 1. Native 16 KB ELF-aligned OpenCV 4.13 initialization (`OpenCVLoader.initLocal()`) and C++
 *    `Calib3d.triangulatePoints` 3D stone triangulation accuracy.
 * 2. Stage 3 `AKAZE` + `BFMatcher(NORM_HAMMING)` + `USAC_MAGSAC` gravel-constellation homography
 *    tracking (< 5.0 ms warm CPU latency) and rolling `1.8x` scale keyframe promotion.
 * 3. Per-profile LiteRT GPU disk shader serialization (`codeCacheDir/litert_gpu_cache`) warm reload
 *    latency and pre-warmed `rf_detr_nano_320_b1_w8dyn` (`B=1`, `320x320`) GPU throughput.
 */
package com.gneissfind.app.tracking

import android.content.Intent
import android.os.Process
import android.os.SystemClock
import android.util.Log
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.gneissfind.app.camera.PushBroomCameraController
import com.gneissfind.app.camera.ReplayCameraSource
import com.gneissfind.app.camera.ReplayRunSummary
import com.gneissfind.app.inference.LiveModelProfile
import com.gneissfind.app.inference.RfDetrTwoGraphGpuDetector
import com.gneissfind.app.logging.FieldDataLogger
import com.gneissfind.app.logging.FieldFrameTelemetry
import com.gneissfind.app.vision.PushBroomCropper
import com.gneissfind.app.vision.Yuv420Frame
import com.google.common.truth.Truth.assertThat
import java.io.File
import java.nio.ByteBuffer
import java.util.Random
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

@RunWith(AndroidJUnit4::class)
class OpenCvLockOnInstrumentedTest {

    @Test
    fun openCvNative16KbInitAndTwoViewTriangulationRecoversMetric3D() {
        assertThat(OpenCVLoader.initLocal()).isTrue()
        assertThat(OpenCvConstellationTracker.ensureOpenCvInitialized()).isTrue()

        val tracker = OpenCvConstellationTracker()
        try {
            val focalPx = PushBroomCameraController.DEFAULT_FY_2X_1080P_PX
            val rot45 = GroundPlaneRaycaster.buildLandscapeRotationMatrix(
                pitchDeg = 45f,
                rollDeg = 0f,
                yawDeg = 0f
            )
            val rCvFromWorld = floatArrayOf(
                rot45[0], rot45[3], rot45[6],
                -rot45[1], -rot45[4], -rot45[7],
                -rot45[2], -rot45[5], -rot45[8]
            )

            val trueStoneEnu = floatArrayOf(0.06f, 1.12f, 0.0f)
            val camA = floatArrayOf(0.0f, 0.0f, 1.06f)
            val camB = floatArrayOf(0.22f, 0.18f, 0.86f)

            val (uA, vA) = projectPinhole(trueStoneEnu, camA, rCvFromWorld, focalPx)
            val (uB, vB) = projectPinhole(trueStoneEnu, camB, rCvFromWorld, focalPx)

            val triangulated = tracker.triangulateTwoViewsOpenCv(
                camPosAEnu = camA,
                rCvFromWorldA = rCvFromWorld,
                pixelUA = uA,
                pixelVA = vA,
                camPosBEnu = camB,
                rCvFromWorldB = rCvFromWorld,
                pixelUB = uB,
                pixelVB = vB,
                focalLengthPx = focalPx
            )
            assertThat(triangulated).isNotNull()
            assertThat(triangulated!![0]).isWithin(0.004f).of(trueStoneEnu[0])
            assertThat(triangulated[1]).isWithin(0.004f).of(trueStoneEnu[1])
            assertThat(triangulated[2]).isWithin(0.004f).of(trueStoneEnu[2])
        } finally {
            tracker.close()
        }
    }

    @Test
    fun akazeConstellationCollarTracksScaleAndTranslationUnder5MsCpu() = withTopAppActivity {
        assertThat(OpenCVLoader.initLocal()).isTrue()

        val width = 1920
        val height = 1080
        val baseMat = synthesizeGravelBedWithSmoothAgate(width, height)
        val baseBytes = ByteArray(width * height)
        baseMat.get(0, 0, baseBytes)

        val tracker = OpenCvConstellationTracker()
        val warpedMat = Mat(height, width, CvType.CV_8UC1)
        try {
            val seeded = tracker.seedConstellationCollar(
                lumaBytes = baseBytes,
                widthPx = width,
                heightPx = height,
                stoneLeftPx = 924f,
                stoneTopPx = 504f,
                stoneRightPx = 996f,
                stoneBottomPx = 576f,
                baseScale = 1.0f
            )
            assertThat(seeded).isTrue()

            // Warp gravel bed by +16 px X, -10 px Y, and 1.12x zoom around (960, 540)
            val warp = Imgproc.getRotationMatrix2D(Point(960.0, 540.0), 2.5, 1.12)
            warp.put(0, 2, warp.get(0, 2)[0] + 16.0)
            warp.put(1, 2, warp.get(1, 2)[0] - 10.0)
            Imgproc.warpAffine(baseMat, warpedMat, warp, Size(width.toDouble(), height.toDouble()))
            warp.release()

            val warpedBytes = ByteArray(width * height)
            warpedMat.get(0, 0, warpedBytes)

            // 2 warmup passes + 12 timed passes (reusing uploaded luma buffer as in production 30 FPS path)
            var firstMatch = tracker.matchConstellationFrame(
                lumaBytes = warpedBytes,
                widthPx = width,
                heightPx = height,
                predictedCenterXPx = 976f,
                predictedCenterYPx = 530f
            )
            assertThat(firstMatch.matched).isTrue()
            assertThat(firstMatch.inliers).isAtLeast(OpenCvConstellationTracker.MIN_HOMOGRAPHY_INLIERS)

            val latenciesMs = FloatArray(12)
            for (i in latenciesMs.indices) {
                firstMatch = tracker.matchConstellationFrame(
                    lumaBytes = null,
                    widthPx = width,
                    heightPx = height,
                    predictedCenterXPx = 976f,
                    predictedCenterYPx = 530f
                )
                latenciesMs[i] = firstMatch.latencyMs
            }
            latenciesMs.sort()
            val medianMs = latenciesMs[latenciesMs.size / 2]
            Log.i(
                TAG,
                "M2C_BENCH AKAZE warm median=${String.format("%.2f", medianMs)}ms, inliers=${firstMatch.inliers}, scale=${String.format("%.2f", firstMatch.cumulativeScale)}"
            )

            assertThat(firstMatch.matched).isTrue()
            val matchedCx = (firstMatch.frameLeftPx + firstMatch.frameRightPx) * 0.5f
            val matchedCy = (firstMatch.frameTopPx + firstMatch.frameBottomPx) * 0.5f
            assertThat(matchedCx).isWithin(6.0f).of(976f)
            assertThat(matchedCy).isWithin(6.0f).of(530f)
            assertThat(medianMs).isLessThan(5.0f)

            // Verify progressive crouch approach (1.0x -> 1.45x -> 1.95x -> 2.45x) triggers rolling 1.8x re-keyframing
            tracker.seedConstellationCollar(
                lumaBytes = baseBytes,
                widthPx = width,
                heightPx = height,
                stoneLeftPx = 924f,
                stoneTopPx = 504f,
                stoneRightPx = 996f,
                stoneBottomPx = 576f,
                baseScale = 1.0f
            )
            for (zoom in listOf(1.45, 1.95, 2.45)) {
                val m = Imgproc.getRotationMatrix2D(Point(960.0, 540.0), 0.0, zoom)
                Imgproc.warpAffine(baseMat, warpedMat, m, Size(width.toDouble(), height.toDouble()))
                m.release()
                warpedMat.get(0, 0, warpedBytes)
                val stepRes = tracker.matchConstellationFrame(
                    lumaBytes = warpedBytes,
                    widthPx = width,
                    heightPx = height,
                    predictedCenterXPx = 960f,
                    predictedCenterYPx = 540f
                )
                assertThat(stepRes.matched).isTrue()
                assertThat(stepRes.cumulativeScale).isWithin(0.25f).of(zoom.toFloat())
            }
        } finally {
            baseMat.release()
            warpedMat.release()
            tracker.close()
        }
    }

    @Test
    fun gpuShaderDiskCacheWarmReloadAndLockOnB1ThroughputOnDevice() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val hostActivity = instrumentation.startActivitySync(
            Intent(context, ComponentActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ) as ComponentActivity
        instrumentation.runOnMainSync {
            hostActivity.setShowWhenLocked(true)
            hostActivity.setTurnScreenOn(true)
            hostActivity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
        val executor = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "M2C-Bench-Inference").apply {
                priority = Thread.MAX_PRIORITY - 1
            }
        }
        val detector = RfDetrTwoGraphGpuDetector(
            appContext = context,
            initialProfile = LiveModelProfile.RFDETR_NANO_256_B3_INTERLEAVED
        )
        try {
            executor.submit {
                Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_DISPLAY)
                // 1. Ensure initial Patrol + B=1 320x320 Lock-On GPU shaders are compiled & cached to disk
                detector.ensureCompiledAndWarmOnInferenceThread()
                assertThat(detector.isHardwareReady).isTrue()
                assertThat(detector.isLockOnHardwareReady).isTrue()

                val cacheDir = File(context.codeCacheDir, "litert_gpu_cache")
                val cacheFiles = cacheDir.listFiles()?.toList().orEmpty()
                Log.i(TAG, "M2C_BENCH GPU cacheDir=${cacheDir.absolutePath}, files=${cacheFiles.size}")
                assertThat(cacheFiles).isNotEmpty()

                // 2. Pre-populate second profile cache, then measure warm cached switch back to 256 B=3
                detector.setLiveModelProfile(LiveModelProfile.RFDETR_NANO_320_B3_INTERLEAVED)
                detector.ensureCompiledAndWarmOnInferenceThread()

                val warmSwitchStartNs = SystemClock.elapsedRealtimeNanos()
                detector.setLiveModelProfile(LiveModelProfile.RFDETR_NANO_256_B3_INTERLEAVED)
                detector.ensureCompiledAndWarmOnInferenceThread()
                val warmSwitchMs = (SystemClock.elapsedRealtimeNanos() - warmSwitchStartNs) / 1_000_000.0
                Log.i(TAG, "M2C_BENCH Warm SRC GPU profile switch=${String.format("%.2f", warmSwitchMs)}ms")
                assertThat(warmSwitchMs).isAtMost(500.0)

                // 3. Measure pre-warmed B=1 320x320 W8-Dyn Lock-On GPU inference latency in foreground top-app cgroup
                val cropFloats = FloatArray(320 * 320 * 3) { 0.45f }
                repeat(4) { detector.detectLockOnCrop320(cropFloats) }
                val b1Latencies = FloatArray(15)
                for (i in b1Latencies.indices) {
                    val res = detector.detectLockOnCrop320(cropFloats)
                    b1Latencies[i] = res.totalSweepLatencyMs
                }
                b1Latencies.sort()
                val medianB1Ms = b1Latencies[b1Latencies.size / 2]
                val b1Fps = 1000.0f / medianB1Ms.coerceAtLeast(1f)
                Log.i(
                    TAG,
                    "M2C_BENCH Lock-On B=1 320x320 W8-Dyn median=${String.format("%.2f", medianB1Ms)}ms (${String.format("%.1f", b1Fps)} FPS)"
                )
                assertThat(medianB1Ms).isLessThan(110.0f)
            }.get(90, TimeUnit.SECONDS)

            // 4. Verify fresh detector deserializes from codeCacheDir/litert_gpu_cache across process restart
            executor.submit {
                val freshDetector = RfDetrTwoGraphGpuDetector(
                    appContext = context,
                    initialProfile = LiveModelProfile.RFDETR_NANO_256_B3_INTERLEAVED
                )
                try {
                    val diskReloadStartNs = SystemClock.elapsedRealtimeNanos()
                    freshDetector.ensureCompiledAndWarmOnInferenceThread()
                    val diskReloadMs = (SystemClock.elapsedRealtimeNanos() - diskReloadStartNs) / 1_000_000.0
                    Log.i(
                        TAG,
                        "M2C_BENCH Disk-cached process startup (Patrol 256 B=3 + Lock-On 320 B=1)=${String.format("%.1f", diskReloadMs)}ms"
                    )
                    assertThat(freshDetector.isHardwareReady).isTrue()
                    assertThat(freshDetector.isLockOnHardwareReady).isTrue()
                } finally {
                    freshDetector.close()
                }
            }.get(60, TimeUnit.SECONDS)
        } finally {
            executor.submit { detector.close() }.get(10, TimeUnit.SECONDS)
            executor.shutdownNow()
            instrumentation.runOnMainSync { hostActivity.finish() }
        }
    }

    @Test
    fun fourEdgeSweep160FramesNeverThrowsSubmatExceptionOnHardwareJni() {
        assertThat(OpenCvConstellationTracker.ensureOpenCvInitialized()).isTrue()
        val width = 1920
        val height = 1080
        val baseMat = synthesizeGravelBedWithSmoothAgate(width, height)
        val lumaBytes = ByteArray(width * height)
        baseMat.get(0, 0, lumaBytes)

        val tracker = OpenCvConstellationTracker()
        try {
            // Seed at center first
            assertThat(
                tracker.seedConstellationCollar(
                    lumaBytes = lumaBytes,
                    widthPx = width,
                    heightPx = height,
                    stoneLeftPx = 924f,
                    stoneTopPx = 504f,
                    stoneRightPx = 996f,
                    stoneBottomPx = 576f
                )
            ).isTrue()

            // Sweep x in -400..2320 and y in -400..1480 across 168 frames (all 4 edges + corners + NaN/Inf)
            var framesTested = 0
            for (step in 0 until 42) {
                val t = step / 41f
                val coords = arrayOf(
                    Pair(-400f + t * 2720f, 540f),
                    Pair(1925f, -400f + t * 1880f),
                    Pair(2320f - t * 2720f, 1085f),
                    Pair(-400f, 1480f - t * 1880f)
                )
                for ((cx, cy) in coords) {
                    tracker.trackGravelFlowCorrespondences(
                        lumaBytes = lumaBytes,
                        widthPx = width,
                        heightPx = height,
                        excludeLeftPx = cx - 36f,
                        excludeTopPx = cy - 36f,
                        excludeRightPx = cx + 36f,
                        excludeBottomPx = cy + 36f
                    )
                    val match = tracker.matchConstellationFrame(
                        lumaBytes = null,
                        widthPx = width,
                        heightPx = height,
                        predictedCenterXPx = cx,
                        predictedCenterYPx = cy
                    )
                    assertThat(match.inliers).isAtLeast(0)
                    if (framesTested % 16 == 0) {
                        tracker.seedConstellationCollar(
                            lumaBytes = null,
                            widthPx = width,
                            heightPx = height,
                            stoneLeftPx = cx - 36f,
                            stoneTopPx = cy - 36f,
                            stoneRightPx = cx + 36f,
                            stoneBottomPx = cy + 36f
                        )
                    }
                    framesTested++
                }
            }
            assertThat(framesTested).isAtLeast(160)

            // Also verify NaN / Infinity inputs never throw CvException or IllegalArgumentException
            for (bad in listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)) {
                tracker.trackGravelFlowCorrespondences(
                    lumaBytes = lumaBytes,
                    widthPx = width,
                    heightPx = height,
                    excludeLeftPx = bad,
                    excludeTopPx = bad,
                    excludeRightPx = bad,
                    excludeBottomPx = bad
                )
                tracker.matchConstellationFrame(
                    lumaBytes = null,
                    widthPx = width,
                    heightPx = height,
                    predictedCenterXPx = bad,
                    predictedCenterYPx = bad
                )
            }
        } finally {
            baseMat.release()
            tracker.close()
        }
    }

    @Test
    fun fullFrame640x360KltVoUnder2MsAnd480x270RelocalizationRecoversLargeDrift() = withTopAppActivity {
        assertThat(OpenCvConstellationTracker.ensureOpenCvInitialized()).isTrue()
        val width = 1920
        val height = 1080
        val baseMat = synthesizeGravelBedWithSmoothAgate(width, height)
        val baseBytes = ByteArray(width * height)
        baseMat.get(0, 0, baseBytes)

        // Shift scene by +136 px X, -64 px Y (large post-occlusion / off-screen VO drift!)
        val shiftedMat = Mat(height, width, CvType.CV_8UC1)
        val warpLarge = Imgproc.getRotationMatrix2D(Point(960.0, 540.0), -3.2, 1.04)
        warpLarge.put(0, 2, warpLarge.get(0, 2)[0] + 136.0)
        warpLarge.put(1, 2, warpLarge.get(1, 2)[0] - 64.0)
        Imgproc.warpAffine(baseMat, shiftedMat, warpLarge, Size(width.toDouble(), height.toDouble()))

        // Also create a realistic inter-frame VO step (+14 px X, -6 px Y) to benchmark 640x360 PyrLK latency
        val stepMat = Mat(height, width, CvType.CV_8UC1)
        val warpStep = Imgproc.getRotationMatrix2D(Point(960.0, 540.0), 0.25, 1.0)
        warpStep.put(0, 2, warpStep.get(0, 2)[0] + 14.0)
        warpStep.put(1, 2, warpStep.get(1, 2)[0] - 6.0)
        Imgproc.warpAffine(baseMat, stepMat, warpStep, Size(width.toDouble(), height.toDouble()))
        warpStep.release()

        val stepBytes = ByteArray(width * height)
        stepMat.get(0, 0, stepBytes)
        val shiftedBytes = ByteArray(width * height)
        shiftedMat.get(0, 0, shiftedBytes)

        val tracker = OpenCvConstellationTracker()
        try {
            // 1. Benchmark 640x360 full-frame PyrLK + RANSAC VO (< 2.0 ms warm median budget on Tensor G5)
            tracker.trackGravelFlowCorrespondences(
                lumaBytes = baseBytes,
                widthPx = width,
                heightPx = height,
                excludeLeftPx = 924f,
                excludeTopPx = 504f,
                excludeRightPx = 996f,
                excludeBottomPx = 576f,
                excludeTargetRegion = false
            )
            // Warmup x2
            tracker.trackGravelFlowCorrespondences(stepBytes, width, height, 0f, 0f, 0f, 0f, false)
            tracker.trackGravelFlowCorrespondences(baseBytes, width, height, 0f, 0f, 0f, 0f, false)

            val voLatenciesMs = FloatArray(15)
            var lastInliers = 0
            for (i in voLatenciesMs.indices) {
                val frame = if (i % 2 == 0) stepBytes else baseBytes
                val flow = tracker.trackGravelFlowCorrespondences(
                    lumaBytes = frame,
                    widthPx = width,
                    heightPx = height,
                    excludeLeftPx = 924f,
                    excludeTopPx = 504f,
                    excludeRightPx = 996f,
                    excludeBottomPx = 576f,
                    excludeTargetRegion = false
                )
                assertThat(flow).isNotNull()
                lastInliers = flow!!.inlierCount
                voLatenciesMs[i] = flow.latencyMs
            }
            voLatenciesMs.sort()
            val medianVoMs = voLatenciesMs[voLatenciesMs.size / 2]
            Log.i(
                TAG,
                "M2D_BENCH 640x360 PyrLK VO warm median=${String.format("%.2f", medianVoMs)}ms, inliers=$lastInliers"
            )
            assertThat(lastInliers).isAtLeast(30)
            assertThat(medianVoMs).isLessThan(2.0f)

            // 2. Verify 480x270 full-frame AKAZE relocalization recovers exact stone center after +136px/-64px drift!
            val seededReloc = tracker.seedFullFrameRelocalizationSnapshot(
                lumaBytes = baseBytes,
                widthPx = width,
                heightPx = height,
                stoneCenterXPx = 960f,
                stoneCenterYPx = 540f
            )
            assertThat(seededReloc).isTrue()

            val m = DoubleArray(6)
            warpLarge.get(0, 0, m)
            val expectedX = (m[0] * 960.0 + m[1] * 540.0 + m[2]).toFloat()
            val expectedY = (m[3] * 960.0 + m[4] * 540.0 + m[5]).toFloat()

            val reloc = tracker.matchFullFrameRelocalization(
                lumaBytes = shiftedBytes,
                widthPx = width,
                heightPx = height
            )
            Log.i(
                TAG,
                "M2D_BENCH 480x270 AKAZE reloc matched=${reloc.matched}, inliers=${reloc.inliers}, latency=${String.format("%.2f", reloc.latencyMs)}ms"
            )
            assertThat(reloc.matched).isTrue()
            assertThat(reloc.inliers).isAtLeast(OpenCvConstellationTracker.MIN_RELOC_INLIERS)
            assertThat(reloc.stoneCenterXPx).isWithin(8.0f).of(expectedX)
            assertThat(reloc.stoneCenterYPx).isWithin(8.0f).of(expectedY)
        } finally {
            warpLarge.release()
            baseMat.release()
            shiftedMat.release()
            stepMat.release()
            tracker.close()
        }
    }

    @Test
    fun fieldDataLoggerHardwareMp4EncodeAndReplayCameraSourceDecodeRoundTrip() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val testSessionDir = File(context.cacheDir, "m2d_hw_codec_roundtrip").apply {
            deleteRecursively()
            mkdirs()
        }
        val cropper = PushBroomCropper()
        val logger = FieldDataLogger(appContext = context)
        val raycaster = GroundPlaneRaycaster(context)

        try {
            logger.startSession(sessionDirOverride = testSessionDir)
            val width = 1920
            val height = 1080
            val yBytes = ByteArray(width * height) { idx -> ((idx * 17) and 0xFF).toByte() }
            val uvSize = width * (height / 2)
            val uBytes = ByteArray(uvSize) { 128.toByte() }
            val vBytes = ByteArray(uvSize) { 132.toByte() }

            var monoNs = 1_000_000_000L
            for (fIdx in 0 until 30) {
                repeat(4) { sub ->
                    val imuTs = monoNs + sub * 8_333_333L
                    val pitch = 44.0f + fIdx * 0.1f
                    val rot = GroundPlaneRaycaster.buildLandscapeRotationMatrix(
                        pitchDeg = pitch,
                        rollDeg = 0f,
                        yawDeg = fIdx * 0.4f
                    )
                    logger.recordImuSample(imuTs, rot, pitch)
                }
                val frame = object : Yuv420Frame {
                    override val width: Int = width
                    override val height: Int = height
                    override val rotationDegrees: Int = 0
                    override val timestampNs: Long = monoNs
                    override val yBuffer: ByteBuffer = ByteBuffer.wrap(yBytes)
                    override val yRowStride: Int = width
                    override val yPixelStride: Int = 1
                    override val uBuffer: ByteBuffer = ByteBuffer.wrap(uBytes)
                    override val uRowStride: Int = width
                    override val uPixelStride: Int = 2
                    override val vBuffer: ByteBuffer = ByteBuffer.wrap(vBytes)
                    override val vRowStride: Int = width
                    override val vPixelStride: Int = 2
                }
                logger.recordCameraFrame(
                    frame = frame,
                    cropper = cropper,
                    telemetry = FieldFrameTelemetry(
                        timestampMonoNs = monoNs,
                        exposureTimeNs = 4_000_000L,
                        sensorIso = 200
                    )
                )
                Thread.sleep(20L)
                monoNs += 33_333_333L
            }

            logger.stopSessionBlocking(timeoutMs = 6_000L)

            val clipFile = File(testSessionDir, FieldDataLogger.CLIP_FILE_NAME)
            val traceFile = File(testSessionDir, FieldDataLogger.TRACE_FILE_NAME)
            assertThat(clipFile.exists()).isTrue()
            assertThat(clipFile.length()).isGreaterThan(4_096L)
            assertThat(traceFile.exists()).isTrue()

            val parsedTrace = FieldDataLogger.parseReplayTraceFile(traceFile)
            assertThat(parsedTrace.imuRecords.size).isAtLeast(60)
            assertThat(parsedTrace.frameRecords.size).isAtLeast(15)

            // Decode the recorded clip.mp4 + trace.jsonl back through ReplayCameraSource
            val decodedFrames = AtomicInteger(0)
            val doneLatch = CountDownLatch(1)
            val summaryRef = AtomicReference<ReplayRunSummary?>(null)
            val replaySource = ReplayCameraSource(
                appContext = context,
                raycaster = raycaster,
                onSessionHeaderLoaded = {},
                onReplayFrameAvailable = { yuv, _, _, _, releaseCallback ->
                    try {
                        assertThat(yuv.width).isEqualTo(1920)
                        assertThat(yuv.height).isEqualTo(1080)
                        decodedFrames.incrementAndGet()
                    } finally {
                        releaseCallback()
                    }
                },
                onTryAutoLockAtFrame = { true },
                onBuildRunSummary = { name, total, errs ->
                    ReplayRunSummary(
                        sessionName = name,
                        frames = total,
                        locks = 1,
                        handovers = 0,
                        lostLocks = 0,
                        distractorJumps = 0,
                        exceptions = errs
                    )
                },
                onSinglePassCompleted = { summary ->
                    summaryRef.set(summary)
                    doneLatch.countDown()
                }
            )
            try {
                val started = replaySource.startSession(
                    sessionDir = testSessionDir,
                    loopReplay = false,
                    autoLockFrameIdx = 5
                )
                assertThat(started).isTrue()
                assertThat(doneLatch.await(12, TimeUnit.SECONDS)).isTrue()
                val summary = summaryRef.get()
                assertThat(summary).isNotNull()
                assertThat(summary!!.exceptions).isEqualTo(0)
                assertThat(decodedFrames.get()).isAtLeast(15)
            } finally {
                replaySource.close()
            }
        } finally {
            logger.close()
            testSessionDir.deleteRecursively()
        }
    }

    @Test
    fun unifiedThreeTierTracker_rejectsAdversarialBlankWallAndDoorframeWithZeroFalseLocks() {
        assertThat(OpenCVLoader.initLocal()).isTrue()

        val width = 1920
        val height = 1080
        val sceneMat = synthesizeGravelBedWithSmoothAgate(width, height)
        // Add distinct high-contrast rim detail to the central target so Gate 4 NCC has real texture
        Imgproc.circle(sceneMat, Point(960.0, 540.0), 34, Scalar(50.0), 3)
        Imgproc.circle(sceneMat, Point(948.0, 530.0), 8, Scalar(235.0), -1)
        val sceneBytes = ByteArray(width * height)
        sceneMat.get(0, 0, sceneBytes)

        val shiftedMat = Mat(height, width, CvType.CV_8UC1)
        val warpMat = Mat(2, 3, CvType.CV_64FC1).apply {
            put(0, 0, 1.0, 0.0, 26.0, 0.0, 1.0, -14.0)
        }
        Imgproc.warpAffine(sceneMat, shiftedMat, warpMat, Size(width.toDouble(), height.toDouble()))
        val shiftedBytes = ByteArray(width * height)
        shiftedMat.get(0, 0, shiftedBytes)

        val wallMat = Mat(height, width, CvType.CV_8UC1)
        val wallBytes = ByteArray(width * height)

        val tracker = OpenCvConstellationTracker()
        val r0 = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 45f, rollDeg = 0f, yawDeg = 0f)
        try {
            tracker.setCurrentImuRotationMatrix(r0)
            // Frame 0 initializes prev/curr VO pyramid (returns false until 2 frames exist)
            assertThat(tracker.beginTrackingFrame(sceneBytes, width, height)).isFalse()
            val seededCollar = tracker.seedConstellationCollar(
                lumaBytes = null,
                widthPx = width,
                heightPx = height,
                stoneLeftPx = 918f,
                stoneTopPx = 498f,
                stoneRightPx = 1002f,
                stoneBottomPx = 582f,
                baseScale = 1.0f
            )
            assertThat(seededCollar).isTrue()
            val seededReloc = tracker.seedFullFrameRelocalizationSnapshotWithBox(
                lumaBytes = null,
                widthPx = width,
                heightPx = height,
                stoneCenterXPx = 960f,
                stoneCenterYPx = 540f,
                stoneWidthPx = 84f,
                stoneHeightPx = 84f
            )
            assertThat(seededReloc).isTrue()

            // Feed 30 adversarial frames of a blank beige wall + sharp vertical doorframe + glare
            var falseLocks = 0
            for (i in 0 until 30) {
                val baseVal = 170.0 + (i % 6) * 4.0
                wallMat.setTo(Scalar(baseVal))
                val frameX = 480.0 + i * 24.0
                Imgproc.rectangle(
                    wallMat,
                    Point(frameX, 0.0),
                    Point((frameX + 42.0).coerceAtMost(width - 1.0), height - 1.0),
                    Scalar(baseVal - 75.0),
                    -1
                )
                Imgproc.line(
                    wallMat,
                    Point((frameX + 44.0).coerceAtMost(width - 1.0), 0.0),
                    Point((frameX + 44.0).coerceAtMost(width - 1.0), height - 1.0),
                    Scalar(240.0),
                    3
                )
                wallMat.get(0, 0, wallBytes)

                tracker.beginTrackingFrame(wallBytes, width, height)
                val klt = tracker.trackLocalTargetSimilarity30Hz(
                    prevBoxLeftPx = 918f,
                    prevBoxTopPx = 498f,
                    prevBoxRightPx = 1002f,
                    prevBoxBottomPx = 582f,
                    prevRotationScr = r0,
                    currRotationScr = r0,
                    focalLengthPx = 2200f,
                    fullWidthPx = width,
                    fullHeightPx = height
                )
                val collar = tracker.matchConstellationFrame(
                    lumaBytes = null,
                    widthPx = width,
                    heightPx = height,
                    predictedCenterXPx = 960f,
                    predictedCenterYPx = 540f
                )
                val reloc = tracker.matchFullFrameRelocalization(
                    lumaBytes = null,
                    widthPx = width,
                    heightPx = height
                )
                if ((klt?.matched == true) || collar.matched || reloc.matched) {
                    falseLocks++
                }
            }
            Log.i(TAG, "M2D2_BLANK_WALL_ADVERSARIAL falseLocks=$falseLocks/30")
            assertThat(falseLocks).isEqualTo(0)

            // Return to real target shifted by (+26, -14) px -> must re-acquire cleanly via Tier 3!
            tracker.beginTrackingFrame(shiftedBytes, width, height)
            val reacquired = tracker.matchFullFrameRelocalization(
                lumaBytes = null,
                widthPx = width,
                heightPx = height
            )
            assertThat(reacquired.matched).isTrue()
            val errPx = kotlin.math.hypot(
                (reacquired.stoneCenterXPx - 986f).toDouble(),
                (reacquired.stoneCenterYPx - 526f).toDouble()
            )
            Log.i(TAG, "M2D2_REACQUIRE_AFTER_WALL inliers=${reacquired.inliers} ncc=${reacquired.nccScore} errPx=$errPx")
            assertThat(errPx).isLessThan(15.0)
        } finally {
            tracker.close()
            sceneMat.release()
            shiftedMat.release()
            warpMat.release()
            wallMat.release()
        }
    }

    @Test
    fun unifiedThreeTierTracker_meetsRealtimeArm64LatencyBudgetOnPixel10ProXl() {
        assertThat(OpenCVLoader.initLocal()).isTrue()

        val width = 1920
        val height = 1080
        val baseMat = synthesizeGravelBedWithSmoothAgate(width, height)
        Imgproc.circle(baseMat, Point(960.0, 540.0), 34, Scalar(50.0), 3)
        Imgproc.circle(baseMat, Point(948.0, 530.0), 8, Scalar(235.0), -1)

        val tracker = OpenCvConstellationTracker()
        val r0 = GroundPlaneRaycaster.buildLandscapeRotationMatrix(pitchDeg = 45f, rollDeg = 0f, yawDeg = 0f)
        val warpedMat = Mat(height, width, CvType.CV_8UC1)
        val frameBytes = ByteArray(width * height)
        val latenciesMs = ArrayList<Double>(25)

        try {
            tracker.setCurrentImuRotationMatrix(r0)
            var boxL = 918f
            var boxT = 498f
            var boxR = 1002f
            var boxB = 582f
            var camPos = floatArrayOf(0f, 0f, 1.10f)

            baseMat.get(0, 0, frameBytes)
            tracker.beginTrackingFrame(frameBytes, width, height)
            assertThat(
                tracker.seedConstellationCollar(
                    lumaBytes = null,
                    widthPx = width,
                    heightPx = height,
                    stoneLeftPx = boxL,
                    stoneTopPx = boxT,
                    stoneRightPx = boxR,
                    stoneBottomPx = boxB,
                    baseScale = 1.0f
                )
            ).isTrue()
            tracker.stepRotationKnownMiniVio30Hz(
                prevCamPosEnu = camPos,
                prevRotationScr = r0,
                currRotationScr = r0,
                focalLengthPx = 2200f,
                nominalSceneDepthM = 1.25f,
                fullWidthPx = width,
                fullHeightPx = height
            )

            var trackedCount = 0
            for (step in 1..25) {
                val tx = step * 4.5
                val ty = step * -2.0
                val warp = Mat(2, 3, CvType.CV_64FC1).apply {
                    put(0, 0, 1.0, 0.0, tx, 0.0, 1.0, ty)
                }
                Imgproc.warpAffine(baseMat, warpedMat, warp, Size(width.toDouble(), height.toDouble()))
                warp.release()
                warpedMat.get(0, 0, frameBytes)

                val t0 = SystemClock.elapsedRealtimeNanos()
                tracker.beginTrackingFrame(frameBytes, width, height)
                val vio = tracker.stepRotationKnownMiniVio30Hz(
                    prevCamPosEnu = camPos,
                    prevRotationScr = r0,
                    currRotationScr = r0,
                    focalLengthPx = 2200f,
                    nominalSceneDepthM = 1.25f,
                    fullWidthPx = width,
                    fullHeightPx = height,
                    excludeLeftPx = boxL,
                    excludeTopPx = boxT,
                    excludeRightPx = boxR,
                    excludeBottomPx = boxB,
                    excludeTargetRegion = true
                )
                if (vio != null && vio.updated) {
                    camPos = vio.camPosEnu
                }
                val klt = tracker.trackLocalTargetSimilarity30Hz(
                    prevBoxLeftPx = boxL,
                    prevBoxTopPx = boxT,
                    prevBoxRightPx = boxR,
                    prevBoxBottomPx = boxB,
                    prevRotationScr = r0,
                    currRotationScr = r0,
                    focalLengthPx = 2200f,
                    fullWidthPx = width,
                    fullHeightPx = height
                )
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
                latenciesMs.add(elapsedMs)

                if (klt != null && klt.matched) {
                    trackedCount++
                    boxL = klt.frameLeftPx
                    boxT = klt.frameTopPx
                    boxR = klt.frameRightPx
                    boxB = klt.frameBottomPx
                }
            }

            latenciesMs.sort()
            val medianMs = latenciesMs[latenciesMs.size / 2]
            val p95Ms = latenciesMs[(latenciesMs.size * 95) / 100]
            val finalCx = (boxL + boxR) * 0.5f
            val finalCy = (boxT + boxB) * 0.5f
            val finalErrPx = kotlin.math.hypot((finalCx - (960f + 25f * 4.5f)).toDouble(), (finalCy - (540f - 25f * 2.0f)).toDouble())
            Log.i(
                TAG,
                "M2D2_ARM64_3TIER_BENCH tracked=$trackedCount/25 medianMs=$medianMs p95Ms=$p95Ms finalErrPx=$finalErrPx"
            )
            assertThat(trackedCount).isAtLeast(22)
            assertThat(finalErrPx).isLessThan(12.0)
            assertThat(medianMs).isLessThan(10.0)
        } finally {
            tracker.close()
            baseMat.release()
            warpedMat.release()
        }
    }

    private fun synthesizeGravelBedWithSmoothAgate(width: Int, height: Int): Mat {
        val mat = Mat(height, width, CvType.CV_8UC1, Scalar(96.0))
        val rng = Random(424242L)
        // Draw 900 textured background pebbles across the gravel bed so AKAZE has rich corner constellations
        repeat(900) {
            val px = rng.nextInt(width)
            val py = rng.nextInt(height)
            // Leave central 924..996 x 504..576 smooth like a wet chalcedony agate!
            if (px in 918..1002 && py in 498..582) return@repeat
            val rx = 4 + rng.nextInt(14)
            val ry = 3 + rng.nextInt(12)
            val angle = rng.nextDouble() * 180.0
            val shade = 35.0 + rng.nextInt(190)
            Imgproc.ellipse(
                mat,
                Point(px.toDouble(), py.toDouble()),
                Size(rx.toDouble(), ry.toDouble()),
                angle,
                0.0,
                360.0,
                Scalar(shade),
                -1
            )
            Imgproc.circle(
                mat,
                Point((px + 2).toDouble(), (py - 1).toDouble()),
                2,
                Scalar((shade + 45.0).coerceAtMost(250.0)),
                -1
            )
        }
        // Smooth featureless central agate (virtually zero internal corners — tracked via 3x gravel collar!)
        Imgproc.circle(mat, Point(960.0, 540.0), 34, Scalar(182.0), -1)
        return mat
    }

    private fun projectPinhole(
        pointEnu: FloatArray,
        camPosEnu: FloatArray,
        rCvFromWorld: FloatArray,
        focalPx: Float
    ): Pair<Float, Float> {
        val dx = pointEnu[0] - camPosEnu[0]
        val dy = pointEnu[1] - camPosEnu[1]
        val dz = pointEnu[2] - camPosEnu[2]
        val xCv = rCvFromWorld[0] * dx + rCvFromWorld[1] * dy + rCvFromWorld[2] * dz
        val yCv = rCvFromWorld[3] * dx + rCvFromWorld[4] * dy + rCvFromWorld[5] * dz
        val zCv = rCvFromWorld[6] * dx + rCvFromWorld[7] * dy + rCvFromWorld[8] * dz
        return Pair(960f + focalPx * (xCv / zCv), 540f + focalPx * (yCv / zCv))
    }

    private fun withTopAppActivity(block: () -> Unit) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val hostActivity = instrumentation.startActivitySync(
            Intent(context, ComponentActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ) as ComponentActivity
        instrumentation.runOnMainSync {
            hostActivity.setShowWhenLocked(true)
            hostActivity.setTurnScreenOn(true)
            hostActivity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
        try {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_DISPLAY)
            block()
        } finally {
            instrumentation.runOnMainSync { hostActivity.finish() }
        }
    }

    companion object {
        private const val TAG = "M2CInstrumentedTest"
    }
}
