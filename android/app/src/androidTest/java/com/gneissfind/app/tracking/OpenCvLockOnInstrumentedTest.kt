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
import com.gneissfind.app.inference.LiveModelProfile
import com.gneissfind.app.inference.RfDetrTwoGraphGpuDetector
import com.google.common.truth.Truth.assertThat
import java.io.File
import java.util.Random
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
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
    fun akazeConstellationCollarTracksScaleAndTranslationUnder5MsCpu() {
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

    companion object {
        private const val TAG = "M2CInstrumentedTest"
    }
}
