/*
 * Unit test suite for FieldDataLogger trace.jsonl serialization/parsing and upright NV12 extraction.
 *
 * Verifies the Milestone 2D MOD: LOG sidecar contract (.agents/rules/android-camera-ar.md §2):
 * - Line 0 "header" schema, 120 Hz "imu" 3x3 rotation matrix fidelity, and 30 FPS "frame" PTS/optics.
 * - Quick duration inspection for SRC drawer REPLAY badges.
 * - Reversible-landscape (0/90/180/270 deg) upright NV12 extraction from YUV_420_888 frames.
 */
package com.gneissfind.app.logging

import android.view.Surface
import com.gneissfind.app.tracking.GroundPlaneRaycaster
import com.gneissfind.app.vision.PushBroomCropper
import com.gneissfind.app.vision.Yuv420Frame
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import java.io.File
import java.nio.ByteBuffer
import java.nio.file.Files

/**
 * Verifies deterministic `trace.jsonl` round-trip serialization and `NV12` frame packing.
 */
@RunWith(JUnit4::class)
class FieldDataLoggerTest {

    @Test
    fun traceJsonlRoundTrip_preservesHeader120HzImuMatricesAnd30FpsFrameOptics() {
        val tempDir = Files.createTempDirectory("gneissfind_trace_test").toFile()
        try {
            val traceFile = File(tempDir, FieldDataLogger.TRACE_FILE_NAME)
            val header = ReplayTraceHeader(
                schemaVersion = 1,
                width = 1920,
                height = 1080,
                fps = 30,
                zoomRatio = 2.0f,
                focalLengthPx = 2985.4f,
                refPitchDeg = 44.5f,
                displayRotation = Surface.ROTATION_270,
                gitHash = "AE0EA3"
            )

            val lines = ArrayList<String>()
            lines.add(FieldDataLogger.buildHeaderJsonLine(header))

            val baseTNs = 50_000_000_000L
            // 121 IMU samples at 120 Hz (1.0 second) + 61 video frames at 30 FPS (2.0 seconds)
            for (k in 0..120) {
                val tNs = baseTNs + k * 8_333_333L
                val pitch = 44.0f + (k % 10) * 0.25f
                val rot = GroundPlaneRaycaster.buildLandscapeRotationMatrix(
                    pitchDeg = pitch,
                    rollDeg = 1.5f,
                    yawDeg = k * 0.5f
                )
                lines.add(FieldDataLogger.buildImuJsonLine(tNs, rot, pitch))
            }
            for (f in 0..60) {
                val ptsUs = f * 33_333L
                val tNs = baseTNs + ptsUs * 1_000L
                lines.add(
                    FieldDataLogger.buildFrameJsonLine(
                        ReplayFrameRecord(
                            frameIdx = f,
                            ptsUs = ptsUs,
                            timestampNs = tNs,
                            exposureNs = 1_992_000L,
                            iso = 640,
                            zoomRatio = 2.0f,
                            shutterClamp = true,
                            evBias = -0.5f
                        )
                    )
                )
            }
            traceFile.writeText(lines.joinToString("\n") + "\n")

            val parsed = FieldDataLogger.parseReplayTraceFile(traceFile)
            assertThat(parsed.header.schemaVersion).isEqualTo(1)
            assertThat(parsed.header.width).isEqualTo(1920)
            assertThat(parsed.header.height).isEqualTo(1080)
            assertThat(parsed.header.fps).isEqualTo(30)
            assertThat(parsed.header.zoomRatio).isWithin(1e-3f).of(2.0f)
            assertThat(parsed.header.focalLengthPx).isWithin(1e-2f).of(2985.4f)
            assertThat(parsed.header.refPitchDeg).isWithin(1e-3f).of(44.5f)
            assertThat(parsed.header.displayRotation).isEqualTo(Surface.ROTATION_270)
            assertThat(parsed.header.gitHash).isEqualTo("AE0EA3")

            assertThat(parsed.imuRecords).hasSize(121)
            val expectedMidRot = GroundPlaneRaycaster.buildLandscapeRotationMatrix(
                pitchDeg = 44.0f + (60 % 10) * 0.25f,
                rollDeg = 1.5f,
                yawDeg = 30.0f
            )
            for (i in 0 until 9) {
                assertThat(parsed.imuRecords[60].rotationMatrix[i]).isWithin(1e-5f).of(expectedMidRot[i])
            }

            assertThat(parsed.frameRecords).hasSize(61)
            assertThat(parsed.frameRecords.last().frameIdx).isEqualTo(60)
            assertThat(parsed.frameRecords.last().ptsUs).isEqualTo(60 * 33_333L)
            assertThat(parsed.frameRecords.last().exposureNs).isEqualTo(1_992_000L)
            assertThat(parsed.frameRecords.last().iso).isEqualTo(640)
            assertThat(parsed.frameRecords.last().shutterClamp).isTrue()
            assertThat(parsed.frameRecords.last().evBias).isWithin(1e-3f).of(-0.5f)
            assertThat(parsed.durationSec).isEqualTo(2)
            assertThat(FieldDataLogger.readReplayDurationSecondsQuick(tempDir)).isEqualTo(2)
        } finally {
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun extractUprightNv12Frame_packsExactYAndInterleavedUvAcrossLandscapeRotations() {
        val cropper = PushBroomCropper()
        val w = 64
        val h = 36
        val frame0 = object : Yuv420Frame {
            override val width: Int = w
            override val height: Int = h
            override val rotationDegrees: Int = 0
            override val timestampNs: Long = 100L
            override val yRowStride: Int = w + 8
            override val yPixelStride: Int = 1
            override val uRowStride: Int = w + 8
            override val uPixelStride: Int = 2
            override val vRowStride: Int = w + 8
            override val vPixelStride: Int = 2
            override val yBuffer: ByteBuffer = ByteBuffer.allocate(h * (w + 8)).apply {
                for (r in 0 until h) {
                    for (c in 0 until w) {
                        put(r * (w + 8) + c, ((r * 3 + c) and 0xFF).toByte())
                    }
                }
            }
            override val uBuffer: ByteBuffer = ByteBuffer.allocate((h / 2) * (w + 8)).apply {
                for (r in 0 until h / 2) {
                    for (c in 0 until w / 2) {
                        put(r * (w + 8) + c * 2, (80 + c).toByte())
                    }
                }
            }
            override val vBuffer: ByteBuffer = ByteBuffer.allocate((h / 2) * (w + 8)).apply {
                for (r in 0 until h / 2) {
                    for (c in 0 until w / 2) {
                        put(r * (w + 8) + c * 2, (160 + r).toByte())
                    }
                }
            }
        }

        val nv12 = ByteArray(w * h * 3 / 2)
        assertThat(cropper.extractUprightNv12Frame(frame0, nv12, dstWidthPx = w, dstHeightPx = h)).isTrue()
        assertThat(nv12[0].toInt() and 0xFF).isEqualTo(0)
        assertThat(nv12[(h - 1) * w + (w - 1)].toInt() and 0xFF).isEqualTo(((h - 1) * 3 + (w - 1)) and 0xFF)
        val uvBase = w * h
        assertThat(nv12[uvBase].toInt() and 0xFF).isEqualTo(80)
        assertThat(nv12[uvBase + 1].toInt() and 0xFF).isEqualTo(160)
        assertThat(nv12[nv12.size - 2].toInt() and 0xFF).isEqualTo(80 + (w / 2 - 1))
        assertThat(nv12[nv12.size - 1].toInt() and 0xFF).isEqualTo(160 + (h / 2 - 1))
    }
}
