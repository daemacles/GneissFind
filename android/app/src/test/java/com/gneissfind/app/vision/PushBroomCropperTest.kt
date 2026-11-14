package com.gneissfind.app.vision

import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.TileDetection
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import java.nio.ByteBuffer

@RunWith(JUnit4::class)
class PushBroomCropperTest {

    private class FakeYuv420Frame(
        override val width: Int,
        override val height: Int,
        override val rotationDegrees: Int = 0,
        override val timestampNs: Long = 1_000_000_000L
    ) : Yuv420Frame {
        override val yRowStride: Int = width
        override val yPixelStride: Int = 1
        override val uRowStride: Int = width / 2
        override val uPixelStride: Int = 1
        override val vRowStride: Int = width / 2
        override val vPixelStride: Int = 1

        override val yBuffer: ByteBuffer = ByteBuffer.allocate(width * height)
        override val uBuffer: ByteBuffer = ByteBuffer.allocate((width / 2) * (height / 2))
        override val vBuffer: ByteBuffer = ByteBuffer.allocate((width / 2) * (height / 2))

        init {
            // Fill deterministic horizontal & vertical luma gradient + neutral chroma
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val luma = ((x * 3 + y * 5) and 0xFF).toByte()
                    yBuffer.put(y * yRowStride + x, luma)
                }
            }
            for (i in 0 until uBuffer.capacity()) {
                uBuffer.put(i, 128.toByte())
                vBuffer.put(i, 140.toByte())
            }
        }
    }

    @Test
    fun extractTiles_4kPortraitAndLandscape_preservesExact32PxTileOverlaps() {
        val cropper = PushBroomCropper(applyImageNetNormalization = false)
        val buffers = PushBroomSweepBuffers()

        // Test portrait 90-degree 4K UHD camera stream (raw 3840x2160 -> upright 2160x3840)
        val frame = FakeYuv420Frame(width = 3840, height = 2160, rotationDegrees = 90)
        val geom = cropper.extractTiles(frame = frame, outBuffers = buffers, populateArgbStrip = true)

        assertThat(geom.uprightFrameWidth).isEqualTo(2160)
        assertThat(geom.uprightFrameHeight).isEqualTo(3840)
        assertThat(geom.stripWidthPx).isEqualTo(1088)
        assertThat(geom.stripHeightPx).isEqualTo(384)
        assertThat(buffers.hasArgbPixels).isTrue()

        val tile0 = buffers.tileRgbFloats[0]
        val tile1 = buffers.tileRgbFloats[1]
        val tile2 = buffers.tileRgbFloats[2]

        // Verify Tile 0 [x = 352..383] matches Tile 1 [x = 0..31] bit-for-bit across all 384 rows
        for (row in 0 until StripRoiGeometry.TILE_SIZE_PX) {
            for (overlapCol in 0 until StripRoiGeometry.TILE_OVERLAP_PX) {
                val t0Idx = (row * 384 + (352 + overlapCol)) * 3
                val t1Idx = (row * 384 + overlapCol) * 3
                assertThat(tile0[t0Idx]).isEqualTo(tile1[t1Idx])
                assertThat(tile0[t0Idx + 1]).isEqualTo(tile1[t1Idx + 1])
                assertThat(tile0[t0Idx + 2]).isEqualTo(tile1[t1Idx + 2])

                // Verify Tile 1 [x = 352..383] matches Tile 2 [x = 0..31]
                val t1RightIdx = (row * 384 + (352 + overlapCol)) * 3
                val t2LeftIdx = (row * 384 + overlapCol) * 3
                assertThat(tile1[t1RightIdx]).isEqualTo(tile2[t2LeftIdx])
            }
        }
    }

    @Test
    fun mergeAndFilterTileDetections_deduplicatesSeamBoxes_mutesConfusers_andFiltersSub20mm() {
        val cropper = PushBroomCropper()

        val rawDetections = listOf(
            // 1. 30 px Agate (~25.5 mm at 0.85 mm/px) straddling Tile 0 & Tile 1 overlap seam (strip x = 354..384)
            TileDetection(
                tileIndex = 0,
                classId = StoneClass.CHALCEDONY_AGATE.id,
                confidence = 0.91f,
                leftPx = 354f,
                topPx = 100f,
                rightPx = 384f,
                bottomPx = 130f
            ),
            TileDetection(
                tileIndex = 1,
                classId = StoneClass.CHALCEDONY_AGATE.id,
                confidence = 0.84f,
                leftPx = 2f,
                topPx = 101f,
                rightPx = 32f,
                bottomPx = 131f
            ),
            // 2. Class 4 hard_confuser (wet orange alder leaf) with 0.98 confidence — MUST be muted!
            TileDetection(
                tileIndex = 2,
                classId = StoneClass.HARD_CONFUSER.id,
                confidence = 0.98f,
                leftPx = 80f,
                topPx = 80f,
                rightPx = 140f,
                bottomPx = 140f
            ),
            // 3. Tiny 14 px Jasper grain (~11.9 mm < 20 mm minimum) — MUST be discarded!
            TileDetection(
                tileIndex = 1,
                classId = StoneClass.JASPER_RED_YELLOW.id,
                confidence = 0.89f,
                leftPx = 200f,
                topPx = 200f,
                rightPx = 214f,
                bottomPx = 214f
            )
        )

        val merged = cropper.mergeAndFilterTileDetections(
            tileDetections = rawDetections,
            gsdMmPerPx = 0.85f
        )

        assertThat(merged).hasSize(1)
        val kept = merged.single()
        assertThat(kept.stoneClass).isEqualTo(StoneClass.CHALCEDONY_AGATE)
        assertThat(kept.confidence).isWithin(1e-4f).of(0.91f)
        assertThat(kept.leftPx).isWithin(1e-3f).of(354f)
        assertThat(kept.estimatedDiameterMm).isWithin(0.1f).of(25.5f)
    }
}
