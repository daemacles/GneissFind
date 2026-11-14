package com.gneissfind.app.vision

import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.SwathTileMode
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
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val luma = ((x * 7 + y * 13) and 0xFF).toByte()
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
    fun fiveTileAndThreeTileExtraction_1080pReversibleLandscape_preservesExactOffsetsAnd32PxOverlaps() {
        val cropper = PushBroomCropper(applyImageNetNormalization = false)
        val buffers = PushBroomSweepBuffers()

        // Verify locked 5-tile frame X offsets [64, 416, 768, 1120, 1472] -> 1792x384 (93.3% of 1920)
        assertThat(SwathTileMode.FIVE_TILE.frameXOffsets.toList())
            .containsExactly(64, 416, 768, 1120, 1472)
            .inOrder()
        assertThat(SwathTileMode.THREE_TILE.frameXOffsets.toList())
            .containsExactly(416, 768, 1120)
            .inOrder()

        // Test both ROTATION_90 (rot=0) and flipped ROTATION_270 (rot=180) on 1920x1080 stream
        for (rot in listOf(0, 180)) {
            val frame = FakeYuv420Frame(width = 1920, height = 1080, rotationDegrees = rot)
            val geom5 = cropper.extractTiles(
                frame = frame,
                outBuffers = buffers,
                cropY = 412,
                tileMode = SwathTileMode.FIVE_TILE,
                populateArgbStrip = true
            )

            assertThat(geom5.uprightFrameWidth).isEqualTo(1920)
            assertThat(geom5.uprightFrameHeight).isEqualTo(1080)
            assertThat(geom5.stripLeftPx).isEqualTo(64)
            assertThat(geom5.stripTopPx).isEqualTo(412)
            assertThat(geom5.stripWidthPx).isEqualTo(1792)
            assertThat(geom5.stripHeightPx).isEqualTo(384)
            assertThat(buffers.activeTileCount).isEqualTo(5)
            assertThat(buffers.hasArgbPixels).isTrue()

            // Verify all 4 adjacent tile seams (0/1, 1/2, 2/3, 3/4) match bit-for-bit across 32px overlap
            for (tileIdx in 0 until 4) {
                val leftTile = buffers.tileRgbFloats[tileIdx]
                val rightTile = buffers.tileRgbFloats[tileIdx + 1]
                for (row in 0 until StripRoiGeometry.TILE_SIZE_PX) {
                    for (overlapCol in 0 until StripRoiGeometry.TILE_OVERLAP_PX) {
                        val leftIdx = (row * 384 + (352 + overlapCol)) * 3
                        val rightIdx = (row * 384 + overlapCol) * 3
                        assertThat(leftTile[leftIdx]).isEqualTo(rightTile[rightIdx])
                        assertThat(leftTile[leftIdx + 1]).isEqualTo(rightTile[rightIdx + 1])
                        assertThat(leftTile[leftIdx + 2]).isEqualTo(rightTile[rightIdx + 2])
                    }
                }
            }

            // Switch inline to 3-tile fast mode (1088x384 at xLeft = 416)
            val geom3 = cropper.extractTiles(
                frame = frame,
                outBuffers = buffers,
                cropY = 348,
                tileMode = SwathTileMode.THREE_TILE,
                populateArgbStrip = false
            )
            assertThat(geom3.stripLeftPx).isEqualTo(416)
            assertThat(geom3.stripWidthPx).isEqualTo(1088)
            assertThat(buffers.activeTileCount).isEqualTo(3)
        }
    }

    @Test
    fun mergeAndFilterTileDetections_fiveTileNms_deduplicatesSeams_mutesConfusers_andApplies059Gsd() {
        val cropper = PushBroomCropper()

        // At 0.59 mm/px GSD (2.0x crop at R = 1.50m), 40 px = 23.6 mm (>= 20 mm), 24 px = 14.2 mm (< 20 mm)
        val rawDetections = listOf(
            // 1. 40 px Carnelian Agate straddling Tile 3 / Tile 4 overlap seam
            //    Tile 3 [344..384] -> strip [1400..1440], frame [1464..1504]
            //    Tile 4 [0..36]    -> strip [1408..1444] (IoU = 32*40 / (1600+1440-1280) = 1280/1760 = 0.727 >= 0.45)
            TileDetection(
                tileIndex = 3,
                classId = StoneClass.CHALCEDONY_AGATE.id,
                confidence = 0.93f,
                leftPx = 344f,
                topPx = 120f,
                rightPx = 384f,
                bottomPx = 160f
            ),
            TileDetection(
                tileIndex = 4,
                classId = StoneClass.CHALCEDONY_AGATE.id,
                confidence = 0.86f,
                leftPx = 0f,
                topPx = 120f,
                rightPx = 36f,
                bottomPx = 160f
            ),
            // 2. Class 4 hard_confuser (wet orange alder leaf) with 0.99 confidence — MUST be muted!
            TileDetection(
                tileIndex = 4,
                classId = StoneClass.HARD_CONFUSER.id,
                confidence = 0.99f,
                leftPx = 120f,
                topPx = 80f,
                rightPx = 200f,
                bottomPx = 160f
            ),
            // 3. Sub-20mm pebble (24 px * 0.59 mm/px = 14.16 mm < 20 mm) — MUST be discarded!
            TileDetection(
                tileIndex = 1,
                classId = StoneClass.JASPER_RED_YELLOW.id,
                confidence = 0.90f,
                leftPx = 200f,
                topPx = 200f,
                rightPx = 224f,
                bottomPx = 224f
            )
        )

        val merged = cropper.mergeAndFilterTileDetections(
            tileDetections = rawDetections,
            tileMode = SwathTileMode.FIVE_TILE,
            gsdMmPerPx = PushBroomCropper.DEFAULT_GSD_MM_PER_PX
        )

        assertThat(merged).hasSize(1)
        val kept = merged.single()
        assertThat(kept.stoneClass).isEqualTo(StoneClass.CHALCEDONY_AGATE)
        assertThat(kept.confidence).isWithin(1e-4f).of(0.93f)
        // Tile 3 strip offset = 3 * 352 = 1056 -> 1056 + 344 = 1400; frame offset = 1120 -> 1120 + 344 = 1464
        assertThat(kept.leftPx).isWithin(1e-3f).of(1400f)
        assertThat(kept.frameLeftPx).isWithin(1e-3f).of(1464f)
        assertThat(kept.estimatedDiameterMm).isWithin(0.1f).of(23.6f)
    }
}
