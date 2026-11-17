/*
 * Unit test suite for [PushBroomCropper] and [PushBroomSweepBuffers].
 *
 * Covers the optical and geometric invariants of the push-broom vision front-end
 * (`AGENTS.md`, `.agents/rules/android-camera-ar.md`):
 * - Reversible landscape (`ROTATION_90` and `ROTATION_270`) 5-tile (`1792x384`) and 3-tile
 *   (`1088x384`) ROI extraction from `1920x1080` `YUV_420_888` frames.
 * - Bit-exact `32 px` overlap seam identity across adjacent `384x384` tiles and raw `[0..1]`
 *   float32 RGB output range.
 * - Cross-tile seam fusion via `IoU >= 0.45`, `IoM >= 0.65`, and adjacent-seam straddle union box
 *   merging, `>= 20 mm` physical diameter gating at `0.59 mm/px` GSD, and `HARD_CONFUSER` muting.
 *
 * Main types:
 * - [PushBroomCropperTest]: JUnit4 test suite and deterministic `YUV_420_888` frame fixture.
 */
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

/**
 * Verifies [PushBroomCropper] tile extraction, reversible landscape orientation normalization,
 * `32 px` seam pixel identity, and cross-tile union box deduplication.
 */
@RunWith(JUnit4::class)
class PushBroomCropperTest {

    /**
     * Deterministic planar `YUV_420_888` test frame whose Y plane encodes `(x * 7 + y * 13) & 0xFF`
     * so seam overlap tests can verify bit-exact pixel alignment across tiles.
     */
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
        val cropper = PushBroomCropper()
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

            // Verify raw [0..1] normalization is default (no CPU ImageNet double-normalization)
            assertThat(buffers.tileRgbFloats[0][0]).isAtLeast(0.0f)
            assertThat(buffers.tileRgbFloats[0][0]).isAtMost(1.0f)

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
    fun mergeAndFilterTileDetections_fiveTileNms_fusesSeamUnion_mutesConfusers_andApplies059Gsd() {
        val cropper = PushBroomCropper()

        // At 0.59 mm/px GSD (2.0x crop at R = 1.50m), 44 px union = 25.96 mm (>= 20 mm), 24 px = 14.16 mm (< 20 mm)
        val rawDetections = listOf(
            // 1. Carnelian Agate straddling Tile 3 / Tile 4 overlap seam
            //    Tile 3 [344..384] -> strip [1400..1440], frame [1464..1504]
            //    Tile 4 [0..36]    -> strip [1408..1444], frame [1472..1508]
            //    Union box         -> strip [1400..1444] (44 px), frame [1464..1508], confidence = 0.93f
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
            // 2. Class 4 hard_confuser with 0.99 confidence — MUST be muted!
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
        assertThat(kept.leftPx).isWithin(1e-3f).of(1400f)
        assertThat(kept.rightPx).isWithin(1e-3f).of(1444f)
        assertThat(kept.frameLeftPx).isWithin(1e-3f).of(1464f)
        assertThat(kept.frameRightPx).isWithin(1e-3f).of(1508f)
        assertThat(kept.estimatedDiameterMm).isWithin(0.1f).of(44f * 0.59f)
    }

    @Test
    fun mergeAndFilterTileDetections_elongated300PxObjectAcrossTileSeam_fusesIntoSingleUnionBox() {
        val cropper = PushBroomCropper()

        // Simulate a 300 px wide elongated object (e.g., toothbrush -> GREEN_GEM_ROCK or remote -> PETRIFIED_WOOD)
        // spanning strip X = [200..500] across Tile 0 [0..384] and Tile 1 [352..736] (seam [352..384]):
        //  - Tile 0 sees [200..384] (width = 184 px, height = 48 px)
        //  - Tile 1 sees [0..148]   -> strip [352..500] (width = 148 px, height = 48 px)
        // Standard IoU is only 32 / 300 = 0.1067 (< 0.45) and IoM is 32 / 148 = 0.2162 (< 0.65),
        // but both fragments straddle the [352..384] seam band with 100% vertical overlap.
        val rawDetections = listOf(
            TileDetection(
                tileIndex = 0,
                classId = StoneClass.GREEN_GEM_ROCK.id,
                confidence = 0.81f,
                leftPx = 200f,
                topPx = 160f,
                rightPx = 384f,
                bottomPx = 208f
            ),
            TileDetection(
                tileIndex = 1,
                classId = StoneClass.GREEN_GEM_ROCK.id,
                confidence = 0.91f,
                leftPx = 0f,
                topPx = 162f,
                rightPx = 148f,
                bottomPx = 208f
            )
        )

        val merged = cropper.mergeAndFilterTileDetections(
            tileDetections = rawDetections,
            tileMode = SwathTileMode.FIVE_TILE,
            gsdMmPerPx = PushBroomCropper.DEFAULT_GSD_MM_PER_PX
        )

        assertThat(merged).hasSize(1)
        val unified = merged.single()
        assertThat(unified.stoneClass).isEqualTo(StoneClass.GREEN_GEM_ROCK)
        assertThat(unified.confidence).isWithin(1e-4f).of(0.91f)
        assertThat(unified.leftPx).isWithin(1e-3f).of(200f)
        assertThat(unified.rightPx).isWithin(1e-3f).of(500f)
        assertThat(unified.topPx).isWithin(1e-3f).of(160f)
        assertThat(unified.bottomPx).isWithin(1e-3f).of(208f)
        assertThat(unified.widthPx).isWithin(1e-3f).of(300f)
        assertThat(unified.estimatedDiameterMm).isWithin(0.1f).of(300f * 0.59f)
    }

    @Test
    fun mergeAndFilterTileDetections_highIoMWithLowIoU_fusesNestedFragmentIntoParentBox() {
        val cropper = PushBroomCropper()

        // Large box in Tile 1: strip [400..520] x [100..220] (120x120 = 14,400 px^2)
        // Smaller clipped fragment in Tile 1: strip [410..450] x [110..150] (40x40 = 1,600 px^2)
        // IoU = 1,600 / 14,400 = 0.111 (< 0.45), but IoM = 1,600 / 1,600 = 1.00 (>= 0.65).
        val rawDetections = listOf(
            TileDetection(
                tileIndex = 1,
                classId = StoneClass.PETRIFIED_WOOD.id,
                confidence = 0.88f,
                leftPx = 48f,
                topPx = 100f,
                rightPx = 168f,
                bottomPx = 220f
            ),
            TileDetection(
                tileIndex = 1,
                classId = StoneClass.PETRIFIED_WOOD.id,
                confidence = 0.64f,
                leftPx = 58f,
                topPx = 110f,
                rightPx = 98f,
                bottomPx = 150f
            )
        )

        val merged = cropper.mergeAndFilterTileDetections(
            tileDetections = rawDetections,
            tileMode = SwathTileMode.FIVE_TILE,
            gsdMmPerPx = PushBroomCropper.DEFAULT_GSD_MM_PER_PX
        )

        assertThat(merged).hasSize(1)
        val unified = merged.single()
        assertThat(unified.stoneClass).isEqualTo(StoneClass.PETRIFIED_WOOD)
        assertThat(unified.confidence).isWithin(1e-4f).of(0.88f)
        assertThat(unified.widthPx).isWithin(1e-3f).of(120f)
    }

    /**
     * Strict Android Camera2 semi-planar `NV21`/`NV12` `YUV_420_888` fixture (`pixelStride = 2`)
     * whose chroma `ByteBuffer.limit()` equals `(width / 2 - 1) * 2 + 1 + (height / 2 - 1) * rowStride`
     * (`1,036,799` for `1920x1080`)—omitting the final interleaved byte at `(uvX = 959, uvY = 539)`
     * where `uvY * rowStride + uvX * 2 == 1,036,798` on one plane and `1,036,799` (out-of-bounds!)
     * when `limit` is truncated by 1 byte on Android HALs (`.agents/rules/android-camera-ar.md §1.1`).
     */
    private class StrictSemiPlanarNv21Frame(
        override val width: Int = 1920,
        override val height: Int = 1080,
        override val rotationDegrees: Int = 0,
        override val timestampNs: Long = 2_000_000_000L
    ) : Yuv420Frame {
        override val yRowStride: Int = width + 64
        override val yPixelStride: Int = 1
        override val uRowStride: Int = width + 64
        override val uPixelStride: Int = 2
        override val vRowStride: Int = width + 64
        override val vPixelStride: Int = 2

        private val exactLumaLimit = (height - 1) * yRowStride + width
        // Exact Android YUV_420_888 semi-planar second-plane limit where last pixel (uvX=w/2-1, uvY=h/2-1)
        // has NO trailing interleaved partner byte, so uvY * rowStride + uvX * 2 == limit - 1, and
        // even tighter (limit - 1) tests the exact overrun case!
        private val strictChromaLimit = (height / 2 - 1) * uRowStride + (width / 2 - 1) * 2

        override val yBuffer: ByteBuffer = ByteBuffer.allocate(exactLumaLimit).apply {
            for (i in 0 until exactLumaLimit) put(i, (120 + (i % 80)).toByte())
            limit(exactLumaLimit)
        }
        override val uBuffer: ByteBuffer = ByteBuffer.allocate(strictChromaLimit).apply {
            for (i in 0 until strictChromaLimit) put(i, if (i % 2 == 0) 110.toByte() else 146.toByte())
            limit(strictChromaLimit)
        }
        override val vBuffer: ByteBuffer = ByteBuffer.allocate(strictChromaLimit).apply {
            for (i in 0 until strictChromaLimit) put(i, if (i % 2 == 0) 146.toByte() else 110.toByte())
            limit(strictChromaLimit)
        }
    }

    @Test
    fun semiPlanarNv21LastPixelChromaLimit_sweeps160FramesAcrossAllFourEdgesAndRotationsWithoutOverrun() {
        val cropper = PushBroomCropper(applyImageNetNormalization = true)
        val lumaOut = ByteArray(1920 * 1080)
        val rgb320Out = FloatArray(320 * 320 * 3)
        val sweepBuffers = PushBroomSweepBuffers()

        // Verify parity-preserving chroma step-back on pixelStride=2 when rawIdx == bufferLimit
        val steppedIdx = PushBroomCropper.safeChromaByteIndex(
            uvX = 959,
            uvY = 539,
            rowStride = 1920,
            pixelStride = 2,
            bufferLimit = 539 * 1920 + 959 * 2 // rawIdx == bufferLimit (out of bounds by 1!)
        )
        assertThat(steppedIdx).isEqualTo(539 * 1920 + 958 * 2) // Preserves even parity (no U/V swap!)

        // Sweep extractTiles at bottom-most cropY=696 (hits y=1079, x=1919 in 0/90/180/270 deg)
        for (rot in listOf(0, 90, 180, 270)) {
            val rawW = if (rot == 90 || rot == 270) 1080 else 1920
            val rawH = if (rot == 90 || rot == 270) 1920 else 1080
            val nvFrame = StrictSemiPlanarNv21Frame(width = rawW, height = rawH, rotationDegrees = rot)
            val geom = cropper.extractTiles(
                frame = nvFrame,
                outBuffers = sweepBuffers,
                cropY = StripRoiGeometry.MAX_CROP_Y_PX,
                tileMode = SwathTileMode.FIVE_TILE,
                populateArgbStrip = true
            )
            assertThat(geom.uprightFrameWidth).isEqualTo(1920)
            assertThat(geom.uprightFrameHeight).isEqualTo(1080)
        }

        // Sweep 160+ Lock-On crop extractions across all 4 edges: x in -400..2320, y in -400..1480 + NaN/Inf
        val nvLandscape = StrictSemiPlanarNv21Frame(width = 1920, height = 1080, rotationDegrees = 0)
        var frameCounter = 0
        for (step in 0 until 42) {
            val t = step / 41f
            val sweepPoints = arrayOf(
                Pair(-400f + t * 2720f, 540f),   // Left -> Right (-400 .. +2320)
                Pair(1919.9f, -400f + t * 1880f), // Top -> Bottom along x = 1919.9 (-400 .. +1480)
                Pair(2320f - t * 2720f, 1079.9f), // Bottom-Right -> Bottom-Left along y = 1079.9
                Pair(-400f, 1480f - t * 1880f)    // Left edge Bottom -> Top
            )
            for ((cx, cy) in sweepPoints) {
                val cropSize = if (frameCounter % 2 == 0) 320 else 640
                val ext = cropper.extractUprightLumaAndLockOnCrop320(
                    frame = nvLandscape,
                    outLumaBytes = if (frameCounter % 8 == 0) lumaOut else ByteArray(0),
                    outCrop320RgbFloats = rgb320Out,
                    cropCenterXPx = cx,
                    cropCenterYPx = cy,
                    sourceCropSizePx = cropSize,
                    extractRgbCrop = true
                )
                assertThat(ext.cropLeftPx).isIn(0..(1920 - cropSize))
                assertThat(ext.cropTopPx).isIn(0..(1080 - cropSize))
                assertThat(rgb320Out[0].isFinite()).isTrue()
                assertThat(rgb320Out[rgb320Out.lastIndex].isFinite()).isTrue()
                frameCounter++
            }
        }
        assertThat(frameCounter).isAtLeast(160)

        // Verify NaN and Infinity crop centers never throw in roundToInt() or coerceIn()
        for (bad in listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)) {
            val ext = cropper.extractUprightLumaAndLockOnCrop320(
                frame = nvLandscape,
                outLumaBytes = ByteArray(0),
                outCrop320RgbFloats = rgb320Out,
                cropCenterXPx = bad,
                cropCenterYPx = bad,
                sourceCropSizePx = 640,
                extractRgbCrop = true
            )
            assertThat(ext.cropLeftPx).isIn(0..(1920 - 640))
            assertThat(ext.cropTopPx).isIn(0..(1080 - 640))
        }
    }
}
