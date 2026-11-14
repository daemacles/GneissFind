package com.gneissfind.app.vision

import androidx.camera.core.ImageProxy
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripDetection
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.SwathTileMode
import com.gneissfind.app.model.TileDetection
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Abstraction over a `YUV_420_888` camera frame so [PushBroomCropper] can operate zero-copy on
 * CameraX [ImageProxy] planes in production and deterministic fake buffers in unit tests.
 */
interface Yuv420Frame {
    val width: Int
    val height: Int
    val rotationDegrees: Int
    val timestampNs: Long

    val yBuffer: ByteBuffer
    val yRowStride: Int
    val yPixelStride: Int

    val uBuffer: ByteBuffer
    val uRowStride: Int
    val uPixelStride: Int

    val vBuffer: ByteBuffer
    val vRowStride: Int
    val vPixelStride: Int
}

/**
 * Lightweight zero-copy wrapper adapting a CameraX [ImageProxy] (`ImageFormat.YUV_420_888`) to [Yuv420Frame].
 */
class ImageProxyYuv420Frame(private val imageProxy: ImageProxy) : Yuv420Frame {
    override val width: Int get() = imageProxy.width
    override val height: Int get() = imageProxy.height
    override val rotationDegrees: Int get() = imageProxy.imageInfo.rotationDegrees
    override val timestampNs: Long get() = imageProxy.imageInfo.timestamp

    private val planes = imageProxy.planes
    override val yBuffer: ByteBuffer = planes[0].buffer
    override val yRowStride: Int = planes[0].rowStride
    override val yPixelStride: Int = planes[0].pixelStride

    override val uBuffer: ByteBuffer = planes[1].buffer
    override val uRowStride: Int = planes[1].rowStride
    override val uPixelStride: Int = planes[1].pixelStride

    override val vBuffer: ByteBuffer = planes[2].buffer
    override val vRowStride: Int = planes[2].rowStride
    override val vPixelStride: Int = planes[2].pixelStride
}

/**
 * Pre-allocated reusable sweep buffers sized for up to 5 overlapping `384x384` RGB float tiles
 * (`1792x384` max swath) plus an optional packed ARGB pixel buffer for `LOG DATA` strip capture.
 */
class PushBroomSweepBuffers(
    val tileRgbFloats: Array<FloatArray> = Array(StripRoiGeometry.MAX_TILES) {
        FloatArray(StripRoiGeometry.TILE_SIZE_PX * StripRoiGeometry.TILE_SIZE_PX * 3)
    },
    val stripArgbPixels: IntArray = IntArray(
        StripRoiGeometry.MAX_STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX
    )
) {
    var activeTileMode: SwathTileMode = SwathTileMode.FIVE_TILE
    val activeTileCount: Int get() = activeTileMode.tileCount
    val activeStripWidthPx: Int get() = activeTileMode.stripWidthPx

    var geometry: StripRoiGeometry = StripRoiGeometry()
    var hasArgbPixels: Boolean = false
    var timestampNs: Long = 0L
}

/**
 * Extracts ONLY the stabilized `1792x384` (5-tile default) or `1088x384` (3-tile fast mode)
 * push-broom strip ROI from a `1920x1080` landscape `YUV_420_888` frame directly into horizontally
 * overlapping `384x384` float tiles (`32 px` overlap, `352 px` stride) without full-frame Bitmap
 * allocation.
 *
 * Maps tile-local `[0..384]` detections into unified strip space, runs class-aware cross-tile NMS
 * (`IoU >= 0.45`), mutes `classId == 4` (`hard_confuser`), and enforces `>= 20 mm` target diameter.
 */
class PushBroomCropper(
    private val applyImageNetNormalization: Boolean = true
) {
    // Reusable intermediate strip RGB cache allocated once for max 1792x384 swath (~8.2 MB total)
    // so the four 32px overlap seams are decoded from YUV_420_888 only once per sweep.
    private val stripNormalizedR = FloatArray(StripRoiGeometry.MAX_STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX)
    private val stripNormalizedG = FloatArray(StripRoiGeometry.MAX_STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX)
    private val stripNormalizedB = FloatArray(StripRoiGeometry.MAX_STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX)

    fun computeStripGeometry(
        rawWidth: Int,
        rawHeight: Int,
        rotationDegrees: Int,
        cropY: Int = StripRoiGeometry.NOMINAL_CROP_Y_PX,
        tileMode: SwathTileMode = SwathTileMode.FIVE_TILE,
        isPitchClamped: Boolean = false
    ): StripRoiGeometry {
        val normalizedRotation = ((rotationDegrees % 360) + 360) % 360
        val uprightW = if (normalizedRotation == 90 || normalizedRotation == 270) rawHeight else rawWidth
        val uprightH = if (normalizedRotation == 90 || normalizedRotation == 270) rawWidth else rawHeight

        val clampedCanonicalCropY = cropY.coerceIn(0, StripRoiGeometry.MAX_CROP_Y_PX)
        val stripWidth = tileMode.stripWidthPx

        val stripLeft = when {
            uprightW == StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX -> tileMode.canonicalStripLeftPx
            uprightW > stripWidth -> ((uprightW - stripWidth) / 2)
            else -> 0
        }

        val maxTop = (uprightH - StripRoiGeometry.STRIP_HEIGHT_PX).coerceAtLeast(0)
        val stripTop = when {
            uprightH == StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX -> clampedCanonicalCropY
            maxTop > 0 -> {
                val frac = clampedCanonicalCropY.toFloat() / StripRoiGeometry.MAX_CROP_Y_PX.toFloat()
                (frac * maxTop).roundToInt().coerceIn(0, maxTop)
            }
            else -> 0
        }

        return StripRoiGeometry(
            uprightFrameWidth = uprightW,
            uprightFrameHeight = uprightH,
            stripLeftPx = stripLeft,
            stripTopPx = stripTop,
            stripWidthPx = stripWidth,
            stripHeightPx = StripRoiGeometry.STRIP_HEIGHT_PX,
            tileMode = tileMode,
            isPitchClamped = isPitchClamped
        )
    }

    /**
     * Extracts the active `1792x384` (5-tile) or `1088x384` (3-tile) strip at stabilized [cropY]
     * from [frame] and slices it into overlapping `384x384` NHWC RGB float tiles inside [outBuffers].
     */
    fun extractTiles(
        frame: Yuv420Frame,
        outBuffers: PushBroomSweepBuffers,
        cropY: Int = StripRoiGeometry.NOMINAL_CROP_Y_PX,
        tileMode: SwathTileMode = SwathTileMode.FIVE_TILE,
        isPitchClamped: Boolean = false,
        populateArgbStrip: Boolean = false
    ): StripRoiGeometry {
        val rawW = frame.width
        val rawH = frame.height
        val rot = ((frame.rotationDegrees % 360) + 360) % 360
        val geometry = computeStripGeometry(
            rawWidth = rawW,
            rawHeight = rawH,
            rotationDegrees = rot,
            cropY = cropY,
            tileMode = tileMode,
            isPitchClamped = isPitchClamped
        )

        outBuffers.activeTileMode = tileMode
        outBuffers.geometry = geometry
        outBuffers.timestampNs = frame.timestampNs
        outBuffers.hasArgbPixels = populateArgbStrip

        val yBuf = frame.yBuffer
        val uBuf = frame.uBuffer
        val vBuf = frame.vBuffer
        val yRowStride = frame.yRowStride
        val yPixStride = frame.yPixelStride
        val uRowStride = frame.uRowStride
        val uPixStride = frame.uPixelStride
        val vRowStride = frame.vRowStride
        val vPixStride = frame.vPixelStride

        val yLimit = yBuf.limit()
        val uLimit = uBuf.limit()
        val vLimit = vBuf.limit()

        val uprightW = geometry.uprightFrameWidth
        val uprightH = geometry.uprightFrameHeight
        val activeStripW = tileMode.stripWidthPx

        // Safe coordinate fallback if an emulator webcam provides < 1792px width
        val scaleX = if (uprightW < activeStripW) {
            uprightW.toFloat() / activeStripW
        } else {
            1.0f
        }
        val scaleY = if (uprightH < StripRoiGeometry.STRIP_HEIGHT_PX) {
            uprightH.toFloat() / StripRoiGeometry.STRIP_HEIGHT_PX
        } else {
            1.0f
        }

        val argbOut = outBuffers.stripArgbPixels

        for (sy in 0 until StripRoiGeometry.STRIP_HEIGHT_PX) {
            val uy = if (scaleY == 1.0f) {
                (geometry.stripTopPx + sy).coerceIn(0, uprightH - 1)
            } else {
                (sy * scaleY).toInt().coerceIn(0, uprightH - 1)
            }
            val stripRowOffset = sy * activeStripW

            for (sx in 0 until activeStripW) {
                val ux = if (scaleX == 1.0f) {
                    (geometry.stripLeftPx + sx).coerceIn(0, uprightW - 1)
                } else {
                    (sx * scaleX).toInt().coerceIn(0, uprightW - 1)
                }

                // Map upright landscape coordinates (ux, uy) back to raw sensor buffer (bx, by)
                val bx: Int
                val by: Int
                when (rot) {
                    90 -> {
                        bx = uy
                        by = (rawH - 1 - ux).coerceIn(0, rawH - 1)
                    }
                    180 -> {
                        bx = (rawW - 1 - ux).coerceIn(0, rawW - 1)
                        by = (rawH - 1 - uy).coerceIn(0, rawH - 1)
                    }
                    270 -> {
                        bx = (rawW - 1 - uy).coerceIn(0, rawW - 1)
                        by = ux
                    }
                    else -> {
                        bx = ux
                        by = uy
                    }
                }

                val yIndex = (by * yRowStride + bx * yPixStride).coerceIn(0, yLimit - 1)
                val uvX = bx shr 1
                val uvY = by shr 1
                val uIndex = (uvY * uRowStride + uvX * uPixStride).coerceIn(0, uLimit - 1)
                val vIndex = (uvY * vRowStride + uvX * vPixStride).coerceIn(0, vLimit - 1)

                val yVal = (yBuf.get(yIndex).toInt() and 0xFF)
                val uVal = (uBuf.get(uIndex).toInt() and 0xFF) - 128
                val vVal = (vBuf.get(vIndex).toInt() and 0xFF) - 128

                // ITU-R BT.601 full-swing integer-fast YUV -> RGB
                val r8 = (yVal + ((1436 * vVal) shr 10)).coerceIn(0, 255)
                val g8 = (yVal - ((352 * uVal + 731 * vVal) shr 10)).coerceIn(0, 255)
                val b8 = (yVal + ((1815 * uVal) shr 10)).coerceIn(0, 255)

                val stripIdx = stripRowOffset + sx
                if (applyImageNetNormalization) {
                    stripNormalizedR[stripIdx] = (r8 * INV_255 - IMAGENET_MEAN_R) * IMAGENET_INV_STD_R
                    stripNormalizedG[stripIdx] = (g8 * INV_255 - IMAGENET_MEAN_G) * IMAGENET_INV_STD_G
                    stripNormalizedB[stripIdx] = (b8 * INV_255 - IMAGENET_MEAN_B) * IMAGENET_INV_STD_B
                } else {
                    stripNormalizedR[stripIdx] = r8 * INV_255
                    stripNormalizedG[stripIdx] = g8 * INV_255
                    stripNormalizedB[stripIdx] = b8 * INV_255
                }

                if (populateArgbStrip) {
                    argbOut[stripIdx] = (0xFF shl 24) or (r8 shl 16) or (g8 shl 8) or b8
                }
            }
        }

        // Slice the decoded strip into 5 (or 3) overlapping 384x384 tiles with 32px seam overlap
        val stripOffsets = tileMode.stripTileOffsets
        for (tileIdx in 0 until tileMode.tileCount) {
            val xOffset = stripOffsets[tileIdx]
            val tileDst = outBuffers.tileRgbFloats[tileIdx]
            var dstIdx = 0
            for (ty in 0 until StripRoiGeometry.TILE_SIZE_PX) {
                val stripRowStart = ty * activeStripW + xOffset
                for (tx in 0 until StripRoiGeometry.TILE_SIZE_PX) {
                    val srcIdx = stripRowStart + tx
                    tileDst[dstIdx++] = stripNormalizedR[srcIdx]
                    tileDst[dstIdx++] = stripNormalizedG[srcIdx]
                    tileDst[dstIdx++] = stripNormalizedB[srcIdx]
                }
            }
        }

        return geometry
    }

    /**
     * Maps tile-local `[0..384]` bounding boxes into unified strip coordinates (`1792x384` in
     * 5-tile mode or `1088x384` in 3-tile mode) and `1920x1080` frame X coordinates
     * (`xOffsets = [64, 416, 768, 1120, 1472]` or `[416, 768, 1120]`), applies class-aware
     * Cross-Tile Non-Maximum Suppression (`IoU >= 0.45`), and discards:
     *  - `classId == 4` ([StoneClass.HARD_CONFUSER])
     *  - Classes disabled in [enabledClasses]
     *  - Stones smaller than [minDiameterMm] (`20.0 mm`)
     */
    fun mergeAndFilterTileDetections(
        tileDetections: List<TileDetection>,
        tileMode: SwathTileMode = SwathTileMode.FIVE_TILE,
        gsdMmPerPx: Float = DEFAULT_GSD_MM_PER_PX,
        nmsIouThreshold: Float = DEFAULT_NMS_IOU_THRESHOLD,
        minConfidence: Float = DEFAULT_MIN_CONFIDENCE,
        minDiameterMm: Float = MIN_STONE_DIAMETER_MM,
        enabledClasses: Set<StoneClass> = StoneClass.alertableClasses.toSet()
    ): List<StripDetection> {
        if (tileDetections.isEmpty()) return emptyList()

        val stripWidthF = tileMode.stripWidthPx.toFloat()
        val stripOffsets = tileMode.stripTileOffsets
        val frameOffsets = tileMode.frameXOffsets

        val mappedCandidates = ArrayList<StripDetection>(tileDetections.size)
        for (det in tileDetections) {
            if (det.confidence < minConfidence) continue
            if (det.tileIndex !in 0 until tileMode.tileCount) continue
            val stoneClass = StoneClass.fromId(det.classId) ?: continue

            val stripOffsetX = stripOffsets[det.tileIndex].toFloat()
            val frameOffsetX = frameOffsets[det.tileIndex].toFloat()

            val stripLeft = (det.leftPx + stripOffsetX).coerceIn(0f, stripWidthF)
            val stripRight = (det.rightPx + stripOffsetX).coerceIn(0f, stripWidthF)
            val stripTop = det.topPx.coerceIn(0f, StripRoiGeometry.STRIP_HEIGHT_PX.toFloat())
            val stripBottom = det.bottomPx.coerceIn(0f, StripRoiGeometry.STRIP_HEIGHT_PX.toFloat())

            val widthPx = stripRight - stripLeft
            val heightPx = stripBottom - stripTop
            if (widthPx <= 1f || heightPx <= 1f) continue

            val frameLeft = (det.leftPx + frameOffsetX)
                .coerceIn(0f, StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX.toFloat())
            val frameRight = (det.rightPx + frameOffsetX)
                .coerceIn(0f, StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX.toFloat())

            val diameterMm = max(widthPx, heightPx) * gsdMmPerPx
            mappedCandidates.add(
                StripDetection(
                    stoneClass = stoneClass,
                    confidence = det.confidence,
                    leftPx = stripLeft,
                    topPx = stripTop,
                    rightPx = stripRight,
                    bottomPx = stripBottom,
                    estimatedDiameterMm = diameterMm,
                    sourceTileIndex = det.tileIndex,
                    frameLeftPx = frameLeft,
                    frameRightPx = frameRight
                )
            )
        }

        if (mappedCandidates.isEmpty()) return emptyList()

        // Sort descending by confidence for greedy Cross-Tile NMS
        mappedCandidates.sortByDescending { it.confidence }

        val nmsKept = ArrayList<StripDetection>(mappedCandidates.size)
        for (candidate in mappedCandidates) {
            var suppressed = false
            for (kept in nmsKept) {
                if (kept.stoneClass == candidate.stoneClass &&
                    candidate.iouWith(kept) >= nmsIouThreshold
                ) {
                    suppressed = true
                    break
                }
            }
            if (!suppressed) {
                nmsKept.add(candidate)
            }
        }

        return nmsKept.filter { det ->
            det.stoneClass.isUserAlertable &&
                det.stoneClass in enabledClasses &&
                det.estimatedDiameterMm >= minDiameterMm
        }
    }

    companion object {
        const val DEFAULT_NMS_IOU_THRESHOLD = 0.45f
        const val DEFAULT_MIN_CONFIDENCE = 0.35f
        const val MIN_STONE_DIAMETER_MM = 20.0f
        const val DEFAULT_SLANT_RANGE_METERS = 1.50f
        const val DEFAULT_ZOOM_RATIO = 2.0f
        const val DEFAULT_GSD_MM_PER_PX = 0.59f // 2.0x crop on 1920x1080 at R = 1.50 m (~1.05 m swath)

        private const val INV_255 = 1.0f / 255.0f
        private const val IMAGENET_MEAN_R = 0.485f
        private const val IMAGENET_MEAN_G = 0.456f
        private const val IMAGENET_MEAN_B = 0.406f
        private const val IMAGENET_INV_STD_R = 1.0f / 0.229f
        private const val IMAGENET_INV_STD_G = 1.0f / 0.224f
        private const val IMAGENET_INV_STD_B = 1.0f / 0.225f

        /**
         * Computes cross-track Ground Sample Distance (`mm/px`) from slant range [slantRangeMeters]
         * and active zoom (`2.0x` -> `0.59 mm/px` at `R = 1.50 m`; `1.0x` -> `1.18 mm/px`).
         */
        fun computeGsdMmPerPx(
            slantRangeMeters: Float,
            zoomRatio: Float = DEFAULT_ZOOM_RATIO
        ): Float {
            val clampedRange = slantRangeMeters.coerceIn(0.40f, 4.00f)
            val safeZoom = zoomRatio.coerceAtLeast(1.0f)
            return DEFAULT_GSD_MM_PER_PX *
                (clampedRange / DEFAULT_SLANT_RANGE_METERS) *
                (DEFAULT_ZOOM_RATIO / safeZoom)
        }
    }
}
