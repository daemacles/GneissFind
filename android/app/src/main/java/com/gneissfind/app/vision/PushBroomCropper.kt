package com.gneissfind.app.vision

import androidx.camera.core.ImageProxy
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripDetection
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.TileDetection
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Abstraction over a YUV_420_888 camera frame so [PushBroomCropper] can operate zero-copy on
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
 * Holds the three overlapping 384x384 RGB normalized float tiles extracted from a single sweep,
 * plus an optional 1088x384 packed ARGB pixel buffer when Field Data Logger mode is active.
 */
class PushBroomSweepBuffers(
    val tileRgbFloats: Array<FloatArray> = Array(StripRoiGeometry.NUM_TILES) {
        FloatArray(StripRoiGeometry.TILE_SIZE_PX * StripRoiGeometry.TILE_SIZE_PX * 3)
    },
    val stripArgbPixels: IntArray = IntArray(StripRoiGeometry.STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX)
) {
    var geometry: StripRoiGeometry = StripRoiGeometry(
        uprightFrameWidth = 3840,
        uprightFrameHeight = 2160,
        stripLeftPx = (3840 - StripRoiGeometry.STRIP_WIDTH_PX) / 2,
        stripTopPx = 280
    )
    var hasArgbPixels: Boolean = false
    var timestampNs: Long = 0L
}

/**
 * Extracts ONLY the 1088x384 push-broom strip ROI from a `YUV_420_888` frame directly into
 * three horizontally overlapping 384x384 float tiles (x in [0,384), [352,736), [704,1088))
 * without ever allocating or converting a full 4K Bitmap.
 *
 * Also maps tile-local detections back into unified 1088x384 strip coordinates, runs cross-tile
 * Non-Maximum Suppression (IoU = 0.50), and filters out `classId == 4` (`hard_confuser`) and
 * sub-20 mm targets.
 */
class PushBroomCropper(
    private val verticalAnchorFraction: Float = StripRoiGeometry.DEFAULT_VERTICAL_ANCHOR,
    private val applyImageNetNormalization: Boolean = true
) {
    // Reusable intermediate strip RGB cache (1088 * 384 * 3 floats = ~4.8 MB allocated once)
    // so the two 32px overlap seams are decoded from YUV only once per sweep.
    private val stripNormalizedR = FloatArray(StripRoiGeometry.STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX)
    private val stripNormalizedG = FloatArray(StripRoiGeometry.STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX)
    private val stripNormalizedB = FloatArray(StripRoiGeometry.STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX)

    fun computeStripGeometry(
        rawWidth: Int,
        rawHeight: Int,
        rotationDegrees: Int
    ): StripRoiGeometry {
        val normalizedRotation = ((rotationDegrees % 360) + 360) % 360
        val uprightW = if (normalizedRotation == 90 || normalizedRotation == 270) rawHeight else rawWidth
        val uprightH = if (normalizedRotation == 90 || normalizedRotation == 270) rawWidth else rawHeight

        val stripLeft = ((uprightW - StripRoiGeometry.STRIP_WIDTH_PX) / 2).coerceAtLeast(0)
        val desiredCenterY = (uprightH * verticalAnchorFraction).roundToInt()
        val maxTop = (uprightH - StripRoiGeometry.STRIP_HEIGHT_PX).coerceAtLeast(0)
        val stripTop = (desiredCenterY - StripRoiGeometry.STRIP_HEIGHT_PX / 2).coerceIn(0, maxTop)

        return StripRoiGeometry(
            uprightFrameWidth = uprightW,
            uprightFrameHeight = uprightH,
            stripLeftPx = stripLeft,
            stripTopPx = stripTop
        )
    }

    /**
     * Extracts the 1088x384 push-broom strip from [frame] and slices it into 3 overlapping
     * 384x384 tiles in NHWC RGB float order inside [outBuffers].
     *
     * @param populateArgbStrip If true (used in `LOG DATA` mode), also populates [PushBroomSweepBuffers.stripArgbPixels]
     *   for lossless PNG / high-bitrate JPEG strip logging without full-frame Bitmap allocation.
     */
    fun extractTiles(
        frame: Yuv420Frame,
        outBuffers: PushBroomSweepBuffers,
        populateArgbStrip: Boolean = false
    ): StripRoiGeometry {
        val rawW = frame.width
        val rawH = frame.height
        val rot = ((frame.rotationDegrees % 360) + 360) % 360
        val geometry = computeStripGeometry(rawW, rawH, rot)

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
        // If a test or fallback camera stream has dimensions smaller than 1088x384, scale sample coordinates safely.
        val scaleX = if (uprightW < StripRoiGeometry.STRIP_WIDTH_PX) {
            uprightW.toFloat() / StripRoiGeometry.STRIP_WIDTH_PX
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
            val stripRowOffset = sy * StripRoiGeometry.STRIP_WIDTH_PX

            for (sx in 0 until StripRoiGeometry.STRIP_WIDTH_PX) {
                val ux = if (scaleX == 1.0f) {
                    (geometry.stripLeftPx + sx).coerceIn(0, uprightW - 1)
                } else {
                    (sx * scaleX).toInt().coerceIn(0, uprightW - 1)
                }

                // Map upright viewfinder coordinates (ux, uy) back to raw sensor buffer (bx, by)
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

        // Slice the 1088x384 strip into the 3 overlapping 384x384 tiles:
        // Tile 0: x in [0, 384), Tile 1: x in [352, 736), Tile 2: x in [704, 1088)
        for (tileIdx in 0 until StripRoiGeometry.NUM_TILES) {
            val xOffset = StripRoiGeometry.TILE_OFFSETS_X[tileIdx]
            val tileDst = outBuffers.tileRgbFloats[tileIdx]
            var dstIdx = 0
            for (ty in 0 until StripRoiGeometry.TILE_SIZE_PX) {
                val stripRowStart = ty * StripRoiGeometry.STRIP_WIDTH_PX + xOffset
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
     * Maps tile-local bounding boxes back to 1088x384 strip coordinates, applies cross-tile
     * Non-Maximum Suppression (`IoU = 0.50`), and discards:
     *  - `classId == 4` ([StoneClass.HARD_CONFUSER])
     *  - Any class disabled in [enabledClasses]
     *  - Any box whose physical diameter at [gsdMmPerPx] is `< 20.0 mm`
     */
    fun mergeAndFilterTileDetections(
        tileDetections: List<TileDetection>,
        gsdMmPerPx: Float = DEFAULT_GSD_MM_PER_PX,
        nmsIouThreshold: Float = DEFAULT_NMS_IOU_THRESHOLD,
        minConfidence: Float = DEFAULT_MIN_CONFIDENCE,
        minDiameterMm: Float = MIN_STONE_DIAMETER_MM,
        enabledClasses: Set<StoneClass> = StoneClass.alertableClasses.toSet()
    ): List<StripDetection> {
        if (tileDetections.isEmpty()) return emptyList()

        val mappedCandidates = ArrayList<StripDetection>(tileDetections.size)
        for (det in tileDetections) {
            if (det.confidence < minConfidence) continue
            if (det.tileIndex !in 0 until StripRoiGeometry.NUM_TILES) continue
            val stoneClass = StoneClass.fromId(det.classId) ?: continue

            val xOffset = StripRoiGeometry.TILE_OFFSETS_X[det.tileIndex].toFloat()
            val stripLeft = (det.leftPx + xOffset).coerceIn(0f, StripRoiGeometry.STRIP_WIDTH_PX.toFloat())
            val stripRight = (det.rightPx + xOffset).coerceIn(0f, StripRoiGeometry.STRIP_WIDTH_PX.toFloat())
            val stripTop = det.topPx.coerceIn(0f, StripRoiGeometry.STRIP_HEIGHT_PX.toFloat())
            val stripBottom = det.bottomPx.coerceIn(0f, StripRoiGeometry.STRIP_HEIGHT_PX.toFloat())

            val widthPx = stripRight - stripLeft
            val heightPx = stripBottom - stripTop
            if (widthPx <= 1f || heightPx <= 1f) continue

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
                    sourceTileIndex = det.tileIndex
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

        // Strict v1 post-NMS filtering:
        // 1. Mute/discard classId == 4 (hard_confuser)
        // 2. Respect user per-class HUD toggles
        // 3. Enforce >= 20 mm minimum target stone physical diameter
        return nmsKept.filter { det ->
            det.stoneClass.isUserAlertable &&
                det.stoneClass in enabledClasses &&
                det.estimatedDiameterMm >= minDiameterMm
        }
    }

    companion object {
        const val DEFAULT_NMS_IOU_THRESHOLD = 0.50f
        const val DEFAULT_MIN_CONFIDENCE = 0.35f
        const val MIN_STONE_DIAMETER_MM = 20.0f
        const val DEFAULT_SLANT_RANGE_METERS = 1.50f
        const val DEFAULT_GSD_MM_PER_PX = 0.85f

        private const val INV_255 = 1.0f / 255.0f
        private const val IMAGENET_MEAN_R = 0.485f
        private const val IMAGENET_MEAN_G = 0.456f
        private const val IMAGENET_MEAN_B = 0.406f
        private const val IMAGENET_INV_STD_R = 1.0f / 0.229f
        private const val IMAGENET_INV_STD_G = 1.0f / 0.224f
        private const val IMAGENET_INV_STD_B = 1.0f / 0.225f

        /**
         * Scales cross-track Ground Sample Distance linearly with slant range R around the
         * locked v1 baseline (0.85 mm/px at R = 1.50 m).
         */
        fun computeGsdMmPerPx(slantRangeMeters: Float): Float {
            val clampedRange = slantRangeMeters.coerceIn(0.40f, 4.00f)
            return DEFAULT_GSD_MM_PER_PX * (clampedRange / DEFAULT_SLANT_RANGE_METERS)
        }
    }
}
