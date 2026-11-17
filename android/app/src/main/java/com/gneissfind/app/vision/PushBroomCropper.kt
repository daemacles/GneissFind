/*
 * YUV_420_888 push-broom strip ROI extractor and cross-tile bounding-box stitcher.
 *
 * Consumes upright landscape CameraX ImageProxy frames and pitch-stabilized cropY offsets from
 * GroundPlaneRaycaster; produces overlapping square NHWC RGB [0..1] float tiles in
 * PushBroomSweepBuffers for StoneDetector and stitches per-tile predictions into unified strip
 * coordinates with IoU, IoM, and seam-straddle union merging.
 */
package com.gneissfind.app.vision

import androidx.camera.core.ImageProxy
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripDetection
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.SwathTileMode
import com.gneissfind.app.model.TileDetection
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Read-only view over a planar or semi-planar `YUV_420_888` camera frame so [PushBroomCropper]
 * can read CameraX [ImageProxy] planes in production and deterministic synthetic buffers in JVM
 * unit tests without copying full-frame byte arrays.
 *
 * Implementations do not synchronize buffer reads; call from a single analysis worker thread.
 */
interface Yuv420Frame {
    /** Raw sensor buffer width in pixels before display rotation. */
    val width: Int
    /** Raw sensor buffer height in pixels before display rotation. */
    val height: Int
    /** Clockwise rotation (`0`, `90`, `180`, or `270` degrees) required to orient the frame upright. */
    val rotationDegrees: Int
    /** Frame capture timestamp in nanoseconds (`SensorEvent` / `CameraCharacteristics` timebase). */
    val timestampNs: Long

    /** Full-resolution luminance (`Y`) plane buffer in raw unrotated sensor coordinates. */
    val yBuffer: ByteBuffer
    /** Byte offset between adjacent rows in [yBuffer]. */
    val yRowStride: Int
    /** Byte offset between adjacent horizontal pixels in [yBuffer] (typically `1`). */
    val yPixelStride: Int

    /** Subsampled `2x2` blue-difference chroma (`Cb` / `U`) plane buffer in raw sensor coordinates. */
    val uBuffer: ByteBuffer
    /** Byte offset between adjacent rows in [uBuffer]. */
    val uRowStride: Int
    /** Byte offset between adjacent horizontal samples in [uBuffer] (`1` planar I420, `2` semi-planar NV12/NV21). */
    val uPixelStride: Int

    /** Subsampled `2x2` red-difference chroma (`Cr` / `V`) plane buffer in raw sensor coordinates. */
    val vBuffer: ByteBuffer
    /** Byte offset between adjacent rows in [vBuffer]. */
    val vRowStride: Int
    /** Byte offset between adjacent horizontal samples in [vBuffer] (`1` planar I420, `2` semi-planar NV12/NV21). */
    val vPixelStride: Int
}

/**
 * Adapter wrapping a CameraX [ImageProxy] (`ImageFormat.YUV_420_888`) as a [Yuv420Frame].
 *
 * State: holds direct references to `imageProxy.planes[0..2].buffer`.
 * Lifecycle: valid only while the underlying [imageProxy] remains open; callers must close
 * [imageProxy] after [PushBroomCropper.extractTiles] returns.
 * Threading: confined to the single-threaded CameraX `ImageAnalysis` executor.
 *
 * @param imageProxy Open CameraX `ImageAnalysis` frame with 3 `YUV_420_888` planes.
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
 * Pre-allocated reusable sweep buffers sized for up to [StripRoiGeometry.MAX_TILES] overlapping
 * [StripRoiGeometry.TILE_SIZE_PX] square NHWC RGB `[0.0, 1.0]` float tiles plus an optional
 * packed ARGB pixel buffer for [com.gneissfind.app.model.OperatingMode.LOG_DATA] strip capture.
 *
 * Allocated once per [com.gneissfind.app.ui.ScannerViewModel] and mutated in-place on the camera
 * analysis thread so per-sweep extraction performs zero heap allocations (`.agents/rules/android-camera-ar.md §2`).
 *
 * @property tileRgbFloats Per-tile NHWC float32 arrays (`[tileIdx][TILE_SIZE_PX * TILE_SIZE_PX * 3]`)
 *   in `[0.0, 1.0]` (or ImageNet-normalized when [PushBroomCropper.applyImageNetNormalization] is true).
 * @property stripArgbPixels Packed `0xAARRGGBB` pixel buffer
 *   (`[StripRoiGeometry.MAX_STRIP_WIDTH_PX] * [StripRoiGeometry.STRIP_HEIGHT_PX]`) populated when
 *   [hasArgbPixels] is true.
 */
class PushBroomSweepBuffers(
    val tileRgbFloats: Array<FloatArray> = Array(StripRoiGeometry.MAX_TILES) {
        FloatArray(StripRoiGeometry.TILE_SIZE_PX * StripRoiGeometry.TILE_SIZE_PX * 3)
    },
    val stripArgbPixels: IntArray = IntArray(
        StripRoiGeometry.MAX_STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX
    )
) {
    /** Active horizontal swath packing mode ([SwathTileMode.FIVE_TILE] or [SwathTileMode.THREE_TILE]) for the current sweep. */
    var activeTileMode: SwathTileMode = SwathTileMode.FIVE_TILE
    /** Number of active tiles (`[activeTileMode].tileCount`) populated in [tileRgbFloats] for the current sweep. */
    val activeTileCount: Int get() = activeTileMode.tileCount
    /** Width in pixels (`[activeTileMode].stripWidthPx`) of the active strip for the current sweep. */
    val activeStripWidthPx: Int get() = activeTileMode.stripWidthPx

    /** Strip ROI geometry in the upright landscape frame for the current sweep. */
    var geometry: StripRoiGeometry = StripRoiGeometry()
    /** Whether [stripArgbPixels] was populated during the most recent [PushBroomCropper.extractTiles] call. */
    var hasArgbPixels: Boolean = false
    /** Sensor timestamp in nanoseconds of the most recent extracted frame. */
    var timestampNs: Long = 0L
}

/**
 * Metadata describing an upright full-frame `1920x1080` luminance (`Y`) copy and clamped
 * `320x320` foveated RGB `[0.0, 1.0]` crop extracted for Stage 2 (`LOCK_ON_NEURAL`) and
 * Stage 3 (`PINPOINT_CONSTELLATION`).
 *
 * @property uprightWidthPx Upright landscape frame width in pixels (`1920`).
 * @property uprightHeightPx Upright landscape frame height in pixels (`1080`).
 * @property cropLeftPx Clamped left X coordinate of the square neural crop in `[0..uprightWidthPx - cropSizePx]`.
 * @property cropTopPx Clamped top Y coordinate of the square neural crop in `[0..uprightHeightPx - cropSizePx]`.
 * @property cropSizePx Edge length in pixels of the square neural crop ([PushBroomCropper.LOCK_ON_CROP_SIZE_PX]).
 * @property timestampNs Frame capture timestamp in nanoseconds.
 */
data class LockOnFrameExtraction(
    val uprightWidthPx: Int,
    val uprightHeightPx: Int,
    val cropLeftPx: Int,
    val cropTopPx: Int,
    val cropSizePx: Int = PushBroomCropper.LOCK_ON_CROP_SIZE_PX,
    val timestampNs: Long
)

/**
 * Extracts only the pitch-stabilized push-broom strip ROI from an upright landscape `YUV_420_888`
 * frame into horizontally overlapping square float tiles without allocating full-frame bitmaps,
 * and stitches per-tile predictions back into unified strip coordinates
 * (`.agents/rules/android-camera-ar.md §2`).
 *
 * Responsibility and state: owns three reusable float planar strip caches ([stripNormalizedR],
 * [stripNormalizedG], [stripNormalizedB]) sized for the maximum
 * `[StripRoiGeometry.MAX_STRIP_WIDTH_PX] x [StripRoiGeometry.STRIP_HEIGHT_PX]` swath so the four
 * [StripRoiGeometry.TILE_OVERLAP_PX] overlap seams are decoded from YUV only once per sweep.
 *
 * Threading: [extractTiles] is not thread-safe because it mutates the internal planar strip caches;
 * call it from a single sequential analysis worker. [computeStripGeometry] and
 * [mergeAndFilterTileDetections] are pure and thread-safe.
 *
 * @param applyImageNetNormalization When false (default), outputs raw `[0.0, 1.0]` RGB floats
 *   matching LiteRT graphs that fuse `(x - mean) / std` inside the model
 *   (`.agents/rules/litert-inference.md §2`). Set true only for legacy models without fused
 *   normalization.
 */
class PushBroomCropper(
    private val applyImageNetNormalization: Boolean = false
) {
    // Reusable intermediate strip RGB cache allocated once for max swath so overlap seams
    // are decoded from YUV_420_888 only once per sweep.
    private val stripNormalizedR = FloatArray(StripRoiGeometry.MAX_STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX)
    private val stripNormalizedG = FloatArray(StripRoiGeometry.MAX_STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX)
    private val stripNormalizedB = FloatArray(StripRoiGeometry.MAX_STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX)

    /**
     * Resolves the upright landscape dimensions and clamped strip ROI bounds for a camera frame.
     *
     * @param rawWidth Sensor buffer width in pixels before rotation.
     * @param rawHeight Sensor buffer height in pixels before rotation.
     * @param rotationDegrees Clockwise rotation (`0`, `90`, `180`, or `270` degrees) to upright landscape.
     * @param cropY Pitch-stabilized top Y coordinate in canonical
     *   `[StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX] x [StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX]`
     *   landscape pixels (`[0..MAX_CROP_Y_PX]`).
     * @param tileMode Active horizontal swath packing mode ([SwathTileMode.FIVE_TILE] or [SwathTileMode.THREE_TILE]).
     * @param isPitchClamped Whether IMU pitch stabilization saturated at the top or bottom frame edge.
     * @return Clamped [StripRoiGeometry] in upright landscape frame pixels.
     */
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
     * Decodes only the active push-broom strip at [cropY] from [frame] and slices it into
     * overlapping square NHWC RGB float tiles inside [outBuffers].
     *
     * Side effects: overwrites `outBuffers.tileRgbFloats[0 until tileMode.tileCount]`, metadata
     * fields on [outBuffers], and (when [populateArgbStrip] is true) `outBuffers.stripArgbPixels`.
     * Mutates internal planar arrays [stripNormalizedR], [stripNormalizedG], and [stripNormalizedB].
     *
     * @param frame Source `YUV_420_888` camera frame.
     * @param outBuffers Pre-allocated destination buffers mutated in-place.
     * @param cropY Pitch-stabilized top Y coordinate in canonical landscape pixels (`[0..MAX_CROP_Y_PX]`).
     * @param tileMode Active horizontal swath packing mode ([SwathTileMode.FIVE_TILE] or [SwathTileMode.THREE_TILE]).
     * @param isPitchClamped Whether IMU pitch stabilization saturated at the frame limits.
     * @param populateArgbStrip True when [com.gneissfind.app.model.OperatingMode.LOG_DATA] needs
     *   packed `0xAARRGGBB` pixels for PNG export.
     * @return Resolved [StripRoiGeometry] for the extracted strip.
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

        val yBuf = frame.yBuffer.duplicate()
        val uBuf = frame.uBuffer.duplicate()
        val vBuf = frame.vBuffer.duplicate()
        val yRowStride = frame.yRowStride
        val yPixStride = frame.yPixelStride
        val uRowStride = frame.uRowStride
        val uPixStride = frame.uPixelStride
        val vRowStride = frame.vRowStride
        val vPixStride = frame.vPixelStride

        val yLimit = yBuf.limit()
        val uLimit = uBuf.limit()
        val vLimit = vBuf.limit()
        if (yLimit <= 0 || uLimit <= 0 || vLimit <= 0 || rawW <= 0 || rawH <= 0) {
            return geometry
        }

        val uprightW = geometry.uprightFrameWidth
        val uprightH = geometry.uprightFrameHeight
        val activeStripW = tileMode.stripWidthPx

        // Safe coordinate fallback if an emulator webcam provides a frame smaller than the strip
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

        // 1. Decode only the active strip rows from YUV_420_888 into the planar float cache.
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

                val yIndex = safeLumaByteIndex(bx, by, yRowStride, yPixStride, yLimit)
                val uvX = bx shr 1
                val uvY = by shr 1
                val uIndex = safeChromaByteIndex(uvX, uvY, uRowStride, uPixStride, uLimit)
                val vIndex = safeChromaByteIndex(uvX, uvY, vRowStride, vPixStride, vLimit)

                val yVal = (yBuf.get(yIndex).toInt() and 0xFF)
                val uVal = (uBuf.get(uIndex).toInt() and 0xFF) - 128
                val vVal = (vBuf.get(vIndex).toInt() and 0xFF) - 128

                // ITU-R BT.601 full-swing integer fixed-point YUV -> RGB (scaled by 2^10 = 1024)
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

        // 2. Slice the decoded strip into overlapping square tiles with TILE_OVERLAP_PX seam overlap.
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
     * Stitches tile-local `[0..TILE_SIZE_PX]` bounding boxes into unified push-broom strip
     * coordinates and full landscape frame X coordinates, fuses same-class fragments across tile
     * seams, and filters out muted confusers, disabled classes, and targets below [minDiameterMm]
     * (`.agents/rules/android-camera-ar.md §2`).
     *
     * Two same-class detections are fused into their union bounding box (`minLeft`, `minTop`,
     * `maxRight`, `maxBottom`, `confidence = max(scoreA, scoreB)`) when any of the following hold:
     * 1. `IoU(boxA, boxB) >= nmsIouThreshold`, OR
     * 2. `IoM(boxA, boxB) >= nmsIomThreshold`, OR
     * 3. Both boxes originate from different tiles, touch or straddle the same
     *    `[xOffset .. xOffset + TILE_OVERLAP_PX]` seam band within [SEAM_STRADDLE_MARGIN_PX], and
     *    have vertical projection overlap `>= MIN_VERTICAL_OVERLAP_RATIO`.
     *
     * @param tileDetections Raw per-tile bounding-box detections in `[0.0, TILE_SIZE_PX]` tile pixels.
     * @param tileMode Active horizontal swath packing mode ([SwathTileMode.FIVE_TILE] or [SwathTileMode.THREE_TILE]).
     * @param gsdMmPerPx Cross-track ground sample distance in `mm/px` used to compute physical size.
     * @param nmsIouThreshold Minimum Intersection-over-Union (`[0.0, 1.0]`) to fuse same-class boxes.
     * @param nmsIomThreshold Minimum Intersection-over-Minimum (`[0.0, 1.0]`) to fuse same-class boxes.
     * @param minConfidence Minimum confidence score (`[0.0, 1.0]`) required to retain a tile box.
     * @param minDiameterMm Minimum physical major-axis diameter (`mm`) required to alert the user.
     * @param enabledClasses Set of user-enabled [StoneClass] targets from the Prospect Matrix.
     * @return Filtered, seam-merged [StripDetection] list sorted by descending confidence. Empty
     *   when no detections pass all gates.
     */
    fun mergeAndFilterTileDetections(
        tileDetections: List<TileDetection>,
        tileMode: SwathTileMode = SwathTileMode.FIVE_TILE,
        gsdMmPerPx: Float = DEFAULT_GSD_MM_PER_PX,
        nmsIouThreshold: Float = DEFAULT_NMS_IOU_THRESHOLD,
        nmsIomThreshold: Float = DEFAULT_NMS_IOM_THRESHOLD,
        minConfidence: Float = DEFAULT_MIN_CONFIDENCE,
        minDiameterMm: Float = MIN_STONE_DIAMETER_MM,
        enabledClasses: Set<StoneClass> = StoneClass.alertableClasses.toSet()
    ): List<StripDetection> {
        if (tileDetections.isEmpty()) return emptyList()

        val stripWidthF = tileMode.stripWidthPx.toFloat()
        val stripOffsets = tileMode.stripTileOffsets
        val frameOffsets = tileMode.frameXOffsets

        // 1. Project tile-local [0..TILE_SIZE_PX] boxes into unified strip and full-frame coordinates.
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

        // 2. Sort descending by confidence and iteratively fuse overlapping or seam-straddling
        //    same-class detections into unified bounding boxes until convergence.
        mappedCandidates.sortByDescending { it.confidence }

        val mergedList = ArrayList<StripDetection>(mappedCandidates.size)
        for (candidate in mappedCandidates) {
            var fused = false
            for (i in mergedList.indices) {
                val kept = mergedList[i]
                if (shouldFuseDetections(kept, candidate, tileMode, nmsIouThreshold, nmsIomThreshold)) {
                    mergedList[i] = unionDetections(kept, candidate, gsdMmPerPx)
                    fused = true
                    break
                }
            }
            if (!fused) {
                mergedList.add(candidate)
            }
        }

        // Fixed-point pass: coalescing two fragments across a seam can expand a box to bridge a
        // third adjacent fragment on wide (> TILE_STRIDE_PX) objects.
        var changed = true
        while (changed && mergedList.size > 1) {
            changed = false
            outer@ for (i in 0 until mergedList.size) {
                for (j in i + 1 until mergedList.size) {
                    val a = mergedList[i]
                    val b = mergedList[j]
                    if (shouldFuseDetections(a, b, tileMode, nmsIouThreshold, nmsIomThreshold)) {
                        mergedList[i] = unionDetections(a, b, gsdMmPerPx)
                        mergedList.removeAt(j)
                        changed = true
                        break@outer
                    }
                }
            }
        }

        mergedList.sortByDescending { it.confidence }

        // 3. Mute HARD_CONFUSER, disabled Prospect Matrix classes, and pebbles below minDiameterMm.
        return mergedList.filter { det ->
            det.stoneClass.isUserAlertable &&
                det.stoneClass in enabledClasses &&
                det.estimatedDiameterMm >= minDiameterMm
        }
    }

    /**
     * Decides whether two detections of the same [StoneClass] represent the same physical object
     * via `IoU`, `IoM`, or cross-tile [StripRoiGeometry.TILE_OVERLAP_PX] seam straddling
     * (`.agents/rules/android-camera-ar.md §2`).
     *
     * @param a First detection in unified strip coordinates.
     * @param b Second detection in unified strip coordinates.
     * @param tileMode Active swath layout defining seam X offsets.
     * @param nmsIouThreshold Minimum `IoU` in `[0.0, 1.0]` to fuse.
     * @param nmsIomThreshold Minimum `IoM` in `[0.0, 1.0]` to fuse.
     * @return True when [a] and [b] should be merged into a single union bounding box.
     */
    private fun shouldFuseDetections(
        a: StripDetection,
        b: StripDetection,
        tileMode: SwathTileMode,
        nmsIouThreshold: Float,
        nmsIomThreshold: Float
    ): Boolean {
        if (a.stoneClass != b.stoneClass) return false
        if (a.iouWith(b) >= nmsIouThreshold) return true
        if (a.iomWith(b) >= nmsIomThreshold) return true

        // Elongated objects (toothbrush, remote, petrified wood limb) spanning two tiles often have
        // low 2D IoU/IoM because each tile only sees its half of the object, but both fragments
        // terminate inside or straddle the shared TILE_OVERLAP_PX seam band with matching vertical bounds.
        if (a.sourceTileIndex != b.sourceTileIndex &&
            a.verticalOverlapRatioWith(b) >= MIN_VERTICAL_OVERLAP_RATIO
        ) {
            val stripOffsets = tileMode.stripTileOffsets
            for (seamIdx in 1 until tileMode.tileCount) {
                val seamLeft = stripOffsets[seamIdx].toFloat()
                val seamRight = seamLeft + StripRoiGeometry.TILE_OVERLAP_PX.toFloat()
                val bandLeft = seamLeft - SEAM_STRADDLE_MARGIN_PX
                val bandRight = seamRight + SEAM_STRADDLE_MARGIN_PX
                val aTouchesSeam = a.rightPx >= bandLeft && a.leftPx <= bandRight
                val bTouchesSeam = b.rightPx >= bandLeft && b.leftPx <= bandRight
                if (aTouchesSeam && bTouchesSeam) {
                    return true
                }
            }
        }
        return false
    }

    /**
     * Combines two overlapping or seam-adjacent [StripDetection] boxes into a single union box
     * preserving the higher confidence score and recomputing physical diameter at [gsdMmPerPx].
     *
     * @param a First detection fragment in unified strip coordinates.
     * @param b Second detection fragment in unified strip coordinates.
     * @param gsdMmPerPx Cross-track ground sample distance in `mm/px`.
     * @return Union [StripDetection] spanning both [a] and [b].
     */
    private fun unionDetections(
        a: StripDetection,
        b: StripDetection,
        gsdMmPerPx: Float
    ): StripDetection {
        val dominant = if (a.confidence >= b.confidence) a else b
        val unionLeft = min(a.leftPx, b.leftPx)
        val unionTop = min(a.topPx, b.topPx)
        val unionRight = max(a.rightPx, b.rightPx)
        val unionBottom = max(a.bottomPx, b.bottomPx)
        val unionFrameLeft = min(a.frameLeftPx, b.frameLeftPx)
        val unionFrameRight = max(a.frameRightPx, b.frameRightPx)
        val unionWidth = (unionRight - unionLeft).coerceAtLeast(0f)
        val unionHeight = (unionBottom - unionTop).coerceAtLeast(0f)

        return StripDetection(
            stoneClass = dominant.stoneClass,
            confidence = max(a.confidence, b.confidence),
            leftPx = unionLeft,
            topPx = unionTop,
            rightPx = unionRight,
            bottomPx = unionBottom,
            estimatedDiameterMm = max(unionWidth, unionHeight) * gsdMmPerPx,
            sourceTileIndex = dominant.sourceTileIndex,
            frameLeftPx = unionFrameLeft,
            frameRightPx = unionFrameRight
        )
    }

    /**
     * Copies the full upright landscape luminance (`Y`) plane into [outLumaBytes] for OpenCV
     * KLT / `AKAZE` tracking and optionally decodes a clamped `320x320` NHWC RGB `[0.0, 1.0]`
     * crop centered on `(cropCenterXPx, cropCenterYPx)` into [outCrop320RgbFloats] in `< 2 ms`
     * so `ImageProxy` can be closed immediately without Gralloc starvation.
     *
     * @param frame Source `YUV_420_888` camera frame.
     * @param outLumaBytes Pre-allocated destination array (`>= uprightW * uprightH` bytes).
     * @param outCrop320RgbFloats Pre-allocated destination array (`>= 320 * 320 * 3` floats).
     * @param cropCenterXPx Desired horizontal center of the square crop in upright frame pixels.
     * @param cropCenterYPx Desired vertical center of the square crop in upright frame pixels.
     * @param sourceCropSizePx Edge length in upright frame pixels of the square ROI sampled into the
     *   `320x320` model tensor (`320` for 1:1 tracking; up to `640` for wide-angle re-acquisition or close-up stones).
     * @param extractRgbCrop Whether to decode the `320x320` RGB float patch (`false` on pure `AKAZE` frames).
     * @return [LockOnFrameExtraction] describing the upright dimensions and clamped crop origin/size.
     */
    fun extractUprightLumaAndLockOnCrop320(
        frame: Yuv420Frame,
        outLumaBytes: ByteArray,
        outCrop320RgbFloats: FloatArray,
        cropCenterXPx: Float,
        cropCenterYPx: Float,
        sourceCropSizePx: Int = LOCK_ON_CROP_SIZE_PX,
        extractRgbCrop: Boolean = true
    ): LockOnFrameExtraction {
        val rawW = frame.width
        val rawH = frame.height
        val rot = ((frame.rotationDegrees % 360) + 360) % 360
        val uprightW = if (rot == 90 || rot == 270) rawH else rawW
        val uprightH = if (rot == 90 || rot == 270) rawW else rawH

        val yDup = frame.yBuffer.duplicate()
        val yRowStride = frame.yRowStride
        val yPixStride = frame.yPixelStride
        val yLimit = yDup.limit()
        val neededLuma = uprightW * uprightH

        if (outLumaBytes.size >= neededLuma) {
            if (rot == 0 && yPixStride == 1 && rawW == uprightW && rawH == uprightH) {
                for (uy in 0 until uprightH) {
                    val srcPos = uy * yRowStride
                    if (srcPos + uprightW <= yLimit) {
                        yDup.position(srcPos)
                        yDup.get(outLumaBytes, uy * uprightW, uprightW)
                    }
                }
            } else {
                var dstIdx = 0
                for (uy in 0 until uprightH) {
                    for (ux in 0 until uprightW) {
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
                        val yIndex = safeLumaByteIndex(bx, by, yRowStride, yPixStride, yLimit)
                        outLumaBytes[dstIdx++] = yDup.get(yIndex)
                    }
                }
            }
        }

        val tensorSize = LOCK_ON_CROP_SIZE_PX
        val maxSquare = min(uprightW, uprightH).coerceAtLeast(1)
        val roiSizePx = sourceCropSizePx.coerceIn(
            min(tensorSize, maxSquare),
            min(LOCK_ON_WIDE_SEARCH_CROP_SIZE_PX, maxSquare)
        )
        val (cropLeft, cropTop) = computeClampedSquareCropOrigin(
            centerXPx = cropCenterXPx,
            centerYPx = cropCenterYPx,
            cropSizePx = roiSizePx,
            frameWidthPx = uprightW,
            frameHeightPx = uprightH
        )

        if (extractRgbCrop &&
            yLimit > 0 &&
            rawW > 0 &&
            rawH > 0 &&
            outCrop320RgbFloats.size >= tensorSize * tensorSize * 3
        ) {
            val uBuf = frame.uBuffer.duplicate()
            val vBuf = frame.vBuffer.duplicate()
            val uRowStride = frame.uRowStride
            val uPixStride = frame.uPixelStride
            val vRowStride = frame.vRowStride
            val vPixStride = frame.vPixelStride
            val uLimit = uBuf.limit()
            val vLimit = vBuf.limit()
            if (uLimit > 0 && vLimit > 0) {
                var dstFloatIdx = 0
                for (cy in 0 until tensorSize) {
                    val srcOffsetY = if (roiSizePx == tensorSize) {
                        cy
                    } else {
                        (cy * roiSizePx + (roiSizePx shr 1)) / tensorSize
                    }
                    val uy = (cropTop + srcOffsetY).coerceIn(0, (uprightH - 1).coerceAtLeast(0))
                    for (cx in 0 until tensorSize) {
                        val srcOffsetX = if (roiSizePx == tensorSize) {
                            cx
                        } else {
                            (cx * roiSizePx + (roiSizePx shr 1)) / tensorSize
                        }
                        val ux = (cropLeft + srcOffsetX).coerceIn(0, (uprightW - 1).coerceAtLeast(0))
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

                        val yIndex = safeLumaByteIndex(bx, by, yRowStride, yPixStride, yLimit)
                        val uvX = bx shr 1
                        val uvY = by shr 1
                        val uIndex = safeChromaByteIndex(uvX, uvY, uRowStride, uPixStride, uLimit)
                        val vIndex = safeChromaByteIndex(uvX, uvY, vRowStride, vPixStride, vLimit)

                        val yVal = yDup.get(yIndex).toInt() and 0xFF
                        val uVal = (uBuf.get(uIndex).toInt() and 0xFF) - 128
                        val vVal = (vBuf.get(vIndex).toInt() and 0xFF) - 128

                        val r8 = (yVal + ((1436 * vVal) shr 10)).coerceIn(0, 255)
                        val g8 = (yVal - ((352 * uVal + 731 * vVal) shr 10)).coerceIn(0, 255)
                        val b8 = (yVal + ((1815 * uVal) shr 10)).coerceIn(0, 255)

                        if (applyImageNetNormalization) {
                            outCrop320RgbFloats[dstFloatIdx++] = (r8 * INV_255 - IMAGENET_MEAN_R) * IMAGENET_INV_STD_R
                            outCrop320RgbFloats[dstFloatIdx++] = (g8 * INV_255 - IMAGENET_MEAN_G) * IMAGENET_INV_STD_G
                            outCrop320RgbFloats[dstFloatIdx++] = (b8 * INV_255 - IMAGENET_MEAN_B) * IMAGENET_INV_STD_B
                        } else {
                            outCrop320RgbFloats[dstFloatIdx++] = r8 * INV_255
                            outCrop320RgbFloats[dstFloatIdx++] = g8 * INV_255
                            outCrop320RgbFloats[dstFloatIdx++] = b8 * INV_255
                        }
                    }
                }
            }
        }

        return LockOnFrameExtraction(
            uprightWidthPx = uprightW,
            uprightHeightPx = uprightH,
            cropLeftPx = cropLeft,
            cropTopPx = cropTop,
            cropSizePx = roiSizePx,
            timestampNs = frame.timestampNs
        )
    }

    companion object {
        /** Native square tensor resolution in pixels for `rf_detr_nano_320_b1_w8dyn` Lock-On tracking. */
        const val LOCK_ON_CROP_SIZE_PX = 320

        /** Expanded `2.0x` (`640x640` -> `320x320`) wide-angle ROI used for close-up stones & re-acquisition. */
        const val LOCK_ON_WIDE_SEARCH_CROP_SIZE_PX = 640

        /** Default cross-tile Intersection-over-Union threshold (`.agents/rules/android-camera-ar.md §2`). */
        const val DEFAULT_NMS_IOU_THRESHOLD = 0.45f

        /** Default cross-tile Intersection-over-Minimum threshold (`.agents/rules/android-camera-ar.md §2`). */
        const val DEFAULT_NMS_IOM_THRESHOLD = 0.65f

        /**
         * Horizontal tolerance in strip pixels outside a `[seamLeft..seamRight]` overlap band
         * within which a tile-clipped box edge still counts as touching the seam (`docs/ENGINEERING_LEDGER.md §4`).
         */
        const val SEAM_STRADDLE_MARGIN_PX = 8.0f

        /**
         * Minimum 1D vertical overlap ratio (`interH / min(heightA, heightB)`) required to fuse
         * two adjacent-tile boxes that straddle a seam (`docs/ENGINEERING_LEDGER.md §4`).
         */
        const val MIN_VERTICAL_OVERLAP_RATIO = 0.60f

        /**
         * Default minimum model confidence score in `[0.0, 1.0]` required to retain a tile box.
         * RATIONALE-UNKNOWN: how was 0.35 chosen relative to the COCO metadata default of 0.35?
         */
        const val DEFAULT_MIN_CONFIDENCE = 0.35f

        /** Minimum physical stone diameter in millimeters required to alert the user (`AGENTS.md §2`). */
        const val MIN_STONE_DIAMETER_MM = 20.0f

        /** Nominal waist-height slant range in meters at a `45 deg` downward look angle (`AGENTS.md §2`). */
        const val DEFAULT_SLANT_RANGE_METERS = 1.50f

        /** Default lossless sensor center-crop zoom ratio (`AGENTS.md §2`, `.agents/rules/android-camera-ar.md §1`). */
        const val DEFAULT_ZOOM_RATIO = 2.0f

        /** Nominal ground sample distance in `mm/px` at [DEFAULT_SLANT_RANGE_METERS] and [DEFAULT_ZOOM_RATIO] (`AGENTS.md §2`). */
        const val DEFAULT_GSD_MM_PER_PX = 0.59f

        private const val INV_255 = 1.0f / 255.0f
        /** Standard ImageNet red channel mean (`.agents/rules/litert-inference.md §2`). */
        private const val IMAGENET_MEAN_R = 0.485f
        /** Standard ImageNet green channel mean (`.agents/rules/litert-inference.md §2`). */
        private const val IMAGENET_MEAN_G = 0.456f
        /** Standard ImageNet blue channel mean (`.agents/rules/litert-inference.md §2`). */
        private const val IMAGENET_MEAN_B = 0.406f
        /** Reciprocal of standard ImageNet red channel standard deviation (`1 / 0.229`). */
        private const val IMAGENET_INV_STD_R = 1.0f / 0.229f
        /** Reciprocal of standard ImageNet green channel standard deviation (`1 / 0.224`). */
        private const val IMAGENET_INV_STD_G = 1.0f / 0.224f
        /** Reciprocal of standard ImageNet blue channel standard deviation (`1 / 0.225`). */
        private const val IMAGENET_INV_STD_B = 1.0f / 0.225f

        /**
         * Computes the clamped `(leftPx, topPx)` origin of a square [cropSizePx] ROI centered on
         * `(centerXPx, centerYPx)` within `[0..frameWidthPx] x [0..frameHeightPx]`.
         *
         * @param centerXPx Desired crop center X in upright frame pixels.
         * @param centerYPx Desired crop center Y in upright frame pixels.
         * @param cropSizePx Square crop edge length in pixels ([LOCK_ON_CROP_SIZE_PX]).
         * @param frameWidthPx Upright landscape frame width in pixels.
         * @param frameHeightPx Upright landscape frame height in pixels.
         * @return Pair `(cropLeftPx, cropTopPx)` guaranteed within frame bounds.
         */
        fun computeClampedSquareCropOrigin(
            centerXPx: Float,
            centerYPx: Float,
            cropSizePx: Int = LOCK_ON_CROP_SIZE_PX,
            frameWidthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
            frameHeightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX
        ): Pair<Int, Int> {
            val safeW = frameWidthPx.coerceAtLeast(1)
            val safeH = frameHeightPx.coerceAtLeast(1)
            val safeSize = cropSizePx.coerceIn(1, min(safeW, safeH))
            val maxLeft = (safeW - safeSize).coerceAtLeast(0)
            val maxTop = (safeH - safeSize).coerceAtLeast(0)
            val finiteCx = if (centerXPx.isFinite()) {
                centerXPx.coerceIn(-100_000f, 100_000f)
            } else {
                safeW * 0.5f
            }
            val finiteCy = if (centerYPx.isFinite()) {
                centerYPx.coerceIn(-100_000f, 100_000f)
            } else {
                safeH * 0.5f
            }
            val rawLeft = (finiteCx - safeSize * 0.5f).roundToInt()
            val rawTop = (finiteCy - safeSize * 0.5f).roundToInt()
            return Pair(rawLeft.coerceIn(0, maxLeft), rawTop.coerceIn(0, maxTop))
        }

        /**
         * Resolves a guaranteed in-bounds `Y` luminance byte offset in `[0, bufferLimit - 1]`.
         */
        internal fun safeLumaByteIndex(
            bx: Int,
            by: Int,
            rowStride: Int,
            pixelStride: Int,
            bufferLimit: Int
        ): Int {
            if (bufferLimit <= 1) return 0
            val safeStride = pixelStride.coerceAtLeast(1)
            return (by * rowStride + bx * safeStride).coerceIn(0, bufferLimit - 1)
        }

        /**
         * Resolves a guaranteed in-bounds `U`/`V` chroma byte offset in `[0, bufferLimit - 1]`
         * while preserving semi-planar `NV12`/`NV21` (`pixelStride == 2`) channel parity on the
         * bottom-right corner (`x = 1918..1919, y = 1078..1079`) where Android's `YUV_420_888`
         * chroma `ByteBuffer.limit()` omits the trailing interleaved byte (`.agents/rules/android-camera-ar.md §1.1`).
         */
        internal fun safeChromaByteIndex(
            uvX: Int,
            uvY: Int,
            rowStride: Int,
            pixelStride: Int,
            bufferLimit: Int
        ): Int {
            if (bufferLimit <= 1) return 0
            val maxValid = bufferLimit - 1
            val safeStride = pixelStride.coerceAtLeast(1)
            var rawIdx = uvY * rowStride + uvX * safeStride
            if (rawIdx > maxValid) {
                val deficitSteps = ((rawIdx - maxValid + safeStride - 1) / safeStride).coerceAtLeast(1)
                rawIdx -= deficitSteps * safeStride
            }
            return rawIdx.coerceIn(0, maxValid)
        }

        /**
         * Computes cross-track Ground Sample Distance (`mm/px`) from slant range [slantRangeMeters]
         * and active zoom ([DEFAULT_ZOOM_RATIO] -> [DEFAULT_GSD_MM_PER_PX] at [DEFAULT_SLANT_RANGE_METERS]).
         *
         * @param slantRangeMeters Slant distance from camera optical center to ground patch in meters
         *   (clamped internally to `[0.18, 4.00]` m to support close-up terminal homing).
         * @param zoomRatio Active digital/sensor crop zoom ratio (clamped internally to `>= 1.0f`).
         * @return Ground sample distance in millimeters per pixel.
         */
        fun computeGsdMmPerPx(
            slantRangeMeters: Float,
            zoomRatio: Float = DEFAULT_ZOOM_RATIO
        ): Float {
            val clampedRange = slantRangeMeters.coerceIn(0.18f, 4.00f)
            val safeZoom = zoomRatio.coerceAtLeast(1.0f)
            return DEFAULT_GSD_MM_PER_PX *
                (clampedRange / DEFAULT_SLANT_RANGE_METERS) *
                (DEFAULT_ZOOM_RATIO / safeZoom)
        }
    }
}
