/*
 * Core domain models for target taxonomy, push-broom strip ROI geometry, per-tile and unified
 * strip bounding-box detections, 3D world-space track points and unit bearing rays, and operating modes.
 *
 * Produced by PushBroomCropper, StoneDetector, GroundPlaneRaycaster, and TemporalTargetTracker;
 * consumed by ScannerViewModel, MetalDetectorSonifier, FieldDataLogger, and ScannerScreen.
 */
package com.gneissfind.app.model

import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Target taxonomy classifying geological targets and hard visual confusers, paired with high-contrast
 * reticle colors designed to remain legible over wet river gravel (AGENTS.md §2).
 *
 * Immutable enum; safe to read from any thread.
 *
 * @property id Zero-based class index matching model output and metadata mappings.
 * @property codePrefix Two-digit LCARS numeric prefix rendered on reticles and Prospect Matrix buttons.
 * @property wireName Canonical `snake_case` identifier written to JSONL field logs and metadata files.
 * @property displayName Human-readable mineral name for telemetry and accessibility.
 * @property lcarsShortName Uppercase console label rendered inside the LCARS Prospect Matrix.
 * @property primaryHexColor ARGB hex color (`0xAARRGGBB`) for the reticle brackets and badge.
 * @property accentHexColor Secondary ARGB hex color (`0xAARRGGBB`) for reticle highlights.
 * @property isUserAlertable Whether detections of this class may trigger visual reticles or audio pings
 *   in [OperatingMode.SCAN]. False for [HARD_CONFUSER] so confusers are silently suppressed.
 */
enum class StoneClass(
    val id: Int,
    val codePrefix: String,
    val wireName: String,
    val displayName: String,
    val lcarsShortName: String,
    val primaryHexColor: Long,
    val accentHexColor: Long,
    val isUserAlertable: Boolean
) {
    /** Translucent chalcedony, carnelian, and banded agate (`AGENTS.md §2`). */
    CHALCEDONY_AGATE(
        id = 0,
        codePrefix = "01",
        wireName = "chalcedony_agate",
        displayName = "Agate / Carnelian",
        lcarsShortName = "AGATE / CARNELIAN",
        primaryHexColor = 0xFF00FFFFL, // Electric Cyan (AGENTS.md §2)
        accentHexColor = 0xFFFFFFFFL,
        isUserAlertable = true
    ),

    /** Iron-rich red, yellow, and orbicular jasper (`AGENTS.md §2`). */
    JASPER_RED_YELLOW(
        id = 1,
        codePrefix = "02",
        wireName = "jasper_red_yellow",
        displayName = "Red / Yellow Jasper",
        lcarsShortName = "RED / YEL JASPER",
        primaryHexColor = 0xFFFFFF00L, // High-Vis Solar Yellow (AGENTS.md §2)
        accentHexColor = 0xFFFFFFFFL,
        isUserAlertable = true
    ),

    /** Dallasite breccia, nephrite jade, and ribbon chert (`AGENTS.md §2`). */
    GREEN_GEM_ROCK(
        id = 2,
        codePrefix = "03",
        wireName = "green_gem_rock",
        displayName = "Dallasite / Nephrite",
        lcarsShortName = "DALLASITE / JADE",
        primaryHexColor = 0xFF00FF88L, // Spring Mint (AGENTS.md §2)
        accentHexColor = 0xFFFFFFFFL,
        isUserAlertable = true
    ),

    /** Silicified wood and limb casts (`AGENTS.md §2`). */
    PETRIFIED_WOOD(
        id = 3,
        codePrefix = "04",
        wireName = "petrified_wood",
        displayName = "Petrified Wood",
        lcarsShortName = "PETRIFIED WOOD",
        primaryHexColor = 0xFFFFAA00L, // Bright Amber-Gold (AGENTS.md §2)
        accentHexColor = 0xFFFFFFFFL,
        isUserAlertable = true
    ),

    /**
     * Explicit negative class for wet dark basalt, plain grey granite/quartzite, wet leaves,
     * specular water glare, and unmapped COCO stand-in classes (`AGENTS.md §2`). Muted in UI and audio.
     */
    HARD_CONFUSER(
        id = 4,
        codePrefix = "05",
        wireName = "hard_confuser",
        displayName = "Hard Confuser (Muted)",
        lcarsShortName = "CONFUSER (MUTED)",
        primaryHexColor = 0xFF777777L,
        accentHexColor = 0xFF555555L,
        isUserAlertable = false
    );

    companion object {
        /**
         * Looks up a [StoneClass] by its zero-based integer [id].
         *
         * @param id Target class index (`0..4`).
         * @return Matching [StoneClass], or `null` if [id] is outside `[0..4]`.
         */
        fun fromId(id: Int): StoneClass? = entries.firstOrNull { it.id == id }

        /** Subset of classes (`0..3`) that can produce user-visible reticles and audio pings. */
        val alertableClasses: List<StoneClass> = entries.filter { it.isUserAlertable }
    }
}

/**
 * Horizontal tile-packing layout across the upright [StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX] by
 * [StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX] landscape camera frame (`.agents/rules/android-camera-ar.md §2`).
 *
 * Immutable enum; safe to read from any thread (`[frameXOffsets]` and `[stripTileOffsets]` must not
 * be mutated by callers).
 *
 * @property tileCount Number of horizontally overlapping square tiles per sweep.
 * @property stripWidthPx Total horizontal span of the unified push-broom strip in pixels.
 * @property canonicalStripLeftPx Left X offset (px) of the strip inside a
 *   [StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX]-wide landscape frame.
 * @property frameXOffsets Left X offset (px) of each tile in full landscape frame coordinates.
 * @property stripTileOffsets Left X offset (px) of each tile relative to [canonicalStripLeftPx].
 * @property lcarsLabel Uppercase telemetry badge describing the active swath width on the `SW` pill.
 */
enum class SwathTileMode(
    val tileCount: Int,
    val stripWidthPx: Int,
    val canonicalStripLeftPx: Int,
    val frameXOffsets: IntArray,
    val stripTileOffsets: IntArray,
    val lcarsLabel: String
) {
    /** Full 5-tile horizontal swath (`1792 x 384 px`, `32 px` seam overlaps, `.agents/rules/android-camera-ar.md §2`). */
    FIVE_TILE(
        tileCount = 5,
        stripWidthPx = 1792,
        canonicalStripLeftPx = 64,
        frameXOffsets = intArrayOf(64, 416, 768, 1120, 1472),
        stripTileOffsets = intArrayOf(0, 352, 704, 1056, 1408),
        lcarsLabel = "5-TILE 1792PX"
    ),

    /** Narrow 3-tile center swath (`1088 x 384 px`, `32 px` seam overlaps, `.agents/rules/android-camera-ar.md §2`). */
    THREE_TILE(
        tileCount = 3,
        stripWidthPx = 1088,
        canonicalStripLeftPx = 416,
        frameXOffsets = intArrayOf(416, 768, 1120),
        stripTileOffsets = intArrayOf(0, 352, 704),
        lcarsLabel = "3-TILE 1088PX"
    );

    /**
     * Returns the alternate [SwathTileMode] when the user taps the swath width (`SW`) pill button.
     *
     * @return [THREE_TILE] when called on [FIVE_TILE], or [FIVE_TILE] when called on [THREE_TILE].
     */
    fun toggled(): SwathTileMode = if (this == FIVE_TILE) THREE_TILE else FIVE_TILE
}

/**
 * Pixel bounds of the active pitch-stabilized push-broom strip inside the upright landscape camera
 * frame.
 *
 * Immutable value object produced by [com.gneissfind.app.vision.PushBroomCropper.computeStripGeometry]
 * and consumed by [com.gneissfind.app.tracking.GroundPlaneRaycaster] and
 * [com.gneissfind.app.ui.ScannerScreen].
 *
 * @property uprightFrameWidth Upright landscape frame width in pixels.
 * @property uprightFrameHeight Upright landscape frame height in pixels.
 * @property stripLeftPx Left edge of the strip in upright landscape frame pixels (`0` = left edge).
 * @property stripTopPx Pitch-stabilized top Y coordinate (`cropY`) in upright landscape frame pixels
 *   (`0` = top edge).
 * @property stripWidthPx Width of the active strip in pixels.
 * @property stripHeightPx Height of the active strip in pixels ([STRIP_HEIGHT_PX]).
 * @property tileMode Active [SwathTileMode] ([SwathTileMode.FIVE_TILE] or [SwathTileMode.THREE_TILE]).
 * @property isPitchClamped True when IMU pitch stabilization saturated at `0` or [MAX_CROP_Y_PX].
 */
data class StripRoiGeometry(
    val uprightFrameWidth: Int = LANDSCAPE_FRAME_WIDTH_PX,
    val uprightFrameHeight: Int = LANDSCAPE_FRAME_HEIGHT_PX,
    val stripLeftPx: Int = SwathTileMode.FIVE_TILE.canonicalStripLeftPx,
    val stripTopPx: Int = NOMINAL_CROP_Y_PX,
    val stripWidthPx: Int = SwathTileMode.FIVE_TILE.stripWidthPx,
    val stripHeightPx: Int = STRIP_HEIGHT_PX,
    val tileMode: SwathTileMode = SwathTileMode.FIVE_TILE,
    val isPitchClamped: Boolean = false
) {
    /** Strip left edge as a fraction (`[0.0, 1.0]`) of [uprightFrameWidth]. */
    val normalizedLeft: Float
        get() = stripLeftPx.toFloat() / uprightFrameWidth.coerceAtLeast(1)

    /** Strip top edge (`cropY`) as a fraction (`[0.0, 1.0]`) of [uprightFrameHeight]. */
    val normalizedTop: Float
        get() = stripTopPx.toFloat() / uprightFrameHeight.coerceAtLeast(1)

    /** Strip width as a fraction (`[0.0, 1.0]`) of [uprightFrameWidth]. */
    val normalizedWidth: Float
        get() = stripWidthPx.toFloat() / uprightFrameWidth.coerceAtLeast(1)

    /** Strip height as a fraction (`[0.0, 1.0]`) of [uprightFrameHeight]. */
    val normalizedHeight: Float
        get() = stripHeightPx.toFloat() / uprightFrameHeight.coerceAtLeast(1)

    companion object {
        /** Canonical upright landscape camera stream width in pixels (`.agents/rules/android-camera-ar.md §2`). */
        const val LANDSCAPE_FRAME_WIDTH_PX = 1920

        /** Canonical upright landscape camera stream height in pixels (`.agents/rules/android-camera-ar.md §2`). */
        const val LANDSCAPE_FRAME_HEIGHT_PX = 1080

        /** Maximum horizontal strip width in pixels ([SwathTileMode.FIVE_TILE]). */
        const val MAX_STRIP_WIDTH_PX = 1792

        /** Horizontal strip width in pixels for [SwathTileMode.FIVE_TILE]. */
        const val STRIP_WIDTH_5_TILE_PX = 1792

        /** Horizontal strip width in pixels for [SwathTileMode.THREE_TILE]. */
        const val STRIP_WIDTH_3_TILE_PX = 1088

        /** Vertical height of the push-broom strip in pixels (`.agents/rules/android-camera-ar.md §2`). */
        const val STRIP_HEIGHT_PX = 384

        /** Square input tile edge length in pixels expected by `RF-DETR-Nano` (`.agents/rules/litert-inference.md §2`). */
        const val TILE_SIZE_PX = 384

        /** Horizontal center-to-center step in pixels between adjacent tiles (`TILE_SIZE_PX - TILE_OVERLAP_PX`). */
        const val TILE_STRIDE_PX = 352

        /** Horizontal seam overlap in pixels between adjacent tiles (`.agents/rules/android-camera-ar.md §2`). */
        const val TILE_OVERLAP_PX = 32

        /** Maximum number of horizontal tiles per sweep ([SwathTileMode.FIVE_TILE]). */
        const val MAX_TILES = 5

        /** Centered strip top Y offset in pixels (`(LANDSCAPE_FRAME_HEIGHT_PX - STRIP_HEIGHT_PX) / 2`). */
        const val NOMINAL_CROP_Y_PX = 348

        /** Maximum valid strip top Y offset in pixels before bottom-edge clamping (`1080 - 384`). */
        const val MAX_CROP_Y_PX = LANDSCAPE_FRAME_HEIGHT_PX - STRIP_HEIGHT_PX // 696
    }
}

/**
 * Unmerged bounding-box prediction inside a single square tile before strip-space stitching.
 *
 * Produced by [com.gneissfind.app.inference.StoneDetector.detectSweep] and consumed by
 * [com.gneissfind.app.vision.PushBroomCropper.mergeAndFilterTileDetections].
 *
 * @property tileIndex Index of the source tile (`0 until activeTileCount`).
 * @property classId Target class index corresponding to [StoneClass.id].
 * @property confidence Model probability in `[0.0, 1.0]`.
 * @property leftPx Left edge in tile-local pixels (`[0.0, TILE_SIZE_PX]`).
 * @property topPx Top edge in tile-local pixels (`[0.0, TILE_SIZE_PX]`).
 * @property rightPx Right edge in tile-local pixels (`[0.0, TILE_SIZE_PX]`).
 * @property bottomPx Bottom edge in tile-local pixels (`[0.0, TILE_SIZE_PX]`).
 */
data class TileDetection(
    val tileIndex: Int,
    val classId: Int,
    val confidence: Float,
    val leftPx: Float,
    val topPx: Float,
    val rightPx: Float,
    val bottomPx: Float
)

/**
 * Bounding-box detection mapped into unified push-broom strip coordinates
 * (`[0..stripWidthPx] x [0..STRIP_HEIGHT_PX]`) and full landscape frame X coordinates
 * (`[0..LANDSCAPE_FRAME_WIDTH_PX]`).
 *
 * Produced by [com.gneissfind.app.vision.PushBroomCropper.mergeAndFilterTileDetections] and
 * consumed by [com.gneissfind.app.tracking.GroundPlaneRaycaster.raycastStripDetection].
 *
 * @property stoneClass Resolved [StoneClass] taxonomy entry.
 * @property confidence Detection confidence score in `[0.0, 1.0]`.
 * @property leftPx Left X coordinate in unified strip pixels.
 * @property topPx Top Y coordinate in unified strip pixels (`[0.0, STRIP_HEIGHT_PX]`).
 * @property rightPx Right X coordinate in unified strip pixels.
 * @property bottomPx Bottom Y coordinate in unified strip pixels (`[0.0, STRIP_HEIGHT_PX]`).
 * @property estimatedDiameterMm Estimated physical major-axis diameter in millimeters at the current GSD.
 * @property sourceTileIndex Index of the highest-confidence contributing tile (`0 until activeTileCount`).
 * @property frameLeftPx Left X coordinate in full upright landscape frame pixels.
 * @property frameRightPx Right X coordinate in full upright landscape frame pixels.
 */
data class StripDetection(
    val stoneClass: StoneClass,
    val confidence: Float,
    val leftPx: Float,
    val topPx: Float,
    val rightPx: Float,
    val bottomPx: Float,
    val estimatedDiameterMm: Float,
    val sourceTileIndex: Int,
    val frameLeftPx: Float = leftPx + SwathTileMode.FIVE_TILE.canonicalStripLeftPx,
    val frameRightPx: Float = rightPx + SwathTileMode.FIVE_TILE.canonicalStripLeftPx
) {
    /** Box width in strip pixels, clamped to `>= 0`. */
    val widthPx: Float get() = (rightPx - leftPx).coerceAtLeast(0f)

    /** Box height in strip pixels, clamped to `>= 0`. */
    val heightPx: Float get() = (bottomPx - topPx).coerceAtLeast(0f)

    /** Longer side (`max(widthPx, heightPx)`) in strip pixels, used to estimate physical stone diameter. */
    val maxSpanPx: Float get() = max(widthPx, heightPx)

    /** Horizontal box center in unified strip pixels (`[0..stripWidthPx]`). */
    val centerXPx: Float get() = (leftPx + rightPx) * 0.5f

    /** Vertical box center in unified strip pixels (`[0..STRIP_HEIGHT_PX]`). */
    val centerYPx: Float get() = (topPx + bottomPx) * 0.5f

    /** Horizontal box center in full upright landscape frame pixels (`[0..LANDSCAPE_FRAME_WIDTH_PX]`). */
    val frameCenterXPx: Float get() = (frameLeftPx + frameRightPx) * 0.5f

    /**
     * Computes standard Intersection-over-Union (`IoU`) with [other] in unified strip pixel coordinates.
     *
     * @param other Candidate detection in the same unified strip coordinate frame.
     * @return Ratio `intersectionArea / unionArea` in `[0.0, 1.0]`, or `0.0` if disjoint.
     */
    fun iouWith(other: StripDetection): Float {
        val interLeft = max(leftPx, other.leftPx)
        val interTop = max(topPx, other.topPx)
        val interRight = min(rightPx, other.rightPx)
        val interBottom = min(bottomPx, other.bottomPx)

        val interW = (interRight - interLeft).coerceAtLeast(0f)
        val interH = (interBottom - interTop).coerceAtLeast(0f)
        val interArea = interW * interH
        if (interArea <= 0f) return 0f

        val unionArea = widthPx * heightPx + other.widthPx * other.heightPx - interArea
        return if (unionArea > 0f) interArea / unionArea else 0f
    }

    /**
     * Computes Intersection-over-Minimum (`IoM = intersectionArea / min(areaA, areaB)`) with [other]
     * in unified strip pixel coordinates so a smaller tile-clipped fragment that is mostly contained
     * inside a larger adjacent-tile box merges cleanly across a [StripRoiGeometry.TILE_OVERLAP_PX] seam
     * (`.agents/rules/android-camera-ar.md §2`).
     *
     * @param other Candidate detection in the same unified strip coordinate frame.
     * @return Ratio `intersectionArea / min(areaA, areaB)` in `[0.0, 1.0]`, or `0.0` if disjoint.
     */
    fun iomWith(other: StripDetection): Float {
        val interLeft = max(leftPx, other.leftPx)
        val interTop = max(topPx, other.topPx)
        val interRight = min(rightPx, other.rightPx)
        val interBottom = min(bottomPx, other.bottomPx)

        val interW = (interRight - interLeft).coerceAtLeast(0f)
        val interH = (interBottom - interTop).coerceAtLeast(0f)
        val interArea = interW * interH
        if (interArea <= 0f) return 0f

        val minArea = min(widthPx * heightPx, other.widthPx * other.heightPx)
        return if (minArea > 0f) (interArea / minArea).coerceIn(0f, 1f) else 0f
    }

    /**
     * Computes the 1D vertical projection overlap ratio `interH / min(heightA, heightB)` with [other]
     * to identify horizontally adjacent fragments of an elongated object straddling a tile seam.
     *
     * @param other Candidate detection in the same unified strip coordinate frame.
     * @return Vertical overlap ratio in `[0.0, 1.0]`, or `0.0` if vertically disjoint.
     */
    fun verticalOverlapRatioWith(other: StripDetection): Float {
        val interTop = max(topPx, other.topPx)
        val interBottom = min(bottomPx, other.bottomPx)
        val interH = (interBottom - interTop).coerceAtLeast(0f)
        if (interH <= 0f) return 0f
        val minH = min(heightPx, other.heightPx)
        return if (minH > 0f) (interH / minH).coerceIn(0f, 1f) else 0f
    }
}

/**
 * 3D point in local world coordinates (meters) produced by [com.gneissfind.app.tracking.GroundPlaneRaycaster].
 *
 * @property xMeters Cross-track (East) horizontal position in meters.
 * @property yMeters Vertical position in meters (`0.0` on the estimated ground plane).
 * @property zMeters Along-track (North) forward position in meters.
 */
data class WorldPoint3D(
    val xMeters: Float,
    val yMeters: Float,
    val zMeters: Float
) {
    /**
     * Returns the Euclidean 3D distance in meters between `this` point and [other].
     *
     * @param other Point in the same world coordinate frame (meters).
     * @return Straight-line 3D distance in meters (`>= 0.0`).
     */
    fun distanceMetersTo(other: WorldPoint3D): Float {
        val dx = xMeters - other.xMeters
        val dy = yMeters - other.yMeters
        val dz = zMeters - other.zMeters
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
}

/**
 * Unit 3D bearing vector in the magnetometer-free `TYPE_GAME_ROTATION_VECTOR` ENU world frame.
 *
 * Anchoring each detection's line-of-sight ray in the IMU world frame (`R_world_from_scr(t_capture) * d_scr`)
 * makes track association invariant to handheld pitch, yaw, and roll between sweeps, and allows
 * 120 Hz ego-motion reprojection (`R_world_from_scr(t_now)^T * d_world`) so target reticles stay
 * locked onto physical objects during inference latency and camera motion.
 *
 * @property eastX World East (`+X_w`) unit component.
 * @property northY World North (`+Y_w`) unit component.
 * @property upZ World Up (`+Z_w`, skyward) unit component (`< 0` for downward ground-pointing rays).
 */
data class WorldRay3D(
    val eastX: Float,
    val northY: Float,
    val upZ: Float
) {
    /**
     * Returns a unit-length copy of this ray, or a canonical `45 deg` downward North ray if
     * the vector norm is degenerate (`<= 1e-6`).
     */
    fun normalized(): WorldRay3D {
        val norm = sqrt(eastX * eastX + northY * northY + upZ * upZ)
        return if (norm > 1e-6f) {
            val inv = 1.0f / norm
            WorldRay3D(eastX * inv, northY * inv, upZ * inv)
        } else {
            WorldRay3D(0f, 0.70710677f, -0.70710677f)
        }
    }

    /**
     * Computes the Euclidean inner product (`this · other`).
     *
     * @param other Ray in the same ENU world frame.
     * @return Scalar dot product (`cos(theta)` when both rays are unit-normalized).
     */
    fun dot(other: WorldRay3D): Float =
        eastX * other.eastX + northY * other.northY + upZ * other.upZ

    /**
     * Returns the great-circle angular separation in degrees (`[0.0, 180.0]`) between `this` ray
     * and [other].
     *
     * @param other Ray in the same ENU world frame.
     * @return Central angle in degrees in `[0.0, 180.0]`.
     */
    fun angularDistanceDegTo(other: WorldRay3D): Float {
        val a = normalized()
        val b = other.normalized()
        val cosAngle = a.dot(b).coerceIn(-1.0f, 1.0f)
        return Math.toDegrees(acos(cosAngle.toDouble())).toFloat()
    }

    /**
     * Blends `this` ray toward [other] with weight [otherWeight] in `[0.0, 1.0]` and normalizes
     * the result back onto the unit sphere.
     *
     * @param other Target ray in the same ENU world frame.
     * @param otherWeight Interpolation weight for [other] (`0.0` = keep `this`, `1.0` = take [other]).
     * @return Unit-normalized blended [WorldRay3D].
     */
    fun blendAndNormalize(other: WorldRay3D, otherWeight: Float): WorldRay3D {
        val w = otherWeight.coerceIn(0f, 1f)
        val keep = 1f - w
        return WorldRay3D(
            eastX = eastX * keep + other.eastX * w,
            northY = northY * keep + other.northY * w,
            upZ = upZ * keep + other.upZ * w
        ).normalized()
    }
}

/**
 * Target track confirmed by multi-frame spatial voting in [com.gneissfind.app.tracking.TemporalTargetTracker].
 *
 * Consumed by [com.gneissfind.app.audio.MetalDetectorSonifier] to drive ping pitch/cadence and by
 * [com.gneissfind.app.ui.ScannerScreen] to draw target reticles inside the push-broom strip.
 *
 * @property trackId Monotonically increasing unique identifier assigned when the track is created.
 * @property stoneClass Confirmed [StoneClass] taxonomy entry.
 * @property worldPosition Smoothed 3D ground coordinate in meters.
 * @property stripLeftPx Current bounding-box left X coordinate in unified strip pixels (reprojected at 120 Hz).
 * @property stripTopPx Current bounding-box top Y coordinate in unified strip pixels (reprojected at 120 Hz).
 * @property stripRightPx Current bounding-box right X coordinate in unified strip pixels (reprojected at 120 Hz).
 * @property stripBottomPx Current bounding-box bottom Y coordinate in unified strip pixels (reprojected at 120 Hz).
 * @property smoothedConfidence Exponentially smoothed confidence score in `[0.0, 1.0]`.
 * @property estimatedDiameterMm Smoothed physical diameter estimate in millimeters.
 * @property slantRangeMeters Slant range from the camera to the ground patch in meters.
 * @property hitCountInWindow Number of sweeps within the active rolling window that matched this track.
 * @property windowSize Rolling window length in sweeps (`M` in `N-of-M` spatial voting).
 * @property lastUpdatedTimestampNs Sensor timestamp (`nanoseconds`) of the most recent matching sweep.
 * @property worldRay Optional 3D unit bearing ray in the `TYPE_GAME_ROTATION_VECTOR` world frame
 *   for 120 Hz ego-motion reticle reprojection. `null` when running synthetic demo profiles.
 * @property angularWidthRad Horizontal angular span (`widthPx / focalLengthPx`, in radians) of the target box.
 * @property angularHeightRad Vertical angular span (`heightPx / focalLengthPx`, in radians) of the target box.
 * @property isVisibleInStrip True when the ego-motion-reprojected bounding box currently intersects
 *   the active `[0..stripWidthPx] x [0..STRIP_HEIGHT_PX]` push-broom strip ROI; false when camera
 *   rotation has moved the target outside the strip so ghost reticles and audio pings are suppressed.
 */
data class ConfirmedTarget(
    val trackId: Long,
    val stoneClass: StoneClass,
    val worldPosition: WorldPoint3D,
    val stripLeftPx: Float,
    val stripTopPx: Float,
    val stripRightPx: Float,
    val stripBottomPx: Float,
    val smoothedConfidence: Float,
    val estimatedDiameterMm: Float,
    val slantRangeMeters: Float,
    val hitCountInWindow: Int,
    val windowSize: Int,
    val lastUpdatedTimestampNs: Long,
    val worldRay: WorldRay3D? = null,
    val angularWidthRad: Float = 0f,
    val angularHeightRad: Float = 0f,
    val isVisibleInStrip: Boolean = true
) {
    /** Ratio `[hitCountInWindow] / [windowSize]` clamped to `[0.0, 1.0]`. */
    val voteRatio: Float
        get() = (hitCountInWindow.toFloat() / windowSize.coerceAtLeast(1)).coerceIn(0f, 1f)
}

/**
 * User-selectable top-level operating mode on the left stem (`M-01` / `M-02`) of the LCARS
 * tactical control deck (`AGENTS.md §2`).
 */
enum class OperatingMode {
    /** Real-time push-broom stone detection, visual reticles, and `AudioTrack` sonification. */
    SCAN,

    /**
     * Field dataset logging mode (`4.0 Hz` lossless strip PNG + `metadata.jsonl` capture,
     * `AGENTS.md §2`). Suppresses visual target reticles while recording raw strips.
     */
    LOG_DATA
}

/**
 * Active inference backend driving push-broom sweep detections.
 *
 * @property label Uppercase telemetry description displayed in status readouts.
 * @property shortCode Compact 3-character code (`SIM` or `GPU`) for stem badges.
 */
enum class DetectorBackend(val label: String, val shortCode: String) {
    /** Deterministic synthetic gravel-bar simulator ([com.gneissfind.app.inference.SyntheticStoneDetector]). */
    SYNTHETIC_DEMO("SYNTHETIC SIM", "SIM"),

    /** Live LiteRT 2.x OpenCL GPU FP16 `RF-DETR-Nano` detector ([com.gneissfind.app.inference.RfDetrTwoGraphGpuDetector]). */
    RF_DETR_NANO_GPU("RF-DETR GPU FP16", "GPU")
}

/**
 * Walk-speed state evaluated against [com.gneissfind.app.tracking.TemporalTargetTracker.MAX_SWEEP_SPEED_MPS]
 * (`AGENTS.md §5`).
 */
sealed interface ScanSpeedState {
    /**
     * Walking speed is within the push-broom overlap gate (`<= MAX_SWEEP_SPEED_MPS`).
     *
     * @property currentSpeedMps Measured or simulated horizontal ground speed in meters per second.
     */
    data class Normal(val currentSpeedMps: Float) : ScanSpeedState

    /**
     * Walking speed exceeds [com.gneissfind.app.tracking.TemporalTargetTracker.MAX_SWEEP_SPEED_MPS],
     * triggering the red `SLOW SWEEP` visual alert and low-frequency warning tone (`AGENTS.md §5`).
     *
     * @property currentSpeedMps Measured or simulated horizontal ground speed in meters per second.
     */
    data class SpeedWarning(val currentSpeedMps: Float) : ScanSpeedState
}

/**
 * Thermal and OLED duty-cycling sub-state while operating in Stage 1 (`PATROL`) scan mode
 * (`.agents/rules/android-camera-ar.md §2`).
 *
 * @property targetCameraFps Target Camera2 `CONTROL_AE_TARGET_FPS_RANGE` rate (`30` active, `15` idle).
 * @property preferredRefreshRateHz Window `preferredRefreshRate` in Hz (`120f` active, `30f` idle).
 * @property statusBadge Uppercase HUD telemetry string describing the active patrol power state.
 */
enum class PatrolPowerState(
    val targetCameraFps: Int,
    val preferredRefreshRateHz: Float,
    val statusBadge: String
) {
    /** Full `30 FPS` camera capture and `120 Hz` RenderNode EIS active. */
    PATROL_ACTIVE(
        targetCameraFps = 30,
        preferredRefreshRateHz = 120f,
        statusBadge = "PATROL • 30 FPS"
    ),

    /**
     * Low-power `15 FPS` camera capture and `30 Hz` display refresh engaged after `5.0 s` with
     * zero alertable candidates (`score >= 0.30`) and no touch input.
     */
    PATROL_IDLE(
        targetCameraFps = 15,
        preferredRefreshRateHz = 30f,
        statusBadge = "EYES-UP PATROL • 15 FPS"
    )
}

/**
 * Stage of the 3-stage `PATROL -> LOCK-ON -> PINPOINT` acquisition funnel
 * (`.agents/rules/android-camera-ar.md §5`).
 *
 * @property shortBadge Compact uppercase stage label rendered on the LCARS HUD.
 */
enum class AcquisitionStage(val shortBadge: String) {
    /** Stage 1: Wide `1792 px` push-broom swath search (`B=3` / `B=5`). */
    PATROL("PATROL"),

    /** Stage 2: Full-screen `16:9` foveated `B=1` `320x320` neural tracking + parallax triangulation. */
    LOCK_ON_NEURAL("LOCK-ON B=1"),

    /** Stage 3: Full-screen `16:9` `30 FPS` OpenCV `AKAZE` gravel-constellation guidance + `1 Hz` neural refresh. */
    PINPOINT_CONSTELLATION("PINPOINT AKAZE")
}

/**
 * Viewport-edge re-acquisition indicator rendered when a locked 3D target moves outside the
 * `16:9` camera frame during the `15.0 s` lost-track grace window.
 *
 * @property edgeXNorm Clamped normalized horizontal position in `[0.0, 1.0]` on the `16:9` viewport.
 * @property edgeYNorm Clamped normalized vertical position in `[0.0, 1.0]` on the `16:9` viewport.
 * @property directionRad Screen-space angle (`atan2(dy, dx)`, radians) pointing toward the off-screen 3D anchor.
 * @property angularOffsetDeg Great-circle angular separation (degrees) from camera optical center to the 3D anchor.
 */
data class OffScreenChevron(
    val edgeXNorm: Float,
    val edgeYNorm: Float,
    val directionRad: Float,
    val angularOffsetDeg: Float
)

/**
 * Immutable HUD state snapshot for Stage 2 (`LOCK_ON_NEURAL`) and Stage 3 (`PINPOINT_CONSTELLATION`)
 * rendered over the full uncropped `16:9` (`1920x1080`) camera viewport.
 *
 * @property stage Current [AcquisitionStage] (`PATROL` when unlocked).
 * @property lockedTrackId Track identifier locked when transitioning out of `PATROL`.
 * @property stoneClass Target [StoneClass] (updated at `1 Hz` during `PINPOINT_CONSTELLATION`).
 * @property smoothedConfidence Exponentially smoothed classification confidence in `[0.0, 1.0]`.
 * @property estimatedDiameterMm Estimated physical stone diameter in millimeters.
 * @property slantRangeMeters Current 3D slant range from camera to target in meters.
 * @property sigma3dMm 3D sphere-of-confusion radius in millimeters from multi-view ray triangulation.
 * @property sigmaRadiusPx Uncertainty circle radius projected into `1920x1080` stream pixels.
 * @property parallaxAngleDeg Accumulated multi-view baseline parallax angle in degrees.
 * @property triangulatedViews Count of distinct camera viewpoints contributing to `sigma3dMm`.
 * @property akazeInliers RANSAC/MAGSAC homography inlier count in `PINPOINT_CONSTELLATION`.
 * @property frameLeftPx Target box left X coordinate in `1920x1080` camera pixels.
 * @property frameTopPx Target box top Y coordinate in `1920x1080` camera pixels.
 * @property frameRightPx Target box right X coordinate in `1920x1080` camera pixels.
 * @property frameBottomPx Target box bottom Y coordinate in `1920x1080` camera pixels.
 * @property cropLeftPx Active foveated crop / constellation collar left X in `1920x1080` pixels.
 * @property cropTopPx Active foveated crop / constellation collar top Y in `1920x1080` pixels.
 * @property cropSizePx Active square crop width/height in `1920x1080` pixels (`320` for `B=1`).
 * @property hasNeuralDetectorBox True when a valid `RF-DETR B=1` neural detection box is anchored in 3D.
 * @property neuralBoxLeftPx Reprojected neural model detection left X in `1920x1080` pixels.
 * @property neuralBoxTopPx Reprojected neural model detection top Y in `1920x1080` pixels.
 * @property neuralBoxRightPx Reprojected neural model detection right X in `1920x1080` pixels.
 * @property neuralBoxBottomPx Reprojected neural model detection bottom Y in `1920x1080` pixels.
 * @property isTargetInFrame True when the projected 3D anchor / tracked box lies inside `1920x1080`.
 * @property isLostTrackGraceActive True while counting down the `15.0 s` lost-track standby window.
 * @property lostTrackRemainingSec Seconds remaining (`[0.0, 15.0]`) before auto-returning to `PATROL`.
 * @property offScreenChevron Directional edge chevron when [isTargetInFrame] is `false`.
 * @property trackingSourceBadge Short telemetry source tag (`"RF-DETR B=1"`, `"AKAZE"`, or `"STANDBY"`).
 */
data class LockOnHudState(
    val stage: AcquisitionStage = AcquisitionStage.PATROL,
    val lockedTrackId: Long = -1L,
    val stoneClass: StoneClass = StoneClass.CHALCEDONY_AGATE,
    val smoothedConfidence: Float = 0f,
    val estimatedDiameterMm: Float = 30f,
    val slantRangeMeters: Float = 1.0f,
    val sigma3dMm: Float = 120f,
    val sigmaRadiusPx: Float = 64f,
    val parallaxAngleDeg: Float = 0f,
    val triangulatedViews: Int = 0,
    val akazeInliers: Int = 0,
    val frameLeftPx: Float = 900f,
    val frameTopPx: Float = 480f,
    val frameRightPx: Float = 1020f,
    val frameBottomPx: Float = 600f,
    val cropLeftPx: Int = 800,
    val cropTopPx: Int = 380,
    val cropSizePx: Int = 320,
    val hasNeuralDetectorBox: Boolean = false,
    val neuralBoxLeftPx: Float = 900f,
    val neuralBoxTopPx: Float = 480f,
    val neuralBoxRightPx: Float = 1020f,
    val neuralBoxBottomPx: Float = 600f,
    val isTargetInFrame: Boolean = true,
    val isLostTrackGraceActive: Boolean = false,
    val lostTrackRemainingSec: Float = 15.0f,
    val offScreenChevron: OffScreenChevron? = null,
    val trackingSourceBadge: String = "RF-DETR B=1",
    val isVisuallyLocked: Boolean = false,
    val hasMetric3dDepth: Boolean = false,
    val visualAnchorLabel: String = "RF-DETR B=1"
) {
    /** True when either Stage 2 (`LOCK_ON_NEURAL`) or Stage 3 (`PINPOINT_CONSTELLATION`) is active. */
    val isLocked: Boolean
        get() = stage != AcquisitionStage.PATROL
}
