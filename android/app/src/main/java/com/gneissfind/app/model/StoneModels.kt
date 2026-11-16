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
 * @property id Zero-based class index matching model output and metadata mappings.
 * @property codePrefix Two-digit LCARS numeric prefix rendered on reticles and matrix buttons.
 * @property wireName Canonical snake_case identifier written to JSONL field logs and metadata files.
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
    CHALCEDONY_AGATE(
        id = 0,
        codePrefix = "01",
        wireName = "chalcedony_agate",
        displayName = "Agate / Carnelian",
        lcarsShortName = "AGATE / CARNELIAN",
        primaryHexColor = 0xFF00FFFFL, // Electric Cyan
        accentHexColor = 0xFFFFFFFFL,
        isUserAlertable = true
    ),
    JASPER_RED_YELLOW(
        id = 1,
        codePrefix = "02",
        wireName = "jasper_red_yellow",
        displayName = "Red / Yellow Jasper",
        lcarsShortName = "RED / YEL JASPER",
        primaryHexColor = 0xFFFFFF00L, // High-Vis Solar Yellow
        accentHexColor = 0xFFFFFFFFL,
        isUserAlertable = true
    ),
    GREEN_GEM_ROCK(
        id = 2,
        codePrefix = "03",
        wireName = "green_gem_rock",
        displayName = "Dallasite / Nephrite",
        lcarsShortName = "DALLASITE / JADE",
        primaryHexColor = 0xFF00FF88L, // Spring Mint
        accentHexColor = 0xFFFFFFFFL,
        isUserAlertable = true
    ),
    PETRIFIED_WOOD(
        id = 3,
        codePrefix = "04",
        wireName = "petrified_wood",
        displayName = "Petrified Wood",
        lcarsShortName = "PETRIFIED WOOD",
        primaryHexColor = 0xFFFFAA00L, // Bright Amber-Gold
        accentHexColor = 0xFFFFFFFFL,
        isUserAlertable = true
    ),
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
        /** Looks up a [StoneClass] by its integer [id], or returns `null` if out of range. */
        fun fromId(id: Int): StoneClass? = entries.firstOrNull { it.id == id }

        /** Subset of classes that can produce user-visible reticles and audio pings. */
        val alertableClasses: List<StoneClass> = entries.filter { it.isUserAlertable }
    }
}

/**
 * Horizontal tile-packing layout across the `1920 x 1080` landscape camera frame
 * (`.agents/rules/android-camera-ar.md §2`).
 *
 * @property tileCount Number of horizontally overlapping square tiles per sweep.
 * @property stripWidthPx Total horizontal span of the unified push-broom strip in pixels.
 * @property canonicalStripLeftPx Left X offset (px) of the strip inside a `1920 px` landscape frame.
 * @property frameXOffsets Left X offset (px) of each tile in full `1920 px` landscape frame coordinates.
 * @property stripTileOffsets Left X offset (px) of each tile relative to [canonicalStripLeftPx].
 * @property lcarsLabel Uppercase telemetry badge describing the active swath width.
 */
enum class SwathTileMode(
    val tileCount: Int,
    val stripWidthPx: Int,
    val canonicalStripLeftPx: Int,
    val frameXOffsets: IntArray,
    val stripTileOffsets: IntArray,
    val lcarsLabel: String
) {
    FIVE_TILE(
        tileCount = 5,
        stripWidthPx = 1792,
        canonicalStripLeftPx = 64,
        frameXOffsets = intArrayOf(64, 416, 768, 1120, 1472),
        stripTileOffsets = intArrayOf(0, 352, 704, 1056, 1408),
        lcarsLabel = "5-TILE 1792PX"
    ),
    THREE_TILE(
        tileCount = 3,
        stripWidthPx = 1088,
        canonicalStripLeftPx = 416,
        frameXOffsets = intArrayOf(416, 768, 1120),
        stripTileOffsets = intArrayOf(0, 352, 704),
        lcarsLabel = "3-TILE 1088PX"
    );

    /** Returns the alternate [SwathTileMode] when the user toggles the swath width pill. */
    fun toggled(): SwathTileMode = if (this == FIVE_TILE) THREE_TILE else FIVE_TILE
}

/**
 * Pixel bounds of the active pitch-stabilized push-broom strip inside the upright landscape camera
 * frame.
 *
 * @property uprightFrameWidth Upright landscape frame width in pixels.
 * @property uprightFrameHeight Upright landscape frame height in pixels.
 * @property stripLeftPx Left edge of the strip in upright landscape frame pixels.
 * @property stripTopPx Pitch-stabilized top Y coordinate (`cropY`) in upright landscape frame pixels.
 * @property stripWidthPx Width of the active strip in pixels.
 * @property stripHeightPx Height of the active strip in pixels ([STRIP_HEIGHT_PX]).
 * @property tileMode Active [SwathTileMode] (`FIVE_TILE` or `THREE_TILE`).
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
    val normalizedLeft: Float
        get() = stripLeftPx.toFloat() / uprightFrameWidth.coerceAtLeast(1)

    val normalizedTop: Float
        get() = stripTopPx.toFloat() / uprightFrameHeight.coerceAtLeast(1)

    val normalizedWidth: Float
        get() = stripWidthPx.toFloat() / uprightFrameWidth.coerceAtLeast(1)

    val normalizedHeight: Float
        get() = stripHeightPx.toFloat() / uprightFrameHeight.coerceAtLeast(1)

    companion object {
        const val LANDSCAPE_FRAME_WIDTH_PX = 1920
        const val LANDSCAPE_FRAME_HEIGHT_PX = 1080
        const val MAX_STRIP_WIDTH_PX = 1792
        const val STRIP_WIDTH_5_TILE_PX = 1792
        const val STRIP_WIDTH_3_TILE_PX = 1088
        const val STRIP_HEIGHT_PX = 384
        const val TILE_SIZE_PX = 384
        const val TILE_STRIDE_PX = 352
        const val TILE_OVERLAP_PX = 32
        const val MAX_TILES = 5
        const val NOMINAL_CROP_Y_PX = 348
        const val MAX_CROP_Y_PX = LANDSCAPE_FRAME_HEIGHT_PX - STRIP_HEIGHT_PX // 696
    }
}

/**
 * Unmerged bounding-box prediction inside a single square tile before strip-space stitching.
 *
 * @property tileIndex Index of the source tile (`0 until activeTileCount`).
 * @property classId Target class index corresponding to [StoneClass.id].
 * @property confidence Model probability in `[0.0, 1.0]`.
 * @property leftPx Left edge in tile-local pixels (`[0.0, 384.0]`).
 * @property topPx Top edge in tile-local pixels (`[0.0, 384.0]`).
 * @property rightPx Right edge in tile-local pixels (`[0.0, 384.0]`).
 * @property bottomPx Bottom edge in tile-local pixels (`[0.0, 384.0]`).
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
 * Bounding-box detection mapped into unified push-broom strip coordinates (`[0..stripWidthPx] x [0..384]`)
 * and full landscape frame X coordinates (`[0..1920]`).
 *
 * @property stoneClass Resolved [StoneClass] taxonomy entry.
 * @property confidence Detection confidence score in `[0.0, 1.0]`.
 * @property leftPx Left X coordinate in unified strip pixels.
 * @property topPx Top Y coordinate in unified strip pixels (`[0.0, 384.0]`).
 * @property rightPx Right X coordinate in unified strip pixels.
 * @property bottomPx Bottom Y coordinate in unified strip pixels (`[0.0, 384.0]`).
 * @property estimatedDiameterMm Estimated physical major-axis diameter in millimeters at the current GSD.
 * @property sourceTileIndex Index of the highest-confidence contributing tile (`0..4`).
 * @property frameLeftPx Left X coordinate in full `1920 px` upright landscape frame pixels.
 * @property frameRightPx Right X coordinate in full `1920 px` upright landscape frame pixels.
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
    val widthPx: Float get() = (rightPx - leftPx).coerceAtLeast(0f)
    val heightPx: Float get() = (bottomPx - topPx).coerceAtLeast(0f)
    val maxSpanPx: Float get() = max(widthPx, heightPx)
    val centerXPx: Float get() = (leftPx + rightPx) * 0.5f
    val centerYPx: Float get() = (topPx + bottomPx) * 0.5f
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
     * inside a larger adjacent-tile box merges cleanly across a 32 px seam
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
    /** Returns the Euclidean 3D distance in meters between `this` point and [other]. */
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
    /** Returns a unit-length copy of this ray. */
    fun normalized(): WorldRay3D {
        val norm = sqrt(eastX * eastX + northY * northY + upZ * upZ)
        return if (norm > 1e-6f) {
            val inv = 1.0f / norm
            WorldRay3D(eastX * inv, northY * inv, upZ * inv)
        } else {
            WorldRay3D(0f, 0.70710677f, -0.70710677f)
        }
    }

    /** Returns the dot product with [other]. */
    fun dot(other: WorldRay3D): Float =
        eastX * other.eastX + northY * other.northY + upZ * other.upZ

    /**
     * Returns the great-circle angular separation in degrees (`[0.0, 180.0]`) between `this` ray
     * and [other].
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
 *   for 120 Hz ego-motion reticle reprojection.
 * @property angularWidthRad Horizontal angular span (`widthPx / focalLengthPx`, in radians) of the target box.
 * @property angularHeightRad Vertical angular span (`heightPx / focalLengthPx`, in radians) of the target box.
 * @property isVisibleInStrip True when the ego-motion-reprojected bounding box currently intersects
 *   the active `[0..stripWidthPx] x [0..384]` push-broom strip ROI; false when camera rotation has
 *   moved the target outside the strip so ghost reticles are suppressed.
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
    val voteRatio: Float
        get() = (hitCountInWindow.toFloat() / windowSize.coerceAtLeast(1)).coerceIn(0f, 1f)
}

/** User-selectable top-level operating mode on the LCARS tactical bridge. */
enum class OperatingMode {
    SCAN,
    LOG_DATA
}

/** Active inference backend driving push-broom sweep detections. */
enum class DetectorBackend(val label: String, val shortCode: String) {
    SYNTHETIC_DEMO("SYNTHETIC SIM", "SIM"),
    RF_DETR_NANO_GPU("RF-DETR GPU FP16", "GPU")
}

/** Walk-speed state evaluated against the maximum push-broom ground speed gate. */
sealed interface ScanSpeedState {
    data class Normal(val currentSpeedMps: Float) : ScanSpeedState
    data class SpeedWarning(val currentSpeedMps: Float) : ScanSpeedState
}
