package com.gneissfind.app.model

import kotlin.math.max
import kotlin.math.min

/**
 * Locked 5-Class Target Taxonomy with Auxiliary Confuser Supervision and Anti-Camouflage Video Reticle Palette:
 *  - `0`: `chalcedony_agate` -> `#00FFFF` Electric Cyan
 *  - `1`: `jasper_red_yellow` -> `#FFFF00` High-Vis Solar Yellow
 *  - `2`: `green_gem_rock` -> `#00FF88` Spring Mint
 *  - `3`: `petrified_wood` -> `#FFAA00` Bright Amber-Gold
 *  - `4`: `hard_confuser` -> Strictly muted in normal `SCAN` UI & audio
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
        fun fromId(id: Int): StoneClass? = entries.firstOrNull { it.id == id }

        val alertableClasses: List<StoneClass> = entries.filter { it.isUserAlertable }
    }
}

/**
 * Cross-track swath tile packing configuration on the `1920 x 1080` landscape stream.
 *  - [FIVE_TILE] (Default): `xOffsets = [64, 416, 768, 1120, 1472]` -> `1792 x 384 px` (`93.3%` width)
 *  - [THREE_TILE] (Fast Mode): `xOffsets = [416, 768, 1120]` -> `1088 x 384 px` (`56.7%` width)
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

    fun toggled(): SwathTileMode = if (this == FIVE_TILE) THREE_TILE else FIVE_TILE
}

/**
 * Geometry of the active stabilized push-broom strip (`1792x384` or `1088x384`) inside the
 * landscape `1920x1080` camera frame.
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
 * Raw bounding box detection inside one of the `384x384` tiles (`[0, 384]` tile-local pixels).
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
 * Bounding box mapped into unified push-broom strip coordinates (`[0..stripWidthPx] x [0..384]`),
 * plus full-frame X coordinates (`frameLeftPx`, `frameRightPx` in `[0..1920]`).
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
}

/**
 * 3D world-space point (in meters) in ARCore or IMU-assisted local world coordinates.
 */
data class WorldPoint3D(
    val xMeters: Float,
    val yMeters: Float,
    val zMeters: Float
) {
    fun distanceMetersTo(other: WorldPoint3D): Float {
        val dx = xMeters - other.xMeters
        val dy = yMeters - other.yMeters
        val dz = zMeters - other.zMeters
        return kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
    }
}

/**
 * Confirmed target tracked across sweeps in 3D ground coordinates.
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
    val lastUpdatedTimestampNs: Long
) {
    val voteRatio: Float
        get() = (hitCountInWindow.toFloat() / windowSize.coerceAtLeast(1)).coerceIn(0f, 1f)
}

enum class OperatingMode {
    SCAN,
    LOG_DATA
}

enum class DetectorBackend(val label: String, val shortCode: String) {
    SYNTHETIC_DEMO("SYNTHETIC SIM", "SIM"),
    RF_DETR_NANO_GPU("RF-DETR GPU FP16", "GPU")
}

sealed interface ScanSpeedState {
    data class Normal(val currentSpeedMps: Float) : ScanSpeedState
    data class SpeedWarning(val currentSpeedMps: Float) : ScanSpeedState
}
