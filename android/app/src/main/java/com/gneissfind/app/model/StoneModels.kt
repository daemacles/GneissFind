package com.gneissfind.app.model

import kotlin.math.max
import kotlin.math.min

/**
 * Locked v1 5-Class Target Taxonomy with Auxiliary Confuser Supervision.
 * Class 4 ([HARD_CONFUSER]) MUST BE MUTED in UI and audio; it exists strictly to absorb false-positive logits.
 */
enum class StoneClass(
    val id: Int,
    val wireName: String,
    val displayName: String,
    val primaryHexColor: Long,
    val accentHexColor: Long,
    val isUserAlertable: Boolean
) {
    CHALCEDONY_AGATE(
        id = 0,
        wireName = "chalcedony_agate",
        displayName = "Agate / Carnelian",
        primaryHexColor = 0xFFFFB300L, // Amber
        accentHexColor = 0xFF00E5FFL,  // Cyan
        isUserAlertable = true
    ),
    JASPER_RED_YELLOW(
        id = 1,
        wireName = "jasper_red_yellow",
        displayName = "Red / Yellow Jasper",
        primaryHexColor = 0xFFFF3D57L, // Crimson
        accentHexColor = 0xFFFFD54FL,  // Gold
        isUserAlertable = true
    ),
    GREEN_GEM_ROCK(
        id = 2,
        wireName = "green_gem_rock",
        displayName = "Dallasite / Nephrite",
        primaryHexColor = 0xFF00E676L, // Emerald
        accentHexColor = 0xFF69F0AEL,
        isUserAlertable = true
    ),
    PETRIFIED_WOOD(
        id = 3,
        wireName = "petrified_wood",
        displayName = "Petrified Wood",
        primaryHexColor = 0xFFD99B52L, // Bronze
        accentHexColor = 0xFFFFCC80L,
        isUserAlertable = true
    ),
    HARD_CONFUSER(
        id = 4,
        wireName = "hard_confuser",
        displayName = "Hard Confuser (Muted)",
        primaryHexColor = 0xFF78909CL,
        accentHexColor = 0xFF546E7AL,
        isUserAlertable = false
    );

    companion object {
        fun fromId(id: Int): StoneClass? = entries.firstOrNull { it.id == id }

        val alertableClasses: List<StoneClass> = entries.filter { it.isUserAlertable }
    }
}

/**
 * Geometry of the 1088x384 push-broom strip inside the upright viewfinder frame.
 */
data class StripRoiGeometry(
    val uprightFrameWidth: Int,
    val uprightFrameHeight: Int,
    val stripLeftPx: Int,
    val stripTopPx: Int,
    val stripWidthPx: Int = STRIP_WIDTH_PX,
    val stripHeightPx: Int = STRIP_HEIGHT_PX
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
        const val STRIP_WIDTH_PX = 1088
        const val STRIP_HEIGHT_PX = 384
        const val TILE_SIZE_PX = 384
        const val TILE_OVERLAP_PX = 32
        const val NUM_TILES = 3
        val TILE_OFFSETS_X = intArrayOf(0, 352, 704)
        const val DEFAULT_VERTICAL_ANCHOR = 0.36f
    }
}

/**
 * Raw bounding box detection inside one of the three 384x384 tiles.
 * Coordinates are in [0, 384] tile-local pixels.
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
 * Bounding box mapped into the unified 1088x384 push-broom strip coordinate system.
 */
data class StripDetection(
    val stoneClass: StoneClass,
    val confidence: Float,
    val leftPx: Float,
    val topPx: Float,
    val rightPx: Float,
    val bottomPx: Float,
    val estimatedDiameterMm: Float,
    val sourceTileIndex: Int
) {
    val widthPx: Float get() = (rightPx - leftPx).coerceAtLeast(0f)
    val heightPx: Float get() = (bottomPx - topPx).coerceAtLeast(0f)
    val maxSpanPx: Float get() = max(widthPx, heightPx)
    val centerXPx: Float get() = (leftPx + rightPx) * 0.5f
    val centerYPx: Float get() = (topPx + bottomPx) * 0.5f

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
 * Confirmed or candidate target tracked across sweeps in 3D ground coordinates.
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

enum class DetectorBackend(val label: String) {
    SYNTHETIC_DEMO("Synthetic Demo"),
    RF_DETR_NANO_GPU("RF-DETR-Nano GPU FP16")
}

sealed interface ScanSpeedState {
    data class Normal(val currentSpeedMps: Float) : ScanSpeedState
    data class SpeedWarning(val currentSpeedMps: Float) : ScanSpeedState
}
