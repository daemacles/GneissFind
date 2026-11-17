/*
 * Native OpenCV 4.13 JNI engine for Stage 2 background gravel KLT + projective triangulation and
 * Stage 3 AKAZE gravel-constellation planar homography terminal guidance.
 *
 * Operates directly on pre-allocated single-channel CV_8UC1 luminance Mats populated from the
 * CameraX YUV_420_888 Y plane so Stage 3 tracks smooth agates via their surrounding 3x gravel
 * collar at 30 FPS (< 5 ms CPU) while throttling B=1 RF-DETR down to 1.0 Hz.
 */
package com.gneissfind.app.tracking

import android.os.SystemClock
import android.util.Log
import com.gneissfind.app.model.StripRoiGeometry
import org.opencv.android.OpenCVLoader
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.AKAZE
import org.opencv.features2d.BFMatcher
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * RANSAC-filtered inter-frame gravel correspondences on the `1920x1080` frame used to solve
 * metric 3D camera translation (`(dEast, dNorth, dUp)`) across Stage 2 (`LOCK_ON_NEURAL`) and
 * Stage 3 (`PINPOINT_CONSTELLATION`), including during target occlusion and off-screen pans.
 *
 * @property prevFramePointsXY Flat `[u0, v0, u1, v1, ...]` coordinates in the previous `1920x1080` frame.
 * @property currFramePointsXY Flat `[u0, v0, u1, v1, ...]` coordinates in the current `1920x1080` frame.
 * @property inlierCount Number of RANSAC planar-gravel inlier correspondences (`prevFramePointsXY.size / 2`).
 * @property latencyMs Wall-clock OpenCV CPU execution latency in milliseconds (`< 2 ms` budget on `640x360`).
 */
class GravelFlowCorrespondences(
    val prevFramePointsXY: FloatArray,
    val currFramePointsXY: FloatArray,
    val inlierCount: Int,
    val latencyMs: Float = 0f
)

/**
 * Result of matching a `480x270` (`1/4`-scale) full-frame `AKAZE` relocalization snapshot
 * against a live `1920x1080` luminance frame (`~5 Hz` during post-occlusion / off-screen re-entry)
 * to snap out accumulated KLT visual odometry drift.
 *
 * @property matched True when `USAC_MAGSAC` homography succeeded with `inliers >= MIN_RELOC_INLIERS`
 *   and physically valid planar scale bounds.
 * @property inliers Number of `USAC_MAGSAC` homography inlier keypoint matches.
 * @property stoneCenterXPx Homography-projected stone horizontal center in `1920x1080` pixels.
 * @property stoneCenterYPx Homography-projected stone vertical center in `1920x1080` pixels.
 * @property latencyMs Wall-clock OpenCV CPU execution latency in milliseconds.
 */
data class RelocalizationMatchResult(
    val matched: Boolean,
    val inliers: Int,
    val stoneCenterXPx: Float = 0f,
    val stoneCenterYPx: Float = 0f,
    val latencyMs: Float = 0f
)

/**
 * Result of matching a live `1920x1080` luminance frame against the locked `AKAZE` gravel
 * constellation collar in Stage 3 (`PINPOINT_CONSTELLATION`).
 *
 * @property matched True when `USAC_MAGSAC` homography succeeded with `inliers >= MIN_HOMOGRAPHY_INLIERS`
 *   and physically valid planar scale/aspect bounds.
 * @property inliers Number of `USAC_MAGSAC` homography inlier keypoint matches.
 * @property frameLeftPx Homography-projected stone bounding-box left X in `1920x1080` pixels.
 * @property frameTopPx Homography-projected stone bounding-box top Y in `1920x1080` pixels.
 * @property frameRightPx Homography-projected stone bounding-box right X in `1920x1080` pixels.
 * @property frameBottomPx Homography-projected stone bounding-box bottom Y in `1920x1080` pixels.
 * @property cumulativeScale Accumulated planar scale factor relative to the initial handover keyframe (`> 1` when closer).
 * @property latencyMs Wall-clock OpenCV CPU execution latency in milliseconds.
 */
data class ConstellationMatchResult(
    val matched: Boolean,
    val inliers: Int,
    val frameLeftPx: Float = 0f,
    val frameTopPx: Float = 0f,
    val frameRightPx: Float = 0f,
    val frameBottomPx: Float = 0f,
    val cumulativeScale: Float = 1.0f,
    val latencyMs: Float = 0f
)

/**
 * Stateful OpenCV C++ JNI tracker managing:
 * 1. **Full-Frame `640x360` Ego-Motion Ground-Plane KLT + RANSAC (`trackGravelFlowCorrespondences`, `< 2 ms` CPU)**
 *    and **`Calib3d.triangulatePoints`**.
 * 2. **Stage 3 `192x192` (`3.0x` target-diameter) `AKAZE` Gravel-Constellation Homography (`seedConstellationCollar` /
 *    `matchConstellationFrame`, `< 5 ms` CPU)** with rolling scale re-keyframing.
 * 3. **Stage 3 `480x270` (`1/4`-scale) Full-Frame `AKAZE` Relocalization Snapshot (`seedFullFrameRelocalizationSnapshot` /
 *    `matchFullFrameRelocalization`)** matched at `~5 Hz` on post-occlusion re-entry to reset VO drift.
 *
 * Threading: Confined to a single sequential tracking worker (`LockOnPinpointController` lock).
 */
open class OpenCvConstellationTracker : AutoCloseable {

    private var fullGrayMat: Mat? = null
    private var prevVoGrayMat: Mat? = null
    private var currVoGrayMat: Mat? = null
    private var voMaskMat: Mat? = null
    private var hasPrevVoFrame: Boolean = false

    private var akaze: AKAZE? = null
    private var bfMatcher: BFMatcher? = null
    private var canonicalPatchMat: Mat? = null
    private var emptyMaskMat: Mat? = null

    // Active reference constellation collar keyframe (updated on rolling scale transitions)
    private var refDescriptors: Mat? = null
    private var refKeypointCoords: Array<Point> = emptyArray()
    private var refCollarWidthPx: Int = 320
    private var refCollarHeightPx: Int = 320
    private var refStoneLocalLeft: Float = 120f
    private var refStoneLocalTop: Float = 120f
    private var refStoneLocalRight: Float = 200f
    private var refStoneLocalBottom: Float = 200f
    private var keyframeBaseScale: Float = 1.0f
    private var latestRelativeScale: Float = 1.0f

    // 480x270 (1/4-scale) full-frame AKAZE relocalization snapshot (<= 220 keypoints)
    private var relocGrayMat: Mat? = null
    private var relocRefDescriptors: Mat? = null
    private var relocRefKeypointCoordsFull: Array<Point> = emptyArray()
    private var relocRefStoneCenterFull: Point = Point(960.0, 540.0)

    /** True when a valid `AKAZE` reference constellation collar is currently seeded. */
    open var hasSeededConstellation: Boolean = false
        protected set

    /** True when a `480x270` full-frame `AKAZE` relocalization snapshot is currently seeded. */
    open var hasRelocalizationSnapshot: Boolean = false
        protected set

    /** Resets all KLT history, active `AKAZE` collar keyframes, and relocalization snapshots when exiting Lock-On. */
    open fun resetSession() {
        hasPrevVoFrame = false
        hasSeededConstellation = false
        hasRelocalizationSnapshot = false
        refKeypointCoords = emptyArray()
        relocRefKeypointCoordsFull = emptyArray()
        keyframeBaseScale = 1.0f
        latestRelativeScale = 1.0f
        refDescriptors?.release()
        refDescriptors = null
        relocRefDescriptors?.release()
        relocRefDescriptors = null
    }

    /**
     * Uploads an upright `1920x1080` luminance byte array into the reusable `CV_8UC1` [fullGrayMat]
     * and computes RANSAC-filtered inter-frame KLT correspondences on a `640x360` (`1/3`-scale)
     * frame across `80–120` corners (`< 2 ms` CPU).
     *
     * @param lumaBytes Upright single-channel luminance buffer (`widthPx * heightPx` bytes).
     * @param widthPx Upright frame width in pixels (`1920`).
     * @param heightPx Upright frame height in pixels (`1080`).
     * @param excludeLeftPx Left X of target stone exclusion region in `1920x1080` pixels.
     * @param excludeTopPx Top Y of target stone exclusion region in `1920x1080` pixels.
     * @param excludeRightPx Right X of target stone exclusion region in `1920x1080` pixels.
     * @param excludeBottomPx Bottom Y of target stone exclusion region in `1920x1080` pixels.
     * @param excludeTargetRegion When `true`, masks out the target stone + specular halo; when `false`
     *   (target occluded or off-screen), tracks corners across the entire `16:9` frame.
     * @return [GravelFlowCorrespondences], or `null` on the first frame or when native OpenCV is unavailable.
     */
    open fun trackGravelFlowCorrespondences(
        lumaBytes: ByteArray,
        widthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        heightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX,
        excludeLeftPx: Float,
        excludeTopPx: Float,
        excludeRightPx: Float,
        excludeBottomPx: Float,
        excludeTargetRegion: Boolean = true
    ): GravelFlowCorrespondences? {
        val startNs = SystemClock.elapsedRealtimeNanos()
        if (!ensureOpenCvInitialized()) return null
        if (widthPx < 64 || heightPx < 36 || lumaBytes.size < widthPx * heightPx) return null

        val fullMat = ensureFullGrayMat(widthPx, heightPx)
        fullMat.put(0, 0, lumaBytes, 0, widthPx * heightPx)

        // 1/3-scale (640x360 on 1920x1080) keeps full-frame ego-motion KLT + RANSAC well below 2.0 ms CPU
        val voW = max(64, widthPx / VO_DOWNSCALE_DIVISOR)
        val voH = max(36, heightPx / VO_DOWNSCALE_DIVISOR)
        val scaleToVoX = voW.toFloat() / widthPx.toFloat()
        val scaleToVoY = voH.toFloat() / heightPx.toFloat()
        val scaleFromVoX = widthPx.toDouble() / voW.toDouble()
        val scaleFromVoY = heightPx.toDouble() / voH.toDouble()

        val currVo = prevOrAllocMat(currVoGrayMat, voH, voW, CvType.CV_8UC1).also {
            currVoGrayMat = it
        }
        val prevVo = prevOrAllocMat(prevVoGrayMat, voH, voW, CvType.CV_8UC1).also {
            prevVoGrayMat = it
        }
        Imgproc.resize(fullMat, currVo, Size(voW.toDouble(), voH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)

        if (!hasPrevVoFrame) {
            currVo.copyTo(prevVo)
            hasPrevVoFrame = true
            return null
        }

        val mask = prevOrAllocMat(voMaskMat, voH, voW, CvType.CV_8UC1).also {
            voMaskMat = it
        }
        mask.setTo(Scalar(255.0))
        // Mask out the stone + 8 px (24 px full-res) specular halo when visible and unoccluded
        if (excludeTargetRegion &&
            excludeLeftPx.isFinite() && excludeTopPx.isFinite() &&
            excludeRightPx.isFinite() && excludeBottomPx.isFinite() &&
            excludeRightPx > 0f && excludeBottomPx > 0f &&
            excludeLeftPx < widthPx.toFloat() && excludeTopPx < heightPx.toFloat()
        ) {
            val rawExL = (excludeLeftPx.finiteOr(0f) * scaleToVoX).roundToInt() - 8
            val rawExT = (excludeTopPx.finiteOr(0f) * scaleToVoY).roundToInt() - 8
            val rawExR = (excludeRightPx.finiteOr(0f) * scaleToVoX).roundToInt() + 8
            val rawExB = (excludeBottomPx.finiteOr(0f) * scaleToVoY).roundToInt() + 8
            val clampedL = rawExL.coerceIn(0, (voW - 2).coerceAtLeast(0))
            val clampedT = rawExT.coerceIn(0, (voH - 2).coerceAtLeast(0))
            val clampedR = rawExR.coerceIn(clampedL, voW)
            val clampedB = rawExB.coerceIn(clampedT, voH)
            if (clampedR - clampedL >= 2 && clampedB - clampedT >= 2) {
                val safeRect = computeSafeSubmatRect(
                    cols = voW,
                    rows = voH,
                    rawX = clampedL,
                    rawY = clampedT,
                    rawW = clampedR - clampedL,
                    rawH = clampedB - clampedT
                )
                if (safeRect != null) {
                    val sub = mask.submat(safeRect)
                    sub.setTo(Scalar(0.0))
                    sub.release()
                }
            }
        }

        val corners = MatOfPoint()
        val prevPts2f = MatOfPoint2f()
        val nextPts2f = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()
        var homography: Mat? = null
        val inlierMask = Mat()

        try {
            Imgproc.goodFeaturesToTrack(
                prevVo,
                corners,
                MAX_VO_GRAVEL_CORNERS,
                0.02,
                11.0,
                mask,
                3
            )
            val cornerArr = corners.toArray()
            if (cornerArr.size < MIN_VO_INLIERS) {
                currVo.copyTo(prevVo)
                return null
            }
            prevPts2f.fromArray(*cornerArr)
            Video.calcOpticalFlowPyrLK(
                prevVo,
                currVo,
                prevPts2f,
                nextPts2f,
                status,
                err,
                Size(17.0, 17.0),
                2
            )

            val nextArr = nextPts2f.toArray()
            val statusArr = status.toArray()
            val errArr = err.toArray()
            val trackedPrev = ArrayList<Point>(cornerArr.size)
            val trackedCurr = ArrayList<Point>(cornerArr.size)
            for (i in cornerArr.indices) {
                if (i < statusArr.size && statusArr[i].toInt() == 1 &&
                    (i >= errArr.size || errArr[i] < 22.0f)
                ) {
                    val p = nextArr[i]
                    if (p.x in 2.0..(voW - 3.0) && p.y in 2.0..(voH - 3.0)) {
                        trackedPrev.add(cornerArr[i])
                        trackedCurr.add(p)
                    }
                }
            }
            currVo.copyTo(prevVo)

            if (trackedPrev.size < MIN_VO_INLIERS) return null

            val srcMat = MatOfPoint2f(*trackedPrev.toTypedArray())
            val dstMat = MatOfPoint2f(*trackedCurr.toTypedArray())
            try {
                homography = Calib3d.findHomography(
                    srcMat,
                    dstMat,
                    Calib3d.RANSAC,
                    2.5,
                    inlierMask
                )
                if (homography == null || homography.empty() || inlierMask.empty()) {
                    return null
                }
                val maskBytes = ByteArray(inlierMask.rows() * inlierMask.cols())
                inlierMask.get(0, 0, maskBytes)
                var count = 0
                for (b in maskBytes) {
                    if (b.toInt() != 0) count++
                }
                if (count < MIN_VO_INLIERS) return null

                val prevXY = FloatArray(count * 2)
                val currXY = FloatArray(count * 2)
                var outIdx = 0
                for (i in maskBytes.indices) {
                    if (maskBytes[i].toInt() != 0) {
                        // Scale 3x from 640x360 back to canonical 1920x1080 pixel coordinates
                        prevXY[outIdx] = (trackedPrev[i].x * scaleFromVoX).toFloat()
                        prevXY[outIdx + 1] = (trackedPrev[i].y * scaleFromVoY).toFloat()
                        currXY[outIdx] = (trackedCurr[i].x * scaleFromVoX).toFloat()
                        currXY[outIdx + 1] = (trackedCurr[i].y * scaleFromVoY).toFloat()
                        outIdx += 2
                    }
                }
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                return GravelFlowCorrespondences(
                    prevFramePointsXY = prevXY,
                    currFramePointsXY = currXY,
                    inlierCount = count,
                    latencyMs = elapsedMs
                )
            } finally {
                srcMat.release()
                dstMat.release()
            }
        } finally {
            corners.release()
            prevPts2f.release()
            nextPts2f.release()
            status.release()
            err.release()
            homography?.release()
            inlierMask.release()
        }
    }

    /**
     * Triangulates a 3D world point (in ENU meters) from two calibrated pinhole viewpoints using
     * OpenCV's C++ `Calib3d.triangulatePoints`.
     *
     * @param camPosAEnu 3-element `(eastM, northM, upM)` camera center for view A.
     * @param rCvFromWorldA 9-element row-major `3x3` `R_cv_from_world` matrix for view A.
     * @param pixelUA Horizontal pixel `u` in view A.
     * @param pixelVA Vertical pixel `v` in view A.
     * @param camPosBEnu 3-element `(eastM, northM, upM)` camera center for view B.
     * @param rCvFromWorldB 9-element row-major `3x3` `R_cv_from_world` matrix for view B.
     * @param pixelUB Horizontal pixel `u` in view B.
     * @param pixelVB Vertical pixel `v` in view B.
     * @param focalLengthPx Calibrated pinhole focal length in pixels.
     * @param principalXPx Principal point `cx` in pixels (`960f`).
     * @param principalYPx Principal point `cy` in pixels (`540f`).
     * @return 3-element `FloatArray(eastM, northM, upM)` in ENU meters, or `null` if degenerate or unavailable.
     */
    fun triangulateTwoViewsOpenCv(
        camPosAEnu: FloatArray,
        rCvFromWorldA: FloatArray,
        pixelUA: Float,
        pixelVA: Float,
        camPosBEnu: FloatArray,
        rCvFromWorldB: FloatArray,
        pixelUB: Float,
        pixelVB: Float,
        focalLengthPx: Float,
        principalXPx: Float = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX * 0.5f,
        principalYPx: Float = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX * 0.5f
    ): FloatArray? {
        if (!ensureOpenCvInitialized()) return null

        val projA = buildNormalizedProjectionMat3x4(rCvFromWorldA, camPosAEnu)
        val projB = buildNormalizedProjectionMat3x4(rCvFromWorldB, camPosBEnu)
        val ptA = Mat(2, 1, CvType.CV_64FC1)
        val ptB = Mat(2, 1, CvType.CV_64FC1)
        val points4D = Mat()

        try {
            val f = focalLengthPx.coerceAtLeast(100f).toDouble()
            ptA.put(0, 0, (pixelUA - principalXPx) / f, (pixelVA - principalYPx) / f)
            ptB.put(0, 0, (pixelUB - principalXPx) / f, (pixelVB - principalYPx) / f)

            Calib3d.triangulatePoints(projA, projB, ptA, ptB, points4D)
            if (points4D.rows() < 4 || points4D.cols() < 1) return null

            val homog = DoubleArray(4)
            points4D.get(0, 0, homog)
            val w = homog[3]
            if (abs(w) < 1e-7) return null

            val east = (homog[0] / w).toFloat()
            val north = (homog[1] / w).toFloat()
            val up = (homog[2] / w).toFloat()
            if (!east.isFinite() || !north.isFinite() || !up.isFinite()) return null
            return floatArrayOf(east, north, up)
        } finally {
            projA.release()
            projB.release()
            ptA.release()
            ptB.release()
            points4D.release()
        }
    }

    /**
     * Extracts `AKAZE` keypoints and binary descriptors over a **`3.0x` target-diameter gravel
     * constellation collar** (`256..448 px` square) centered on the locked stone on the upright
     * `1920x1080` luminance plane.
     *
     * @param lumaBytes Upright `1920x1080` luminance byte array (`null` reuses the already-uploaded [fullGrayMat]).
     * @param widthPx Upright frame width in pixels (`1920`).
     * @param heightPx Upright frame height in pixels (`1080`).
     * @param stoneLeftPx Stone bounding-box left X in `1920x1080` pixels.
     * @param stoneTopPx Stone bounding-box top Y in `1920x1080` pixels.
     * @param stoneRightPx Stone bounding-box right X in `1920x1080` pixels.
     * @param stoneBottomPx Stone bounding-box bottom Y in `1920x1080` pixels.
     * @param baseScale Cumulative scale factor of this keyframe (`1.0f` on initial Stage 2 -> Stage 3 handover).
     * @return True when `>= MIN_SEED_KEYPOINTS` `AKAZE` descriptors were extracted on the collar.
     */
    open fun seedConstellationCollar(
        lumaBytes: ByteArray?,
        widthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        heightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX,
        stoneLeftPx: Float,
        stoneTopPx: Float,
        stoneRightPx: Float,
        stoneBottomPx: Float,
        baseScale: Float = 1.0f
    ): Boolean {
        if (!ensureOpenCvInitialized()) return false
        if (widthPx < 4 || heightPx < 4) return false
        val fullMat = ensureFullGrayMat(widthPx, heightPx)
        if (lumaBytes != null && lumaBytes.size >= widthPx * heightPx) {
            fullMat.put(0, 0, lumaBytes, 0, widthPx * heightPx)
        }

        val sL = stoneLeftPx.finiteOr(widthPx * 0.5f - 24f)
        val sT = stoneTopPx.finiteOr(heightPx * 0.5f - 24f)
        val sR = stoneRightPx.finiteOr(sL + 48f)
        val sB = stoneBottomPx.finiteOr(sT + 48f)

        val stoneW = (sR - sL).coerceIn(16f, widthPx.toFloat())
        val stoneH = (sB - sT).coerceIn(16f, heightPx.toFloat())
        val stoneCenterX = (sL + sR) * 0.5f
        val stoneCenterY = (sT + sB) * 0.5f

        val maxSquare = min(widthPx, heightPx).coerceAtLeast(2)
        val collarSide = (COLLAR_DIAMETER_MULTIPLIER * max(stoneW, stoneH))
            .finiteOr(MIN_COLLAR_SIZE_PX.toFloat())
            .roundToInt()
            .coerceIn(min(MIN_COLLAR_SIZE_PX, maxSquare), max(MIN_COLLAR_SIZE_PX, MAX_COLLAR_SIZE_PX))
            .coerceAtMost(maxSquare)

        val rawCollarLeft = (stoneCenterX - collarSide * 0.5f).finiteOr(0f).roundToInt()
        val rawCollarTop = (stoneCenterY - collarSide * 0.5f).finiteOr(0f).roundToInt()
        val collarRect = computeSafeSubmatRect(
            cols = widthPx,
            rows = heightPx,
            rawX = rawCollarLeft,
            rawY = rawCollarTop,
            rawW = collarSide,
            rawH = collarSide
        ) ?: return false
        val collarLeft = collarRect.x
        val collarTop = collarRect.y
        val effectiveCollarW = collarRect.width
        val effectiveCollarH = collarRect.height

        val detector = ensureAkaze()
        val patchMat = prevOrAllocMat(
            canonicalPatchMat,
            CANONICAL_PATCH_PX,
            CANONICAL_PATCH_PX,
            CvType.CV_8UC1
        ).also { canonicalPatchMat = it }
        val maskMat = ensureEmptyMaskMat()
        val collarSub = fullMat.submat(collarRect)
        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()

        try {
            Imgproc.resize(
                collarSub,
                patchMat,
                Size(CANONICAL_PATCH_PX.toDouble(), CANONICAL_PATCH_PX.toDouble()),
                0.0,
                0.0,
                Imgproc.INTER_LINEAR
            )
            detector.detectAndCompute(patchMat, maskMat, keypoints, descriptors)
            val kpArray = keypoints.toArray()
            if (kpArray.size < MIN_SEED_KEYPOINTS || descriptors.empty()) {
                descriptors.release()
                return false
            }

            val scaleXFromCanonical = effectiveCollarW.toDouble() / CANONICAL_PATCH_PX.toDouble()
            val scaleYFromCanonical = effectiveCollarH.toDouble() / CANONICAL_PATCH_PX.toDouble()
            refDescriptors?.release()
            refDescriptors = descriptors
            refKeypointCoords = Array(kpArray.size) { idx ->
                val p = kpArray[idx].pt
                Point(p.x * scaleXFromCanonical, p.y * scaleYFromCanonical)
            }
            refCollarWidthPx = effectiveCollarW
            refCollarHeightPx = effectiveCollarH
            refStoneLocalLeft = sL - collarLeft
            refStoneLocalTop = sT - collarTop
            refStoneLocalRight = sR - collarLeft
            refStoneLocalBottom = sB - collarTop
            keyframeBaseScale = baseScale.finiteOr(1.0f)
            latestRelativeScale = 1.0f
            hasSeededConstellation = true
            return true
        } finally {
            collarSub.release()
            keypoints.release()
        }
    }

    /**
     * Matches a live `1920x1080` luminance frame against the active `AKAZE` gravel-constellation
     * collar using `BFMatcher(NORM_HAMMING)` + Lowe ratio test (`0.78`) + `Calib3d.findHomography`
     * with `USAC_MAGSAC` (`3.5 px` threshold), and automatically updates the rolling scale keyframe
     * whenever approach scale grows beyond `1.8x` (or shrinks below `0.56x`).
     *
     * @param lumaBytes Upright `1920x1080` luminance byte array (`null` reuses the already-uploaded [fullGrayMat]).
     * @param widthPx Upright frame width in pixels (`1920`).
     * @param heightPx Upright frame height in pixels (`1080`).
     * @param predictedCenterXPx Predicted stone horizontal center in `1920x1080` pixels.
     * @param predictedCenterYPx Predicted stone vertical center in `1920x1080` pixels.
     * @return [ConstellationMatchResult] with projected stone bounds, `USAC_MAGSAC` inlier count, and CPU latency.
     */
    open fun matchConstellationFrame(
        lumaBytes: ByteArray?,
        widthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        heightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX,
        predictedCenterXPx: Float,
        predictedCenterYPx: Float
    ): ConstellationMatchResult {
        val startNs = SystemClock.elapsedRealtimeNanos()
        if (!hasSeededConstellation || !ensureOpenCvInitialized()) {
            return ConstellationMatchResult(matched = false, inliers = 0)
        }
        val activeRefDesc = refDescriptors ?: return ConstellationMatchResult(matched = false, inliers = 0)
        if (activeRefDesc.empty() || refKeypointCoords.size < MIN_SEED_KEYPOINTS) {
            return ConstellationMatchResult(matched = false, inliers = 0)
        }

        if (widthPx < 4 || heightPx < 4) {
            return ConstellationMatchResult(matched = false, inliers = 0)
        }
        val fullMat = ensureFullGrayMat(widthPx, heightPx)
        if (lumaBytes != null && lumaBytes.size >= widthPx * heightPx) {
            fullMat.put(0, 0, lumaBytes, 0, widthPx * heightPx)
        }

        val maxSquare = min(widthPx, heightPx).coerceAtLeast(2)
        val searchSide = (refCollarWidthPx * latestRelativeScale.finiteOr(1f).coerceIn(0.75f, 1.95f) * 1.18f)
            .finiteOr(MIN_COLLAR_SIZE_PX.toFloat())
            .roundToInt()
            .coerceIn(min(MIN_COLLAR_SIZE_PX, maxSquare), max(MIN_COLLAR_SIZE_PX, MAX_SEARCH_SIZE_PX))
            .coerceAtMost(maxSquare)

        val safePredX = predictedCenterXPx.finiteOr(widthPx * 0.5f)
        val safePredY = predictedCenterYPx.finiteOr(heightPx * 0.5f)
        val rawSearchLeft = (safePredX - searchSide * 0.5f).finiteOr(0f).roundToInt()
        val rawSearchTop = (safePredY - searchSide * 0.5f).finiteOr(0f).roundToInt()
        val searchRect = computeSafeSubmatRect(
            cols = widthPx,
            rows = heightPx,
            rawX = rawSearchLeft,
            rawY = rawSearchTop,
            rawW = searchSide,
            rawH = searchSide
        ) ?: return ConstellationMatchResult(matched = false, inliers = 0)
        val searchLeft = searchRect.x
        val searchTop = searchRect.y
        val effectiveSearchW = searchRect.width
        val effectiveSearchH = searchRect.height

        val detector = ensureAkaze()
        val matcher = ensureBfMatcher()
        val patchMat = prevOrAllocMat(
            canonicalPatchMat,
            CANONICAL_PATCH_PX,
            CANONICAL_PATCH_PX,
            CvType.CV_8UC1
        ).also { canonicalPatchMat = it }
        val maskMat = ensureEmptyMaskMat()
        val searchSub = fullMat.submat(searchRect)
        val liveKeypoints = MatOfKeyPoint()
        val liveDescriptors = Mat()
        val knnMatches = ArrayList<MatOfDMatch>()
        var homography: Mat? = null
        val inlierMask = Mat()

        try {
            Imgproc.resize(
                searchSub,
                patchMat,
                Size(CANONICAL_PATCH_PX.toDouble(), CANONICAL_PATCH_PX.toDouble()),
                0.0,
                0.0,
                Imgproc.INTER_LINEAR
            )
            detector.detectAndCompute(patchMat, maskMat, liveKeypoints, liveDescriptors)
            val liveKpArr = liveKeypoints.toArray()
            if (liveKpArr.size < MIN_HOMOGRAPHY_INLIERS || liveDescriptors.empty()) {
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                return ConstellationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
            }

            matcher.knnMatch(activeRefDesc, liveDescriptors, knnMatches, 2)

            val searchScaleXFromCanonical = effectiveSearchW.toDouble() / CANONICAL_PATCH_PX.toDouble()
            val searchScaleYFromCanonical = effectiveSearchH.toDouble() / CANONICAL_PATCH_PX.toDouble()
            val goodSrc = ArrayList<Point>(knnMatches.size)
            val goodDst = ArrayList<Point>(knnMatches.size)
            for (pairMat in knnMatches) {
                val arr = pairMat.toArray()
                if (arr.size >= 2 && arr[0].distance < LOWE_RATIO_THRESHOLD * arr[1].distance) {
                    val qIdx = arr[0].queryIdx
                    val tIdx = arr[0].trainIdx
                    if (qIdx in refKeypointCoords.indices && tIdx in liveKpArr.indices) {
                        goodSrc.add(refKeypointCoords[qIdx])
                        val dstPt = liveKpArr[tIdx].pt
                        goodDst.add(
                            Point(
                                searchLeft + dstPt.x * searchScaleXFromCanonical,
                                searchTop + dstPt.y * searchScaleYFromCanonical
                            )
                        )
                    }
                }
            }

            if (goodSrc.size < MIN_HOMOGRAPHY_INLIERS) {
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                return ConstellationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
            }

            val srcPtsMat = MatOfPoint2f(*goodSrc.toTypedArray())
            val dstPtsMat = MatOfPoint2f(*goodDst.toTypedArray())
            try {
                // Estimate 4-DOF similarity transform (translation + rotation + uniform scale,
                // refined via Levenberg-Marquardt) rather than 8-DOF perspective homography so
                // high-ISO keypoint noise never introduces projective keystone corner wobble.
                homography = Calib3d.estimateAffinePartial2D(
                    srcPtsMat,
                    dstPtsMat,
                    inlierMask,
                    Calib3d.RANSAC,
                    MAGSAC_REPROJ_THRESHOLD_PX
                )
                if (homography == null || homography.empty() ||
                    homography.rows() < 2 || homography.cols() < 3 || inlierMask.empty()
                ) {
                    val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return ConstellationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
                }

                val maskBytes = ByteArray(inlierMask.rows() * inlierMask.cols())
                inlierMask.get(0, 0, maskBytes)
                var inliers = 0
                for (b in maskBytes) {
                    if (b.toInt() != 0) inliers++
                }
                if (inliers < MIN_HOMOGRAPHY_INLIERS) {
                    val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return ConstellationMatchResult(matched = false, inliers = inliers, latencyMs = elapsedMs)
                }

                // Extract 2x3 similarity matrix [a, -b, tx; b, a, ty]
                val m = DoubleArray(6)
                homography.get(0, 0, m)
                val m00 = m[0]
                val m01 = m[1]
                val tx = m[2]
                val m10 = m[3]
                val m11 = m[4]
                val ty = m[5]

                val scaleX = hypot(m00, m10).toFloat()
                val scaleY = hypot(m01, m11).toFloat()
                val relScale = sqrt((scaleX * scaleY).coerceAtLeast(0f))
                if (!relScale.isFinite() ||
                    relScale !in MIN_RELATIVE_SCALE..MAX_RELATIVE_SCALE ||
                    (scaleX / scaleY.coerceAtLeast(1e-3f)) !in 0.75f..1.35f
                ) {
                    val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return ConstellationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
                }

                val cxLocal = (refStoneLocalLeft + refStoneLocalRight) * 0.5
                val cyLocal = (refStoneLocalTop + refStoneLocalBottom) * 0.5
                val projCx = (m00 * cxLocal + m01 * cyLocal + tx).toFloat()
                val projCy = (m10 * cxLocal + m11 * cyLocal + ty).toFloat()
                if (!projCx.isFinite() || !projCy.isFinite()) {
                    val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return ConstellationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
                }

                // Reject spurious repetitive-texture jumps far from predicted target position
                val jumpDistPx = hypot(projCx - safePredX, projCy - safePredY)
                val maxJumpPx = max(96f, effectiveSearchW * 0.28f)
                if (jumpDistPx > maxJumpPx) {
                    val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return ConstellationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
                }

                val refW = (refStoneLocalRight - refStoneLocalLeft).coerceAtLeast(8f)
                val refH = (refStoneLocalBottom - refStoneLocalTop).coerceAtLeast(8f)
                val halfW = refW * 0.5f * relScale
                val halfH = refH * 0.5f * relScale
                val minX = projCx - halfW
                val maxX = projCx + halfW
                val minY = projCy - halfH
                val maxY = projCy + halfH

                latestRelativeScale = relScale
                val cumulativeScale = (keyframeBaseScale * relScale).coerceIn(0.25f, 12.0f)

                // Rolling scale re-keyframing on significant approach/retreat scale changes
                if (relScale >= ROLLING_KEYFRAME_GROWTH_RATIO || relScale <= ROLLING_KEYFRAME_SHRINK_RATIO) {
                    val centerInFrame = projCx in 64f..(widthPx - 64f) &&
                        projCy in 64f..(heightPx - 64f)
                    if (centerInFrame) {
                        seedConstellationCollar(
                            lumaBytes = null,
                            widthPx = widthPx,
                            heightPx = heightPx,
                            stoneLeftPx = minX,
                            stoneTopPx = minY,
                            stoneRightPx = maxX,
                            stoneBottomPx = maxY,
                            baseScale = cumulativeScale
                        )
                    }
                }

                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                return ConstellationMatchResult(
                    matched = true,
                    inliers = inliers,
                    frameLeftPx = minX,
                    frameTopPx = minY,
                    frameRightPx = maxX,
                    frameBottomPx = maxY,
                    cumulativeScale = cumulativeScale,
                    latencyMs = elapsedMs
                )
            } finally {
                srcPtsMat.release()
                dstPtsMat.release()
            }
        } finally {
            for (m in knnMatches) m.release()
            searchSub.release()
            liveKeypoints.release()
            liveDescriptors.release()
            homography?.release()
            inlierMask.release()
        }
    }

    /**
     * Captures a `480x270` (`1/4`-scale) full-frame `AKAZE` relocalization snapshot (`<= 220`
     * strongest keypoints) anchored to `(stoneCenterXPx, stoneCenterYPx)` so post-occlusion or
     * off-screen re-entry can snap out accumulated KLT visual odometry drift in a single `~5 Hz` check.
     *
     * @param lumaBytes Upright `1920x1080` luminance byte array (`null` reuses the already-uploaded [fullGrayMat]).
     * @param widthPx Upright frame width in pixels (`1920`).
     * @param heightPx Upright frame height in pixels (`1080`).
     * @param stoneCenterXPx Locked stone horizontal center in `1920x1080` pixels.
     * @param stoneCenterYPx Locked stone vertical center in `1920x1080` pixels.
     * @return True when `>= MIN_RELOC_SEED_KEYPOINTS` descriptors were stored.
     */
    open fun seedFullFrameRelocalizationSnapshot(
        lumaBytes: ByteArray?,
        widthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        heightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX,
        stoneCenterXPx: Float,
        stoneCenterYPx: Float
    ): Boolean {
        if (!ensureOpenCvInitialized()) return false
        if (widthPx < 64 || heightPx < 36) return false
        val fullMat = ensureFullGrayMat(widthPx, heightPx)
        if (lumaBytes != null && lumaBytes.size >= widthPx * heightPx) {
            fullMat.put(0, 0, lumaBytes, 0, widthPx * heightPx)
        }

        val relocW = max(64, widthPx / RELOC_DOWNSCALE_DIVISOR)
        val relocH = max(36, heightPx / RELOC_DOWNSCALE_DIVISOR)
        val rMat = prevOrAllocMat(relocGrayMat, relocH, relocW, CvType.CV_8UC1).also {
            relocGrayMat = it
        }
        val maskMat = ensureEmptyMaskMat()
        val detector = ensureAkaze()
        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()

        try {
            Imgproc.resize(fullMat, rMat, Size(relocW.toDouble(), relocH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            detector.detectAndCompute(rMat, maskMat, keypoints, descriptors)
            val kpArray = keypoints.toArray()
            if (kpArray.size < MIN_RELOC_SEED_KEYPOINTS || descriptors.empty()) {
                descriptors.release()
                return false
            }

            val scaleFromRelocX = widthPx.toDouble() / relocW.toDouble()
            val scaleFromRelocY = heightPx.toDouble() / relocH.toDouble()

            if (kpArray.size <= MAX_RELOC_KEYPOINTS) {
                relocRefDescriptors?.release()
                relocRefDescriptors = descriptors
                relocRefKeypointCoordsFull = Array(kpArray.size) { idx ->
                    val p = kpArray[idx].pt
                    Point(p.x * scaleFromRelocX, p.y * scaleFromRelocY)
                }
            } else {
                // Retain the strongest MAX_RELOC_KEYPOINTS (220) by AKAZE response to bound 5 Hz matching latency
                val topIndices = kpArray.indices
                    .sortedByDescending { kpArray[it].response }
                    .take(MAX_RELOC_KEYPOINTS)
                val trimmedDesc = Mat(MAX_RELOC_KEYPOINTS, descriptors.cols(), descriptors.type())
                val rowBytes = ByteArray(descriptors.cols() * descriptors.elemSize().toInt())
                val coords = Array(MAX_RELOC_KEYPOINTS) { outRow ->
                    val srcIdx = topIndices[outRow]
                    descriptors.get(srcIdx, 0, rowBytes)
                    trimmedDesc.put(outRow, 0, rowBytes)
                    val p = kpArray[srcIdx].pt
                    Point(p.x * scaleFromRelocX, p.y * scaleFromRelocY)
                }
                descriptors.release()
                relocRefDescriptors?.release()
                relocRefDescriptors = trimmedDesc
                relocRefKeypointCoordsFull = coords
            }

            relocRefStoneCenterFull = Point(
                stoneCenterXPx.finiteOr(widthPx * 0.5f).toDouble(),
                stoneCenterYPx.finiteOr(heightPx * 0.5f).toDouble()
            )
            hasRelocalizationSnapshot = true
            return true
        } finally {
            keypoints.release()
        }
    }

    /**
     * Matches the live `1920x1080` frame (downsampled to `480x270`) against the stored full-frame
     * `AKAZE` relocalization snapshot via `BFMatcher(NORM_HAMMING)` + Lowe `0.78` + `USAC_MAGSAC`
     * homography to recover the exact stone pixel center even after `100+ px` VO drift.
     *
     * @param lumaBytes Upright `1920x1080` luminance byte array (`null` reuses the already-uploaded [fullGrayMat]).
     * @param widthPx Upright frame width in pixels (`1920`).
     * @param heightPx Upright frame height in pixels (`1080`).
     * @return [RelocalizationMatchResult] with projected stone coordinates and CPU latency.
     */
    open fun matchFullFrameRelocalization(
        lumaBytes: ByteArray?,
        widthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        heightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX
    ): RelocalizationMatchResult {
        val startNs = SystemClock.elapsedRealtimeNanos()
        if (!hasRelocalizationSnapshot || !ensureOpenCvInitialized()) {
            return RelocalizationMatchResult(matched = false, inliers = 0)
        }
        val activeRefDesc = relocRefDescriptors
            ?: return RelocalizationMatchResult(matched = false, inliers = 0)
        if (activeRefDesc.empty() || relocRefKeypointCoordsFull.size < MIN_RELOC_SEED_KEYPOINTS ||
            widthPx < 64 || heightPx < 36
        ) {
            return RelocalizationMatchResult(matched = false, inliers = 0)
        }

        val fullMat = ensureFullGrayMat(widthPx, heightPx)
        if (lumaBytes != null && lumaBytes.size >= widthPx * heightPx) {
            fullMat.put(0, 0, lumaBytes, 0, widthPx * heightPx)
        }

        val relocW = max(64, widthPx / RELOC_DOWNSCALE_DIVISOR)
        val relocH = max(36, heightPx / RELOC_DOWNSCALE_DIVISOR)
        val rMat = prevOrAllocMat(relocGrayMat, relocH, relocW, CvType.CV_8UC1).also {
            relocGrayMat = it
        }
        val maskMat = ensureEmptyMaskMat()
        val detector = ensureAkaze()
        val matcher = ensureBfMatcher()
        val liveKeypoints = MatOfKeyPoint()
        val liveDescriptors = Mat()
        val knnMatches = ArrayList<MatOfDMatch>()
        var homography: Mat? = null
        val inlierMask = Mat()

        try {
            Imgproc.resize(fullMat, rMat, Size(relocW.toDouble(), relocH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            detector.detectAndCompute(rMat, maskMat, liveKeypoints, liveDescriptors)
            val liveKpArr = liveKeypoints.toArray()
            if (liveKpArr.size < MIN_RELOC_INLIERS || liveDescriptors.empty()) {
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                return RelocalizationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
            }

            matcher.knnMatch(activeRefDesc, liveDescriptors, knnMatches, 2)
            val scaleFromRelocX = widthPx.toDouble() / relocW.toDouble()
            val scaleFromRelocY = heightPx.toDouble() / relocH.toDouble()
            val goodSrc = ArrayList<Point>(knnMatches.size)
            val goodDst = ArrayList<Point>(knnMatches.size)
            for (pairMat in knnMatches) {
                val arr = pairMat.toArray()
                if (arr.size >= 2 && arr[0].distance < LOWE_RATIO_THRESHOLD * arr[1].distance) {
                    val qIdx = arr[0].queryIdx
                    val tIdx = arr[0].trainIdx
                    if (qIdx in relocRefKeypointCoordsFull.indices && tIdx in liveKpArr.indices) {
                        goodSrc.add(relocRefKeypointCoordsFull[qIdx])
                        val dstPt = liveKpArr[tIdx].pt
                        goodDst.add(Point(dstPt.x * scaleFromRelocX, dstPt.y * scaleFromRelocY))
                    }
                }
            }

            if (goodSrc.size < MIN_RELOC_INLIERS) {
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                return RelocalizationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
            }

            val srcPtsMat = MatOfPoint2f(*goodSrc.toTypedArray())
            val dstPtsMat = MatOfPoint2f(*goodDst.toTypedArray())
            val srcCenterMat = MatOfPoint2f(relocRefStoneCenterFull)
            val dstCenterMat = MatOfPoint2f()
            try {
                homography = Calib3d.findHomography(
                    srcPtsMat,
                    dstPtsMat,
                    Calib3d.USAC_MAGSAC,
                    RELOC_MAGSAC_REPROJ_THRESHOLD_PX,
                    inlierMask
                )
                if (homography == null || homography.empty() ||
                    homography.rows() < 3 || homography.cols() < 3 || inlierMask.empty()
                ) {
                    val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return RelocalizationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
                }

                val maskBytes = ByteArray(inlierMask.rows() * inlierMask.cols())
                inlierMask.get(0, 0, maskBytes)
                var inliers = 0
                for (b in maskBytes) {
                    if (b.toInt() != 0) inliers++
                }
                if (inliers < MIN_RELOC_INLIERS) {
                    val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return RelocalizationMatchResult(matched = false, inliers = inliers, latencyMs = elapsedMs)
                }

                // Reject degenerate / flipped homographies via upper-left 2x2 Jacobian determinant
                val hVals = DoubleArray(9)
                homography.get(0, 0, hVals)
                val h22 = hVals[8]
                if (abs(h22) < 1e-6) {
                    val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return RelocalizationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
                }
                val det2x2 = (hVals[0] * hVals[4] - hVals[1] * hVals[3]) / (h22 * h22)
                if (!det2x2.isFinite() || det2x2 !in 0.12..8.50) {
                    val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return RelocalizationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
                }

                Core.perspectiveTransform(srcCenterMat, dstCenterMat, homography)
                val projPts = dstCenterMat.toArray()
                if (projPts.isEmpty()) {
                    val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return RelocalizationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
                }
                val projX = projPts[0].x.toFloat()
                val projY = projPts[0].y.toFloat()
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                if (!projX.isFinite() || !projY.isFinite() ||
                    projX !in -320f..(widthPx + 320f) ||
                    projY !in -240f..(heightPx + 240f)
                ) {
                    return RelocalizationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
                }
                return RelocalizationMatchResult(
                    matched = true,
                    inliers = inliers,
                    stoneCenterXPx = projX,
                    stoneCenterYPx = projY,
                    latencyMs = elapsedMs
                )
            } finally {
                srcPtsMat.release()
                dstPtsMat.release()
                srcCenterMat.release()
                dstCenterMat.release()
            }
        } finally {
            for (m in knnMatches) m.release()
            liveKeypoints.release()
            liveDescriptors.release()
            homography?.release()
            inlierMask.release()
        }
    }

    override fun close() {
        resetSession()
        fullGrayMat?.release()
        fullGrayMat = null
        prevVoGrayMat?.release()
        prevVoGrayMat = null
        currVoGrayMat?.release()
        currVoGrayMat = null
        voMaskMat?.release()
        voMaskMat = null
        relocGrayMat?.release()
        relocGrayMat = null
        canonicalPatchMat?.release()
        canonicalPatchMat = null
        emptyMaskMat?.release()
        emptyMaskMat = null
    }

    private fun ensureFullGrayMat(widthPx: Int, heightPx: Int): Mat {
        val existing = fullGrayMat
        if (existing != null && existing.cols() == widthPx && existing.rows() == heightPx) {
            return existing
        }
        existing?.release()
        return Mat(heightPx, widthPx, CvType.CV_8UC1).also { fullGrayMat = it }
    }

    private fun ensureEmptyMaskMat(): Mat {
        return emptyMaskMat ?: Mat().also { emptyMaskMat = it }
    }

    private fun prevOrAllocMat(existing: Mat?, rows: Int, cols: Int, type: Int): Mat {
        if (existing != null && existing.rows() == rows && existing.cols() == cols && existing.type() == type) {
            return existing
        }
        existing?.release()
        return Mat(rows, cols, type)
    }

    private fun ensureAkaze(): AKAZE {
        return akaze ?: AKAZE.create(
            AKAZE.DESCRIPTOR_MLDB,
            0,
            3,
            0.00035f,
            2,
            2
        ).also { akaze = it }
    }

    private fun ensureBfMatcher(): BFMatcher {
        return bfMatcher ?: BFMatcher.create(Core.NORM_HAMMING, false).also { bfMatcher = it }
    }

    private fun buildNormalizedProjectionMat3x4(
        rCvFromWorld: FloatArray,
        camPosEnu: FloatArray
    ): Mat {
        // P = [R | -R * C] in normalized camera coordinates ((u - cx)/f, (v - cy)/f)
        val cx = camPosEnu[0].toDouble()
        val cy = camPosEnu[1].toDouble()
        val cz = camPosEnu[2].toDouble()
        val r00 = rCvFromWorld[0].toDouble()
        val r01 = rCvFromWorld[1].toDouble()
        val r02 = rCvFromWorld[2].toDouble()
        val r10 = rCvFromWorld[3].toDouble()
        val r11 = rCvFromWorld[4].toDouble()
        val r12 = rCvFromWorld[5].toDouble()
        val r20 = rCvFromWorld[6].toDouble()
        val r21 = rCvFromWorld[7].toDouble()
        val r22 = rCvFromWorld[8].toDouble()

        val tx = -(r00 * cx + r01 * cy + r02 * cz)
        val ty = -(r10 * cx + r11 * cy + r12 * cz)
        val tz = -(r20 * cx + r21 * cy + r22 * cz)

        return Mat(3, 4, CvType.CV_64FC1).apply {
            put(
                0,
                0,
                r00, r01, r02, tx,
                r10, r11, r12, ty,
                r20, r21, r22, tz
            )
        }
    }

    companion object {
        private const val TAG = "OpenCvConstellation"

        /** Gravel-constellation context collar multiplier (`3.0x` stone diameter, `.agents/rules/android-camera-ar.md §5`). */
        const val COLLAR_DIAMETER_MULTIPLIER = 3.0f
        const val CANONICAL_PATCH_PX = 192
        const val MIN_COLLAR_SIZE_PX = 256
        const val MAX_COLLAR_SIZE_PX = 576
        const val MAX_SEARCH_SIZE_PX = 680

        /** Minimum `USAC_MAGSAC` homography inliers required to hold Stage 3 `PINPOINT_CONSTELLATION` lock. */
        const val MIN_HOMOGRAPHY_INLIERS = 12
        const val MIN_SEED_KEYPOINTS = 16

        /** Lowe second-nearest-neighbor descriptor distance ratio (`0.78f`, `.agents/rules/android-camera-ar.md §5`). */
        const val LOWE_RATIO_THRESHOLD = 0.78f
        const val MAGSAC_REPROJ_THRESHOLD_PX = 3.5

        /** Rolling scale keyframe thresholds (`1.28x` scale hops during `1.5m -> 0.25m` crouch approach). */
        const val ROLLING_KEYFRAME_GROWTH_RATIO = 1.28f
        const val ROLLING_KEYFRAME_SHRINK_RATIO = 0.78f
        private const val MIN_RELATIVE_SCALE = 0.35f
        private const val MAX_RELATIVE_SCALE = 3.50f

        private const val VO_DOWNSCALE_DIVISOR = 3 // 1920x1080 -> 640x360 (< 2 ms CPU)
        private const val MAX_VO_GRAVEL_CORNERS = 108 // 80..120 corner target budget
        private const val MIN_VO_INLIERS = 10

        private const val RELOC_DOWNSCALE_DIVISOR = 4 // 1920x1080 -> 480x270 (~5 Hz relocalization)
        const val MAX_RELOC_KEYPOINTS = 220
        const val MIN_RELOC_SEED_KEYPOINTS = 18
        const val MIN_RELOC_INLIERS = 14
        private const val RELOC_MAGSAC_REPROJ_THRESHOLD_PX = 6.0 // Full-res equivalent of 1.5 px on 480x270

        private val openCvLoadAttempted = AtomicBoolean(false)

        @Volatile
        private var openCvLoaded: Boolean = false

        /**
         * Clamps a candidate `Rect(rawX, rawY, rawW, rawH)` strictly inside `[0..cols] x [0..rows]`
         * with `width >= 2` and `height >= 2` so `Mat.submat(Rect)` can never throw a C++ `CvException`
         * (`0 <= x && x + width <= m.cols && 0 <= y && y + height <= m.rows`) at right/bottom edges
         * (`.agents/rules/android-camera-ar.md §1.2`).
         */
        fun computeSafeSubmatRect(
            cols: Int,
            rows: Int,
            rawX: Int,
            rawY: Int,
            rawW: Int,
            rawH: Int
        ): Rect? {
            if (cols < 2 || rows < 2) return null
            val safeX = rawX.coerceIn(0, cols - 2)
            val safeY = rawY.coerceIn(0, rows - 2)
            val safeW = rawW.coerceIn(2, cols - safeX)
            val safeH = rawH.coerceIn(2, rows - safeY)
            return Rect(safeX, safeY, safeW, safeH)
        }

        private fun Float.finiteOr(fallback: Float): Float =
            if (isFinite()) coerceIn(-100_000f, 100_000f) else fallback

        /**
         * Initializes the native `libopencv_java4.so` library once via `OpenCVLoader.initLocal()`.
         * Safe to call on both Android devices (returns `true`) and desktop JVM unit tests (returns `false`).
         */
        fun ensureOpenCvInitialized(): Boolean {
            if (openCvLoaded) return true
            if (openCvLoadAttempted.compareAndSet(false, true)) {
                openCvLoaded = runCatching {
                    OpenCVLoader.initLocal()
                }.onFailure { err ->
                    Log.w(TAG, "OpenCVLoader.initLocal unavailable in current runtime: ${err.message}")
                }.getOrDefault(false)
                if (openCvLoaded) {
                    Log.i(TAG, "OpenCV 4.13 native runtime initialized via OpenCVLoader.initLocal()")
                }
            }
            return openCvLoaded
        }
    }
}
