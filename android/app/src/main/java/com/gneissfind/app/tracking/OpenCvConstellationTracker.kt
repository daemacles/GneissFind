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
 * metric 3D camera translation (`(dEast, dNorth, dUp)`) during Stage 2 (`LOCK_ON_NEURAL`).
 *
 * @property prevFramePointsXY Flat `[u0, v0, u1, v1, ...]` coordinates in the previous `1920x1080` frame.
 * @property currFramePointsXY Flat `[u0, v0, u1, v1, ...]` coordinates in the current `1920x1080` frame.
 * @property inlierCount Number of RANSAC planar-gravel inlier correspondences (`prevFramePointsXY.size / 2`).
 */
class GravelFlowCorrespondences(
    val prevFramePointsXY: FloatArray,
    val currFramePointsXY: FloatArray,
    val inlierCount: Int
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
 * Stateful OpenCV C++ JNI tracker managing both:
 * 1. **Stage 2 Ground-Plane KLT + RANSAC (`trackGravelFlowCorrespondences`)** and **`Calib3d.triangulatePoints`**.
 * 2. **Stage 3 `AKAZE` Gravel-Constellation Homography (`seedConstellationCollar` / `matchConstellationFrame`)**
 *    with automatic rolling scale re-keyframing every `1.8x` scale change (`1.5m -> 0.25m`).
 *
 * Threading: Confined to a single sequential tracking worker (`LockOnPinpointController` lock).
 */
class OpenCvConstellationTracker : AutoCloseable {

    private var fullGrayMat: Mat? = null
    private var prevHalfGrayMat: Mat? = null
    private var currHalfGrayMat: Mat? = null
    private var halfMaskMat: Mat? = null
    private var hasPrevHalfFrame: Boolean = false

    private var akaze: AKAZE? = null
    private var bfMatcher: BFMatcher? = null
    private var canonicalPatchMat: Mat? = null
    private var emptyMaskMat: Mat? = null

    // Active reference constellation keyframe (updated on rolling 1.8x scale transitions)
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

    /** True when a valid `AKAZE` reference constellation collar is currently seeded. */
    var hasSeededConstellation: Boolean = false
        private set

    /** Resets all KLT history and active `AKAZE` constellation keyframes when exiting Lock-On. */
    fun resetSession() {
        hasPrevHalfFrame = false
        hasSeededConstellation = false
        refKeypointCoords = emptyArray()
        keyframeBaseScale = 1.0f
        latestRelativeScale = 1.0f
        refDescriptors?.release()
        refDescriptors = null
    }

    /**
     * Uploads an upright `1920x1080` luminance byte array into the reusable `CV_8UC1` [fullGrayMat]
     * and computes RANSAC-filtered inter-frame KLT gravel correspondences outside the target stone box.
     *
     * @param lumaBytes Upright single-channel luminance buffer (`widthPx * heightPx` bytes).
     * @param widthPx Upright frame width in pixels (`1920`).
     * @param heightPx Upright frame height in pixels (`1080`).
     * @param excludeLeftPx Left X of target stone exclusion region in `1920x1080` pixels.
     * @param excludeTopPx Top Y of target stone exclusion region in `1920x1080` pixels.
     * @param excludeRightPx Right X of target stone exclusion region in `1920x1080` pixels.
     * @param excludeBottomPx Bottom Y of target stone exclusion region in `1920x1080` pixels.
     * @return [GravelFlowCorrespondences], or `null` on the first frame or when native OpenCV is unavailable.
     */
    fun trackGravelFlowCorrespondences(
        lumaBytes: ByteArray,
        widthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        heightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX,
        excludeLeftPx: Float,
        excludeTopPx: Float,
        excludeRightPx: Float,
        excludeBottomPx: Float
    ): GravelFlowCorrespondences? {
        if (!ensureOpenCvInitialized()) return null
        if (lumaBytes.size < widthPx * heightPx) return null

        val fullMat = ensureFullGrayMat(widthPx, heightPx)
        fullMat.put(0, 0, lumaBytes, 0, widthPx * heightPx)

        val halfW = widthPx / 2
        val halfH = heightPx / 2
        val currHalf = prevOrAllocMat(currHalfGrayMat, halfH, halfW, CvType.CV_8UC1).also {
            currHalfGrayMat = it
        }
        val prevHalf = prevOrAllocMat(prevHalfGrayMat, halfH, halfW, CvType.CV_8UC1).also {
            prevHalfGrayMat = it
        }
        Imgproc.resize(fullMat, currHalf, Size(halfW.toDouble(), halfH.toDouble()))

        if (!hasPrevHalfFrame) {
            currHalf.copyTo(prevHalf)
            hasPrevHalfFrame = true
            return null
        }

        val mask = prevOrAllocMat(halfMaskMat, halfH, halfW, CvType.CV_8UC1).also {
            halfMaskMat = it
        }
        mask.setTo(Scalar(255.0))
        // Mask out the stone + 24 px specular halo on the half-res frame so wet glints never bias VO
        val exL = ((excludeLeftPx * 0.5f).roundToInt() - 12).coerceIn(0, halfW - 1)
        val exT = ((excludeTopPx * 0.5f).roundToInt() - 12).coerceIn(0, halfH - 1)
        val exR = ((excludeRightPx * 0.5f).roundToInt() + 12).coerceIn(exL + 1, halfW)
        val exB = ((excludeBottomPx * 0.5f).roundToInt() + 12).coerceIn(exT + 1, halfH)
        if (exR > exL && exB > exT) {
            val sub = mask.submat(exT, exB, exL, exR)
            sub.setTo(Scalar(0.0))
            sub.release()
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
                prevHalf,
                corners,
                MAX_VO_GRAVEL_CORNERS,
                0.02,
                14.0,
                mask,
                3
            )
            val cornerArr = corners.toArray()
            if (cornerArr.size < MIN_VO_INLIERS) {
                currHalf.copyTo(prevHalf)
                return null
            }
            prevPts2f.fromArray(*cornerArr)
            Video.calcOpticalFlowPyrLK(
                prevHalf,
                currHalf,
                prevPts2f,
                nextPts2f,
                status,
                err,
                Size(21.0, 21.0),
                3
            )

            val nextArr = nextPts2f.toArray()
            val statusArr = status.toArray()
            val errArr = err.toArray()
            val trackedPrev = ArrayList<Point>(cornerArr.size)
            val trackedCurr = ArrayList<Point>(cornerArr.size)
            for (i in cornerArr.indices) {
                if (i < statusArr.size && statusArr[i].toInt() == 1 &&
                    (i >= errArr.size || errArr[i] < 24.0f)
                ) {
                    val p = nextArr[i]
                    if (p.x in 2.0..(halfW - 3.0) && p.y in 2.0..(halfH - 3.0)) {
                        trackedPrev.add(cornerArr[i])
                        trackedCurr.add(p)
                    }
                }
            }
            currHalf.copyTo(prevHalf)

            if (trackedPrev.size < MIN_VO_INLIERS) return null

            val srcMat = MatOfPoint2f(*trackedPrev.toTypedArray())
            val dstMat = MatOfPoint2f(*trackedCurr.toTypedArray())
            try {
                homography = Calib3d.findHomography(
                    srcMat,
                    dstMat,
                    Calib3d.RANSAC,
                    3.0,
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
                        // Scale 2x from half-res back to canonical 1920x1080 pixel coordinates
                        prevXY[outIdx] = (trackedPrev[i].x * 2.0).toFloat()
                        prevXY[outIdx + 1] = (trackedPrev[i].y * 2.0).toFloat()
                        currXY[outIdx] = (trackedCurr[i].x * 2.0).toFloat()
                        currXY[outIdx + 1] = (trackedCurr[i].y * 2.0).toFloat()
                        outIdx += 2
                    }
                }
                return GravelFlowCorrespondences(
                    prevFramePointsXY = prevXY,
                    currFramePointsXY = currXY,
                    inlierCount = count
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
    fun seedConstellationCollar(
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
        val fullMat = ensureFullGrayMat(widthPx, heightPx)
        if (lumaBytes != null && lumaBytes.size >= widthPx * heightPx) {
            fullMat.put(0, 0, lumaBytes, 0, widthPx * heightPx)
        }

        val stoneW = (stoneRightPx - stoneLeftPx).coerceAtLeast(16f)
        val stoneH = (stoneBottomPx - stoneTopPx).coerceAtLeast(16f)
        val stoneCenterX = (stoneLeftPx + stoneRightPx) * 0.5f
        val stoneCenterY = (stoneTopPx + stoneBottomPx) * 0.5f

        val collarSide = (COLLAR_DIAMETER_MULTIPLIER * max(stoneW, stoneH))
            .roundToInt()
            .coerceIn(MIN_COLLAR_SIZE_PX, MAX_COLLAR_SIZE_PX)
            .coerceAtMost(min(widthPx, heightPx))

        val collarLeft = (stoneCenterX - collarSide * 0.5f).roundToInt()
            .coerceIn(0, (widthPx - collarSide).coerceAtLeast(0))
        val collarTop = (stoneCenterY - collarSide * 0.5f).roundToInt()
            .coerceIn(0, (heightPx - collarSide).coerceAtLeast(0))

        val detector = ensureAkaze()
        val patchMat = prevOrAllocMat(
            canonicalPatchMat,
            CANONICAL_PATCH_PX,
            CANONICAL_PATCH_PX,
            CvType.CV_8UC1
        ).also { canonicalPatchMat = it }
        val maskMat = ensureEmptyMaskMat()
        val collarSub = fullMat.submat(Rect(collarLeft, collarTop, collarSide, collarSide))
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

            val scaleFromCanonical = collarSide.toDouble() / CANONICAL_PATCH_PX.toDouble()
            refDescriptors?.release()
            refDescriptors = descriptors
            refKeypointCoords = Array(kpArray.size) { idx ->
                val p = kpArray[idx].pt
                Point(p.x * scaleFromCanonical, p.y * scaleFromCanonical)
            }
            refCollarWidthPx = collarSide
            refCollarHeightPx = collarSide
            refStoneLocalLeft = stoneLeftPx - collarLeft
            refStoneLocalTop = stoneTopPx - collarTop
            refStoneLocalRight = stoneRightPx - collarLeft
            refStoneLocalBottom = stoneBottomPx - collarTop
            keyframeBaseScale = baseScale
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
    fun matchConstellationFrame(
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

        val fullMat = ensureFullGrayMat(widthPx, heightPx)
        if (lumaBytes != null && lumaBytes.size >= widthPx * heightPx) {
            fullMat.put(0, 0, lumaBytes, 0, widthPx * heightPx)
        }

        val searchSide = (refCollarWidthPx * latestRelativeScale.coerceIn(0.75f, 1.95f) * 1.18f)
            .roundToInt()
            .coerceIn(MIN_COLLAR_SIZE_PX, MAX_SEARCH_SIZE_PX)
            .coerceAtMost(min(widthPx, heightPx))

        val searchLeft = (predictedCenterXPx - searchSide * 0.5f).roundToInt()
            .coerceIn(0, (widthPx - searchSide).coerceAtLeast(0))
        val searchTop = (predictedCenterYPx - searchSide * 0.5f).roundToInt()
            .coerceIn(0, (heightPx - searchSide).coerceAtLeast(0))

        val detector = ensureAkaze()
        val matcher = ensureBfMatcher()
        val patchMat = prevOrAllocMat(
            canonicalPatchMat,
            CANONICAL_PATCH_PX,
            CANONICAL_PATCH_PX,
            CvType.CV_8UC1
        ).also { canonicalPatchMat = it }
        val maskMat = ensureEmptyMaskMat()
        val searchSub = fullMat.submat(Rect(searchLeft, searchTop, searchSide, searchSide))
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

            val searchScaleFromCanonical = searchSide.toDouble() / CANONICAL_PATCH_PX.toDouble()
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
                                searchLeft + dstPt.x * searchScaleFromCanonical,
                                searchTop + dstPt.y * searchScaleFromCanonical
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
                homography = Calib3d.findHomography(
                    srcPtsMat,
                    dstPtsMat,
                    Calib3d.USAC_MAGSAC,
                    MAGSAC_REPROJ_THRESHOLD_PX,
                    inlierMask
                )
                if (homography == null || homography.empty() || inlierMask.empty()) {
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

                // Project the 4 corners + center of the stone through H into 1920x1080 coordinates
                val cxLocal = (refStoneLocalLeft + refStoneLocalRight) * 0.5
                val cyLocal = (refStoneLocalTop + refStoneLocalBottom) * 0.5
                val localPts = MatOfPoint2f(
                    Point(refStoneLocalLeft.toDouble(), refStoneLocalTop.toDouble()),
                    Point(refStoneLocalRight.toDouble(), refStoneLocalTop.toDouble()),
                    Point(refStoneLocalRight.toDouble(), refStoneLocalBottom.toDouble()),
                    Point(refStoneLocalLeft.toDouble(), refStoneLocalBottom.toDouble()),
                    Point(cxLocal, cyLocal)
                )
                val projectedPts = MatOfPoint2f()
                try {
                    Core.perspectiveTransform(localPts, projectedPts, homography)
                    val projArr = projectedPts.toArray()
                    if (projArr.size < 5) {
                        val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                        return ConstellationMatchResult(matched = false, inliers = inliers, latencyMs = elapsedMs)
                    }

                    // Validate homography scale from projected top/left edge lengths vs reference stone size
                    val topEdgeLen = hypot(projArr[1].x - projArr[0].x, projArr[1].y - projArr[0].y).toFloat()
                    val leftEdgeLen = hypot(projArr[3].x - projArr[0].x, projArr[3].y - projArr[0].y).toFloat()
                    val refW = (refStoneLocalRight - refStoneLocalLeft).coerceAtLeast(8f)
                    val refH = (refStoneLocalBottom - refStoneLocalTop).coerceAtLeast(8f)
                    val relScaleX = topEdgeLen / refW
                    val relScaleY = leftEdgeLen / refH
                    val relScale = sqrt((relScaleX * relScaleY).coerceAtLeast(0f))

                    if (!relScale.isFinite() ||
                        relScale !in MIN_RELATIVE_SCALE..MAX_RELATIVE_SCALE ||
                        (relScaleX / relScaleY.coerceAtLeast(1e-3f)) !in 0.45f..2.25f
                    ) {
                        val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                        return ConstellationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
                    }

                    var minX = projArr[0].x.toFloat()
                    var maxX = projArr[0].x.toFloat()
                    var minY = projArr[0].y.toFloat()
                    var maxY = projArr[0].y.toFloat()
                    for (k in 1..3) {
                        val px = projArr[k].x.toFloat()
                        val py = projArr[k].y.toFloat()
                        minX = min(minX, px)
                        maxX = max(maxX, px)
                        minY = min(minY, py)
                        maxY = max(maxY, py)
                    }

                    latestRelativeScale = relScale
                    val cumulativeScale = (keyframeBaseScale * relScale).coerceIn(0.25f, 12.0f)

                    // Rolling scale re-keyframing every 1.8x scale growth (1.5m -> 0.25m approach)
                    if (inliers >= MIN_REKEYFRAME_INLIERS &&
                        (relScale >= ROLLING_KEYFRAME_GROWTH_RATIO || relScale <= ROLLING_KEYFRAME_SHRINK_RATIO)
                    ) {
                        val centerInFrame = projArr[4].x in 64.0..(widthPx - 64.0) &&
                            projArr[4].y in 64.0..(heightPx - 64.0)
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
                    localPts.release()
                    projectedPts.release()
                }
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

    override fun close() {
        resetSession()
        fullGrayMat?.release()
        fullGrayMat = null
        prevHalfGrayMat?.release()
        prevHalfGrayMat = null
        currHalfGrayMat?.release()
        currHalfGrayMat = null
        halfMaskMat?.release()
        halfMaskMat = null
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
        const val MAX_COLLAR_SIZE_PX = 448
        const val MAX_SEARCH_SIZE_PX = 512

        /** Minimum `USAC_MAGSAC` homography inliers required to hold Stage 3 `PINPOINT_CONSTELLATION` lock. */
        const val MIN_HOMOGRAPHY_INLIERS = 12
        const val MIN_SEED_KEYPOINTS = 16
        const val MIN_REKEYFRAME_INLIERS = 14

        /** Lowe second-nearest-neighbor descriptor distance ratio (`0.78f`, `.agents/rules/android-camera-ar.md §5`). */
        const val LOWE_RATIO_THRESHOLD = 0.78f
        const val MAGSAC_REPROJ_THRESHOLD_PX = 3.5

        /** Rolling scale keyframe thresholds (`1.45x` scale hops during `1.5m -> 0.25m` crouch approach). */
        const val ROLLING_KEYFRAME_GROWTH_RATIO = 1.45f
        const val ROLLING_KEYFRAME_SHRINK_RATIO = 0.68f
        private const val MIN_RELATIVE_SCALE = 0.35f
        private const val MAX_RELATIVE_SCALE = 3.50f

        private const val MAX_VO_GRAVEL_CORNERS = 140
        private const val MIN_VO_INLIERS = 10

        private val openCvLoadAttempted = AtomicBoolean(false)

        @Volatile
        private var openCvLoaded: Boolean = false

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
