/*
 * Native OpenCV 4.13 JNI Unified 3-Tier Spatial Anchor Engine:
 * - Tier 1 (30 FPS): IMU-predicted local target PyrLK + Forward-Backward check + 4-DOF Similarity
 *   + IMU roll gate + 32x32 zero-mean NCC appearance check (< 1.2 ms CPU on 640x360).
 * - Tier 2 (30 FPS): Rotation-Known Sparse 3D Mini-VIO on 640x360 (~90 persistent 3D ENU landmarks,
 *   Huber IRLS 3-DOF camera translation solver with zero z=0 ground-plane blowup, < 1.8 ms CPU).
 * - Tier 3 (5-30 Hz): 4-Gate Multi-Keyframe ORB Bank (up to 6 collar keyframes at 224x224 + up to 4
 *   full-frame 480x270 relocalization keyframes, verified by inlier spread, matrix conditioning,
 *   IMU in-plane roll consistency, and 32x32 zero-mean NCC target appearance, < 3.0 ms CPU).
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
import org.opencv.core.TermCriteria
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * RANSAC-filtered inter-frame gravel correspondences on the `1920x1080` frame used by legacy
 * ground-plane VO callers and deterministic JVM unit tests.
 */
class GravelFlowCorrespondences(
    val prevFramePointsXY: FloatArray,
    val currFramePointsXY: FloatArray,
    val inlierCount: Int,
    val latencyMs: Float = 0f
)

/**
 * Tier 1 (`30 FPS`) result from IMU-predicted local `PyrLK` + forward-backward consistency +
 * 4-DOF Similarity RANSAC + `32x32` zero-mean NCC appearance check on the `640x360` luminance pyramid.
 */
data class LocalSimilarityTrackResult(
    val matched: Boolean,
    val inliers: Int,
    val frameLeftPx: Float = 0f,
    val frameTopPx: Float = 0f,
    val frameRightPx: Float = 0f,
    val frameBottomPx: Float = 0f,
    val scaleRatio: Float = 1.0f,
    val nccScore: Float = 0f,
    val latencyMs: Float = 0f
)

/**
 * Tier 2 (`30 FPS`) result from the Rotation-Known Sparse 3D Mini-VIO camera translation solver.
 */
data class MiniVioStepResult(
    val updated: Boolean,
    val camPosEnu: FloatArray,
    val activeLandmarks: Int,
    val triangulatedLandmarks: Int,
    val inlierLandmarks: Int,
    val latencyMs: Float = 0f
)

/**
 * Result of matching a `480x270` (`1/4`-scale) full-frame `ORB` multi-keyframe relocalization bank
 * against a live `1920x1080` luminance frame (`~5 Hz` during post-occlusion / off-screen re-entry)
 * gated by the 4-Gate Geometric + Appearance Verifier.
 */
data class RelocalizationMatchResult(
    val matched: Boolean,
    val inliers: Int,
    val stoneCenterXPx: Float = 0f,
    val stoneCenterYPx: Float = 0f,
    val stoneWidthPx: Float = 0f,
    val stoneHeightPx: Float = 0f,
    val nccScore: Float = 0f,
    val latencyMs: Float = 0f
)

/**
 * Result of matching a live `1920x1080` luminance frame against the locked 4-Gate Multi-Keyframe
 * `ORB` collar bank.
 */
data class ConstellationMatchResult(
    val matched: Boolean,
    val inliers: Int,
    val frameLeftPx: Float = 0f,
    val frameTopPx: Float = 0f,
    val frameRightPx: Float = 0f,
    val frameBottomPx: Float = 0f,
    val cumulativeScale: Float = 1.0f,
    val nccScore: Float = 0f,
    val matchedKeyframeIndex: Int = -1,
    val latencyMs: Float = 0f
)

/**
 * Stateful OpenCV C++ JNI engine powering the Unified 3-Tier Spatial Anchor:
 * 1. **Tier 1 (`30 FPS`, `< 1.2 ms` CPU):** IMU-predicted local `PyrLK` + FB `< 1.5 px` + 4-DOF Similarity + NCC.
 * 2. **Tier 2 (`30 FPS`, `< 1.8 ms` CPU):** Rotation-Known Sparse 3D Mini-VIO (`640x360`, ~90 landmarks, Huber IRLS).
 * 3. **Tier 3 (`5–30 Hz`, `< 3.0 ms` CPU):** 4-Gate Multi-Keyframe `ORB` Collar + `480x270` Full-Frame Relocalization Bank.
 *
 * Threading: Confined to a single sequential tracking worker (`LockOnPinpointController` lock).
 */
open class OpenCvConstellationTracker : AutoCloseable {

    private class MiniVioLandmark(
        var pxVoX: Double,
        var pxVoY: Double,
        val worldPosEnu: FloatArray,
        val anchorCamPosEnu: FloatArray,
        val anchorRayEnu: FloatArray,
        var age: Int = 0,
        var isTriangulated: Boolean = false
    )

    private class CollarKeyframe(
        val descriptors: Mat,
        val keypointCoords: Array<Point>,
        val collarWidthPx: Int,
        val collarHeightPx: Int,
        val stoneLocalLeft: Float,
        val stoneLocalTop: Float,
        val stoneLocalRight: Float,
        val stoneLocalBottom: Float,
        val nccPatch32: FloatArray?,
        val rWorldFromScr: FloatArray?,
        val baseScale: Float
    ) {
        fun release() {
            descriptors.release()
        }
    }

    private class RelocKeyframe(
        val descriptors: Mat,
        val keypointCoordsFull: Array<Point>,
        val stoneCenterFull: Point,
        val stoneWidthFull: Float,
        val stoneHeightFull: Float,
        val nccPatch32: FloatArray?,
        val rWorldFromScr: FloatArray?
    ) {
        fun release() {
            descriptors.release()
        }
    }

    private var fullGrayMat: Mat? = null
    private var latestFullLumaRef: ByteArray? = null
    private var latestFullWidthPx: Int = 0
    private var latestFullHeightPx: Int = 0
    private var fullMatDirty: Boolean = false

    private var prevVoGrayMat: Mat? = null
    private var currVoGrayMat: Mat? = null
    private var prevVoDetectMat: Mat? = null
    private var voMaskMat: Mat? = null
    private var voSubsampleBuffer: ByteArray? = null
    private var cachedVoCorners: Array<Point> = emptyArray()
    private var hasPrevVoFrame: Boolean = false
    private var currVoWidthPx: Int = 0
    private var currVoHeightPx: Int = 0

    // Tier 2: Rotation-Known Sparse 3D Mini-VIO persistent landmarks
    private val vioLandmarks = ArrayList<MiniVioLandmark>(MAX_VIO_LANDMARKS + 16)
    private var vioMaskMat: Mat? = null

    // Tier 3: ORB + BFMatcher + 4-Gate Multi-Keyframe Banks (Collar + Full-Frame Relocalization)
    private var orbCollar: ORB? = null
    private var orbReloc: ORB? = null
    private var bfMatcher: BFMatcher? = null
    private var canonicalPatchMat: Mat? = null
    private var nccPatchMat: Mat? = null
    private var nccByteScratch: ByteArray = ByteArray(NCC_PATCH_SIZE * NCC_PATCH_SIZE)
    private var emptyMaskMat: Mat? = null

    private val collarKeyframeBank = ArrayList<CollarKeyframe>(MAX_COLLAR_KEYFRAMES)
    private val relocKeyframeBank = ArrayList<RelocKeyframe>(MAX_RELOC_KEYFRAMES)
    private var latestRelativeScale: Float = 1.0f
    private var lastMatchedKeyframeIndex: Int = 0

    // Optional current IMU orientation matrix (3x3 R_world_from_scr) supplied by LockOnPinpointController
    private var currentImuRotationScr: FloatArray? = null

    // 480x270 (1/4-scale) full-frame relocalization buffer
    private var relocGrayMat: Mat? = null

    /** True when at least one valid collar keyframe is seeded in the Tier 3 bank. */
    open var hasSeededConstellation: Boolean = false
        protected set

    /** True when at least one `480x270` full-frame relocalization keyframe is seeded. */
    open var hasRelocalizationSnapshot: Boolean = false
        protected set

    /** Number of collar keyframes currently stored in the Tier 3 multi-keyframe bank. */
    val collarKeyframeCount: Int
        get() = collarKeyframeBank.size

    /** Number of full-frame relocalization keyframes currently stored. */
    val relocKeyframeCount: Int
        get() = relocKeyframeBank.size

    /** Updates the current `3x3` `R_world_from_scr` IMU rotation matrix for Gate 3 roll verification. */
    open fun setCurrentImuRotationMatrix(rWorldFromScr: FloatArray?) {
        currentImuRotationScr = if (rWorldFromScr != null && rWorldFromScr.size >= 9) {
            rWorldFromScr.copyOf(9)
        } else {
            null
        }
    }

    /** Resets all KLT/VIO landmarks, collar keyframe banks, and relocalization snapshots. */
    open fun resetSession() {
        hasPrevVoFrame = false
        hasSeededConstellation = false
        hasRelocalizationSnapshot = false
        latestFullLumaRef = null
        fullMatDirty = false
        cachedVoCorners = emptyArray()
        vioLandmarks.clear()
        latestRelativeScale = 1.0f
        lastMatchedKeyframeIndex = 0
        currentImuRotationScr = null
        for (kf in collarKeyframeBank) kf.release()
        collarKeyframeBank.clear()
        for (rkf in relocKeyframeBank) rkf.release()
        relocKeyframeBank.clear()
    }

    /**
     * Ingests a new `1920x1080` luminance frame into the `640x360` (`1/3`-scale) pyramid buffers
     * (`prevVoGrayMat` <- old `currVoGrayMat`, `currVoGrayMat` <- new frame) ONCE per camera frame
     * so Tier 1 (`PyrLK` similarity) and Tier 2 (Sparse 3D Mini-VIO) share the exact same buffer
     * with zero redundant downsampling or JNI upload.
     *
     * @return `true` when both `prevVoGrayMat` and `currVoGrayMat` are valid for inter-frame tracking.
     */
    open fun beginTrackingFrame(
        lumaBytes: ByteArray,
        widthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        heightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX
    ): Boolean {
        if (!ensureOpenCvInitialized()) return false
        if (widthPx < 64 || heightPx < 36 || lumaBytes.size < widthPx * heightPx) return false

        latestFullLumaRef = lumaBytes
        latestFullWidthPx = widthPx
        latestFullHeightPx = heightPx
        fullMatDirty = true

        val voW = max(64, widthPx / VO_DOWNSCALE_DIVISOR)
        val voH = max(36, heightPx / VO_DOWNSCALE_DIVISOR)
        currVoWidthPx = voW
        currVoHeightPx = voH

        val prevVo = prevOrAllocMat(prevVoGrayMat, voH, voW, CvType.CV_8UC1).also {
            prevVoGrayMat = it
        }
        val currVo = prevOrAllocMat(currVoGrayMat, voH, voW, CvType.CV_8UC1).also {
            currVoGrayMat = it
        }
        if (hasPrevVoFrame) {
            currVo.copyTo(prevVo)
        }

        uploadToVoGrayMat(lumaBytes, widthPx, heightPx, voW, voH, currVo)

        if (!hasPrevVoFrame) {
            currVo.copyTo(prevVo)
            cachedVoCorners = emptyArray()
            vioLandmarks.clear()
            hasPrevVoFrame = true
            return false
        }
        return true
    }

    /**
     * **Tier 1 (`30 FPS`, `< 1.2 ms` CPU):** IMU-predicted local `PyrLK` + Forward-Backward check
     * (`< 1.5 px`) + 4-DOF Similarity RANSAC + IMU in-plane roll gate (`<= 12 deg`) + `32x32`
     * zero-mean NCC appearance check on the `640x360` frame pair prepared by [beginTrackingFrame].
     *
     * Operates from **Frame 1 of Lock-On** across both Stage 2 and Stage 3.
     */
    open fun trackLocalTargetSimilarity30Hz(
        prevBoxLeftPx: Float,
        prevBoxTopPx: Float,
        prevBoxRightPx: Float,
        prevBoxBottomPx: Float,
        prevRotationScr: FloatArray,
        currRotationScr: FloatArray,
        focalLengthPx: Float,
        fullWidthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        fullHeightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX
    ): LocalSimilarityTrackResult? {
        val startNs = SystemClock.elapsedRealtimeNanos()
        if (!hasPrevVoFrame || !ensureOpenCvInitialized()) return null
        val prevVo = prevVoGrayMat ?: return null
        val currVo = currVoGrayMat ?: return null
        val voW = prevVo.cols()
        val voH = prevVo.rows()
        if (voW < 64 || voH < 36 || prevRotationScr.size < 9 || currRotationScr.size < 9) return null

        val sL = prevBoxLeftPx.finiteOr(fullWidthPx * 0.5f - 40f)
        val sT = prevBoxTopPx.finiteOr(fullHeightPx * 0.5f - 40f)
        val sR = prevBoxRightPx.finiteOr(sL + 80f)
        val sB = prevBoxBottomPx.finiteOr(sT + 80f)
        val boxW = (sR - sL).coerceAtLeast(16f)
        val boxH = (sB - sT).coerceAtLeast(16f)
        val prevCx = (sL + sR) * 0.5f
        val prevCy = (sT + sB) * 0.5f
        if (prevCx !in -40f..(fullWidthPx + 40f) || prevCy !in -40f..(fullHeightPx + 40f)) {
            return LocalSimilarityTrackResult(matched = false, inliers = 0)
        }

        val scaleToVoX = voW.toDouble() / fullWidthPx.toDouble()
        val scaleToVoY = voH.toDouble() / fullHeightPx.toDouble()
        val scaleFromVoX = fullWidthPx.toDouble() / voW.toDouble()
        val scaleFromVoY = fullHeightPx.toDouble() / voH.toDouble()

        val roiSpanVo = max(
            MIN_LOCAL_KLT_ROI_VO_PX.toDouble(),
            max(boxW * scaleToVoX, boxH * scaleToVoY) * LOCAL_KLT_COLLAR_MULT
        ).coerceAtMost(min(voW, voH).toDouble())
        val halfRoiVo = (roiSpanVo * 0.5).roundToInt()
        val prevCxVo = prevCx * scaleToVoX
        val prevCyVo = prevCy * scaleToVoY
        val roiRect = computeSafeSubmatRect(
            cols = voW,
            rows = voH,
            rawX = (prevCxVo - halfRoiVo).roundToInt(),
            rawY = (prevCyVo - halfRoiVo).roundToInt(),
            rawW = halfRoiVo * 2,
            rawH = halfRoiVo * 2
        ) ?: return LocalSimilarityTrackResult(matched = false, inliers = 0)

        val rRelCv = computeRelativeCvRotation(prevRotationScr, currRotationScr)
        val fVo = focalLengthPx.coerceAtLeast(100f).toDouble() * scaleToVoX
        val cxVo = voW * 0.5
        val cyVo = voH * 0.5
        val imuRollDeltaDeg = Math.toDegrees(atan2(rRelCv[3], rRelCv[0])).toFloat()

        // Predict center via pure IMU rotation for sanity gating
        val imuPredCenter = warpPointWithRelativeCvRotation(prevCxVo, prevCyVo, rRelCv, fVo, cxVo, cyVo)
            ?: return LocalSimilarityTrackResult(matched = false, inliers = 0)
        val imuPredFullX = (imuPredCenter.x * scaleFromVoX).toFloat()
        val imuPredFullY = (imuPredCenter.y * scaleFromVoY).toFloat()

        val roiSub = prevVo.submat(roiRect)
        val cornersMat = MatOfPoint()
        val prevPtsMat = MatOfPoint2f()
        val nextPtsMat = MatOfPoint2f()
        val backPtsMat = MatOfPoint2f()
        val statusFwd = MatOfByte()
        val errFwd = MatOfFloat()
        val statusBwd = MatOfByte()
        val errBwd = MatOfFloat()
        val inlierMask = Mat()
        var affine: Mat? = null

        try {
            Imgproc.goodFeaturesToTrack(
                roiSub,
                cornersMat,
                MAX_LOCAL_KLT_CORNERS,
                0.015,
                5.0,
                ensureEmptyMaskMat(),
                3
            )
            val localPts = cornersMat.toArray()
            if (localPts.size < MIN_LOCAL_KLT_INLIERS) {
                val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                return LocalSimilarityTrackResult(matched = false, inliers = 0, latencyMs = ms)
            }

            val prevVoPts = ArrayList<Point>(localPts.size)
            val initNextPts = ArrayList<Point>(localPts.size)
            for (p in localPts) {
                val gx = p.x + roiRect.x
                val gy = p.y + roiRect.y
                val warped = warpPointWithRelativeCvRotation(gx, gy, rRelCv, fVo, cxVo, cyVo) ?: continue
                if (warped.x in 2.0..(voW - 3.0) && warped.y in 2.0..(voH - 3.0)) {
                    prevVoPts.add(Point(gx, gy))
                    initNextPts.add(warped)
                }
            }
            if (prevVoPts.size < MIN_LOCAL_KLT_INLIERS) {
                val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                return LocalSimilarityTrackResult(matched = false, inliers = 0, latencyMs = ms)
            }

            prevPtsMat.fromArray(*prevVoPts.toTypedArray())
            nextPtsMat.fromArray(*initNextPts.toTypedArray())
            val criteria = TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 20, 0.03)
            Video.calcOpticalFlowPyrLK(
                prevVo,
                currVo,
                prevPtsMat,
                nextPtsMat,
                statusFwd,
                errFwd,
                Size(15.0, 15.0),
                3,
                criteria,
                Video.OPTFLOW_USE_INITIAL_FLOW,
                1e-4
            )
            Video.calcOpticalFlowPyrLK(
                currVo,
                prevVo,
                nextPtsMat,
                backPtsMat,
                statusBwd,
                errBwd,
                Size(15.0, 15.0),
                3,
                criteria,
                0,
                1e-4
            )

            val nextArr = nextPtsMat.toArray()
            val backArr = backPtsMat.toArray()
            val stF = statusFwd.toArray()
            val erF = errFwd.toArray()
            val stB = statusBwd.toArray()

            val goodSrc = ArrayList<Point>(prevVoPts.size)
            val goodDst = ArrayList<Point>(prevVoPts.size)
            for (i in prevVoPts.indices) {
                if (i < stF.size && stF[i].toInt() == 1 &&
                    i < stB.size && stB[i].toInt() == 1 &&
                    (i >= erF.size || erF[i] < 24.0f)
                ) {
                    val np = nextArr[i]
                    val bp = backArr[i]
                    val fbErr = hypot(bp.x - prevVoPts[i].x, bp.y - prevVoPts[i].y)
                    if (fbErr <= MAX_FB_ERROR_VO_PX &&
                        np.x in 2.0..(voW - 3.0) &&
                        np.y in 2.0..(voH - 3.0)
                    ) {
                        goodSrc.add(prevVoPts[i])
                        goodDst.add(np)
                    }
                }
            }

            if (goodSrc.size < MIN_LOCAL_KLT_INLIERS) {
                val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                return LocalSimilarityTrackResult(matched = false, inliers = 0, latencyMs = ms)
            }

            val srcM = MatOfPoint2f(*goodSrc.toTypedArray())
            val dstM = MatOfPoint2f(*goodDst.toTypedArray())
            try {
                affine = Calib3d.estimateAffinePartial2D(
                    srcM,
                    dstM,
                    inlierMask,
                    Calib3d.RANSAC,
                    2.5
                )
                if (affine == null || affine.empty() || affine.rows() < 2 || affine.cols() < 3 || inlierMask.empty()) {
                    val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return LocalSimilarityTrackResult(matched = false, inliers = 0, latencyMs = ms)
                }

                val maskBytes = ByteArray(inlierMask.rows() * inlierMask.cols())
                inlierMask.get(0, 0, maskBytes)
                var inliers = 0
                for (b in maskBytes) if (b.toInt() != 0) inliers++
                if (inliers < MIN_LOCAL_KLT_INLIERS) {
                    val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return LocalSimilarityTrackResult(matched = false, inliers = inliers, latencyMs = ms)
                }

                val m = DoubleArray(6)
                affine.get(0, 0, m)
                val scale = hypot(m[0], m[3]).toFloat()
                if (!scale.isFinite() || scale !in 0.85f..1.18f) {
                    val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return LocalSimilarityTrackResult(matched = false, inliers = 0, latencyMs = ms)
                }

                // IMU in-plane roll gate: reject background-shear or specular-glint spurious rotations
                val visRollDeg = Math.toDegrees(atan2(m[3], m[0])).toFloat()
                val rollErrDeg = abs(wrapAngleDegrees(visRollDeg - imuRollDeltaDeg))
                if (rollErrDeg > MAX_LOCAL_KLT_ROLL_ERR_DEG) {
                    val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return LocalSimilarityTrackResult(matched = false, inliers = 0, latencyMs = ms)
                }

                val newCxVo = m[0] * prevCxVo + m[1] * prevCyVo + m[2]
                val newCyVo = m[3] * prevCxVo + m[4] * prevCyVo + m[5]
                val newCx = (newCxVo * scaleFromVoX).toFloat()
                val newCy = (newCyVo * scaleFromVoY).toFloat()
                if (!newCx.isFinite() || !newCy.isFinite()) {
                    val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return LocalSimilarityTrackResult(matched = false, inliers = 0, latencyMs = ms)
                }

                val imuJumpPx = hypot(newCx - imuPredFullX, newCy - imuPredFullY)
                val maxImuJumpPx = max(72f, max(boxW, boxH) * 0.55f)
                if (imuJumpPx > maxImuJumpPx) {
                    val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                    return LocalSimilarityTrackResult(matched = false, inliers = 0, latencyMs = ms)
                }

                val newHalfW = (boxW * 0.5f * scale).coerceIn(10f, fullWidthPx * 0.45f)
                val newHalfH = (boxH * 0.5f * scale).coerceIn(10f, fullHeightPx * 0.45f)
                val outL = newCx - newHalfW
                val outT = newCy - newHalfH
                val outR = newCx + newHalfW
                val outB = newCy + newHalfH

                // Appearance gate: verify against multi-keyframe 32x32 NCC bank so sudden occlusion
                // (hand / blank wall / floor) breaks Tier 1 immediately instead of sticking to occluder!
                var nccScore = 1.0f
                if (collarKeyframeBank.isNotEmpty() &&
                    newCx in 0f..fullWidthPx.toFloat() &&
                    newCy in 0f..fullHeightPx.toFloat()
                ) {
                    val livePatch = extractNormalizedNccPatch32(
                        grayMat = currVo,
                        boxLeftFull = outL,
                        boxTopFull = outT,
                        boxRightFull = outR,
                        boxBottomFull = outB,
                        fullWidthPx = fullWidthPx,
                        fullHeightPx = fullHeightPx
                    )
                    if (livePatch == null) {
                        val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                        return LocalSimilarityTrackResult(matched = false, inliers = 0, nccScore = 0f, latencyMs = ms)
                    }
                    nccScore = computeBestBankNccScore(livePatch)
                    if (nccScore < MIN_LOCAL_KLT_NCC_SCORE) {
                        val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                        return LocalSimilarityTrackResult(
                            matched = false,
                            inliers = 0,
                            nccScore = nccScore,
                            latencyMs = ms
                        )
                    }
                }

                val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                return LocalSimilarityTrackResult(
                    matched = true,
                    inliers = inliers,
                    frameLeftPx = outL,
                    frameTopPx = outT,
                    frameRightPx = outR,
                    frameBottomPx = outB,
                    scaleRatio = scale,
                    nccScore = nccScore,
                    latencyMs = ms
                )
            } finally {
                srcM.release()
                dstM.release()
            }
        } finally {
            roiSub.release()
            cornersMat.release()
            prevPtsMat.release()
            nextPtsMat.release()
            backPtsMat.release()
            statusFwd.release()
            errFwd.release()
            statusBwd.release()
            errBwd.release()
            inlierMask.release()
            affine?.release()
        }
    }

    /**
     * **Tier 2 (`30 FPS`, `< 1.8 ms` CPU):** Rotation-Known Sparse 3D Mini-VIO on `640x360`.
     *
     * Tracks up to `90` persistent 3D ENU scene landmarks across frames with IMU-predicted
     * `OPTFLOW_USE_INITIAL_FLOW` + forward-backward `< 1.5 px`, solves 3-DOF camera translation
     * `C_curr` via 3-iteration Huber IRLS with Tikhonov damping toward `C_prev` (eliminating
     * pure-rotation drift AND the `z = 0 @ 1.06 m` ground-plane assumption), and continuously
     * triangulates landmark depths from real camera baselines (`>= 4 cm`, `>= 1.5 deg` parallax).
     */
    open fun stepRotationKnownMiniVio30Hz(
        prevCamPosEnu: FloatArray,
        prevRotationScr: FloatArray,
        currRotationScr: FloatArray,
        focalLengthPx: Float,
        nominalSceneDepthM: Float,
        fullWidthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        fullHeightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX,
        excludeLeftPx: Float = -1f,
        excludeTopPx: Float = -1f,
        excludeRightPx: Float = -1f,
        excludeBottomPx: Float = -1f,
        excludeTargetRegion: Boolean = false
    ): MiniVioStepResult? {
        val startNs = SystemClock.elapsedRealtimeNanos()
        if (!hasPrevVoFrame || !ensureOpenCvInitialized()) return null
        val prevVo = prevVoGrayMat ?: return null
        val currVo = currVoGrayMat ?: return null
        val voW = prevVo.cols()
        val voH = prevVo.rows()
        if (voW < 64 || voH < 36 || prevCamPosEnu.size < 3 || prevRotationScr.size < 9 || currRotationScr.size < 9) {
            return null
        }

        val scaleToVoX = voW.toDouble() / fullWidthPx.toDouble()
        val scaleToVoY = voH.toDouble() / fullHeightPx.toDouble()
        val scaleFromVoX = fullWidthPx.toFloat() / voW.toFloat()
        val scaleFromVoY = fullHeightPx.toFloat() / voH.toFloat()
        val fVo = focalLengthPx.coerceAtLeast(100f).toDouble() * scaleToVoX
        val cxVo = voW * 0.5
        val cyVo = voH * 0.5
        val rRelCv = computeRelativeCvRotation(prevRotationScr, currRotationScr)
        val rotStepDeg = LockOnPinpointController.rotationAngleBetweenMatricesDeg(prevRotationScr, currRotationScr)

        val outCam = prevCamPosEnu.copyOf(3)
        var updatedCam = false
        var inlierCount = 0

        if (vioLandmarks.size >= MIN_VIO_SOLVE_LANDMARKS) {
            val prevPtsArr = ArrayList<Point>(vioLandmarks.size)
            val initNextArr = ArrayList<Point>(vioLandmarks.size)
            val validIndices = ArrayList<Int>(vioLandmarks.size)
            for (i in vioLandmarks.indices) {
                val lm = vioLandmarks[i]
                val warped = warpPointWithRelativeCvRotation(lm.pxVoX, lm.pxVoY, rRelCv, fVo, cxVo, cyVo)
                    ?: continue
                if (warped.x in 3.0..(voW - 4.0) && warped.y in 3.0..(voH - 4.0)) {
                    prevPtsArr.add(Point(lm.pxVoX, lm.pxVoY))
                    initNextArr.add(warped)
                    validIndices.add(i)
                }
            }

            if (prevPtsArr.size >= MIN_VIO_SOLVE_LANDMARKS) {
                val prevPtsMat = MatOfPoint2f(*prevPtsArr.toTypedArray())
                val nextPtsMat = MatOfPoint2f(*initNextArr.toTypedArray())
                val backPtsMat = MatOfPoint2f()
                val stFwd = MatOfByte()
                val erFwd = MatOfFloat()
                val stBwd = MatOfByte()
                val erBwd = MatOfFloat()
                try {
                    val criteria = TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 18, 0.03)
                    Video.calcOpticalFlowPyrLK(
                        prevVo, currVo, prevPtsMat, nextPtsMat, stFwd, erFwd,
                        Size(13.0, 13.0), 3, criteria, Video.OPTFLOW_USE_INITIAL_FLOW, 1e-4
                    )
                    Video.calcOpticalFlowPyrLK(
                        currVo, prevVo, nextPtsMat, backPtsMat, stBwd, erBwd,
                        Size(13.0, 13.0), 3, criteria, 0, 1e-4
                    )
                    val nextA = nextPtsMat.toArray()
                    val backA = backPtsMat.toArray()
                    val sf = stFwd.toArray()
                    val ef = erFwd.toArray()
                    val sb = stBwd.toArray()

                    val survivors = ArrayList<MiniVioLandmark>(prevPtsArr.size)
                    val currRays = ArrayList<FloatArray>(prevPtsArr.size)
                    val rotResidualsPx = FloatArray(prevPtsArr.size)
                    var resCount = 0

                    for (k in validIndices.indices) {
                        if (k < sf.size && sf[k].toInt() == 1 &&
                            k < sb.size && sb[k].toInt() == 1 &&
                            (k >= ef.size || ef[k] < 24.0f)
                        ) {
                            val np = nextA[k]
                            val bp = backA[k]
                            val fbErr = hypot(bp.x - prevPtsArr[k].x, bp.y - prevPtsArr[k].y)
                            if (fbErr <= MAX_FB_ERROR_VO_PX &&
                                np.x in 3.0..(voW - 4.0) &&
                                np.y in 3.0..(voH - 4.0)
                            ) {
                                val lm = vioLandmarks[validIndices[k]]
                                lm.pxVoX = np.x
                                lm.pxVoY = np.y
                                lm.age++
                                val ray = GroundPlaneRaycaster.pixelToWorldRay(
                                    frameXPx = (np.x * scaleFromVoX).toFloat(),
                                    frameYPx = (np.y * scaleFromVoY).toFloat(),
                                    focalLengthPx = focalLengthPx,
                                    rotationMatrix = currRotationScr,
                                    frameWidthPx = fullWidthPx,
                                    frameHeightPx = fullHeightPx
                                )
                                survivors.add(lm)
                                currRays.add(floatArrayOf(ray.eastX, ray.northY, ray.upZ))
                                rotResidualsPx[resCount++] = hypot(
                                    (np.x - initNextArr[k].x).toFloat(),
                                    (np.y - initNextArr[k].y).toFloat()
                                )
                            }
                        }
                    }

                    vioLandmarks.clear()
                    vioLandmarks.addAll(survivors)

                    if (survivors.size >= MIN_VIO_SOLVE_LANDMARKS) {
                        val medianRotResidVoPx = if (resCount > 0) {
                            rotResidualsPx.sort(0, resCount)
                            rotResidualsPx[resCount / 2]
                        } else {
                            0f
                        }
                        // Pure-rotation deadband: if IMU-derotated flow median is < 0.40 px on 640x360
                        // (~1.2 px on 1080p), camera is rotating in place -> hold C_curr exact!
                        if (medianRotResidVoPx >= VIO_PURE_ROTATION_DEADBAND_VO_PX && rotStepDeg <= 14.0f) {
                            val priorLambda = 0.10 + 0.045 * rotStepDeg.toDouble()
                            val solved = solveHuberIrlsCameraCenter(
                                landmarks = survivors,
                                currRays = currRays,
                                priorCamPosEnu = prevCamPosEnu,
                                priorLambda = priorLambda
                            )
                            if (solved != null) {
                                val stepX = (solved.first[0] - prevCamPosEnu[0]).coerceIn(-0.14f, 0.14f)
                                val stepY = (solved.first[1] - prevCamPosEnu[1]).coerceIn(-0.14f, 0.14f)
                                val stepZ = (solved.first[2] - prevCamPosEnu[2]).coerceIn(-0.08f, 0.08f)
                                outCam[0] = prevCamPosEnu[0] + stepX
                                outCam[1] = prevCamPosEnu[1] + stepY
                                outCam[2] = (prevCamPosEnu[2] + stepZ).coerceIn(0.05f, 2.50f)
                                inlierCount = solved.second
                                updatedCam = true

                                // Refine triangulated 3D positions for landmarks with real parallax baseline
                                for (i in survivors.indices) {
                                    val lm = survivors[i]
                                    val ray = currRays[i]
                                    val bx = outCam[0] - lm.anchorCamPosEnu[0]
                                    val by = outCam[1] - lm.anchorCamPosEnu[1]
                                    val bz = outCam[2] - lm.anchorCamPosEnu[2]
                                    val baseline = sqrt(bx * bx + by * by + bz * bz)
                                    if (baseline >= MIN_VIO_TRIANGULATE_BASELINE_M) {
                                        val dotRays = (lm.anchorRayEnu[0] * ray[0] +
                                            lm.anchorRayEnu[1] * ray[1] +
                                            lm.anchorRayEnu[2] * ray[2]).coerceIn(-1f, 1f)
                                        val parallaxDeg = Math.toDegrees(acos(dotRays.toDouble())).toFloat()
                                        if (parallaxDeg >= MIN_VIO_TRIANGULATE_PARALLAX_DEG) {
                                            val mid = triangulateTwoRaysMidpoint(
                                                originA = lm.anchorCamPosEnu,
                                                rayA = lm.anchorRayEnu,
                                                originB = outCam,
                                                rayB = ray
                                            )
                                            if (mid != null) {
                                                val depthCurr = (mid[0] - outCam[0]) * ray[0] +
                                                    (mid[1] - outCam[1]) * ray[1] +
                                                    (mid[2] - outCam[2]) * ray[2]
                                                if (depthCurr in 0.25f..8.0f) {
                                                    val alpha = if (lm.isTriangulated) 0.35f else 0.70f
                                                    lm.worldPosEnu[0] = lm.worldPosEnu[0] * (1f - alpha) + mid[0] * alpha
                                                    lm.worldPosEnu[1] = lm.worldPosEnu[1] * (1f - alpha) + mid[1] * alpha
                                                    lm.worldPosEnu[2] = lm.worldPosEnu[2] * (1f - alpha) + mid[2] * alpha
                                                    lm.isTriangulated = true
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            inlierCount = survivors.size
                        }
                    }
                } finally {
                    prevPtsMat.release()
                    nextPtsMat.release()
                    backPtsMat.release()
                    stFwd.release()
                    erFwd.release()
                    stBwd.release()
                    erBwd.release()
                }
            }
        }

        // Replenish sparse 3D landmarks whenever count drops below target floor
        if (vioLandmarks.size < MIN_VIO_REPLENISH_LANDMARKS) {
            val mask = prevOrAllocMat(vioMaskMat, voH, voW, CvType.CV_8UC1).also { vioMaskMat = it }
            mask.setTo(Scalar(255.0))
            // Keep 12 px border clear
            Imgproc.rectangle(mask, Point(0.0, 0.0), Point(voW.toDouble(), 10.0), Scalar(0.0), -1)
            Imgproc.rectangle(mask, Point(0.0, (voH - 10).toDouble()), Point(voW.toDouble(), voH.toDouble()), Scalar(0.0), -1)
            Imgproc.rectangle(mask, Point(0.0, 0.0), Point(10.0, voH.toDouble()), Scalar(0.0), -1)
            Imgproc.rectangle(mask, Point((voW - 10).toDouble(), 0.0), Point(voW.toDouble(), voH.toDouble()), Scalar(0.0), -1)
            for (lm in vioLandmarks) {
                Imgproc.circle(mask, Point(lm.pxVoX, lm.pxVoY), 14, Scalar(0.0), -1)
            }
            if (excludeTargetRegion &&
                excludeLeftPx.isFinite() && excludeTopPx.isFinite() &&
                excludeRightPx.isFinite() && excludeBottomPx.isFinite() &&
                excludeRightPx > 0f && excludeBottomPx > 0f &&
                excludeLeftPx < fullWidthPx && excludeTopPx < fullHeightPx
            ) {
                val exRect = computeSafeSubmatRect(
                    cols = voW,
                    rows = voH,
                    rawX = (excludeLeftPx * scaleToVoX - 6.0).roundToInt(),
                    rawY = (excludeTopPx * scaleToVoY - 6.0).roundToInt(),
                    rawW = ((excludeRightPx - excludeLeftPx) * scaleToVoX + 12.0).roundToInt(),
                    rawH = ((excludeBottomPx - excludeTopPx) * scaleToVoY + 12.0).roundToInt()
                )
                if (exRect != null) {
                    val sub = mask.submat(exRect)
                    sub.setTo(Scalar(0.0))
                    sub.release()
                }
            }

            val needed = (MAX_VIO_LANDMARKS - vioLandmarks.size).coerceAtLeast(1)
            val newCorners = MatOfPoint()
            try {
                Imgproc.goodFeaturesToTrack(currVo, newCorners, needed, 0.015, 16.0, mask, 3)
                val initDepth = nominalSceneDepthM.coerceIn(0.45f, 2.80f)
                for (pt in newCorners.toArray()) {
                    val ray = GroundPlaneRaycaster.pixelToWorldRay(
                        frameXPx = (pt.x * scaleFromVoX).toFloat(),
                        frameYPx = (pt.y * scaleFromVoY).toFloat(),
                        focalLengthPx = focalLengthPx,
                        rotationMatrix = currRotationScr,
                        frameWidthPx = fullWidthPx,
                        frameHeightPx = fullHeightPx
                    )
                    val wx = outCam[0] + initDepth * ray.eastX
                    val wy = outCam[1] + initDepth * ray.northY
                    val wz = outCam[2] + initDepth * ray.upZ
                    vioLandmarks.add(
                        MiniVioLandmark(
                            pxVoX = pt.x,
                            pxVoY = pt.y,
                            worldPosEnu = floatArrayOf(wx, wy, wz),
                            anchorCamPosEnu = outCam.copyOf(3),
                            anchorRayEnu = floatArrayOf(ray.eastX, ray.northY, ray.upZ),
                            age = 0,
                            isTriangulated = false
                        )
                    )
                }
            } finally {
                newCorners.release()
            }
        }

        val triCount = vioLandmarks.count { it.isTriangulated }
        val ms = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
        return MiniVioStepResult(
            updated = updatedCam,
            camPosEnu = outCam,
            activeLandmarks = vioLandmarks.size,
            triangulatedLandmarks = triCount,
            inlierLandmarks = inlierCount,
            latencyMs = ms
        )
    }

    /**
     * Legacy ground-plane KLT correspondence tracker retained for backward compatibility with
     * deterministic JVM unit tests and fallback callers.
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

        val hasPair = beginTrackingFrame(lumaBytes, widthPx, heightPx)
        if (!hasPair) return null

        val prevVo = prevVoGrayMat ?: return null
        val currVo = currVoGrayMat ?: return null
        val voW = prevVo.cols()
        val voH = prevVo.rows()
        val scaleToVoX = voW.toFloat() / widthPx.toFloat()
        val scaleToVoY = voH.toFloat() / heightPx.toFloat()
        val scaleFromVoX = widthPx.toDouble() / voW.toDouble()
        val scaleFromVoY = heightPx.toDouble() / voH.toDouble()

        val hasExclusionBox = excludeTargetRegion &&
            excludeLeftPx.isFinite() && excludeTopPx.isFinite() &&
            excludeRightPx.isFinite() && excludeBottomPx.isFinite() &&
            excludeRightPx > 0f && excludeBottomPx > 0f &&
            excludeLeftPx < widthPx.toFloat() && excludeTopPx < heightPx.toFloat()
        val exVoL = if (hasExclusionBox) (excludeLeftPx.finiteOr(0f) * scaleToVoX - 6.0) else Double.NaN
        val exVoT = if (hasExclusionBox) (excludeTopPx.finiteOr(0f) * scaleToVoY - 6.0) else Double.NaN
        val exVoR = if (hasExclusionBox) (excludeRightPx.finiteOr(0f) * scaleToVoX + 6.0) else Double.NaN
        val exVoB = if (hasExclusionBox) (excludeBottomPx.finiteOr(0f) * scaleToVoY + 6.0) else Double.NaN

        val reusableCorners: Array<Point>? = if (cachedVoCorners.size >= MIN_CACHED_VO_CORNERS) {
            if (!hasExclusionBox) {
                cachedVoCorners
            } else {
                val filtered = cachedVoCorners.filter { p ->
                    !(p.x in exVoL..exVoR && p.y in exVoT..exVoB)
                }
                if (filtered.size >= MIN_CACHED_VO_CORNERS) filtered.toTypedArray() else null
            }
        } else {
            null
        }

        val corners = MatOfPoint()
        val prevPts2f = MatOfPoint2f()
        val nextPts2f = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()
        var homography: Mat? = null
        val inlierMask = Mat()

        try {
            val cornerArr: Array<Point> = if (reusableCorners != null) {
                reusableCorners
            } else {
                val detW = max(32, voW / 2)
                val detH = max(18, voH / 2)
                val scaleToDetX = detW.toFloat() / widthPx.toFloat()
                val scaleToDetY = detH.toFloat() / heightPx.toFloat()
                val detPrev = prevOrAllocMat(prevVoDetectMat, detH, detW, CvType.CV_8UC1).also {
                    prevVoDetectMat = it
                }
                Imgproc.resize(prevVo, detPrev, Size(detW.toDouble(), detH.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)

                val mask = prevOrAllocMat(voMaskMat, detH, detW, CvType.CV_8UC1).also {
                    voMaskMat = it
                }
                mask.setTo(Scalar(255.0))
                if (hasExclusionBox) {
                    val rawExL = (excludeLeftPx.finiteOr(0f) * scaleToDetX).roundToInt() - 4
                    val rawExT = (excludeTopPx.finiteOr(0f) * scaleToDetY).roundToInt() - 4
                    val rawExR = (excludeRightPx.finiteOr(0f) * scaleToDetX).roundToInt() + 4
                    val rawExB = (excludeBottomPx.finiteOr(0f) * scaleToDetY).roundToInt() + 4
                    val clampedL = rawExL.coerceIn(0, (detW - 2).coerceAtLeast(0))
                    val clampedT = rawExT.coerceIn(0, (detH - 2).coerceAtLeast(0))
                    val clampedR = rawExR.coerceIn(clampedL, detW)
                    val clampedB = rawExB.coerceIn(clampedT, detH)
                    if (clampedR - clampedL >= 2 && clampedB - clampedT >= 2) {
                        val safeRect = computeSafeSubmatRect(
                            cols = detW,
                            rows = detH,
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

                Imgproc.goodFeaturesToTrack(
                    detPrev, corners, MAX_VO_GRAVEL_CORNERS, 0.02, 6.0, mask, 3
                )
                val rawCornerArr = corners.toArray()
                if (rawCornerArr.size < MIN_VO_INLIERS) {
                    cachedVoCorners = emptyArray()
                    return null
                }
                val detToVoX = voW.toDouble() / detW.toDouble()
                val detToVoY = voH.toDouble() / detH.toDouble()
                Array(rawCornerArr.size) { idx ->
                    val c = rawCornerArr[idx]
                    Point(c.x * detToVoX, c.y * detToVoY)
                }
            }

            prevPts2f.fromArray(*cornerArr)
            Video.calcOpticalFlowPyrLK(
                prevVo, currVo, prevPts2f, nextPts2f, status, err, Size(11.0, 11.0), 3
            )

            val nextArr = nextPts2f.toArray()
            val statusArr = status.toArray()
            val errArr = err.toArray()
            val trackedPrev = ArrayList<Point>(cornerArr.size)
            val trackedCurr = ArrayList<Point>(cornerArr.size)
            for (i in cornerArr.indices) {
                if (i < statusArr.size && statusArr[i].toInt() == 1 &&
                    (i >= errArr.size || errArr[i] < 28.0f)
                ) {
                    val p = nextArr[i]
                    if (p.x in 2.0..(voW - 3.0) && p.y in 2.0..(voH - 3.0) &&
                        !(hasExclusionBox && p.x in exVoL..exVoR && p.y in exVoT..exVoB)
                    ) {
                        trackedPrev.add(cornerArr[i])
                        trackedCurr.add(p)
                    }
                }
            }

            if (trackedPrev.size < MIN_VO_INLIERS) {
                cachedVoCorners = emptyArray()
                return null
            }

            val srcMat = MatOfPoint2f(*trackedPrev.toTypedArray())
            val dstMat = MatOfPoint2f(*trackedCurr.toTypedArray())
            try {
                homography = Calib3d.findHomography(srcMat, dstMat, Calib3d.RANSAC, 2.5, inlierMask)
                if (homography == null || homography.empty() || inlierMask.empty()) {
                    cachedVoCorners = emptyArray()
                    return null
                }
                val maskBytes = ByteArray(inlierMask.rows() * inlierMask.cols())
                inlierMask.get(0, 0, maskBytes)
                var count = 0
                for (b in maskBytes) if (b.toInt() != 0) count++
                if (count < MIN_VO_INLIERS) {
                    cachedVoCorners = emptyArray()
                    return null
                }

                val prevXY = FloatArray(count * 2)
                val currXY = FloatArray(count * 2)
                val survivingCurr = ArrayList<Point>(count)
                var outIdx = 0
                for (i in maskBytes.indices) {
                    if (maskBytes[i].toInt() != 0) {
                        val currPt = trackedCurr[i]
                        survivingCurr.add(currPt)
                        prevXY[outIdx] = (trackedPrev[i].x * scaleFromVoX).toFloat()
                        prevXY[outIdx + 1] = (trackedPrev[i].y * scaleFromVoY).toFloat()
                        currXY[outIdx] = (currPt.x * scaleFromVoX).toFloat()
                        currXY[outIdx + 1] = (currPt.y * scaleFromVoY).toFloat()
                        outIdx += 2
                    }
                }
                cachedVoCorners = survivingCurr.toTypedArray()
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
     * **Tier 3:** Extracts `ORB` keypoints/descriptors over a `3.0x` target-diameter collar plus a
     * `32x32` zero-mean unit-norm NCC target patch and inserts it into the Multi-Keyframe Bank
     * (preserving Keyframe 0 permanently and retaining up to [MAX_COLLAR_KEYFRAMES] viewpoints).
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
        val fullMat = prepareFullGrayMat(lumaBytes, widthPx, heightPx)

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

        val detector = ensureOrbCollar()
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
            val coords = Array(kpArray.size) { idx ->
                val p = kpArray[idx].pt
                Point(p.x * scaleXFromCanonical, p.y * scaleYFromCanonical)
            }
            val nccPatch = extractNormalizedNccPatch32(
                grayMat = fullMat,
                boxLeftFull = sL,
                boxTopFull = sT,
                boxRightFull = sR,
                boxBottomFull = sB,
                fullWidthPx = widthPx,
                fullHeightPx = heightPx
            )
            val newKf = CollarKeyframe(
                descriptors = descriptors,
                keypointCoords = coords,
                collarWidthPx = effectiveCollarW,
                collarHeightPx = effectiveCollarH,
                stoneLocalLeft = sL - collarLeft,
                stoneLocalTop = sT - collarTop,
                stoneLocalRight = sR - collarLeft,
                stoneLocalBottom = sB - collarTop,
                nccPatch32 = nccPatch,
                rWorldFromScr = currentImuRotationScr?.copyOf(9),
                baseScale = baseScale.finiteOr(1.0f)
            )
            if (collarKeyframeBank.size >= MAX_COLLAR_KEYFRAMES) {
                // Always preserve Keyframe 0 (initial user-verified anchor), evict oldest intermediate
                val evicted = collarKeyframeBank.removeAt(1)
                evicted.release()
            }
            collarKeyframeBank.add(newKf)
            lastMatchedKeyframeIndex = collarKeyframeBank.lastIndex
            latestRelativeScale = 1.0f
            hasSeededConstellation = true
            return true
        } finally {
            collarSub.release()
            keypoints.release()
        }
    }

    /**
     * **Tier 3 (`< 3.0 ms` CPU):** Matches a live `1920x1080` luminance frame against the Multi-Keyframe
     * `ORB` collar bank using `BFMatcher(NORM_HAMMING)` + Lowe `0.78` + `estimateAffinePartial2D`
     * RANSAC, strictly verified by the **4-Gate Verifier**:
     * 1. Inlier count (`>= 12`) & destination spatial spread (`std >= 6%` search ROI).
     * 2. Scale/aspect physical bounds (`0.38..2.65`).
     * 3. IMU in-plane roll consistency (`<= 25 deg`) when IMU orientation is available.
     * 4. `32x32` zero-mean NCC target appearance check (`>= 0.40f`, rejecting blank walls `std < 3.5`).
     */
    open fun matchConstellationFrame(
        lumaBytes: ByteArray?,
        widthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        heightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX,
        predictedCenterXPx: Float,
        predictedCenterYPx: Float
    ): ConstellationMatchResult {
        val startNs = SystemClock.elapsedRealtimeNanos()
        if (!hasSeededConstellation || collarKeyframeBank.isEmpty() || !ensureOpenCvInitialized()) {
            return ConstellationMatchResult(matched = false, inliers = 0)
        }
        if (widthPx < 4 || heightPx < 4) {
            return ConstellationMatchResult(matched = false, inliers = 0)
        }
        val fullMat = prepareFullGrayMat(lumaBytes, widthPx, heightPx)

        val primaryKf = collarKeyframeBank[lastMatchedKeyframeIndex.coerceIn(0, collarKeyframeBank.lastIndex)]
        val maxSquare = min(widthPx, heightPx).coerceAtLeast(2)
        val searchSide = (primaryKf.collarWidthPx * latestRelativeScale.finiteOr(1f).coerceIn(0.75f, 1.95f) * 1.20f)
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

        val detector = ensureOrbCollar()
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

            val searchScaleXFromCanonical = effectiveSearchW.toDouble() / CANONICAL_PATCH_PX.toDouble()
            val searchScaleYFromCanonical = effectiveSearchH.toDouble() / CANONICAL_PATCH_PX.toDouble()
            val liveFullCoords = Array(liveKpArr.size) { idx ->
                val p = liveKpArr[idx].pt
                Point(searchLeft + p.x * searchScaleXFromCanonical, searchTop + p.y * searchScaleYFromCanonical)
            }

            // Try the last matched keyframe first (fast path, ~1.5 ms), then fall back to remaining
            // keyframes in the bank if the first doesn't reach strong lock!
            val candidateOrder = buildKeyframeSearchOrder(
                size = collarKeyframeBank.size,
                preferredIdx = lastMatchedKeyframeIndex
            )
            var bestResult: ConstellationMatchResult? = null
            val liveImuR = currentImuRotationScr

            for (kfIdx in candidateOrder) {
                val kf = collarKeyframeBank[kfIdx]
                val cand = matchSingleCollarKeyframe4Gate(
                    kf = kf,
                    kfIndex = kfIdx,
                    fullMat = fullMat,
                    widthPx = widthPx,
                    heightPx = heightPx,
                    effectiveSearchW = effectiveSearchW,
                    safePredX = safePredX,
                    safePredY = safePredY,
                    liveDescriptors = liveDescriptors,
                    liveFullCoords = liveFullCoords,
                    matcher = matcher,
                    liveImuR = liveImuR
                )
                if (cand != null) {
                    if (bestResult == null || cand.inliers > bestResult.inliers) {
                        bestResult = cand
                    }
                    // Fast early exit when we hit a strong verified match (>= 18 inliers & NCC >= 0.55)
                    if (cand.inliers >= 18 && cand.nccScore >= 0.55f) {
                        break
                    }
                }
            }

            val winner = bestResult
            if (winner == null) {
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                return ConstellationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
            }

            lastMatchedKeyframeIndex = winner.matchedKeyframeIndex.coerceIn(0, collarKeyframeBank.lastIndex)
            val matchedKf = collarKeyframeBank[lastMatchedKeyframeIndex]
            val relScale = (winner.cumulativeScale / matchedKf.baseScale.coerceAtLeast(0.1f))
                .coerceIn(MIN_RELATIVE_SCALE, MAX_RELATIVE_SCALE)
            latestRelativeScale = relScale

            // Rolling multi-keyframe bank expansion on significant scale or viewpoint angle change
            val projCx = (winner.frameLeftPx + winner.frameRightPx) * 0.5f
            val projCy = (winner.frameTopPx + winner.frameBottomPx) * 0.5f
            val needsScaleKf = relScale >= ROLLING_KEYFRAME_GROWTH_RATIO || relScale <= ROLLING_KEYFRAME_SHRINK_RATIO
            val needsAngleKf = liveImuR != null && isNewViewpointAngleForCollarBank(liveImuR)
            if ((needsScaleKf || needsAngleKf) &&
                projCx in 64f..(widthPx - 64f) &&
                projCy in 64f..(heightPx - 64f) &&
                winner.nccScore >= 0.50f
            ) {
                seedConstellationCollar(
                    lumaBytes = null,
                    widthPx = widthPx,
                    heightPx = heightPx,
                    stoneLeftPx = winner.frameLeftPx,
                    stoneTopPx = winner.frameTopPx,
                    stoneRightPx = winner.frameRightPx,
                    stoneBottomPx = winner.frameBottomPx,
                    baseScale = winner.cumulativeScale
                )
            }

            val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
            return winner.copy(latencyMs = elapsedMs)
        } finally {
            searchSub.release()
            liveKeypoints.release()
            liveDescriptors.release()
        }
    }

    /**
     * **Tier 3:** Captures a `480x270` (`1/4`-scale) full-frame `ORB` relocalization snapshot
     * (top `260` keypoints + `32x32` zero-mean NCC patch + IMU orientation) into the
     * multi-keyframe relocalization bank (up to [MAX_RELOC_KEYFRAMES]).
     */
    open fun seedFullFrameRelocalizationSnapshot(
        lumaBytes: ByteArray?,
        widthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        heightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX,
        stoneCenterXPx: Float,
        stoneCenterYPx: Float
    ): Boolean {
        val activeCollar = collarKeyframeBank.lastOrNull()
        val estW = activeCollar?.let { (it.stoneLocalRight - it.stoneLocalLeft).coerceIn(28f, 420f) } ?: 96f
        val estH = activeCollar?.let { (it.stoneLocalBottom - it.stoneLocalTop).coerceIn(28f, 420f) } ?: 96f
        return seedFullFrameRelocalizationSnapshotWithBox(
            lumaBytes = lumaBytes,
            widthPx = widthPx,
            heightPx = heightPx,
            stoneCenterXPx = stoneCenterXPx,
            stoneCenterYPx = stoneCenterYPx,
            stoneWidthPx = estW,
            stoneHeightPx = estH
        )
    }

    /**
     * Captures a `480x270` full-frame `ORB` relocalization keyframe with explicit target box width/height
     * for exact Gate 4 `32x32` NCC verification during post-occlusion re-entry.
     */
    open fun seedFullFrameRelocalizationSnapshotWithBox(
        lumaBytes: ByteArray?,
        widthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        heightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX,
        stoneCenterXPx: Float,
        stoneCenterYPx: Float,
        stoneWidthPx: Float,
        stoneHeightPx: Float
    ): Boolean {
        if (!ensureOpenCvInitialized()) return false
        if (widthPx < 64 || heightPx < 36) return false
        val fullMat = prepareFullGrayMat(lumaBytes, widthPx, heightPx)

        val relocW = max(64, widthPx / RELOC_DOWNSCALE_DIVISOR)
        val relocH = max(36, heightPx / RELOC_DOWNSCALE_DIVISOR)
        val rMat = prevOrAllocMat(relocGrayMat, relocH, relocW, CvType.CV_8UC1).also {
            relocGrayMat = it
        }
        val maskMat = ensureEmptyMaskMat()
        val detector = ensureOrbReloc()
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

            val storedDescriptors: Mat
            val storedCoords: Array<Point>
            if (kpArray.size <= MAX_RELOC_KEYPOINTS) {
                storedDescriptors = descriptors
                storedCoords = Array(kpArray.size) { idx ->
                    val p = kpArray[idx].pt
                    Point(p.x * scaleFromRelocX, p.y * scaleFromRelocY)
                }
            } else {
                val topIndices = kpArray.indices
                    .sortedByDescending { kpArray[it].response }
                    .take(MAX_RELOC_KEYPOINTS)
                val trimmedDesc = Mat(MAX_RELOC_KEYPOINTS, descriptors.cols(), descriptors.type())
                val rowBytes = ByteArray(descriptors.cols() * descriptors.elemSize().toInt())
                storedCoords = Array(MAX_RELOC_KEYPOINTS) { outRow ->
                    val srcIdx = topIndices[outRow]
                    descriptors.get(srcIdx, 0, rowBytes)
                    trimmedDesc.put(outRow, 0, rowBytes)
                    val p = kpArray[srcIdx].pt
                    Point(p.x * scaleFromRelocX, p.y * scaleFromRelocY)
                }
                descriptors.release()
                storedDescriptors = trimmedDesc
            }

            val safeCx = stoneCenterXPx.finiteOr(widthPx * 0.5f)
            val safeCy = stoneCenterYPx.finiteOr(heightPx * 0.5f)
            val safeW = stoneWidthPx.finiteOr(96f).coerceIn(24f, widthPx * 0.6f)
            val safeH = stoneHeightPx.finiteOr(96f).coerceIn(24f, heightPx * 0.6f)
            val nccPatch = extractNormalizedNccPatch32(
                grayMat = fullMat,
                boxLeftFull = safeCx - safeW * 0.5f,
                boxTopFull = safeCy - safeH * 0.5f,
                boxRightFull = safeCx + safeW * 0.5f,
                boxBottomFull = safeCy + safeH * 0.5f,
                fullWidthPx = widthPx,
                fullHeightPx = heightPx
            )

            val newRelocKf = RelocKeyframe(
                descriptors = storedDescriptors,
                keypointCoordsFull = storedCoords,
                stoneCenterFull = Point(safeCx.toDouble(), safeCy.toDouble()),
                stoneWidthFull = safeW,
                stoneHeightFull = safeH,
                nccPatch32 = nccPatch,
                rWorldFromScr = currentImuRotationScr?.copyOf(9)
            )
            if (relocKeyframeBank.size >= MAX_RELOC_KEYFRAMES) {
                val evicted = relocKeyframeBank.removeAt(1)
                evicted.release()
            }
            relocKeyframeBank.add(newRelocKf)
            hasRelocalizationSnapshot = true
            return true
        } finally {
            keypoints.release()
        }
    }

    /**
     * **Tier 3 (`~5 Hz` on occlusion/off-screen re-entry):** Matches the live `480x270` frame
     * against the multi-keyframe full-frame `ORB` relocalization bank with strict **4-Gate
     * Verification** (inlier spatial spread, SVD singular-value ratio `<= 2.4` + perspective bound,
     * IMU in-plane roll consistency `<= 25 deg`, and `32x32` zero-mean NCC `>= 0.40f`).
     */
    open fun matchFullFrameRelocalization(
        lumaBytes: ByteArray?,
        widthPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_WIDTH_PX,
        heightPx: Int = StripRoiGeometry.LANDSCAPE_FRAME_HEIGHT_PX
    ): RelocalizationMatchResult {
        val startNs = SystemClock.elapsedRealtimeNanos()
        if (!hasRelocalizationSnapshot || relocKeyframeBank.isEmpty() || !ensureOpenCvInitialized()) {
            return RelocalizationMatchResult(matched = false, inliers = 0)
        }
        if (widthPx < 64 || heightPx < 36) {
            return RelocalizationMatchResult(matched = false, inliers = 0)
        }

        val fullMat = prepareFullGrayMat(lumaBytes, widthPx, heightPx)
        val relocW = max(64, widthPx / RELOC_DOWNSCALE_DIVISOR)
        val relocH = max(36, heightPx / RELOC_DOWNSCALE_DIVISOR)
        val rMat = prevOrAllocMat(relocGrayMat, relocH, relocW, CvType.CV_8UC1).also {
            relocGrayMat = it
        }
        val maskMat = ensureEmptyMaskMat()
        val detector = ensureOrbReloc()
        val matcher = ensureBfMatcher()
        val liveKeypoints = MatOfKeyPoint()
        val liveDescriptors = Mat()

        try {
            Imgproc.resize(fullMat, rMat, Size(relocW.toDouble(), relocH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            detector.detectAndCompute(rMat, maskMat, liveKeypoints, liveDescriptors)
            val liveKpArr = liveKeypoints.toArray()
            if (liveKpArr.size < MIN_RELOC_INLIERS || liveDescriptors.empty()) {
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
                return RelocalizationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
            }

            val scaleFromRelocX = widthPx.toDouble() / relocW.toDouble()
            val scaleFromRelocY = heightPx.toDouble() / relocH.toDouble()
            val liveFullCoords = Array(liveKpArr.size) { idx ->
                val p = liveKpArr[idx].pt
                Point(p.x * scaleFromRelocX, p.y * scaleFromRelocY)
            }

            var bestReloc: RelocalizationMatchResult? = null
            val liveImuR = currentImuRotationScr
            // Check newest keyframe first, then Keyframe 0 and remaining bank entries
            for (idx in relocKeyframeBank.indices.reversed()) {
                val rkf = relocKeyframeBank[idx]
                val cand = matchSingleRelocKeyframe4Gate(
                    rkf = rkf,
                    fullMat = fullMat,
                    widthPx = widthPx,
                    heightPx = heightPx,
                    liveDescriptors = liveDescriptors,
                    liveFullCoords = liveFullCoords,
                    matcher = matcher,
                    liveImuR = liveImuR
                )
                if (cand != null && (bestReloc == null || cand.inliers > bestReloc.inliers)) {
                    bestReloc = cand
                    if (cand.inliers >= 22 && cand.nccScore >= 0.55f) break
                }
            }

            val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000f
            return bestReloc?.copy(latencyMs = elapsedMs)
                ?: RelocalizationMatchResult(matched = false, inliers = 0, latencyMs = elapsedMs)
        } finally {
            liveKeypoints.release()
            liveDescriptors.release()
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
        prevVoDetectMat?.release()
        prevVoDetectMat = null
        voMaskMat?.release()
        voMaskMat = null
        vioMaskMat?.release()
        vioMaskMat = null
        relocGrayMat?.release()
        relocGrayMat = null
        canonicalPatchMat?.release()
        canonicalPatchMat = null
        nccPatchMat?.release()
        nccPatchMat = null
        emptyMaskMat?.release()
        emptyMaskMat = null
    }

    private fun matchSingleCollarKeyframe4Gate(
        kf: CollarKeyframe,
        kfIndex: Int,
        fullMat: Mat,
        widthPx: Int,
        heightPx: Int,
        effectiveSearchW: Int,
        safePredX: Float,
        safePredY: Float,
        liveDescriptors: Mat,
        liveFullCoords: Array<Point>,
        matcher: BFMatcher,
        liveImuR: FloatArray?
    ): ConstellationMatchResult? {
        if (kf.descriptors.empty() || kf.keypointCoords.size < MIN_SEED_KEYPOINTS) return null

        val knnMatches = ArrayList<MatOfDMatch>()
        var affine: Mat? = null
        val inlierMask = Mat()

        try {
            matcher.knnMatch(kf.descriptors, liveDescriptors, knnMatches, 2)
            val goodSrc = ArrayList<Point>(knnMatches.size)
            val goodDst = ArrayList<Point>(knnMatches.size)
            for (pairMat in knnMatches) {
                val arr = pairMat.toArray()
                if (arr.size >= 2 && arr[0].distance < LOWE_RATIO_THRESHOLD * arr[1].distance) {
                    val qIdx = arr[0].queryIdx
                    val tIdx = arr[0].trainIdx
                    if (qIdx in kf.keypointCoords.indices && tIdx in liveFullCoords.indices) {
                        goodSrc.add(kf.keypointCoords[qIdx])
                        goodDst.add(liveFullCoords[tIdx])
                    }
                }
            }
            if (goodSrc.size < MIN_HOMOGRAPHY_INLIERS) return null

            val srcPtsMat = MatOfPoint2f(*goodSrc.toTypedArray())
            val dstPtsMat = MatOfPoint2f(*goodDst.toTypedArray())
            try {
                affine = Calib3d.estimateAffinePartial2D(
                    srcPtsMat,
                    dstPtsMat,
                    inlierMask,
                    Calib3d.RANSAC,
                    MAGSAC_REPROJ_THRESHOLD_PX
                )
                if (affine == null || affine.empty() || affine.rows() < 2 || affine.cols() < 3 || inlierMask.empty()) {
                    return null
                }

                val maskBytes = ByteArray(inlierMask.rows() * inlierMask.cols())
                inlierMask.get(0, 0, maskBytes)
                var inliers = 0
                var sumX = 0.0
                var sumY = 0.0
                for (i in maskBytes.indices) {
                    if (maskBytes[i].toInt() != 0) {
                        inliers++
                        sumX += goodDst[i].x
                        sumY += goodDst[i].y
                    }
                }
                if (inliers < MIN_HOMOGRAPHY_INLIERS) return null

                // GATE 1: Inlier spatial spread (reject clustered glint/line matches)
                val meanX = sumX / inliers
                val meanY = sumY / inliers
                var varSum = 0.0
                for (i in maskBytes.indices) {
                    if (maskBytes[i].toInt() != 0) {
                        val dx = goodDst[i].x - meanX
                        val dy = goodDst[i].y - meanY
                        varSum += dx * dx + dy * dy
                    }
                }
                val spreadPx = sqrt(varSum / inliers).toFloat()
                if (spreadPx < effectiveSearchW * MIN_INLIER_SPREAD_FRACTION) return null

                // GATE 2: Similarity matrix scale & aspect conditioning
                val m = DoubleArray(6)
                affine.get(0, 0, m)
                val scaleX = hypot(m[0], m[3]).toFloat()
                val scaleY = hypot(m[1], m[4]).toFloat()
                val relScale = sqrt((scaleX * scaleY).coerceAtLeast(0f))
                if (!relScale.isFinite() ||
                    relScale !in MIN_RELATIVE_SCALE..MAX_RELATIVE_SCALE ||
                    (scaleX / scaleY.coerceAtLeast(1e-3f)) !in 0.78f..1.28f
                ) {
                    return null
                }

                // GATE 3: IMU in-plane roll consistency (when both KF and live IMU matrices exist)
                if (kf.rWorldFromScr != null && liveImuR != null) {
                    val rRelCv = computeRelativeCvRotation(kf.rWorldFromScr, liveImuR)
                    val expectedRollDeg = Math.toDegrees(atan2(rRelCv[3], rRelCv[0])).toFloat()
                    val obsRollDeg = Math.toDegrees(atan2(m[3], m[0])).toFloat()
                    if (abs(wrapAngleDegrees(obsRollDeg - expectedRollDeg)) > MAX_GATE3_IMU_ROLL_ERR_DEG) {
                        return null
                    }
                }

                val cxLocal = (kf.stoneLocalLeft + kf.stoneLocalRight) * 0.5
                val cyLocal = (kf.stoneLocalTop + kf.stoneLocalBottom) * 0.5
                val projCx = (m[0] * cxLocal + m[1] * cyLocal + m[2]).toFloat()
                val projCy = (m[3] * cxLocal + m[4] * cyLocal + m[5]).toFloat()
                if (!projCx.isFinite() || !projCy.isFinite()) return null

                val jumpDistPx = hypot(projCx - safePredX, projCy - safePredY)
                val maxJumpPx = max(120f, effectiveSearchW * 0.34f)
                if (jumpDistPx > maxJumpPx) return null

                val refW = (kf.stoneLocalRight - kf.stoneLocalLeft).coerceAtLeast(8f)
                val refH = (kf.stoneLocalBottom - kf.stoneLocalTop).coerceAtLeast(8f)
                val halfW = refW * 0.5f * relScale
                val halfH = refH * 0.5f * relScale
                val minX = projCx - halfW
                val maxX = projCx + halfW
                val minY = projCy - halfH
                val maxY = projCy + halfH

                // GATE 4: 32x32 zero-mean NCC target appearance verification
                var nccScore = 1.0f
                if (kf.nccPatch32 != null) {
                    val livePatch = extractNormalizedNccPatch32(
                        grayMat = fullMat,
                        boxLeftFull = minX,
                        boxTopFull = minY,
                        boxRightFull = maxX,
                        boxBottomFull = maxY,
                        fullWidthPx = widthPx,
                        fullHeightPx = heightPx
                    ) ?: return null
                    nccScore = max(dotNcc(livePatch, kf.nccPatch32), computeBestBankNccScore(livePatch))
                    if (nccScore < MIN_GATE4_NCC_SCORE) return null
                }

                val cumulativeScale = (kf.baseScale * relScale).coerceIn(0.25f, 12.0f)
                return ConstellationMatchResult(
                    matched = true,
                    inliers = inliers,
                    frameLeftPx = minX,
                    frameTopPx = minY,
                    frameRightPx = maxX,
                    frameBottomPx = maxY,
                    cumulativeScale = cumulativeScale,
                    nccScore = nccScore,
                    matchedKeyframeIndex = kfIndex
                )
            } finally {
                srcPtsMat.release()
                dstPtsMat.release()
            }
        } finally {
            for (m in knnMatches) m.release()
            affine?.release()
            inlierMask.release()
        }
    }

    private fun matchSingleRelocKeyframe4Gate(
        rkf: RelocKeyframe,
        fullMat: Mat,
        widthPx: Int,
        heightPx: Int,
        liveDescriptors: Mat,
        liveFullCoords: Array<Point>,
        matcher: BFMatcher,
        liveImuR: FloatArray?
    ): RelocalizationMatchResult? {
        if (rkf.descriptors.empty() || rkf.keypointCoordsFull.size < MIN_RELOC_SEED_KEYPOINTS) return null

        val knnMatches = ArrayList<MatOfDMatch>()
        var homography: Mat? = null
        val inlierMask = Mat()

        try {
            matcher.knnMatch(rkf.descriptors, liveDescriptors, knnMatches, 2)
            val goodSrc = ArrayList<Point>(knnMatches.size)
            val goodDst = ArrayList<Point>(knnMatches.size)
            for (pairMat in knnMatches) {
                val arr = pairMat.toArray()
                if (arr.size >= 2 && arr[0].distance < LOWE_RATIO_THRESHOLD * arr[1].distance) {
                    val qIdx = arr[0].queryIdx
                    val tIdx = arr[0].trainIdx
                    if (qIdx in rkf.keypointCoordsFull.indices && tIdx in liveFullCoords.indices) {
                        goodSrc.add(rkf.keypointCoordsFull[qIdx])
                        goodDst.add(liveFullCoords[tIdx])
                    }
                }
            }
            if (goodSrc.size < MIN_RELOC_INLIERS) return null

            val srcPtsMat = MatOfPoint2f(*goodSrc.toTypedArray())
            val dstPtsMat = MatOfPoint2f(*goodDst.toTypedArray())
            val srcCenterMat = MatOfPoint2f(rkf.stoneCenterFull)
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
                    return null
                }

                val maskBytes = ByteArray(inlierMask.rows() * inlierMask.cols())
                inlierMask.get(0, 0, maskBytes)
                var inliers = 0
                var sumX = 0.0
                var sumY = 0.0
                for (i in maskBytes.indices) {
                    if (maskBytes[i].toInt() != 0) {
                        inliers++
                        sumX += goodDst[i].x
                        sumY += goodDst[i].y
                    }
                }
                if (inliers < MIN_RELOC_INLIERS) return null

                // GATE 1: Inlier spatial spread across frame
                val meanX = sumX / inliers
                val meanY = sumY / inliers
                var varSum = 0.0
                for (i in maskBytes.indices) {
                    if (maskBytes[i].toInt() != 0) {
                        val dx = goodDst[i].x - meanX
                        val dy = goodDst[i].y - meanY
                        varSum += dx * dx + dy * dy
                    }
                }
                val spreadPx = sqrt(varSum / inliers).toFloat()
                if (spreadPx < widthPx * 0.06f) return null

                // GATE 2: Homography Jacobian conditioning (det, singular value ratio <= 2.4, perspective bound)
                val hVals = DoubleArray(9)
                homography.get(0, 0, hVals)
                val h22 = hVals[8]
                if (abs(h22) < 1e-6) return null
                val invH22 = 1.0 / h22
                val j00 = hVals[0] * invH22
                val j01 = hVals[1] * invH22
                val j10 = hVals[3] * invH22
                val j11 = hVals[4] * invH22
                val h20 = hVals[6] * invH22
                val h21 = hVals[7] * invH22

                if (!verifyHomography2x2Conditioning(j00, j01, j10, j11, h20, h21, widthPx)) {
                    return null
                }

                // GATE 3: IMU in-plane roll consistency
                if (rkf.rWorldFromScr != null && liveImuR != null) {
                    val rRelCv = computeRelativeCvRotation(rkf.rWorldFromScr, liveImuR)
                    val expectedRollDeg = Math.toDegrees(atan2(rRelCv[3], rRelCv[0])).toFloat()
                    val obsRollDeg = Math.toDegrees(atan2(j10, j00)).toFloat()
                    if (abs(wrapAngleDegrees(obsRollDeg - expectedRollDeg)) > MAX_GATE3_IMU_ROLL_ERR_DEG) {
                        return null
                    }
                }

                Core.perspectiveTransform(srcCenterMat, dstCenterMat, homography)
                val projPts = dstCenterMat.toArray()
                if (projPts.isEmpty()) return null
                val projX = projPts[0].x.toFloat()
                val projY = projPts[0].y.toFloat()
                if (!projX.isFinite() || !projY.isFinite() ||
                    projX !in -120f..(widthPx + 120f) ||
                    projY !in -120f..(heightPx + 120f)
                ) {
                    return null
                }

                val relScale = sqrt(abs(j00 * j11 - j01 * j10)).toFloat().coerceIn(0.4f, 2.5f)
                val projW = (rkf.stoneWidthFull * relScale).coerceIn(20f, widthPx * 0.7f)
                val projH = (rkf.stoneHeightFull * relScale).coerceIn(20f, heightPx * 0.7f)

                // GATE 4: 32x32 zero-mean NCC target appearance verification (kills blank-wall false locks!)
                var nccScore = 1.0f
                val refPatch = rkf.nccPatch32
                if (refPatch != null && projX in 0f..widthPx.toFloat() && projY in 0f..heightPx.toFloat()) {
                    val livePatch = extractNormalizedNccPatch32(
                        grayMat = fullMat,
                        boxLeftFull = projX - projW * 0.5f,
                        boxTopFull = projY - projH * 0.5f,
                        boxRightFull = projX + projW * 0.5f,
                        boxBottomFull = projY + projH * 0.5f,
                        fullWidthPx = widthPx,
                        fullHeightPx = heightPx
                    ) ?: return null
                    nccScore = max(dotNcc(livePatch, refPatch), computeBestBankNccScore(livePatch))
                    if (nccScore < MIN_GATE4_NCC_SCORE) return null
                }

                return RelocalizationMatchResult(
                    matched = true,
                    inliers = inliers,
                    stoneCenterXPx = projX,
                    stoneCenterYPx = projY,
                    stoneWidthPx = projW,
                    stoneHeightPx = projH,
                    nccScore = nccScore
                )
            } finally {
                srcPtsMat.release()
                dstPtsMat.release()
                srcCenterMat.release()
                dstCenterMat.release()
            }
        } finally {
            for (m in knnMatches) m.release()
            homography?.release()
            inlierMask.release()
        }
    }

    private fun extractNormalizedNccPatch32(
        grayMat: Mat,
        boxLeftFull: Float,
        boxTopFull: Float,
        boxRightFull: Float,
        boxBottomFull: Float,
        fullWidthPx: Int,
        fullHeightPx: Int
    ): FloatArray? {
        val matW = grayMat.cols()
        val matH = grayMat.rows()
        if (matW < 8 || matH < 8 || fullWidthPx < 8 || fullHeightPx < 8) return null

        val cx = (boxLeftFull + boxRightFull) * 0.5f
        val cy = (boxTopFull + boxBottomFull) * 0.5f
        // Include 1.25x rim padding so smooth agates retain sharp rim contrast in the 32x32 template
        val padHalfW = max(12f, (boxRightFull - boxLeftFull) * 0.5f * NCC_BOX_PADDING_MULT)
        val padHalfH = max(12f, (boxBottomFull - boxTopFull) * 0.5f * NCC_BOX_PADDING_MULT)

        val sx = matW.toFloat() / fullWidthPx.toFloat()
        val sy = matH.toFloat() / fullHeightPx.toFloat()
        val rect = computeSafeSubmatRect(
            cols = matW,
            rows = matH,
            rawX = ((cx - padHalfW) * sx).roundToInt(),
            rawY = ((cy - padHalfH) * sy).roundToInt(),
            rawW = max(4, (padHalfW * 2f * sx).roundToInt()),
            rawH = max(4, (padHalfH * 2f * sy).roundToInt())
        ) ?: return null

        val sub = grayMat.submat(rect)
        val resized = prevOrAllocMat(nccPatchMat, NCC_PATCH_SIZE, NCC_PATCH_SIZE, CvType.CV_8UC1).also {
            nccPatchMat = it
        }
        return try {
            Imgproc.resize(
                sub,
                resized,
                Size(NCC_PATCH_SIZE.toDouble(), NCC_PATCH_SIZE.toDouble()),
                0.0,
                0.0,
                Imgproc.INTER_LINEAR
            )
            resized.get(0, 0, nccByteScratch)
            normalizeZeroMeanUnitNcc(nccByteScratch)
        } finally {
            sub.release()
        }
    }

    private fun computeBestBankNccScore(livePatch: FloatArray): Float {
        var best = -1.0f
        for (kf in collarKeyframeBank) {
            val ref = kf.nccPatch32 ?: continue
            val score = dotNcc(livePatch, ref)
            if (score > best) best = score
        }
        for (rkf in relocKeyframeBank) {
            val ref = rkf.nccPatch32 ?: continue
            val score = dotNcc(livePatch, ref)
            if (score > best) best = score
        }
        return if (best < -0.99f) 1.0f else best
    }

    private fun isNewViewpointAngleForCollarBank(liveImuR: FloatArray): Boolean {
        if (collarKeyframeBank.isEmpty()) return true
        var minAngleDeg = Float.POSITIVE_INFINITY
        for (kf in collarKeyframeBank) {
            val r = kf.rWorldFromScr ?: continue
            val angle = LockOnPinpointController.rotationAngleBetweenMatricesDeg(r, liveImuR)
            if (angle < minAngleDeg) minAngleDeg = angle
        }
        return minAngleDeg.isFinite() && minAngleDeg >= NEW_KEYFRAME_VIEWPOINT_ANGLE_DEG
    }

    private fun uploadToVoGrayMat(
        lumaBytes: ByteArray,
        widthPx: Int,
        heightPx: Int,
        voW: Int,
        voH: Int,
        dstVo: Mat
    ) {
        if (widthPx == voW * VO_DOWNSCALE_DIVISOR && heightPx == voH * VO_DOWNSCALE_DIVISOR) {
            val voSize = voW * voH
            val subBuf = voSubsampleBuffer?.takeIf { it.size == voSize }
                ?: ByteArray(voSize).also { voSubsampleBuffer = it }
            var dstIdx = 0
            var srcCenterRow = widthPx
            val rowStep = widthPx * VO_DOWNSCALE_DIVISOR
            repeat(voH) {
                var sx = 1
                repeat(voW) {
                    subBuf[dstIdx++] = lumaBytes[srcCenterRow + sx]
                    sx += VO_DOWNSCALE_DIVISOR
                }
                srcCenterRow += rowStep
            }
            dstVo.put(0, 0, subBuf, 0, voSize)
        } else {
            val fullMat = prepareFullGrayMat(lumaBytes, widthPx, heightPx)
            Imgproc.resize(fullMat, dstVo, Size(voW.toDouble(), voH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        }
    }

    private fun solveHuberIrlsCameraCenter(
        landmarks: List<MiniVioLandmark>,
        currRays: List<FloatArray>,
        priorCamPosEnu: FloatArray,
        priorLambda: Double
    ): Pair<FloatArray, Int>? {
        val n = landmarks.size
        if (n < MIN_VIO_SOLVE_LANDMARKS) return null

        val cam = doubleArrayOf(
            priorCamPosEnu[0].toDouble(),
            priorCamPosEnu[1].toDouble(),
            priorCamPosEnu[2].toDouble()
        )
        var inliers = 0
        val huberDeltaM = 0.045

        repeat(3) {
            var a00 = priorLambda; var a01 = 0.0; var a02 = 0.0
            var a11 = priorLambda; var a12 = 0.0; var a22 = priorLambda
            var b0 = priorLambda * priorCamPosEnu[0].toDouble()
            var b1 = priorLambda * priorCamPosEnu[1].toDouble()
            var b2 = priorLambda * priorCamPosEnu[2].toDouble()
            inliers = 0

            for (i in 0 until n) {
                val lm = landmarks[i]
                val ray = currRays[i]
                val dx = ray[0].toDouble()
                val dy = ray[1].toDouble()
                val dz = ray[2].toDouble()
                val gx = lm.worldPosEnu[0].toDouble()
                val gy = lm.worldPosEnu[1].toDouble()
                val gz = lm.worldPosEnu[2].toDouble()

                // Orthogonal distance from landmark X_i to ray (cam + t * ray)
                val vx = gx - cam[0]
                val vy = gy - cam[1]
                val vz = gz - cam[2]
                val proj = vx * dx + vy * dy + vz * dz
                if (proj <= 0.12) continue
                val ex = vx - proj * dx
                val ey = vy - proj * dy
                val ez = vz - proj * dz
                val residM = sqrt(ex * ex + ey * ey + ez * ez)
                val baseWeight = if (lm.isTriangulated) 1.5 else 0.75
                val huberW = if (residM <= huberDeltaM) 1.0 else (huberDeltaM / residM)
                val w = baseWeight * huberW
                if (residM <= huberDeltaM * 2.5) inliers++

                val p00 = (1.0 - dx * dx) * w
                val p01 = (-dx * dy) * w
                val p02 = (-dx * dz) * w
                val p11 = (1.0 - dy * dy) * w
                val p12 = (-dy * dz) * w
                val p22 = (1.0 - dz * dz) * w

                a00 += p00; a01 += p01; a02 += p02
                a11 += p11; a12 += p12; a22 += p22

                b0 += p00 * gx + p01 * gy + p02 * gz
                b1 += p01 * gx + p11 * gy + p12 * gz
                b2 += p02 * gx + p12 * gy + p22 * gz
            }

            val sol = solveSymmetric3x3(a00, a01, a02, a11, a12, a22, b0, b1, b2) ?: return null
            cam[0] = sol[0]
            cam[1] = sol[1]
            cam[2] = sol[2]
        }

        if (inliers < MIN_VIO_SOLVE_LANDMARKS) return null
        return Pair(floatArrayOf(cam[0].toFloat(), cam[1].toFloat(), cam[2].toFloat()), inliers)
    }

    private fun prepareFullGrayMat(lumaBytes: ByteArray?, widthPx: Int, heightPx: Int): Mat {
        val fullMat = ensureFullGrayMat(widthPx, heightPx)
        if (lumaBytes != null && lumaBytes.size >= widthPx * heightPx) {
            fullMat.put(0, 0, lumaBytes, 0, widthPx * heightPx)
            latestFullLumaRef = lumaBytes
            latestFullWidthPx = widthPx
            latestFullHeightPx = heightPx
            fullMatDirty = false
        } else if (fullMatDirty) {
            val cachedLuma = latestFullLumaRef
            if (cachedLuma != null &&
                latestFullWidthPx == widthPx &&
                latestFullHeightPx == heightPx &&
                cachedLuma.size >= widthPx * heightPx
            ) {
                fullMat.put(0, 0, cachedLuma, 0, widthPx * heightPx)
            }
            fullMatDirty = false
        }
        return fullMat
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

    private fun ensureOrbCollar(): ORB {
        return orbCollar ?: ORB.create(
            360,
            1.2f,
            6,
            15,
            0,
            2,
            ORB.HARRIS_SCORE,
            15,
            12
        ).also { orbCollar = it }
    }

    private fun ensureOrbReloc(): ORB {
        return orbReloc ?: ORB.create(
            550,
            1.2f,
            8,
            15,
            0,
            2,
            ORB.HARRIS_SCORE,
            15,
            10
        ).also { orbReloc = it }
    }

    private fun ensureBfMatcher(): BFMatcher {
        return bfMatcher ?: BFMatcher.create(Core.NORM_HAMMING, false).also { bfMatcher = it }
    }

    private fun buildNormalizedProjectionMat3x4(
        rCvFromWorld: FloatArray,
        camPosEnu: FloatArray
    ): Mat {
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

        /** Gravel-constellation context collar multiplier (`3.0x` stone diameter). */
        const val COLLAR_DIAMETER_MULTIPLIER = 3.0f
        const val CANONICAL_PATCH_PX = 192
        const val MIN_COLLAR_SIZE_PX = 256
        const val MAX_COLLAR_SIZE_PX = 576
        const val MAX_SEARCH_SIZE_PX = 680

        /** Minimum RANSAC similarity inliers required to hold Tier 3 collar lock. */
        const val MIN_HOMOGRAPHY_INLIERS = 12
        const val MIN_SEED_KEYPOINTS = 16

        /** Lowe second-nearest-neighbor descriptor distance ratio (`0.78f`). */
        const val LOWE_RATIO_THRESHOLD = 0.78f
        const val MAGSAC_REPROJ_THRESHOLD_PX = 3.5

        /** Multi-keyframe bank capacities & expansion thresholds. */
        const val MAX_COLLAR_KEYFRAMES = 6
        const val MAX_RELOC_KEYFRAMES = 4
        const val ROLLING_KEYFRAME_GROWTH_RATIO = 1.25f
        const val ROLLING_KEYFRAME_SHRINK_RATIO = 0.80f
        const val NEW_KEYFRAME_VIEWPOINT_ANGLE_DEG = 18.0f
        private const val MIN_RELATIVE_SCALE = 0.35f
        private const val MAX_RELATIVE_SCALE = 3.50f

        /** 4-Gate Geometric + Appearance Verification constants. */
        const val NCC_PATCH_SIZE = 32
        private const val NCC_BOX_PADDING_MULT = 1.25f
        const val MIN_NCC_PATCH_STD = 3.5f
        const val MIN_GATE4_NCC_SCORE = 0.40f
        const val MIN_LOCAL_KLT_NCC_SCORE = 0.25f
        const val MAX_GATE3_IMU_ROLL_ERR_DEG = 25.0f
        const val MAX_LOCAL_KLT_ROLL_ERR_DEG = 12.0f
        const val MIN_INLIER_SPREAD_FRACTION = 0.06f
        const val MAX_HOMOGRAPHY_SV_RATIO = 2.40

        /** Tier 1 Local PyrLK constants on 640x360. */
        private const val MAX_LOCAL_KLT_CORNERS = 60
        const val MIN_LOCAL_KLT_INLIERS = 7
        private const val MIN_LOCAL_KLT_ROI_VO_PX = 96
        private const val LOCAL_KLT_COLLAR_MULT = 1.75
        private const val MAX_FB_ERROR_VO_PX = 1.5

        /** Tier 2 Sparse 3D Mini-VIO constants on 640x360. */
        private const val VO_DOWNSCALE_DIVISOR = 3 // 1920x1080 -> 640x360
        private const val MAX_VIO_LANDMARKS = 90
        private const val MIN_VIO_REPLENISH_LANDMARKS = 65
        private const val MIN_VIO_SOLVE_LANDMARKS = 12
        private const val VIO_PURE_ROTATION_DEADBAND_VO_PX = 0.38f
        private const val MIN_VIO_TRIANGULATE_BASELINE_M = 0.04f
        private const val MIN_VIO_TRIANGULATE_PARALLAX_DEG = 1.5f

        private const val MAX_VO_GRAVEL_CORNERS = 96
        private const val MIN_CACHED_VO_CORNERS = 60
        private const val MIN_VO_INLIERS = 10

        private const val RELOC_DOWNSCALE_DIVISOR = 4 // 1920x1080 -> 480x270 (~5 Hz relocalization)
        const val MAX_RELOC_KEYPOINTS = 260
        const val MIN_RELOC_SEED_KEYPOINTS = 18
        const val MIN_RELOC_INLIERS = 14
        private const val RELOC_MAGSAC_REPROJ_THRESHOLD_PX = 6.0

        private val openCvLoadAttempted = AtomicBoolean(false)

        @Volatile
        private var openCvLoaded: Boolean = false

        /**
         * Clamps a candidate `Rect(rawX, rawY, rawW, rawH)` strictly inside `[0..cols] x [0..rows]`
         * with `width >= 2` and `height >= 2` so `Mat.submat(Rect)` can never throw a C++ `CvException`.
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

        /**
         * Verifies Gate 2 conditioning of a normalized `3x3` homography's upper-left `2x2` Jacobian
         * `[[j00, j01], [j10, j11]]` and perspective terms `(h20, h21)`:
         * - Determinant in `[0.16 .. 6.25]` (no flips or collapse)
         * - Singular value ratio `svMax / svMin <= MAX_HOMOGRAPHY_SV_RATIO` (`2.4`)
         * - Perspective warp bound `hypot(h20, h21) * widthPx < 0.45`
         */
        fun verifyHomography2x2Conditioning(
            j00: Double,
            j01: Double,
            j10: Double,
            j11: Double,
            h20: Double,
            h21: Double,
            widthPx: Int
        ): Boolean {
            val det = j00 * j11 - j01 * j10
            if (!det.isFinite() || det !in 0.16..6.25) return false
            val t = j00 * j00 + j01 * j01 + j10 * j10 + j11 * j11
            val disc = sqrt(max(0.0, t * t - 4.0 * det * det))
            val svMaxSq = (t + disc) * 0.5
            val svMinSq = (t - disc) * 0.5
            if (svMinSq <= 1e-6) return false
            val svRatio = sqrt(svMaxSq / svMinSq)
            if (!svRatio.isFinite() || svRatio > MAX_HOMOGRAPHY_SV_RATIO) return false
            val persp = hypot(h20, h21) * widthPx.toDouble()
            return persp.isFinite() && persp < 0.45
        }

        /**
         * Normalizes an unsigned 8-bit patch into a zero-mean unit-L2 `FloatArray` for NCC scoring.
         * Returns `null` when the patch has insufficient texture (`std < MIN_NCC_PATCH_STD`, e.g., blank wall).
         */
        fun normalizeZeroMeanUnitNcc(bytes: ByteArray): FloatArray? {
            val n = bytes.size
            if (n < 16) return null
            var sum = 0.0
            for (i in 0 until n) sum += (bytes[i].toInt() and 0xFF)
            val mean = sum / n
            val out = FloatArray(n)
            var sumSq = 0.0
            for (i in 0 until n) {
                val d = (bytes[i].toInt() and 0xFF) - mean
                out[i] = d.toFloat()
                sumSq += d * d
            }
            val std = sqrt(sumSq / n)
            if (!std.isFinite() || std < MIN_NCC_PATCH_STD) return null
            val invNorm = (1.0 / sqrt(sumSq)).toFloat()
            for (i in 0 until n) out[i] *= invNorm
            return out
        }

        /** Computes dot product of two equal-length zero-mean unit-L2 NCC vectors (`[-1..1]`). */
        fun dotNcc(a: FloatArray, b: FloatArray): Float {
            if (a.size != b.size || a.isEmpty()) return -1f
            var acc = 0f
            for (i in a.indices) acc += a[i] * b[i]
            return acc.coerceIn(-1f, 1f)
        }

        /**
         * Computes relative OpenCV camera-frame rotation `R_rel_cv = R_cv_curr * R_cv_prev^T`
         * from two `3x3` `R_world_from_scr` rotation matrices.
         */
        fun computeRelativeCvRotation(
            prevRWorldFromScr: FloatArray,
            currRWorldFromScr: FloatArray
        ): DoubleArray {
            val rPrev = scrToCvFromWorldDouble(prevRWorldFromScr)
            val rCurr = scrToCvFromWorldDouble(currRWorldFromScr)
            val out = DoubleArray(9)
            for (row in 0..2) {
                for (col in 0..2) {
                    out[row * 3 + col] =
                        rCurr[row * 3 + 0] * rPrev[col * 3 + 0] +
                        rCurr[row * 3 + 1] * rPrev[col * 3 + 1] +
                        rCurr[row * 3 + 2] * rPrev[col * 3 + 2]
                }
            }
            return out
        }

        /** Wraps an angle difference in degrees to `(-180, 180]`. */
        fun wrapAngleDegrees(angleDeg: Float): Float {
            var a = angleDeg % 360f
            if (a > 180f) a -= 360f
            if (a <= -180f) a += 360f
            return a
        }

        private fun scrToCvFromWorldDouble(r: FloatArray): DoubleArray = doubleArrayOf(
            r[0].toDouble(), r[3].toDouble(), r[6].toDouble(),
            -r[1].toDouble(), -r[4].toDouble(), -r[7].toDouble(),
            -r[2].toDouble(), -r[5].toDouble(), -r[8].toDouble()
        )

        private fun warpPointWithRelativeCvRotation(
            u: Double,
            v: Double,
            rRelCv: DoubleArray,
            f: Double,
            cx: Double,
            cy: Double
        ): Point? {
            val x = (u - cx) / f
            val y = (v - cy) / f
            val xp = rRelCv[0] * x + rRelCv[1] * y + rRelCv[2]
            val yp = rRelCv[3] * x + rRelCv[4] * y + rRelCv[5]
            val zp = rRelCv[6] * x + rRelCv[7] * y + rRelCv[8]
            if (zp <= 0.15) return null
            return Point(cx + f * (xp / zp), cy + f * (yp / zp))
        }

        private fun buildKeyframeSearchOrder(size: Int, preferredIdx: Int): IntArray {
            val order = IntArray(size)
            if (size <= 0) return order
            val pref = preferredIdx.coerceIn(0, size - 1)
            order[0] = pref
            var outIdx = 1
            for (i in size - 1 downTo 0) {
                if (i != pref) order[outIdx++] = i
            }
            return order
        }

        private fun triangulateTwoRaysMidpoint(
            originA: FloatArray,
            rayA: FloatArray,
            originB: FloatArray,
            rayB: FloatArray
        ): FloatArray? {
            val w0x = (originA[0] - originB[0]).toDouble()
            val w0y = (originA[1] - originB[1]).toDouble()
            val w0z = (originA[2] - originB[2]).toDouble()
            val b = (rayA[0] * rayB[0] + rayA[1] * rayB[1] + rayA[2] * rayB[2]).toDouble()
            val denom = 1.0 - b * b
            if (denom < 1e-4) return null
            val d = rayA[0] * w0x + rayA[1] * w0y + rayA[2] * w0z
            val e = rayB[0] * w0x + rayB[1] * w0y + rayB[2] * w0z
            val s = (b * e - d) / denom
            val t = (e - b * d) / denom
            if (s <= 0.12 || t <= 0.12 || s > 10.0 || t > 10.0) return null
            val mx = 0.5 * ((originA[0] + s * rayA[0]) + (originB[0] + t * rayB[0]))
            val my = 0.5 * ((originA[1] + s * rayA[1]) + (originB[1] + t * rayB[1]))
            val mz = 0.5 * ((originA[2] + s * rayA[2]) + (originB[2] + t * rayB[2]))
            if (!mx.isFinite() || !my.isFinite() || !mz.isFinite()) return null
            return floatArrayOf(mx.toFloat(), my.toFloat(), mz.toFloat())
        }

        private fun solveSymmetric3x3(
            a00: Double, a01: Double, a02: Double,
            a11: Double, a12: Double, a22: Double,
            b0: Double, b1: Double, b2: Double
        ): DoubleArray? {
            val c00 = a11 * a22 - a12 * a12
            val c01 = -(a01 * a22 - a12 * a02)
            val c02 = a01 * a12 - a11 * a02
            val det = a00 * c00 + a01 * c01 + a02 * c02
            if (!det.isFinite() || abs(det) < 1e-9) return null
            val c11 = a00 * a22 - a02 * a02
            val c12 = -(a00 * a12 - a01 * a02)
            val c22 = a00 * a11 - a01 * a01
            val inv = 1.0 / det
            val x = (c00 * b0 + c01 * b1 + c02 * b2) * inv
            val y = (c01 * b0 + c11 * b1 + c12 * b2) * inv
            val z = (c02 * b0 + c12 * b1 + c22 * b2) * inv
            if (!x.isFinite() || !y.isFinite() || !z.isFinite()) return null
            return doubleArrayOf(x, y, z)
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
