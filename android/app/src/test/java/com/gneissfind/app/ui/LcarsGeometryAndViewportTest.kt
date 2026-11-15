/*
 * Unit tests for LCARS swept elbow geometry, continuous sub-pixel EIS viewport transforms, and
 * semantic HSV value stepping across adjacent UI elements.
 */
package com.gneissfind.app.ui

import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.SwathTileMode
import com.gneissfind.app.tracking.GroundPlaneRaycaster
import com.gneissfind.app.ui.lcars.LcarsElbowOrientation
import com.gneissfind.app.ui.lcars.resolveLcarsElbowMetricsPx
import com.gneissfind.app.ui.theme.LCARS_ADJACENT_VALUE_STEP
import com.gneissfind.app.ui.theme.LcarsColors
import com.gneissfind.app.ui.theme.LcarsValueLevel
import com.gneissfind.app.ui.theme.colorToHsv
import com.gneissfind.app.ui.theme.withValueLevel
import com.google.common.truth.Truth.assertThat
import org.junit.Test

class LcarsGeometryAndViewportTest {

    @Test
    fun `LcarsElbow concave inner fillet includes crook web and excludes inner viewport`() {
        val metrics = resolveLcarsElbowMetricsPx(
            widthPx = 180f,
            heightPx = 80f,
            stemWidthPx = 68f,
            barHeightPx = 22f,
            outerRadiusPx = 32f,
            innerRadiusPx = 14f
        )

        assertThat(metrics.containsPoint(1f, 1f, LcarsElbowOrientation.TopLeft)).isFalse()
        assertThat(metrics.containsPoint(20f, 20f, LcarsElbowOrientation.TopLeft)).isTrue()

        // Concave inner crook fillet at (68, 22) with rIn=14 has circle center at (82, 36)
        assertThat(metrics.containsPoint(70f, 24f, LcarsElbowOrientation.TopLeft)).isTrue()
        assertThat(metrics.containsPoint(82f, 36f, LcarsElbowOrientation.TopLeft)).isFalse()

        // BottomLeft mirror symmetry: crook is at (68, 80 - 22 = 58)
        assertThat(metrics.containsPoint(1f, 79f, LcarsElbowOrientation.BottomLeft)).isFalse()
        assertThat(metrics.containsPoint(70f, 56f, LcarsElbowOrientation.BottomLeft)).isTrue()
        assertThat(metrics.containsPoint(82f, 44f, LcarsElbowOrientation.BottomLeft)).isFalse()
    }

    @Test
    fun `computePanoramicViewportTransform keeps 384px strip fully visible across 0 to 696 cropY`() {
        val viewW = 1900f
        val viewH = 410f

        for (mode in SwathTileMode.entries) {
            for (cropY in listOf(0, 174, StripRoiGeometry.NOMINAL_CROP_Y_PX, 520, StripRoiGeometry.MAX_CROP_Y_PX)) {
                val t = computePanoramicViewportTransform(
                    viewportWidthPx = viewW,
                    viewportHeightPx = viewH,
                    cropY = cropY,
                    swathTileMode = mode
                )

                assertThat(t.stripTopPx).isAtLeast(-0.01f)
                assertThat(t.stripTopPx + t.stripHeightPx).isAtMost(viewH + 0.01f)
                assertThat(t.stripTopPx).isWithin(0.01f).of(t.frameTopPx + cropY * t.scale)
                assertThat(t.stripLeftPx).isWithin(0.01f)
                    .of(t.frameLeftPx + mode.canonicalStripLeftPx * t.scale)
            }
        }
    }

    @Test
    fun `sub-pixel continuous EIS pitch updates translate frameTopPx smoothly without integer steps`() {
        val viewW = 1900f
        val viewH = 410f

        val pitchSamples = listOf(45.00f, 45.01f, 45.02f, 45.03f, 45.04f)
        val frameTops = pitchSamples.map { pitch ->
            val stab = GroundPlaneRaycaster.computeStabilizedCropYFloat(
                currentPitchDeg = pitch,
                referencePitchDeg = 45.00f
            )
            computePanoramicViewportTransform(
                viewportWidthPx = viewW,
                viewportHeightPx = viewH,
                cropYFloat = stab.clampedCropYFloat,
                swathTileMode = SwathTileMode.FIVE_TILE
            ).frameTopPx
        }

        // Every 0.01 deg tilt produces a strictly monotonic sub-pixel GPU translation delta
        for (i in 1 until frameTops.size) {
            assertThat(frameTops[i]).isGreaterThan(frameTops[i - 1])
            assertThat(frameTops[i] - frameTops[i - 1]).isLessThan(1.0f)
        }
    }

    @Test
    fun `withValueLevel preserves HSV hue and saturation while stepping value by parameterized percentage`() {
        val families = listOf(
            LcarsColors.GoldenAmber,
            LcarsColors.Periwinkle,
            LcarsColors.Lavender,
            LcarsColors.IceBlue,
            LcarsColors.ApricotPeach,
            LcarsColors.CopperTerracotta,
            LcarsColors.MutedSlate
        )

        for (stepFraction in listOf(LCARS_ADJACENT_VALUE_STEP, 0.09f)) {
            for (baseColor in families) {
                val minus = colorToHsv(baseColor.withValueLevel(LcarsValueLevel.MINUS, stepFraction))
                val base = colorToHsv(baseColor.withValueLevel(LcarsValueLevel.BASE, stepFraction))
                val plus = colorToHsv(baseColor.withValueLevel(LcarsValueLevel.PLUS, stepFraction))

                // Hue and saturation remain invariant across all 3 value tiers
                assertThat(minus[0]).isWithin(0.6f).of(base[0])
                assertThat(plus[0]).isWithin(0.6f).of(base[0])
                assertThat(minus[1]).isWithin(0.005f).of(base[1])
                assertThat(plus[1]).isWithin(0.005f).of(base[1])

                // HSV Value steps cleanly by +-stepFraction around BASE
                assertThat(base[2] - minus[2]).isWithin(0.005f).of(stepFraction)
                assertThat(plus[2] - base[2]).isWithin(0.005f).of(stepFraction)
            }
        }
    }
}
