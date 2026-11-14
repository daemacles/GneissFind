package com.gneissfind.app.ui

import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.SwathTileMode
import com.gneissfind.app.ui.lcars.LcarsElbowOrientation
import com.gneissfind.app.ui.lcars.resolveLcarsElbowMetricsPx
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

        // TopLeft: (1, 1) is outside outer 32px convex corner; (20, 20) is inside stem/arm
        assertThat(metrics.containsPoint(1f, 1f, LcarsElbowOrientation.TopLeft)).isFalse()
        assertThat(metrics.containsPoint(20f, 20f, LcarsElbowOrientation.TopLeft)).isTrue()

        // Concave inner crook fillet at (68, 22) with rIn=14 has circle center at (82, 36):
        // Point (70, 24) is tucked into the concave fillet web -> INSIDE the elbow!
        assertThat(metrics.containsPoint(70f, 24f, LcarsElbowOrientation.TopLeft)).isTrue()
        // Point (82, 36) is at the center of the inner fillet circle -> OUTSIDE the elbow (viewport area)!
        assertThat(metrics.containsPoint(82f, 36f, LcarsElbowOrientation.TopLeft)).isFalse()

        // BottomLeft mirror symmetry: crook is at (68, 80 - 22 = 58)
        assertThat(metrics.containsPoint(1f, 79f, LcarsElbowOrientation.BottomLeft)).isFalse()
        assertThat(metrics.containsPoint(70f, 56f, LcarsElbowOrientation.BottomLeft)).isTrue()
        assertThat(metrics.containsPoint(82f, 44f, LcarsElbowOrientation.BottomLeft)).isFalse()
    }

    @Test
    fun `computePanoramicViewportTransform keeps 384px strip fully visible across 0 to 696 cropY`() {
        // Typical Pixel 10 Pro XL panoramic camera slot (~1900 x 410 physical px)
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

                // Strip top & bottom stay strictly inside [0, viewH]
                assertThat(t.stripTopPx).isAtLeast(-0.01f)
                assertThat(t.stripTopPx + t.stripHeightPx).isAtMost(viewH + 0.01f)

                // Strip top matches exact optical row offset inside rendered 1920x1080 frame
                assertThat(t.stripTopPx).isWithin(0.01f).of(t.frameTopPx + cropY * t.scale)
                assertThat(t.stripLeftPx).isWithin(0.01f)
                    .of(t.frameLeftPx + mode.canonicalStripLeftPx * t.scale)
            }
        }
    }
}
