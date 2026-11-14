package com.gneissfind.app.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import com.gneissfind.app.model.ConfirmedTarget
import com.gneissfind.app.model.ScanSpeedState
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Real-time `AudioTrack` metal-detector sonification engine for GneissFind:
 *  - **Pitch (`780..320 Hz`):** Inversely proportional to estimated metric stone diameter
 *    (`20 mm` pebble -> `780 Hz` bright high chirp; `60 mm` cobble -> `320 Hz` deep resonant ping).
 *  - **Ping Cadence (`2..14 Hz`):** Proportional to temporal vote / detection confidence.
 *  - **Speed Gate Warning Buzz:** Emits a distinct `145 Hz` low-frequency warning buzz whenever
 *    walking speed exceeds `1.10 m/s` (`SLOW SWEEP`).
 */
class MetalDetectorSonifier(
    private val sampleRateHz: Int = DEFAULT_SAMPLE_RATE_HZ
) : AutoCloseable {

    data class SonifierTelemetry(
        val isAudibleActive: Boolean = false,
        val currentPitchHz: Float = 0f,
        val currentPingCadenceHz: Float = 0f,
        val isSpeedWarningBuzzing: Boolean = false
    )

    @Volatile
    var isMuted: Boolean = false

    @Volatile
    private var targetPitchHz: Float = 0f

    @Volatile
    private var targetCadenceHz: Float = 0f

    @Volatile
    private var speedGateWarningActive: Boolean = false

    @Volatile
    private var hasActiveTarget: Boolean = false

    @Volatile
    var latestTelemetry: SonifierTelemetry = SonifierTelemetry()
        private set

    private var audioThread: Thread? = null

    @Volatile
    private var running: Boolean = false

    fun start() {
        if (running) return
        running = true
        audioThread = Thread(::runAudioSynthesisLoop, "GneissFind-Sonifier").apply {
            isDaemon = true
            priority = Thread.MAX_PRIORITY
            start()
        }
    }

    /**
     * Updates real-time sonification parameters from the highest-priority [ConfirmedTarget]
     * and current [ScanSpeedState]. Never emits audio for Class 4 (`hard_confuser`).
     */
    fun updateTargets(
        confirmedTargets: List<ConfirmedTarget>,
        speedState: ScanSpeedState
    ) {
        val speedWarning = speedState is ScanSpeedState.SpeedWarning
        speedGateWarningActive = speedWarning

        val primary = confirmedTargets.firstOrNull { it.stoneClass.isUserAlertable }
        if (primary == null || isMuted) {
            hasActiveTarget = false
            targetPitchHz = 0f
            targetCadenceHz = 0f
            latestTelemetry = SonifierTelemetry(
                isAudibleActive = !isMuted && speedWarning,
                currentPitchHz = if (!isMuted && speedWarning) SPEED_WARNING_BUZZ_HZ else 0f,
                currentPingCadenceHz = if (!isMuted && speedWarning) SPEED_WARNING_CADENCE_HZ else 0f,
                isSpeedWarningBuzzing = !isMuted && speedWarning
            )
            return
        }

        // Combine smoothed neural confidence with N-of-M vote ratio so 3/6 -> moderate ping rate, 6/6 -> urgent 14 Hz
        val combinedConfidence = (primary.smoothedConfidence * 0.55f + primary.voteRatio * 0.45f)
            .coerceIn(0f, 1f)
        val pitch = diameterMmToPitchHz(primary.estimatedDiameterMm)
        val cadence = confidenceToPingCadenceHz(combinedConfidence)

        targetPitchHz = pitch
        targetCadenceHz = cadence
        hasActiveTarget = true

        latestTelemetry = SonifierTelemetry(
            isAudibleActive = true,
            currentPitchHz = pitch,
            currentPingCadenceHz = cadence,
            isSpeedWarningBuzzing = speedWarning
        )
    }

    fun stop() {
        running = false
        audioThread?.interrupt()
        audioThread = null
    }

    override fun close() {
        stop()
    }

    private fun runAudioSynthesisLoop() {
        val chunkFrames = (sampleRateHz * CHUNK_DURATION_SEC).roundToInt().coerceAtLeast(256)
        val minBufBytes = AudioTrack.getMinBufferSize(
            sampleRateHz,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(chunkFrames * 4)

        val track = try {
            AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AUDIO_USAGE)
                        .setContentType(AUDIO_CONTENT_TYPE)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(sampleRateHz)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setTransferMode(AudioTrack.MODE_STREAM)
                .setBufferSizeInBytes(minBufBytes)
                .build()
        } catch (t: Throwable) {
            Log.w(TAG, "AudioTrack unavailable on this device/environment", t)
            return
        }

        val pcmChunk = ShortArray(chunkFrames)
        var phaseRad = 0.0
        var warningPhaseRad = 0.0
        var samplesIntoCycle = 0

        try {
            track.play()
            while (running && !Thread.currentThread().isInterrupted) {
                val muted = isMuted
                val warnBuzz = speedGateWarningActive && !muted
                val targetActive = hasActiveTarget && !muted
                val freqHz = targetPitchHz
                val cadenceHz = targetCadenceHz

                if (!warnBuzz && !targetActive) {
                    pcmChunk.fill(0)
                    samplesIntoCycle = 0
                } else if (warnBuzz) {
                    // Distinct low-frequency dual-harmonic warning buzz ("SLOW SWEEP")
                    val cycleSamples = (sampleRateHz / SPEED_WARNING_CADENCE_HZ).roundToInt().coerceAtLeast(1)
                    val burstSamples = (cycleSamples * 0.60f).roundToInt()
                    val phaseStep = 2.0 * PI * SPEED_WARNING_BUZZ_HZ / sampleRateHz

                    for (i in 0 until chunkFrames) {
                        val posInCycle = samplesIntoCycle % cycleSamples
                        samplesIntoCycle = (samplesIntoCycle + 1) % cycleSamples
                        if (posInCycle < burstSamples) {
                            val env = computeSmoothBurstEnvelope(posInCycle, burstSamples)
                            val fundamental = sin(warningPhaseRad)
                            val overtone = 0.45 * sin(warningPhaseRad * 2.0)
                            val sample = ((fundamental + overtone) * 0.45 * env * Short.MAX_VALUE)
                                .roundToInt()
                                .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                            pcmChunk[i] = sample.toShort()
                            warningPhaseRad = (warningPhaseRad + phaseStep) % (2.0 * PI)
                        } else {
                            pcmChunk[i] = 0
                        }
                    }
                } else {
                    val periodSamples = (sampleRateHz / cadenceHz.coerceIn(MIN_PING_HZ, MAX_PING_HZ))
                        .roundToInt()
                        .coerceAtLeast(1)
                    val burstSamples = min(
                        (sampleRateHz * TONE_BURST_DURATION_SEC).roundToInt(),
                        (periodSamples * 0.72f).roundToInt().coerceAtLeast(64)
                    )
                    val phaseStep = 2.0 * PI * freqHz.coerceIn(MIN_PITCH_HZ, MAX_PITCH_HZ) / sampleRateHz

                    for (i in 0 until chunkFrames) {
                        if (samplesIntoCycle >= periodSamples) {
                            samplesIntoCycle = 0
                        }
                        if (samplesIntoCycle < burstSamples) {
                            val env = computeSmoothBurstEnvelope(samplesIntoCycle, burstSamples)
                            // Pure fundamental + subtle 2nd harmonic for crisp outdoor gravel-bar audibility
                            val wave = sin(phaseRad) + 0.18 * sin(phaseRad * 2.0)
                            val sample = (wave * 0.65 * env * Short.MAX_VALUE)
                                .roundToInt()
                                .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                            pcmChunk[i] = sample.toShort()
                            phaseRad = (phaseRad + phaseStep) % (2.0 * PI)
                        } else {
                            pcmChunk[i] = 0
                            phaseRad = 0.0
                        }
                        samplesIntoCycle++
                    }
                }

                track.write(pcmChunk, 0, chunkFrames)
            }
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        } catch (t: Throwable) {
            Log.w(TAG, "AudioTrack synthesis loop stopped", t)
        } finally {
            runCatching { track.stop() }
            runCatching { track.release() }
        }
    }

    companion object {
        private const val TAG = "MetalDetectorSonifier"
        const val AUDIO_USAGE = AudioAttributes.USAGE_MEDIA
        const val AUDIO_CONTENT_TYPE = AudioAttributes.CONTENT_TYPE_MUSIC
        const val DEFAULT_SAMPLE_RATE_HZ = 44_100
        private const val CHUNK_DURATION_SEC = 0.020f // 20 ms low-latency streaming slices
        private const val TONE_BURST_DURATION_SEC = 0.048f // 48 ms crisp metallic ping

        const val MIN_DIAMETER_MM = 20.0f
        const val MAX_DIAMETER_MM = 60.0f
        const val MAX_PITCH_HZ = 780.0f // 20 mm pebble -> 780 Hz
        const val MIN_PITCH_HZ = 320.0f // 60 mm+ cobble -> 320 Hz

        const val MIN_PING_HZ = 2.0f
        const val MAX_PING_HZ = 14.0f

        const val SPEED_WARNING_BUZZ_HZ = 145.0f
        const val SPEED_WARNING_CADENCE_HZ = 4.5f

        /**
         * Maps metric stone diameter `20..60 mm` inversely to pitch `780..320 Hz`.
         */
        fun diameterMmToPitchHz(diameterMm: Float): Float {
            val clamped = diameterMm.coerceIn(MIN_DIAMETER_MM, MAX_DIAMETER_MM)
            val t = (clamped - MIN_DIAMETER_MM) / (MAX_DIAMETER_MM - MIN_DIAMETER_MM)
            return MAX_PITCH_HZ - t * (MAX_PITCH_HZ - MIN_PITCH_HZ)
        }

        /**
         * Maps normalized temporal vote / detection confidence `0..1` linearly to `2..14 Hz` ping rate.
         */
        fun confidenceToPingCadenceHz(confidence: Float): Float {
            val clamped = confidence.coerceIn(0.0f, 1.0f)
            return MIN_PING_HZ + clamped * (MAX_PING_HZ - MIN_PING_HZ)
        }

        /**
         * Click-free raised-cosine attack + smooth cosine release envelope for each tone burst.
         */
        fun computeSmoothBurstEnvelope(sampleIndex: Int, burstLengthSamples: Int): Double {
            if (burstLengthSamples <= 0 || sampleIndex !in 0 until burstLengthSamples) return 0.0
            val attackSamples = (burstLengthSamples * 0.15).roundToInt().coerceAtLeast(8)
            return if (sampleIndex < attackSamples) {
                0.5 * (1.0 - cos(PI * sampleIndex / attackSamples))
            } else {
                val releaseSpan = (burstLengthSamples - attackSamples).coerceAtLeast(1)
                val releasePos = sampleIndex - attackSamples
                0.5 * (1.0 + cos(PI * releasePos / releaseSpan))
            }
        }

        /**
         * Synthesizes a deterministic single PCM16 tone burst buffer for offline / unit-test verification.
         */
        fun synthesizeSinglePingBurstPcm16(
            diameterMm: Float,
            sampleRateHz: Int = DEFAULT_SAMPLE_RATE_HZ,
            durationMs: Int = 48
        ): ShortArray {
            val pitchHz = diameterMmToPitchHz(diameterMm)
            val totalSamples = ((sampleRateHz * durationMs) / 1000).coerceAtLeast(16)
            val out = ShortArray(totalSamples)
            val phaseStep = 2.0 * PI * pitchHz / sampleRateHz
            var phase = 0.0
            for (i in 0 until totalSamples) {
                val env = computeSmoothBurstEnvelope(i, totalSamples)
                val s = (sin(phase) * 0.75 * env * Short.MAX_VALUE)
                    .roundToInt()
                    .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                out[i] = s.toShort()
                phase += phaseStep
            }
            return out
        }
    }
}
