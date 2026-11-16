/*
 * Real-time PCM16 AudioTrack metal-detector sonification engine for GneissFind.
 *
 * Consumes confirmed mineral targets (ConfirmedTarget) and walking speed gate state
 * (ScanSpeedState) from ScannerViewModel, and synthesizes continuous low-latency PCM16
 * audio bursts on a dedicated high-priority daemon thread, while publishing SonifierTelemetry
 * back to the LCARS HUD.
 *
 * Main types:
 * - MetalDetectorSonifier: stateful streaming AudioTrack synthesizer and pure pitch/cadence mapping functions.
 * - MetalDetectorSonifier.SonifierTelemetry: immutable snapshot of active pitch, cadence, and warning buzz state.
 */
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
 * Real-time [AudioTrack] metal-detector sonification engine for GneissFind:
 *  - **Pitch ([MAX_PITCH_HZ]..[MIN_PITCH_HZ]):** Inversely proportional to estimated metric stone diameter
 *    ([MIN_DIAMETER_MM] pebble -> [MAX_PITCH_HZ] bright high chirp; [MAX_DIAMETER_MM] cobble -> [MIN_PITCH_HZ] deep resonant ping).
 *  - **Ping Cadence ([MIN_PING_HZ]..[MAX_PING_HZ]):** Proportional to temporal vote / detection confidence.
 *  - **Speed Gate Warning Buzz:** Emits a distinct [SPEED_WARNING_BUZZ_HZ] low-frequency warning buzz whenever
 *    walking speed enters [ScanSpeedState.SpeedWarning] (`SLOW SWEEP`).
 *
 * Lifecycle:
 * - Call [start] to spawn the background `GneissFind-Sonifier` synthesis thread.
 * - Call [updateTargets] on each sweep/frame to update volatile synthesis targets.
 * - Call [stop] or [close] to interrupt the synthesis thread and release the hardware [AudioTrack].
 *
 * Threading:
 * - [updateTargets] and [isMuted] are safe to read/write from any thread via `@Volatile` fields.
 * - PCM synthesis and blocking [AudioTrack.write] calls run exclusively on the internal `GneissFind-Sonifier` thread.
 *
 * @param sampleRateHz Output PCM sample rate in Hz (`> 0`, defaults to [DEFAULT_SAMPLE_RATE_HZ]).
 */
class MetalDetectorSonifier(
    private val sampleRateHz: Int = DEFAULT_SAMPLE_RATE_HZ
) : AutoCloseable {

    /**
     * Immutable telemetry snapshot of the sonifier's current acoustic output for LCARS HUD display.
     *
     * @property isAudibleActive `true` when unmuted and emitting either a target ping or a speed warning buzz.
     * @property currentPitchHz Active fundamental tone frequency in Hz (`0f` when silent).
     * @property currentPingCadenceHz Active burst repetition rate in Hz (`0f` when silent).
     * @property isSpeedWarningBuzzing `true` when the overspeed warning buzz is currently active.
     */
    data class SonifierTelemetry(
        val isAudibleActive: Boolean = false,
        val currentPitchHz: Float = 0f,
        val currentPingCadenceHz: Float = 0f,
        val isSpeedWarningBuzzing: Boolean = false
    )

    /** Whether audio synthesis is globally muted by the user (`AUDIO ON` / `MUTED` HUD pill). */
    @Volatile
    var isMuted: Boolean = false

    /** Target fundamental frequency in Hz (`[MIN_PITCH_HZ]..[MAX_PITCH_HZ]`, or `0f` when no alertable target is active). */
    @Volatile
    private var targetPitchHz: Float = 0f

    /** Target ping repetition rate in Hz (`[MIN_PING_HZ]..[MAX_PING_HZ]`, or `0f` when no alertable target is active). */
    @Volatile
    private var targetCadenceHz: Float = 0f

    /** `true` when [ScanSpeedState.SpeedWarning] is active on the most recent [updateTargets] call. */
    @Volatile
    private var speedGateWarningActive: Boolean = false

    /** `true` when at least one user-alertable [ConfirmedTarget] is present and [isMuted] is `false`. */
    @Volatile
    private var hasActiveTarget: Boolean = false

    /** Most recent [SonifierTelemetry] snapshot published by [updateTargets] for HUD readouts. */
    @Volatile
    var latestTelemetry: SonifierTelemetry = SonifierTelemetry()
        private set

    /** Background daemon thread running [runAudioSynthesisLoop], or `null` when stopped. */
    private var audioThread: Thread? = null

    /** Lifecycle flag controlling the `while (running)` loop inside [runAudioSynthesisLoop]. */
    @Volatile
    private var running: Boolean = false

    /**
     * Spawns and starts the high-priority daemon synthesis thread if not already running.
     *
     * Side effects:
     * - Allocates and starts a new `GneissFind-Sonifier` [Thread] at [Thread.MAX_PRIORITY].
     * - Idempotent if called while already [running].
     */
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
     * Updates real-time sonification parameters from the highest-priority user-alertable [ConfirmedTarget]
     * and current [ScanSpeedState]. Never emits target audio for Class 4 ([ com.gneissfind.app.model.StoneClass.HARD_CONFUSER ]).
     *
     * @param confirmedTargets Current list of confirmed targets ordered by priority from `TemporalTargetTracker`.
     * @param speedState Current walking speed gate state; [ScanSpeedState.SpeedWarning] overrides target pings with a low buzz.
     *
     * Side effects:
     * - Updates volatile synthesis parameters ([targetPitchHz], [targetCadenceHz], [speedGateWarningActive], [hasActiveTarget])
     *   and publishes a new [latestTelemetry] snapshot.
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

        // Combine smoothed neural confidence with N-of-M vote ratio so 3/6 -> moderate ping rate, 6/6 -> urgent MAX_PING_HZ.
        // RATIONALE-UNKNOWN: how were the 0.55f / 0.45f blend weights between smoothedConfidence and voteRatio chosen?
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

    /**
     * Signals the background synthesis loop to exit and interrupts [audioThread].
     *
     * Side effects:
     * - Sets [running] to `false`, interrupts [audioThread], and clears the thread reference.
     */
    fun stop() {
        running = false
        audioThread?.interrupt()
        audioThread = null
    }

    override fun close() {
        stop()
    }

    /**
     * Main streaming synthesis loop executed on `GneissFind-Sonifier`.
     *
     * Allocates a mono 16-bit PCM [AudioTrack] with [CHUNK_DURATION_SEC] write slices and continuously
     * synthesizes silence, the dual-harmonic overspeed warning buzz, or raised-cosine-enveloped
     * mineral target pings until [running] becomes `false` or the thread is interrupted.
     * If [AudioTrack] initialization fails (e.g. in headless JVM unit tests), logs a warning and returns cleanly.
     */
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

        /** Android [AudioAttributes] usage routing sonification through media volume controls (`USAGE_MEDIA`). */
        const val AUDIO_USAGE = AudioAttributes.USAGE_MEDIA

        /** Android [AudioAttributes] content type for continuous synthesized tone bursts (`CONTENT_TYPE_MUSIC`). */
        const val AUDIO_CONTENT_TYPE = AudioAttributes.CONTENT_TYPE_MUSIC

        /** Default mono PCM output sample rate (`44,100 Hz`). */
        const val DEFAULT_SAMPLE_RATE_HZ = 44_100

        /** Streaming write slice duration in seconds (`20 ms`) balancing low latency against scheduler jitter. */
        private const val CHUNK_DURATION_SEC = 0.020f

        /** Nominal duration of each metallic target ping burst in seconds (`48 ms`). */
        // RATIONALE-UNKNOWN: why was 48 ms chosen as the nominal tone burst duration?
        private const val TONE_BURST_DURATION_SEC = 0.048f

        /** Minimum metric stone diameter (`20.0 mm`) clamped at the high-pitch ceiling ([MAX_PITCH_HZ]) per `AGENTS.md` §1. */
        const val MIN_DIAMETER_MM = 20.0f

        /** Maximum metric stone diameter (`60.0 mm`) clamped at the low-pitch floor ([MIN_PITCH_HZ]) per `AGENTS.md` §1. */
        const val MAX_DIAMETER_MM = 60.0f

        /** Highest target ping frequency (`780.0 Hz`) corresponding to a [MIN_DIAMETER_MM] pebble. */
        // RATIONALE-UNKNOWN: how were the exact 780 Hz ceiling and 320 Hz floor frequencies selected?
        const val MAX_PITCH_HZ = 780.0f

        /** Lowest target ping frequency (`320.0 Hz`) corresponding to a [MAX_DIAMETER_MM]+ cobble. */
        const val MIN_PITCH_HZ = 320.0f

        /** Minimum ping repetition rate (`2.0 Hz`) at threshold confidence (`0.0`). */
        // RATIONALE-UNKNOWN: how were the 2.0..14.0 Hz ping cadence bounds selected?
        const val MIN_PING_HZ = 2.0f

        /** Maximum ping repetition rate (`14.0 Hz`) at peak confidence (`1.0`). */
        const val MAX_PING_HZ = 14.0f

        /** Fundamental frequency (`145.0 Hz`) of the overspeed gate warning buzz (`SLOW SWEEP`), well below [MIN_PITCH_HZ]. */
        const val SPEED_WARNING_BUZZ_HZ = 145.0f

        /** Burst repetition rate (`4.5 Hz`) of the overspeed gate warning buzz. */
        const val SPEED_WARNING_CADENCE_HZ = 4.5f

        /**
         * Maps metric stone diameter `[MIN_DIAMETER_MM]..[MAX_DIAMETER_MM]` inversely and linearly
         * to pitch `[MAX_PITCH_HZ]..[MIN_PITCH_HZ]`.
         *
         * @param diameterMm Estimated stone diameter in millimeters; values outside `[MIN_DIAMETER_MM]..[MAX_DIAMETER_MM]` are clamped.
         * @return Target ping fundamental frequency in Hz within `[MIN_PITCH_HZ]..[MAX_PITCH_HZ]`.
         */
        fun diameterMmToPitchHz(diameterMm: Float): Float {
            val clamped = diameterMm.coerceIn(MIN_DIAMETER_MM, MAX_DIAMETER_MM)
            val t = (clamped - MIN_DIAMETER_MM) / (MAX_DIAMETER_MM - MIN_DIAMETER_MM)
            return MAX_PITCH_HZ - t * (MAX_PITCH_HZ - MIN_PITCH_HZ)
        }

        /**
         * Maps normalized temporal vote / detection confidence `0..1` linearly to `[MIN_PING_HZ]..[MAX_PING_HZ]` ping rate.
         *
         * @param confidence Combined confidence score in `[0.0, 1.0]` (clamped if outside range).
         * @return Target ping cadence in Hz within `[MIN_PING_HZ]..[MAX_PING_HZ]`.
         */
        fun confidenceToPingCadenceHz(confidence: Float): Float {
            val clamped = confidence.coerceIn(0.0f, 1.0f)
            return MIN_PING_HZ + clamped * (MAX_PING_HZ - MIN_PING_HZ)
        }

        /**
         * Click-free raised-cosine attack (`15%` of burst, minimum `8` samples) plus smooth cosine
         * release (`85%` of burst) amplitude envelope for each tone burst.
         *
         * @param sampleIndex Zero-based sample offset within the current tone burst.
         * @param burstLengthSamples Total number of samples in the tone burst.
         * @return Normalized envelope gain in `[0.0, 1.0]`, or `0.0` if [sampleIndex] is outside `0 until burstLengthSamples`.
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
         *
         * @param diameterMm Estimated stone diameter in millimeters mapped via [diameterMmToPitchHz].
         * @param sampleRateHz Sample rate in Hz (defaults to [DEFAULT_SAMPLE_RATE_HZ]).
         * @param durationMs Burst duration in milliseconds (defaults to `48 ms`, minimum `16` samples).
         * @return Newly allocated [ShortArray] containing the windowed mono PCM16 waveform.
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
