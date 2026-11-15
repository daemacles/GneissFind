/*
 * LiteRT 2.x GPU (FP16) detector running RF-DETR-Nano (384x384) across the push-broom tile swath.
 *
 * Consumes pre-allocated NHWC RGB [0..1] float tiles from PushBroomSweepBuffers and class-index
 * routing rules from assets/rf_detr_coco_metadata.json; produces tile-local TileDetection lists
 * and stage latency telemetry for ScannerViewModel.
 */
package com.gneissfind.app.inference

import android.content.Context
import android.util.Log
import com.gneissfind.app.model.DetectorBackend
import com.gneissfind.app.model.StoneClass
import com.gneissfind.app.model.StripRoiGeometry
import com.gneissfind.app.model.TileDetection
import com.gneissfind.app.vision.PushBroomSweepBuffers
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Arrays
import java.util.Locale
import kotlin.math.exp

/**
 * Parsed runtime configuration and MS-COCO stand-in class routing table loaded from
 * `assets/rf_detr_coco_metadata.json` (`.agents/rules/litert-inference.md §4`).
 *
 * @property modelName Identifier of the exported model family.
 * @property inputResolution Square tile width and height in pixels (`384`).
 * @property batchSize Configured static batch dimension (`5` for batched sweep, `1` for sequential).
 * @property topology Graph execution topology: `"two_graph"` or `"single_graph"`.
 * @property singleGraphAsset Asset filename for the single-graph `.tflite` model.
 * @property part1Asset Asset filename for Part 1 (backbone + encoder + proposal scores).
 * @property part2Asset Asset filename for Part 2 (decoder + detection head).
 * @property encoderGridH Spatial height of the encoder proposal grid (`24`).
 * @property encoderGridW Spatial width of the encoder proposal grid (`24`).
 * @property numEncoderTokens Total encoder spatial tokens (`encoderGridH * encoderGridW = 576`).
 * @property hiddenDim Transformer decoder embedding width (`256`).
 * @property proposalDefaultWh Default normalized proposal width and height (`0.05f`).
 * @property numQueries Number of top-scoring object queries passed to the decoder (`300`).
 * @property numClasses Number of output class logits/probabilities per query (`91` for sparse COCO).
 * @property scoresAreProbabilities True when the exported graph already applies `sigmoid(logits)`.
 * @property scoreThreshold Minimum confidence score required to emit a [TileDetection].
 * @property cocoClassIdToStoneClassId Lookup table indexed by model `class_id` returning the mapped
 *   [StoneClass.id] (`0..3` for the 4 stand-in targets, or `4` for [StoneClass.HARD_CONFUSER]).
 * @property unmappedConfidentToHardConfuser Whether confident detections of unmapped classes route
 *   to [StoneClass.HARD_CONFUSER].
 */
data class CocoStandInMetadata(
    val modelName: String = "rf_detr_nano_coco",
    val inputResolution: Int = StripRoiGeometry.TILE_SIZE_PX,
    val batchSize: Int = StripRoiGeometry.MAX_TILES,
    val topology: String = TOPOLOGY_TWO_GRAPH,
    val singleGraphAsset: String = SINGLE_GRAPH_ASSET_NAME,
    val part1Asset: String = PART1_ASSET_NAME,
    val part2Asset: String = PART2_ASSET_NAME,
    val encoderGridH: Int = DEFAULT_ENCODER_GRID_SIZE,
    val encoderGridW: Int = DEFAULT_ENCODER_GRID_SIZE,
    val numEncoderTokens: Int = DEFAULT_ENCODER_TOKENS,
    val hiddenDim: Int = DEFAULT_QUERY_HIDDEN_DIM,
    val proposalDefaultWh: Float = DEFAULT_PROPOSAL_WH,
    val numQueries: Int = DEFAULT_COCO_TOP_K_QUERIES,
    val numClasses: Int = DEFAULT_COCO_NUM_CLASSES,
    val scoresAreProbabilities: Boolean = true,
    val scoreThreshold: Float = DEFAULT_SCORE_THRESHOLD,
    val cocoClassIdToStoneClassId: IntArray = buildDefaultCocoToStoneClassLut(),
    val unmappedConfidentToHardConfuser: Boolean = true
) {
    companion object {
        const val METADATA_ASSET_NAME = "rf_detr_coco_metadata.json"
        const val BENCHMARK_OVERRIDE_FILE_NAME = "benchmark_override.json"
        const val TOPOLOGY_TWO_GRAPH = "two_graph"
        const val TOPOLOGY_SINGLE_GRAPH = "single_graph"

        const val SINGLE_GRAPH_ASSET_NAME = "rf_detr_nano_coco.tflite"
        const val PART1_ASSET_NAME = "rf_detr_nano_part1_backbone_encoder.tflite"
        const val PART2_ASSET_NAME = "rf_detr_nano_part2_decoder_head.tflite"

        const val DEFAULT_ENCODER_GRID_SIZE = 24
        const val DEFAULT_ENCODER_TOKENS = 576
        const val DEFAULT_QUERY_HIDDEN_DIM = 256
        const val DEFAULT_PROPOSAL_WH = 0.05f
        const val DEFAULT_COCO_TOP_K_QUERIES = 300
        const val DEFAULT_COCO_NUM_CLASSES = 91
        const val DEFAULT_SCORE_THRESHOLD = 0.35f

        /**
         * Builds the default 91-class sparse MS-COCO lookup table verified against `rfdetr 1.12.1`:
         *  - `47` (`cup`) -> `0` ([StoneClass.CHALCEDONY_AGATE])
         *  - `53` (`apple`) -> `1` ([StoneClass.JASPER_RED_YELLOW])
         *  - `90` (`toothbrush`) -> `2` ([StoneClass.GREEN_GEM_ROCK])
         *  - `75` (`remote`) -> `3` ([StoneClass.PETRIFIED_WOOD])
         *  - All other indices -> `4` ([StoneClass.HARD_CONFUSER])
         */
        fun buildDefaultCocoToStoneClassLut(numClasses: Int = DEFAULT_COCO_NUM_CLASSES): IntArray {
            val lutSize = maxOf(numClasses + 1, 128)
            return IntArray(lutSize) { StoneClass.HARD_CONFUSER.id }.apply {
                this[47] = StoneClass.CHALCEDONY_AGATE.id
                this[53] = StoneClass.JASPER_RED_YELLOW.id
                this[90] = StoneClass.GREEN_GEM_ROCK.id
                this[75] = StoneClass.PETRIFIED_WOOD.id
            }
        }
    }
}

/**
 * LiteRT 2.x GPU (FP16) detector executing `RF-DETR-Nano` ($384 \times 384$ tiles) in either
 * 2-Graph split topology (`Part 1` backbone/encoder + Kotlin Host Top-300 proposal gather + `Part 2`
 * decoder) or 1-Graph end-to-end topology, supporting both static `B=5` batched sweep dispatch and
 * `5 x B=1` sequential tile dispatch (`.agents/rules/litert-inference.md §3–§5`).
 *
 * Responsibility and state: owns the LiteRT [Environment], [CompiledModel] handles, native
 * [TensorBuffer] bindings, and pre-allocated direct `ByteBuffer` / `FloatArray` staging buffers so
 * [detectSweep] performs zero heap allocations per frame.
 *
 * Lifecycle: lazily initializes LiteRT GPU models on the first [detectSweep] call so OpenCL/EGL
 * contexts bind to the CameraX single-threaded analyzer worker; release via [close].
 *
 * Threading: confined to the single-threaded CameraX analysis executor.
 *
 * @param appContext Application context used to read `assets/` and locate `codeCacheDir` for
 *   serialized GPU shader caching.
 * @param confidenceThreshold Minimum probability threshold (`[0.0, 1.0]`) to retain a query box.
 */
class RfDetrTwoGraphGpuDetector(
    private val appContext: Context,
    private val confidenceThreshold: Float = CocoStandInMetadata.DEFAULT_SCORE_THRESHOLD
) : StoneDetector {

    override val backend: DetectorBackend = DetectorBackend.RF_DETR_NANO_GPU

    private var environment: Environment? = null
    private var singleModel: CompiledModel? = null
    private var part1Model: CompiledModel? = null
    private var part2Model: CompiledModel? = null
    private var singleInputs: List<TensorBuffer> = emptyList()
    private var singleOutputs: List<TensorBuffer> = emptyList()
    private var part1Inputs: List<TensorBuffer> = emptyList()
    private var part1Outputs: List<TensorBuffer> = emptyList()
    private var part2Inputs: List<TensorBuffer> = emptyList()
    private var part2Outputs: List<TensorBuffer> = emptyList()

    private var metadata: CocoStandInMetadata = CocoStandInMetadata(scoreThreshold = confidenceThreshold)
    private var activeTopology: String = CocoStandInMetadata.TOPOLOGY_TWO_GRAPH
    private var activeAcceleratorLabel: String = "GPU FP16"
    private var compiledBatchSize: Int = StripRoiGeometry.MAX_TILES
    private var lastOverrideMtimeMs: Long = -1L

    private var initializedOnThread = false
    private var hardwareGraphsLoaded = false
    private var initMessage: String = "RF-DETR NANO GPU [COCO PROXY]"

    // Pre-allocated direct ByteBuffer and float staging arrays for zero-GC B=5 or B=1 execution
    private val directInputByteBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(StripRoiGeometry.MAX_TILES * TILE_FLOAT_COUNT * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
    private val directInputFloatBuffer: FloatBuffer = directInputByteBuffer.asFloatBuffer()
    private val batchedInputFloats = FloatArray(StripRoiGeometry.MAX_TILES * TILE_FLOAT_COUNT)
    private val singleTileInputFloats = FloatArray(TILE_FLOAT_COUNT)

    // Pre-allocated 2-graph Host Top-300 bridge tables (sized for max B=5, 576 tokens, K=300, dim=256)
    private var baseProposalsTable: FloatArray = buildNormalizedProposalGrid(
        gridH = CocoStandInMetadata.DEFAULT_ENCODER_GRID_SIZE,
        gridW = CocoStandInMetadata.DEFAULT_ENCODER_GRID_SIZE,
        defaultWh = CocoStandInMetadata.DEFAULT_PROPOSAL_WH
    )
    private val batchedSelectedPrenormFloats = FloatArray(
        StripRoiGeometry.MAX_TILES *
            CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES *
            CocoStandInMetadata.DEFAULT_QUERY_HIDDEN_DIM
    )
    private val batchedSelectedProposalsFloats = FloatArray(
        StripRoiGeometry.MAX_TILES * CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES * 4
    )
    private val singleSelectedPrenormFloats = FloatArray(
        CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES *
            CocoStandInMetadata.DEFAULT_QUERY_HIDDEN_DIM
    )
    private val singleSelectedProposalsFloats = FloatArray(
        CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES * 4
    )
    private val batchedTopIndices = IntArray(
        StripRoiGeometry.MAX_TILES * CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES
    )
    private val scratchHeapScores = FloatArray(CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES)
    private val scratchHeapIndices = IntArray(CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES)

    // Legacy fallback profiler buffers used only when .tflite assets are absent in unit tests
    private val stubEncoderLogits = FloatArray(DEFAULT_ENCODER_ANCHORS * NUM_CLASSES)
    private val stubEncoderFeatures = FloatArray(DEFAULT_ENCODER_ANCHORS * QUERY_HIDDEN_DIM)
    private val selectedQueryFeaturesBuffer: FloatBuffer =
        FloatBuffer.allocate(TOP_K_QUERIES * QUERY_HIDDEN_DIM)
    private val selectedTopIndices = IntArray(TOP_K_QUERIES)

    private var rollingLatencyMs: Float = 0f
    private var sweepCount: Long = 0L

    override val isHardwareReady: Boolean
        get() = hardwareGraphsLoaded

    override val statusSummary: String
        get() = initMessage

    /**
     * Checks whether an external `benchmark_override.json` file was updated by `adb` so the
     * 4-way benchmark matrix (`1-Graph vs 2-Graph` x `B=5 vs 5xB=1`) can hot-swap models on the
     * analyzer thread without restarting the camera session.
     */
    private fun checkBenchmarkOverrideHotSwap() {
        val overrideFile = resolveBenchmarkOverrideFile()
        val mtime = if (overrideFile != null && overrideFile.exists()) overrideFile.lastModified() else 0L
        if (initializedOnThread && mtime != lastOverrideMtimeMs) {
            Log.i(TAG, "Detected benchmark_override.json change (mtime=$mtime); reloading LiteRT models")
            releaseModelsOnly()
            initializedOnThread = false
            sweepCount = 0L
            rollingLatencyMs = 0f
        }
    }

    private fun resolveBenchmarkOverrideFile(): File? {
        val extDir = appContext.getExternalFilesDir(null)
        if (extDir != null) {
            val extFile = File(extDir, CocoStandInMetadata.BENCHMARK_OVERRIDE_FILE_NAME)
            if (extFile.exists()) return extFile
        }
        val intFile = File(appContext.filesDir, CocoStandInMetadata.BENCHMARK_OVERRIDE_FILE_NAME)
        return if (intFile.exists()) intFile else null
    }

    /**
     * Lazily initializes LiteRT [Environment] and [CompiledModel] instances on the single-threaded
     * CameraX analyzer worker so GPU OpenCL/EGL contexts maintain strict thread affinity.
     */
    private fun ensureInitializedOnAnalyzerThread() {
        checkBenchmarkOverrideHotSwap()
        if (initializedOnThread) return
        initializedOnThread = true

        val overrideFile = resolveBenchmarkOverrideFile()
        lastOverrideMtimeMs = if (overrideFile != null && overrideFile.exists()) overrideFile.lastModified() else 0L

        try {
            val assetList = appContext.assets.list("")?.toSet().orEmpty()
            if (CocoStandInMetadata.METADATA_ASSET_NAME in assetList) {
                val jsonText = appContext.assets.open(CocoStandInMetadata.METADATA_ASSET_NAME)
                    .bufferedReader()
                    .use { it.readText() }
                metadata = parseCocoMetadataJson(jsonText, confidenceThreshold)
            }
            baseProposalsTable = buildNormalizedProposalGrid(
                gridH = metadata.encoderGridH,
                gridW = metadata.encoderGridW,
                defaultWh = metadata.proposalDefaultWh
            )

            val env = Environment.create(appContext, enableCompilerCache = true)
            environment = env

            // Optional benchmark override JSON allows hot-swapping 1g/2g and b1/b5 from filesDir/externalFilesDir
            val overrideJson = overrideFile?.readText()?.let { JSONObject(it) }
            val desiredTopology = overrideJson?.optString("topology", metadata.topology) ?: metadata.topology
            val singleFileOverride = overrideJson?.optString("single_graph_path", "")?.takeIf { it.isNotEmpty() }
            val part1FileOverride = overrideJson?.optString("part1_path", "")?.takeIf { it.isNotEmpty() }
            val part2FileOverride = overrideJson?.optString("part2_path", "")?.takeIf { it.isNotEmpty() }

            val hasTwoGraphOverride = part1FileOverride != null && part2FileOverride != null &&
                File(part1FileOverride).exists() && File(part2FileOverride).exists()
            val hasSingleGraphOverride = singleFileOverride != null && File(singleFileOverride).exists()

            val hasTwoGraphAssets = metadata.part1Asset in assetList && metadata.part2Asset in assetList
            val hasSingleGraphAsset = metadata.singleGraphAsset in assetList

            val useTwoGraph = when {
                desiredTopology == CocoStandInMetadata.TOPOLOGY_SINGLE_GRAPH &&
                    (hasSingleGraphOverride || hasSingleGraphAsset) -> false
                desiredTopology == CocoStandInMetadata.TOPOLOGY_TWO_GRAPH &&
                    (hasTwoGraphOverride || hasTwoGraphAssets) -> true
                hasTwoGraphOverride || hasTwoGraphAssets -> true
                hasSingleGraphOverride || hasSingleGraphAsset -> false
                else -> false
            }

            if (useTwoGraph && (hasTwoGraphOverride || hasTwoGraphAssets)) {
                val (m1, m2, accelLabel) = loadTwoGraphModels(
                    env = env,
                    part1FilePath = part1FileOverride.takeIf { hasTwoGraphOverride },
                    part2FilePath = part2FileOverride.takeIf { hasTwoGraphOverride }
                )
                part1Model = m1
                part2Model = m2
                part1Inputs = m1.createInputBuffers()
                part1Outputs = m1.createOutputBuffers()
                part2Inputs = m2.createInputBuffers()
                part2Outputs = m2.createOutputBuffers()

                val p1MaxScoresSize = part1Outputs.getOrNull(2)?.readFloat()?.size ?: 0
                compiledBatchSize = if (p1MaxScoresSize >= metadata.numEncoderTokens) {
                    (p1MaxScoresSize / metadata.numEncoderTokens).coerceIn(1, StripRoiGeometry.MAX_TILES)
                } else {
                    overrideJson?.optInt("batch_size", metadata.batchSize) ?: metadata.batchSize
                }
                activeTopology = CocoStandInMetadata.TOPOLOGY_TWO_GRAPH
                activeAcceleratorLabel = accelLabel
                hardwareGraphsLoaded = true
                initMessage = "RF-DETR 2G B=$compiledBatchSize • $accelLabel [COCO]"
                Log.i(TAG, "Initialized $initMessage (override=${overrideFile?.absolutePath ?: "none"})")
            } else if (hasSingleGraphOverride || hasSingleGraphAsset) {
                val (sModel, accelLabel) = loadSingleGraphModel(
                    env = env,
                    singleFilePath = singleFileOverride.takeIf { hasSingleGraphOverride }
                )
                singleModel = sModel
                singleInputs = sModel.createInputBuffers()
                singleOutputs = sModel.createOutputBuffers()

                val boxesSize = singleOutputs.getOrNull(0)?.readFloat()?.size ?: 0
                val perBatchBoxesFloats = metadata.numQueries * 4
                compiledBatchSize = if (boxesSize >= perBatchBoxesFloats && perBatchBoxesFloats > 0) {
                    (boxesSize / perBatchBoxesFloats).coerceIn(1, StripRoiGeometry.MAX_TILES)
                } else {
                    overrideJson?.optInt("batch_size", metadata.batchSize) ?: metadata.batchSize
                }
                activeTopology = CocoStandInMetadata.TOPOLOGY_SINGLE_GRAPH
                activeAcceleratorLabel = accelLabel
                hardwareGraphsLoaded = true
                initMessage = "RF-DETR 1G B=$compiledBatchSize • $accelLabel [COCO]"
                Log.i(TAG, "Initialized $initMessage (override=${overrideFile?.absolutePath ?: "none"})")
            } else {
                hardwareGraphsLoaded = false
                initMessage = "LiteRT GPU Stub (missing .tflite in assets/)"
                Log.w(TAG, "$initMessage — falling back to host bridge profiler")
            }
        } catch (t: Throwable) {
            hardwareGraphsLoaded = false
            initMessage = "LiteRT GPU Stub (${t.javaClass.simpleName})"
            Log.w(TAG, "LiteRT GPU initialization fell back to stub profiler", t)
        }
    }

    private fun buildGpuOptions(cacheKey: String): CompiledModel.Options {
        val cacheDir = appContext.codeCacheDir.absolutePath
        return CompiledModel.Options(Accelerator.GPU).apply {
            gpuOptions = CompiledModel.GpuOptions(
                precision = CompiledModel.GpuOptions.Precision.FP16,
                serializationDir = cacheDir,
                modelCacheKey = cacheKey,
                serializeProgramCache = true,
                priority = CompiledModel.GpuOptions.Priority.HIGH
            )
        }
    }

    private fun buildCpuFallbackOptions(): CompiledModel.Options {
        val cacheFile = File(appContext.codeCacheDir, "xnnpack_weights.cache").absolutePath
        return CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(
                numThreads = 4,
                xnnPackWeightCachePath = cacheFile
            )
        }
    }

    private fun loadTwoGraphModels(
        env: Environment,
        part1FilePath: String?,
        part2FilePath: String?
    ): Triple<CompiledModel, CompiledModel, String> {
        return try {
            val opt1 = buildGpuOptions("rf_detr_part1")
            val opt2 = buildGpuOptions("rf_detr_part2")
            val m1 = if (part1FilePath != null) {
                CompiledModel.create(part1FilePath, opt1, env)
            } else {
                CompiledModel.create(appContext.assets, metadata.part1Asset, opt1, env)
            }
            val m2 = if (part2FilePath != null) {
                CompiledModel.create(part2FilePath, opt2, env)
            } else {
                CompiledModel.create(appContext.assets, metadata.part2Asset, opt2, env)
            }
            Triple(m1, m2, "GPU FP16")
        } catch (gpuErr: Throwable) {
            Log.w(TAG, "GPU FP16 delegate unavailable (${gpuErr.message}); falling back to CPU XNNPACK")
            val cpuOpt = buildCpuFallbackOptions()
            val m1 = if (part1FilePath != null) {
                CompiledModel.create(part1FilePath, cpuOpt, env)
            } else {
                CompiledModel.create(appContext.assets, metadata.part1Asset, cpuOpt, env)
            }
            val m2 = if (part2FilePath != null) {
                CompiledModel.create(part2FilePath, cpuOpt, env)
            } else {
                CompiledModel.create(appContext.assets, metadata.part2Asset, cpuOpt, env)
            }
            Triple(m1, m2, "CPU XNN")
        }
    }

    private fun loadSingleGraphModel(
        env: Environment,
        singleFilePath: String?
    ): Pair<CompiledModel, String> {
        return try {
            val opt = buildGpuOptions("rf_detr_single")
            val m = if (singleFilePath != null) {
                CompiledModel.create(singleFilePath, opt, env)
            } else {
                CompiledModel.create(appContext.assets, metadata.singleGraphAsset, opt, env)
            }
            m to "GPU FP16"
        } catch (gpuErr: Throwable) {
            Log.w(TAG, "GPU FP16 delegate unavailable for 1-graph (${gpuErr.message}); falling back to CPU XNNPACK")
            val cpuOpt = buildCpuFallbackOptions()
            val m = if (singleFilePath != null) {
                CompiledModel.create(singleFilePath, cpuOpt, env)
            } else {
                CompiledModel.create(appContext.assets, metadata.singleGraphAsset, cpuOpt, env)
            }
            m to "CPU XNN"
        }
    }

    /**
     * Stages tile float buffers into the pre-allocated direct [ByteBuffer] and flat [batchedInputFloats]
     * array for zero-allocation `B=5` batched dispatch, zero-padding inactive trailing tiles when
     * `activeTiles < 5` (`THREE_TILE` mode).
     */
    private fun stageBatchedTileInputs(sweepBuffers: PushBroomSweepBuffers, activeTiles: Int) {
        directInputFloatBuffer.clear()
        for (tileIdx in 0 until StripRoiGeometry.MAX_TILES) {
            val dstOffset = tileIdx * TILE_FLOAT_COUNT
            if (tileIdx < activeTiles) {
                val src = sweepBuffers.tileRgbFloats[tileIdx]
                System.arraycopy(src, 0, batchedInputFloats, dstOffset, TILE_FLOAT_COUNT)
                directInputFloatBuffer.put(src, 0, TILE_FLOAT_COUNT)
            } else {
                Arrays.fill(batchedInputFloats, dstOffset, dstOffset + TILE_FLOAT_COUNT, 0f)
            }
        }
    }

    override fun detectSweep(sweepBuffers: PushBroomSweepBuffers): SweepInferenceResult {
        ensureInitializedOnAnalyzerThread()

        val sweepStartNs = System.nanoTime()
        var part1NsAcc = 0L
        var bridgeNsAcc = 0L
        var part2NsAcc = 0L

        val detections = ArrayList<TileDetection>(16)
        val activeTiles = sweepBuffers.activeTileCount.coerceIn(1, StripRoiGeometry.MAX_TILES)

        val m1 = part1Model
        val m2 = part2Model
        val sModel = singleModel

        if (hardwareGraphsLoaded && activeTopology == CocoStandInMetadata.TOPOLOGY_TWO_GRAPH && m1 != null && m2 != null) {
            if (compiledBatchSize >= StripRoiGeometry.MAX_TILES) {
                // Static B=5 batched 2-Graph GPU execution (single JNI/GPU dispatch for all 5 tiles)
                val t0 = System.nanoTime()
                stageBatchedTileInputs(sweepBuffers, activeTiles)
                part1Inputs[0].writeFloat(batchedInputFloats)
                m1.run(part1Inputs, part1Outputs)
                val encMemory = part1Outputs[0].readFloat()
                val encPrenorm = part1Outputs[1].readFloat()
                val encMaxScores = part1Outputs[2].readFloat()
                val t1 = System.nanoTime()
                part1NsAcc += (t1 - t0)

                // Stage 2: Host Top-300 query proposal gather across B=5 tiles
                selectTopKProposalsFromMaxScores(
                    encMaxScores = encMaxScores,
                    outputMemoryPrenorm = encPrenorm,
                    baseProposals = baseProposalsTable,
                    batchSize = StripRoiGeometry.MAX_TILES,
                    activeBatchItems = activeTiles,
                    numTokens = metadata.numEncoderTokens,
                    hiddenDim = metadata.hiddenDim,
                    topK = metadata.numQueries,
                    outSelectedPrenorm = batchedSelectedPrenormFloats,
                    outSelectedProposals = batchedSelectedProposalsFloats,
                    outSelectedIndices = batchedTopIndices,
                    scratchHeapScores = scratchHeapScores,
                    scratchHeapIndices = scratchHeapIndices
                )
                val t2 = System.nanoTime()
                bridgeNsAcc += (t2 - t1)

                // Stage 3: Decoder + Detection Head on GPU FP16
                part2Inputs[0].writeFloat(encMemory)
                part2Inputs[1].writeFloat(batchedSelectedPrenormFloats)
                part2Inputs[2].writeFloat(batchedSelectedProposalsFloats)
                m2.run(part2Inputs, part2Outputs)
                val predBoxes = part2Outputs[0].readFloat()
                val predScores = part2Outputs[1].readFloat()
                val t3 = System.nanoTime()
                part2NsAcc += (t3 - t2)

                decodeCocoBatchPredictions(
                    activeTileCount = activeTiles,
                    boxesCxCyWhNorm = predBoxes,
                    classScores = predScores,
                    metadata = metadata,
                    confidenceThreshold = confidenceThreshold,
                    outDetections = detections
                )
            } else {
                // Sequential 5 x B=1 2-Graph GPU execution
                for (tileIdx in 0 until activeTiles) {
                    val tileFloats = sweepBuffers.tileRgbFloats[tileIdx]
                    val t0 = System.nanoTime()
                    System.arraycopy(tileFloats, 0, singleTileInputFloats, 0, TILE_FLOAT_COUNT)
                    part1Inputs[0].writeFloat(singleTileInputFloats)
                    m1.run(part1Inputs, part1Outputs)
                    val encMemory = part1Outputs[0].readFloat()
                    val encPrenorm = part1Outputs[1].readFloat()
                    val encMaxScores = part1Outputs[2].readFloat()
                    val t1 = System.nanoTime()
                    part1NsAcc += (t1 - t0)

                    selectTopKProposalsFromMaxScores(
                        encMaxScores = encMaxScores,
                        outputMemoryPrenorm = encPrenorm,
                        baseProposals = baseProposalsTable,
                        batchSize = 1,
                        activeBatchItems = 1,
                        numTokens = metadata.numEncoderTokens,
                        hiddenDim = metadata.hiddenDim,
                        topK = metadata.numQueries,
                        outSelectedPrenorm = singleSelectedPrenormFloats,
                        outSelectedProposals = singleSelectedProposalsFloats,
                        outSelectedIndices = batchedTopIndices,
                        scratchHeapScores = scratchHeapScores,
                        scratchHeapIndices = scratchHeapIndices
                    )
                    val t2 = System.nanoTime()
                    bridgeNsAcc += (t2 - t1)

                    part2Inputs[0].writeFloat(encMemory)
                    part2Inputs[1].writeFloat(singleSelectedPrenormFloats)
                    part2Inputs[2].writeFloat(singleSelectedProposalsFloats)
                    m2.run(part2Inputs, part2Outputs)
                    val predBoxes = part2Outputs[0].readFloat()
                    val predScores = part2Outputs[1].readFloat()
                    val t3 = System.nanoTime()
                    part2NsAcc += (t3 - t2)

                    decodeCocoSingleTilePredictions(
                        tileIndex = tileIdx,
                        batchElementIndex = 0,
                        boxesCxCyWhNorm = predBoxes,
                        classScores = predScores,
                        metadata = metadata,
                        confidenceThreshold = confidenceThreshold,
                        outDetections = detections
                    )
                }
            }
        } else if (hardwareGraphsLoaded && activeTopology == CocoStandInMetadata.TOPOLOGY_SINGLE_GRAPH && sModel != null) {
            if (compiledBatchSize >= StripRoiGeometry.MAX_TILES) {
                // Static B=5 batched 1-Graph execution
                val t0 = System.nanoTime()
                stageBatchedTileInputs(sweepBuffers, activeTiles)
                singleInputs[0].writeFloat(batchedInputFloats)
                sModel.run(singleInputs, singleOutputs)
                val predBoxes = singleOutputs[0].readFloat()
                val predScores = singleOutputs[1].readFloat()
                val t1 = System.nanoTime()
                part1NsAcc += (t1 - t0)

                decodeCocoBatchPredictions(
                    activeTileCount = activeTiles,
                    boxesCxCyWhNorm = predBoxes,
                    classScores = predScores,
                    metadata = metadata,
                    confidenceThreshold = confidenceThreshold,
                    outDetections = detections
                )
            } else {
                // Sequential 5 x B=1 1-Graph execution
                for (tileIdx in 0 until activeTiles) {
                    val tileFloats = sweepBuffers.tileRgbFloats[tileIdx]
                    val t0 = System.nanoTime()
                    System.arraycopy(tileFloats, 0, singleTileInputFloats, 0, TILE_FLOAT_COUNT)
                    singleInputs[0].writeFloat(singleTileInputFloats)
                    sModel.run(singleInputs, singleOutputs)
                    val predBoxes = singleOutputs[0].readFloat()
                    val predScores = singleOutputs[1].readFloat()
                    val t1 = System.nanoTime()
                    part1NsAcc += (t1 - t0)

                    decodeCocoSingleTilePredictions(
                        tileIndex = tileIdx,
                        batchElementIndex = 0,
                        boxesCxCyWhNorm = predBoxes,
                        classScores = predScores,
                        metadata = metadata,
                        confidenceThreshold = confidenceThreshold,
                        outDetections = detections
                    )
                }
            }
        } else {
            // Stubbed profiler fallback when .tflite assets are absent in unit tests
            for (tileIdx in 0 until activeTiles) {
                val tileFloats = sweepBuffers.tileRgbFloats[tileIdx]
                val t0 = System.nanoTime()
                val sampleEnergy = tileFloats[0] + tileFloats[tileFloats.size / 2] + tileFloats[tileFloats.size - 1]
                stubEncoderLogits[0] = sampleEnergy * 0.001f
                val t1 = System.nanoTime()
                part1NsAcc += (t1 - t0)

                selectedQueryFeaturesBuffer.clear()
                val outBacking = selectedQueryFeaturesBuffer.array()
                selectTopKQueriesOnHost(
                    encoderLogits = stubEncoderLogits,
                    encoderFeatures = stubEncoderFeatures,
                    numAnchors = DEFAULT_ENCODER_ANCHORS,
                    numClasses = NUM_CLASSES,
                    hiddenDim = QUERY_HIDDEN_DIM,
                    topK = TOP_K_QUERIES,
                    outQueryFeatures = outBacking,
                    outSelectedIndices = selectedTopIndices
                )
                val t2 = System.nanoTime()
                bridgeNsAcc += (t2 - t1)

                val t3 = System.nanoTime()
                part2NsAcc += (t3 - t2)
            }
        }

        val totalMs = ((System.nanoTime() - sweepStartNs) / 1_000_000.0).toFloat()
        val p1Ms = (part1NsAcc / 1_000_000.0).toFloat()
        val bridgeMs = (bridgeNsAcc / 1_000_000.0).toFloat()
        val p2Ms = (part2NsAcc / 1_000_000.0).toFloat()

        sweepCount++
        rollingLatencyMs = if (sweepCount == 1L) {
            totalMs
        } else {
            rollingLatencyMs * 0.85f + totalMs * 0.15f
        }

        if (sweepCount <= 5L || sweepCount % 5L == 0L) {
            Log.i(
                TAG,
                String.format(
                    Locale.US,
                    "RFDETR_BENCH sweep=%d topology=%s batch=%d tiles=%d accel=%s total_ms=%.2f p1_ms=%.2f topk_ms=%.2f p2_ms=%.2f rolling_ms=%.2f dets=%d",
                    sweepCount,
                    activeTopology,
                    compiledBatchSize,
                    activeTiles,
                    activeAcceleratorLabel.replace(' ', '_'),
                    totalMs,
                    p1Ms,
                    bridgeMs,
                    p2Ms,
                    rollingLatencyMs,
                    detections.size
                )
            )
        }

        return SweepInferenceResult(
            tileDetections = detections,
            part1LatencyMs = p1Ms,
            hostTopKBridgeLatencyMs = bridgeMs,
            part2LatencyMs = p2Ms,
            totalSweepLatencyMs = totalMs,
            rollingAvgLatencyMs = rollingLatencyMs,
            isHardwareGraphActive = hardwareGraphsLoaded
        )
    }

    private fun releaseModelsOnly() {
        singleInputs.forEach { runCatching { it.close() } }
        singleOutputs.forEach { runCatching { it.close() } }
        part1Inputs.forEach { runCatching { it.close() } }
        part1Outputs.forEach { runCatching { it.close() } }
        part2Inputs.forEach { runCatching { it.close() } }
        part2Outputs.forEach { runCatching { it.close() } }
        runCatching { singleModel?.close() }
        runCatching { part1Model?.close() }
        runCatching { part2Model?.close() }
        runCatching { environment?.close() }
        singleInputs = emptyList()
        singleOutputs = emptyList()
        part1Inputs = emptyList()
        part1Outputs = emptyList()
        part2Inputs = emptyList()
        part2Outputs = emptyList()
        singleModel = null
        part1Model = null
        part2Model = null
        environment = null
        hardwareGraphsLoaded = false
    }

    override fun close() {
        releaseModelsOnly()
    }

    companion object {
        private const val TAG = "RfDetrTwoGraphGpu"
        const val PART1_ASSET_NAME = CocoStandInMetadata.PART1_ASSET_NAME
        const val PART2_ASSET_NAME = CocoStandInMetadata.PART2_ASSET_NAME
        const val SINGLE_GRAPH_ASSET_NAME = CocoStandInMetadata.SINGLE_GRAPH_ASSET_NAME

        private const val TILE_FLOAT_COUNT =
            StripRoiGeometry.TILE_SIZE_PX * StripRoiGeometry.TILE_SIZE_PX * 3

        const val TOP_K_QUERIES = 100
        const val DEFAULT_ENCODER_ANCHORS = 900
        const val QUERY_HIDDEN_DIM = 256
        val NUM_CLASSES: Int = StoneClass.entries.size // 5 classes (0..3 targets + 4 hard_confuser)

        /**
         * Parses `rf_detr_coco_metadata.json` into a [CocoStandInMetadata] instance, building the
         * `O(1)` integer array lookup table mapping sparse MS-COCO class indices (`cup=47`,
         * `apple=53`, `toothbrush=90`, `remote=75`) to [StoneClass.id] and all other indices to
         * [StoneClass.HARD_CONFUSER].
         *
         * @param jsonString Raw JSON content of `rf_detr_coco_metadata.json`.
         * @param defaultThreshold Fallback score threshold if omitted from JSON.
         * @return Parsed [CocoStandInMetadata].
         */
        fun parseCocoMetadataJson(
            jsonString: String,
            defaultThreshold: Float = CocoStandInMetadata.DEFAULT_SCORE_THRESHOLD
        ): CocoStandInMetadata {
            val root = JSONObject(jsonString)
            val modelName = root.optString("model_name", "rf_detr_nano_coco")
            val inputRes = root.optInt("input_resolution", StripRoiGeometry.TILE_SIZE_PX)
            val batchSize = root.optInt("batch_size", StripRoiGeometry.MAX_TILES)
            val topology = root.optString("topology", CocoStandInMetadata.TOPOLOGY_TWO_GRAPH)

            val assetsObj = root.optJSONObject("asset_files")
            val singleAsset = assetsObj?.optString(
                "single_graph",
                CocoStandInMetadata.SINGLE_GRAPH_ASSET_NAME
            ) ?: CocoStandInMetadata.SINGLE_GRAPH_ASSET_NAME
            val part1Asset = assetsObj?.optString(
                "part1",
                CocoStandInMetadata.PART1_ASSET_NAME
            ) ?: CocoStandInMetadata.PART1_ASSET_NAME
            val part2Asset = assetsObj?.optString(
                "part2",
                CocoStandInMetadata.PART2_ASSET_NAME
            ) ?: CocoStandInMetadata.PART2_ASSET_NAME

            val gridArr = root.optJSONArray("encoder_grid_hw")
            val gridH = gridArr?.optInt(0, CocoStandInMetadata.DEFAULT_ENCODER_GRID_SIZE)
                ?: CocoStandInMetadata.DEFAULT_ENCODER_GRID_SIZE
            val gridW = gridArr?.optInt(1, CocoStandInMetadata.DEFAULT_ENCODER_GRID_SIZE)
                ?: CocoStandInMetadata.DEFAULT_ENCODER_GRID_SIZE
            val numTokens = root.optInt("num_encoder_tokens", gridH * gridW)
            val hiddenDim = root.optInt("hidden_dim", CocoStandInMetadata.DEFAULT_QUERY_HIDDEN_DIM)
            val proposalWh = root.optDouble(
                "proposal_default_wh",
                CocoStandInMetadata.DEFAULT_PROPOSAL_WH.toDouble()
            ).toFloat()
            val numQueries = root.optInt("num_queries", CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES)
            val numClasses = root.optInt("num_classes", CocoStandInMetadata.DEFAULT_COCO_NUM_CLASSES)
            val scoresAreProbs = root.optBoolean("scores_are_probabilities", true)
            val threshold = root.optDouble("score_threshold", defaultThreshold.toDouble()).toFloat()
            val unmappedToConfuser = root.optBoolean("unmapped_confident_to_hard_confuser", true)

            val lutSize = maxOf(numClasses + 1, 128)
            val defaultUnmappedId = if (unmappedToConfuser) StoneClass.HARD_CONFUSER.id else -1
            val lut = IntArray(lutSize) { defaultUnmappedId }

            val standInMap = root.optJSONObject("stand_in_map")
            if (standInMap != null) {
                for (stoneClass in StoneClass.alertableClasses) {
                    val entry = standInMap.optJSONObject(stoneClass.name)
                    if (entry != null && entry.has("class_id")) {
                        val cocoId = entry.getInt("class_id")
                        val targetId = entry.optInt("stone_class_id", stoneClass.id)
                        if (cocoId in lut.indices) {
                            lut[cocoId] = targetId
                        }
                    }
                }
            } else {
                val fallback = CocoStandInMetadata.buildDefaultCocoToStoneClassLut(numClasses)
                System.arraycopy(fallback, 0, lut, 0, minOf(fallback.size, lut.size))
            }

            return CocoStandInMetadata(
                modelName = modelName,
                inputResolution = inputRes,
                batchSize = batchSize,
                topology = topology,
                singleGraphAsset = singleAsset,
                part1Asset = part1Asset,
                part2Asset = part2Asset,
                encoderGridH = gridH,
                encoderGridW = gridW,
                numEncoderTokens = numTokens,
                hiddenDim = hiddenDim,
                proposalDefaultWh = proposalWh,
                numQueries = numQueries,
                numClasses = numClasses,
                scoresAreProbabilities = scoresAreProbs,
                scoreThreshold = threshold,
                cocoClassIdToStoneClassId = lut,
                unmappedConfidentToHardConfuser = unmappedToConfuser
            )
        }

        /**
         * Precomputes the static `[gridH * gridW, 4]` normalized `[cx, cy, w, h]` proposal anchor
         * grid matching `RFDETRNano`'s $24 \times 24$ ($576$-token) encoder grid.
         */
        fun buildNormalizedProposalGrid(
            gridH: Int = CocoStandInMetadata.DEFAULT_ENCODER_GRID_SIZE,
            gridW: Int = CocoStandInMetadata.DEFAULT_ENCODER_GRID_SIZE,
            defaultWh: Float = CocoStandInMetadata.DEFAULT_PROPOSAL_WH
        ): FloatArray {
            val out = FloatArray(gridH * gridW * 4)
            var idx = 0
            for (y in 0 until gridH) {
                val cy = (y + 0.5f) / gridH.toFloat()
                for (x in 0 until gridW) {
                    val cx = (x + 0.5f) / gridW.toFloat()
                    out[idx++] = cx
                    out[idx++] = cy
                    out[idx++] = defaultWh
                    out[idx++] = defaultWh
                }
            }
            return out
        }

        /**
         * Zero-allocation Host CPU Top-K ($K=300$ out of $576$) proposal bridge between Part 1 and
         * Part 2 of the 2-Graph `RF-DETR-Nano` LiteRT split.
         *
         * For each batch item `b in 0 until activeBatchItems`, selects the [topK] spatial tokens
         * with highest `encMaxScores[b, token]` (breaking ties by smaller token index to match
         * `torch.topk(..., sorted=True)`) and gathers both their `outputMemoryPrenorm` vectors
         * (`[topK, hiddenDim]`) and their normalized `baseProposals` (`[topK, 4]`).
         */
        fun selectTopKProposalsFromMaxScores(
            encMaxScores: FloatArray,
            outputMemoryPrenorm: FloatArray,
            baseProposals: FloatArray,
            batchSize: Int,
            activeBatchItems: Int,
            numTokens: Int,
            hiddenDim: Int,
            topK: Int,
            outSelectedPrenorm: FloatArray,
            outSelectedProposals: FloatArray,
            outSelectedIndices: IntArray,
            scratchHeapScores: FloatArray = FloatArray(topK),
            scratchHeapIndices: IntArray = IntArray(topK)
        ) {
            val k = topK.coerceAtMost(numTokens)
            val safeBatch = activeBatchItems.coerceIn(1, batchSize)

            for (b in 0 until batchSize) {
                val prenormBatchDst = b * k * hiddenDim
                val proposalBatchDst = b * k * 4
                val indexBatchDst = b * k

                if (b >= safeBatch) {
                    Arrays.fill(outSelectedPrenorm, prenormBatchDst, prenormBatchDst + k * hiddenDim, 0f)
                    Arrays.fill(outSelectedProposals, proposalBatchDst, proposalBatchDst + k * 4, 0f)
                    Arrays.fill(outSelectedIndices, indexBatchDst, indexBatchDst + k, 0)
                    continue
                }

                for (i in 0 until k) {
                    scratchHeapScores[i] = Float.NEGATIVE_INFINITY
                    scratchHeapIndices[i] = Int.MAX_VALUE
                }

                val scoreBase = b * numTokens
                for (tokenIdx in 0 until numTokens) {
                    val s = encMaxScores[scoreBase + tokenIdx]
                    if (s > scratchHeapScores[0] ||
                        (s == scratchHeapScores[0] && tokenIdx < scratchHeapIndices[0])
                    ) {
                        scratchHeapScores[0] = s
                        scratchHeapIndices[0] = tokenIdx
                        siftDownMinHeapTieAware(scratchHeapScores, scratchHeapIndices, 0, k)
                    }
                }

                // Sort selected K indices descending by score (tie-breaking by ascending token index)
                for (i in 0 until k - 1) {
                    var best = i
                    for (j in i + 1 until k) {
                        val sj = scratchHeapScores[j]
                        val sb = scratchHeapScores[best]
                        if (sj > sb || (sj == sb && scratchHeapIndices[j] < scratchHeapIndices[best])) {
                            best = j
                        }
                    }
                    if (best != i) {
                        val tmpS = scratchHeapScores[i]
                        scratchHeapScores[i] = scratchHeapScores[best]
                        scratchHeapScores[best] = tmpS
                        val tmpI = scratchHeapIndices[i]
                        scratchHeapIndices[i] = scratchHeapIndices[best]
                        scratchHeapIndices[best] = tmpI
                    }
                }

                val prenormBatchSrc = b * numTokens * hiddenDim
                for (i in 0 until k) {
                    val tokenIdx = scratchHeapIndices[i]
                    outSelectedIndices[indexBatchDst + i] = tokenIdx
                    System.arraycopy(
                        outputMemoryPrenorm,
                        prenormBatchSrc + tokenIdx * hiddenDim,
                        outSelectedPrenorm,
                        prenormBatchDst + i * hiddenDim,
                        hiddenDim
                    )
                    System.arraycopy(
                        baseProposals,
                        tokenIdx * 4,
                        outSelectedProposals,
                        proposalBatchDst + i * 4,
                        4
                    )
                }
            }
        }

        private fun siftDownMinHeapTieAware(
            heapScores: FloatArray,
            heapIndices: IntArray,
            startIdx: Int,
            size: Int
        ) {
            var idx = startIdx
            while (true) {
                val left = (idx shl 1) + 1
                if (left >= size) break
                val right = left + 1
                var worst = left
                if (right < size && isWorseCandidate(
                        heapScores[right],
                        heapIndices[right],
                        heapScores[left],
                        heapIndices[left]
                    )
                ) {
                    worst = right
                }
                if (!isWorseCandidate(
                        heapScores[ worst],
                        heapIndices[worst],
                        heapScores[idx],
                        heapIndices[idx]
                    )
                ) {
                    break
                }

                val tmpS = heapScores[idx]
                heapScores[idx] = heapScores[worst]
                heapScores[worst] = tmpS

                val tmpI = heapIndices[idx]
                heapIndices[idx] = heapIndices[worst]
                heapIndices[worst] = tmpI

                idx = worst
            }
        }

        private fun isWorseCandidate(scoreA: Float, idxA: Int, scoreB: Float, idxB: Int): Boolean {
            return scoreA < scoreB || (scoreA == scoreB && idxA > idxB)
        }

        /**
         * Decodes batched (`[B, numQueries, 4]` and `[B, numQueries, numClasses]`) COCO stand-in
         * predictions across `0 until activeTileCount` into [TileDetection] instances using the
         * class routing table in [metadata].
         */
        fun decodeCocoBatchPredictions(
            activeTileCount: Int,
            boxesCxCyWhNorm: FloatArray,
            classScores: FloatArray,
            metadata: CocoStandInMetadata,
            confidenceThreshold: Float = metadata.scoreThreshold,
            outDetections: MutableList<TileDetection>
        ) {
            for (tileIdx in 0 until activeTileCount) {
                decodeCocoSingleTilePredictions(
                    tileIndex = tileIdx,
                    batchElementIndex = tileIdx,
                    boxesCxCyWhNorm = boxesCxCyWhNorm,
                    classScores = classScores,
                    metadata = metadata,
                    confidenceThreshold = confidenceThreshold,
                    outDetections = outDetections
                )
            }
        }

        /**
         * Decodes a single tile's `[numQueries, 4]` normalized `[cx, cy, w, h]` boxes and
         * `[numQueries, numClasses]` probabilities/logits at [batchElementIndex] into [outDetections].
         */
        fun decodeCocoSingleTilePredictions(
            tileIndex: Int,
            batchElementIndex: Int,
            boxesCxCyWhNorm: FloatArray,
            classScores: FloatArray,
            metadata: CocoStandInMetadata,
            confidenceThreshold: Float = metadata.scoreThreshold,
            outDetections: MutableList<TileDetection>
        ) {
            val numQueries = metadata.numQueries
            val numClasses = metadata.numClasses
            val tileSize = StripRoiGeometry.TILE_SIZE_PX.toFloat()
            val lut = metadata.cocoClassIdToStoneClassId
            val scoresAreProbs = metadata.scoresAreProbabilities

            val batchBoxOffset = batchElementIndex * numQueries * 4
            val batchScoreOffset = batchElementIndex * numQueries * numClasses
            if (batchBoxOffset + numQueries * 4 > boxesCxCyWhNorm.size ||
                batchScoreOffset + numQueries * numClasses > classScores.size
            ) {
                return
            }

            for (q in 0 until numQueries) {
                val scoreBase = batchScoreOffset + q * numClasses
                var bestCocoClass = 0
                var bestRaw = classScores[scoreBase]
                for (c in 1 until numClasses) {
                    val v = classScores[scoreBase + c]
                    if (v > bestRaw) {
                        bestRaw = v
                        bestCocoClass = c
                    }
                }

                val prob = if (scoresAreProbs) {
                    bestRaw
                } else {
                    1.0f / (1.0f + exp(-bestRaw))
                }
                if (prob < confidenceThreshold) continue

                val mappedStoneClassId = if (bestCocoClass in lut.indices) {
                    lut[bestCocoClass]
                } else if (metadata.unmappedConfidentToHardConfuser) {
                    StoneClass.HARD_CONFUSER.id
                } else {
                    -1
                }
                if (mappedStoneClassId < 0) continue

                val boxBase = batchBoxOffset + q * 4
                val cx = boxesCxCyWhNorm[boxBase] * tileSize
                val cy = boxesCxCyWhNorm[boxBase + 1] * tileSize
                val w = boxesCxCyWhNorm[boxBase + 2] * tileSize
                val h = boxesCxCyWhNorm[boxBase + 3] * tileSize

                outDetections.add(
                    TileDetection(
                        tileIndex = tileIndex,
                        classId = mappedStoneClassId,
                        confidence = prob,
                        leftPx = (cx - w * 0.5f).coerceIn(0f, tileSize),
                        topPx = (cy - h * 0.5f).coerceIn(0f, tileSize),
                        rightPx = (cx + w * 0.5f).coerceIn(0f, tileSize),
                        bottomPx = (cy + h * 0.5f).coerceIn(0f, tileSize)
                    )
                )
            }
        }

        /**
         * Host CPU Top-K ($K=100$) object query selection bridge for raw `[numAnchors, numClasses]`
         * encoder logits. Selects the [topK] anchor positions with highest max-class logit and
         * gathers their feature vectors contiguously into [outQueryFeatures].
         */
        fun selectTopKQueriesOnHost(
            encoderLogits: FloatArray,
            encoderFeatures: FloatArray,
            numAnchors: Int,
            numClasses: Int,
            hiddenDim: Int,
            topK: Int,
            outQueryFeatures: FloatArray,
            outSelectedIndices: IntArray
        ) {
            val k = topK.coerceAtMost(numAnchors)
            val scores = FloatArray(numAnchors)
            for (a in 0 until numAnchors) {
                val base = a * numClasses
                var maxLogit = Float.NEGATIVE_INFINITY
                for (c in 0 until numClasses) {
                    val v = encoderLogits[base + c]
                    if (v > maxLogit) maxLogit = v
                }
                scores[a] = maxLogit
            }

            // Partial min-heap of size K to find top-K anchor indices in O(N log K) without boxing
            val heapScores = FloatArray(k) { Float.NEGATIVE_INFINITY }
            val heapIndices = IntArray(k) { 0 }

            for (a in 0 until numAnchors) {
                val s = scores[a]
                if (s > heapScores[0]) {
                    heapScores[0] = s
                    heapIndices[0] = a
                    siftDownMinHeap(heapScores, heapIndices, 0, k)
                }
            }

            // Sort selected K indices descending by score for deterministic decoder ordering
            for (i in 0 until k - 1) {
                var best = i
                for (j in i + 1 until k) {
                    if (heapScores[j] > heapScores[best]) best = j
                }
                if (best != i) {
                    val tmpS = heapScores[i]
                    heapScores[i] = heapScores[best]
                    heapScores[best] = tmpS
                    val tmpI = heapIndices[i]
                    heapIndices[i] = heapIndices[best]
                    heapIndices[best] = tmpI
                }
            }

            for (i in 0 until k) {
                val anchorIdx = heapIndices[i]
                outSelectedIndices[i] = anchorIdx
                val srcOffset = anchorIdx * hiddenDim
                val dstOffset = i * hiddenDim
                System.arraycopy(encoderFeatures, srcOffset, outQueryFeatures, dstOffset, hiddenDim)
            }
        }

        private fun siftDownMinHeap(
            heapScores: FloatArray,
            heapIndices: IntArray,
            startIdx: Int,
            size: Int
        ) {
            var idx = startIdx
            while (true) {
                val left = (idx shl 1) + 1
                if (left >= size) break
                val right = left + 1
                var smallest = left
                if (right < size && heapScores[right] < heapScores[left]) {
                    smallest = right
                }
                if (heapScores[idx] <= heapScores[smallest]) break

                val tmpS = heapScores[idx]
                heapScores[idx] = heapScores[smallest]
                heapScores[smallest] = tmpS

                val tmpI = heapIndices[idx]
                heapIndices[idx] = heapIndices[smallest]
                heapIndices[smallest] = tmpI

                idx = smallest
            }
        }

        /**
         * Decodes normalized `[cx, cy, w, h]` boxes and raw 5-class logits into [TileDetection]
         * instances in `[0, 384]` tile pixel coordinates.
         */
        fun decodeTilePredictions(
            tileIndex: Int,
            boxesCxCyWhNorm: FloatArray,
            classLogits: FloatArray,
            confidenceThreshold: Float,
            outDetections: MutableList<TileDetection>
        ) {
            val numQueries = minOf(boxesCxCyWhNorm.size / 4, classLogits.size / NUM_CLASSES)
            val tileSize = StripRoiGeometry.TILE_SIZE_PX.toFloat()

            for (q in 0 until numQueries) {
                val logitBase = q * NUM_CLASSES
                var bestClass = 0
                var bestLogit = classLogits[logitBase]
                for (c in 1 until NUM_CLASSES) {
                    val v = classLogits[logitBase + c]
                    if (v > bestLogit) {
                        bestLogit = v
                        bestClass = c
                    }
                }

                val prob = 1.0f / (1.0f + exp(-bestLogit))
                if (prob < confidenceThreshold) continue

                val boxBase = q * 4
                val cx = boxesCxCyWhNorm[boxBase] * tileSize
                val cy = boxesCxCyWhNorm[boxBase + 1] * tileSize
                val w = boxesCxCyWhNorm[boxBase + 2] * tileSize
                val h = boxesCxCyWhNorm[boxBase + 3] * tileSize

                outDetections.add(
                    TileDetection(
                        tileIndex = tileIndex,
                        classId = bestClass,
                        confidence = prob,
                        leftPx = (cx - w * 0.5f).coerceIn(0f, tileSize),
                        topPx = (cy - h * 0.5f).coerceIn(0f, tileSize),
                        rightPx = (cx + w * 0.5f).coerceIn(0f, tileSize),
                        bottomPx = (cy + h * 0.5f).coerceIn(0f, tileSize)
                    )
                )
            }
        }
    }
}
