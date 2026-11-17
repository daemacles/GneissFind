/*
 * Multi-resolution, multi-architecture LiteRT 2.3.x GPU (FP16 / Quantized) & Tensor G5 NPU detector
 * running RF-DETR-Nano (384/320/256, K=300/100, B=5/6/3-Interleaved/1) and YOLO26n/s across the
 * push-broom tile swath.
 *
 * Consumes pre-allocated NHWC RGB [0..1] float tiles from PushBroomSweepBuffers and class-index
 * routing rules from assets/rf_detr_coco_metadata.json or LiveModelProfile; produces tile-local
 * TileDetection lists and stage latency telemetry for ScannerViewModel.
 *
 * Main types:
 * - LiveModelProfile: selectable on-device model/resolution/scheduling profiles for the LCARS SOURCE drawer.
 * - CocoStandInMetadata: parsed runtime configuration and MS-COCO-to-StoneClass routing table.
 * - RfDetrTwoGraphGpuDetector: zero-allocation 2-Graph/1-Graph LiteRT GPU/NPU detector and host Top-K proposal bridge.
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
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Selectable live hardware detector profiles exposed in the LCARS `SOURCE` drawer and benchmark harness.
 *
 * @property id Deterministic short token used for per-profile LiteRT OpenCL shader disk cache keys.
 * @property modelName Canonical profile identifier logged in telemetry.
 * @property statusBadge Human-readable status string shown in the LCARS `SOURCE` drawer.
 * @property topology Execution topology (`"two_graph"` or `"single_graph"`).
 * @property inputResolution Square tile input resolution in pixels (`384`, `320`, or `256`).
 * @property batchSize Static batch dimension (`5`, `6`, `3`, or `1`).
 * @property numQueries Decoder Top-K query count (`300` or `100`) or dense anchor count (`3024` for YOLO26).
 * @property numClasses Output class count (`91` for sparse COCO RF-DETR, `80` for contiguous COCO YOLO26).
 * @property singleGraphAsset Asset filename when [topology] is `"single_graph"`.
 * @property part1Asset Asset filename for Part 1 when [topology] is `"two_graph"`.
 * @property part2Asset Asset filename for Part 2 when [topology] is `"two_graph"`.
 * @property interleaveEvenOdd True when a `B=3` model alternates even (`[0,2,4]`) and odd (`[1,3,priority]`)
 *   tiles across consecutive sweeps while caching opposite-parity detections for 1 sweep.
 */
enum class LiveModelProfile(
    val id: String,
    val modelName: String,
    val statusBadge: String,
    val topology: String,
    val inputResolution: Int,
    val batchSize: Int,
    val numQueries: Int,
    val numClasses: Int,
    val singleGraphAsset: String,
    val part1Asset: String,
    val part2Asset: String,
    val interleaveEvenOdd: Boolean = false
) {
    /**
     * Milestone 2B Apache-2.0 High-Recall Balanced Winner: `RF-DETR-Nano` with `3x320x320` Even/Odd
     * Priority Sub-Swath Interleaving, `W8-Dyn` quantized weights (`28.3 MB`), and `K=100` Top-K
     * queries (`100%` Small `38-45px` recall, `141.5 ms` / `7.07 FPS` cold, `-60.1%` FLOPs/sweep).
     */
    RFDETR_NANO_320_B3_INTERLEAVED(
        id = "rfdetr_320_b3_w8dyn",
        modelName = "rf_detr_nano_b3_320_k100_w8dyn_interleaved",
        statusBadge = "RF-DETR 320 B=3 W8-DYN • GPU FP16",
        topology = CocoStandInMetadata.TOPOLOGY_TWO_GRAPH,
        inputResolution = 320,
        batchSize = 3,
        numQueries = 100,
        numClasses = 91,
        singleGraphAsset = CocoStandInMetadata.SINGLE_GRAPH_ASSET_NAME,
        part1Asset = "rf_detr_nano_part1_b3_320_w8dyn.tflite",
        part2Asset = "rf_detr_nano_part2_b3_320_k100_w8dyn.tflite",
        interleaveEvenOdd = true
    ),

    /**
     * Milestone 2B Apache-2.0 Full-Frame Every-Sweep Winner: `RF-DETR-Nano` at `5x256x256` (`K=100`,
     * `W8-Dyn` `28.5 MB`) evaluating all 5 tiles on every single sweep (`158-162 ms` / `6.15-6.33 FPS`
     * cold, `206 ms` / `4.85 FPS` 180s soaked, `87.5%` Small `38-45px` recall).
     */
    RFDETR_NANO_256_B5_K100(
        id = "rfdetr_256_b5_w8dyn",
        modelName = "rf_detr_nano_b5_256_k100_w8dyn",
        statusBadge = "RF-DETR 256 B=5 W8-DYN • GPU FP16",
        topology = CocoStandInMetadata.TOPOLOGY_TWO_GRAPH,
        inputResolution = 256,
        batchSize = 5,
        numQueries = 100,
        numClasses = 91,
        singleGraphAsset = CocoStandInMetadata.SINGLE_GRAPH_ASSET_NAME,
        part1Asset = "rf_detr_nano_part1_b5_256_w8dyn.tflite",
        part2Asset = "rf_detr_nano_part2_b5_256_k100_w8dyn.tflite",
        interleaveEvenOdd = false
    ),

    /**
     * Milestone 2B Apache-2.0 Ultra-Fast Sustained Winner: `RF-DETR-Nano` with `3x256x256` Even/Odd
     * Priority Sub-Swath Interleaving, `W8-Dyn` weights (`28.1 MB`), and `K=100` (`108.2 ms` /
     * `9.24 FPS` cold, `152.2 ms` / `6.57 FPS` sustained after 180s thermal soak).
     */
    RFDETR_NANO_256_B3_INTERLEAVED(
        id = "rfdetr_256_b3_w8dyn",
        modelName = "rf_detr_nano_b3_256_k100_w8dyn_interleaved",
        statusBadge = "RF-DETR 256 B=3 W8-DYN • GPU FP16",
        topology = CocoStandInMetadata.TOPOLOGY_TWO_GRAPH,
        inputResolution = 256,
        batchSize = 3,
        numQueries = 100,
        numClasses = 91,
        singleGraphAsset = CocoStandInMetadata.SINGLE_GRAPH_ASSET_NAME,
        part1Asset = "rf_detr_nano_part1_b3_256_w8dyn.tflite",
        part2Asset = "rf_detr_nano_part2_b3_256_k100_w8dyn.tflite",
        interleaveEvenOdd = true
    ),

    /**
     * Milestone 2B Apache-2.0 Full-Frame `5x320x320` (`K=100`): `100%` Small `38-45px` recall
     * across all 5 tiles every sweep (`-33.4%` FLOPs/sweep).
     */
    RFDETR_NANO_320_B5_K100(
        id = "rfdetr_320_b5_k100",
        modelName = "rf_detr_nano_b5_320_k100",
        statusBadge = "RF-DETR 320 B=5 K100 • GPU FP16",
        topology = CocoStandInMetadata.TOPOLOGY_TWO_GRAPH,
        inputResolution = 320,
        batchSize = 5,
        numQueries = 100,
        numClasses = 91,
        singleGraphAsset = CocoStandInMetadata.SINGLE_GRAPH_ASSET_NAME,
        part1Asset = "rf_detr_nano_part1_b5_320.tflite",
        part2Asset = "rf_detr_nano_part2_b5_320_k100.tflite",
        interleaveEvenOdd = false
    ),

    /**
     * Milestone 2B Single-Graph CNN Speed Reference (`AGPL-3.0` benchmark): `YOLO26s` at `5x384x384`
     * on GPU FP16 (`41.15 GFLOPs` per 5-tile sweep).
     */
    YOLO26S_384_B5_GPU(
        id = "yolo26s_384_b5",
        modelName = "yolo26s_b5_384",
        statusBadge = "YOLO26-S 384 B=5 • GPU FP16 [AGPL]",
        topology = CocoStandInMetadata.TOPOLOGY_SINGLE_GRAPH,
        inputResolution = 384,
        batchSize = 5,
        numQueries = 3024,
        numClasses = 80,
        singleGraphAsset = "yolo26s_b5_384.tflite",
        part1Asset = CocoStandInMetadata.PART1_ASSET_NAME,
        part2Asset = CocoStandInMetadata.PART2_ASSET_NAME,
        interleaveEvenOdd = false
    ),

    /**
     * Milestone 2 Baseline Reference: `RF-DETR-Nano` at `5x384x384` (`K=300` queries, `148.55 GFLOPs`).
     */
    RFDETR_NANO_384_BASELINE(
        id = "rfdetr_384_b5_k300",
        modelName = "rf_detr_nano_b5_384_k300",
        statusBadge = "RF-DETR 384 B=5 K300 • BASELINE",
        topology = CocoStandInMetadata.TOPOLOGY_TWO_GRAPH,
        inputResolution = 384,
        batchSize = 5,
        numQueries = 300,
        numClasses = 91,
        singleGraphAsset = CocoStandInMetadata.SINGLE_GRAPH_ASSET_NAME,
        part1Asset = CocoStandInMetadata.PART1_ASSET_NAME,
        part2Asset = CocoStandInMetadata.PART2_ASSET_NAME,
        interleaveEvenOdd = false
    )
}

/**
 * Parsed runtime configuration and MS-COCO stand-in class routing table loaded from
 * `assets/rf_detr_coco_metadata.json` (`.agents/rules/litert-inference.md §4`).
 *
 * @property modelName Identifier of the exported model family.
 * @property inputResolution Square tile width and height in pixels (`384`, `320`, or `256`).
 * @property batchSize Configured static batch dimension (`5`, `6`, `3`, or `1`).
 * @property topology Graph execution topology: `"two_graph"` or `"single_graph"`.
 * @property singleGraphAsset Asset filename for the single-graph `.tflite` model.
 * @property part1Asset Asset filename for Part 1 (backbone + encoder + proposal scores).
 * @property part2Asset Asset filename for Part 2 (decoder + detection head).
 * @property encoderGridH Spatial height of the encoder proposal grid (`24` at `384`, `20` at `320`, `16` at `256`).
 * @property encoderGridW Spatial width of the encoder proposal grid (`24` at `384`, `20` at `320`, `16` at `256`).
 * @property numEncoderTokens Total encoder spatial tokens (`encoderGridH * encoderGridW`).
 * @property hiddenDim Transformer decoder embedding width (`256`).
 * @property proposalDefaultWh Default normalized proposal width and height (`0.05f`).
 * @property numQueries Number of top-scoring object queries passed to the decoder (`300` or `100`, or `3024` for YOLO26).
 * @property numClasses Number of output class logits/probabilities per query (`91` for sparse COCO, `80` for contiguous COCO).
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
        const val LOCK_ON_METADATA_ASSET_NAME = "rf_detr_320_b1_metadata.json"
        const val BENCHMARK_OVERRIDE_FILE_NAME = "benchmark_override.json"
        const val TOPOLOGY_TWO_GRAPH = "two_graph"
        const val TOPOLOGY_SINGLE_GRAPH = "single_graph"

        const val SINGLE_GRAPH_ASSET_NAME = "rf_detr_nano_coco.tflite"
        const val PART1_ASSET_NAME = "rf_detr_nano_part1_backbone_encoder.tflite"
        const val PART2_ASSET_NAME = "rf_detr_nano_part2_decoder_head.tflite"
        const val LOCK_ON_PART1_ASSET_NAME = "rf_detr_nano_part1_b1_320_w8dyn.tflite"
        const val LOCK_ON_PART2_ASSET_NAME = "rf_detr_nano_part2_b1_320_k100_w8dyn.tflite"
        const val LOCK_ON_CROP_RESOLUTION_PX = 320
        const val LOCK_ON_TOP_K_QUERIES = 100

        const val DEFAULT_ENCODER_GRID_SIZE = 24
        const val DEFAULT_ENCODER_TOKENS = 576
        const val DEFAULT_QUERY_HIDDEN_DIM = 256
        const val DEFAULT_PROPOSAL_WH = 0.05f
        const val DEFAULT_COCO_TOP_K_QUERIES = 300
        const val DEFAULT_COCO_NUM_CLASSES = 91
        const val CONTIGUOUS_COCO_NUM_CLASSES = 80
        const val DEFAULT_SCORE_THRESHOLD = 0.35f

        /**
         * Builds the default 91-class sparse MS-COCO lookup table verified against `rfdetr 1.12.1`
         * (or the 80-class contiguous MS-COCO lookup table when `numClasses == 80` for `YOLO26`):
         * - Sparse 91-class (`numClasses != 80`):
         *   `47` (`cup`) -> `0`, `53` (`apple`) -> `1`, `90` (`toothbrush`) -> `2`, `75` (`remote`) -> `3`.
         * - Contiguous 80-class (`numClasses == 80`):
         *   `41` (`cup`) -> `0`, `47` (`apple`) -> `1`, `79` (`toothbrush`) -> `2`, `65` (`remote`) -> `3`.
         *
         * @param numClasses Number of output class slots in the model head.
         * @return Integer lookup table mapping model class ID to [StoneClass.id].
         */
        fun buildDefaultCocoToStoneClassLut(numClasses: Int = DEFAULT_COCO_NUM_CLASSES): IntArray {
            val lutSize = maxOf(numClasses + 1, 128)
            return IntArray(lutSize) { StoneClass.HARD_CONFUSER.id }.apply {
                if (numClasses == CONTIGUOUS_COCO_NUM_CLASSES) {
                    this[41] = StoneClass.CHALCEDONY_AGATE.id
                    this[47] = StoneClass.JASPER_RED_YELLOW.id
                    this[79] = StoneClass.GREEN_GEM_ROCK.id
                    this[65] = StoneClass.PETRIFIED_WOOD.id
                } else {
                    this[47] = StoneClass.CHALCEDONY_AGATE.id
                    this[53] = StoneClass.JASPER_RED_YELLOW.id
                    this[90] = StoneClass.GREEN_GEM_ROCK.id
                    this[75] = StoneClass.PETRIFIED_WOOD.id
                }
            }
        }
    }
}

/**
 * LiteRT 2.3.x GPU (FP16 / Quantized) & Google Tensor G5 NPU detector executing `RF-DETR-Nano`
 * ($384 \times 384$, $320 \times 320$, or $256 \times 256$ tiles) and `YOLO26n/s` in either 2-Graph
 * split topology (`Part 1` backbone/encoder + Kotlin Host Top-K proposal gather + `Part 2` decoder)
 * or 1-Graph end-to-end topology, supporting static `B=5`/`B=6` batched dispatch, `B=3` Even/Odd
 * Priority Sub-Swath Interleaving, and `5 x B=1` sequential dispatch (`.agents/rules/litert-inference.md §3–§5`).
 *
 * Responsibility and state: owns the LiteRT [Environment], [CompiledModel] handles, native
 * [TensorBuffer] bindings, precomputed bilinear resampling tables, and pre-allocated staging arrays
 * so [detectSweep] performs zero heap allocations per frame.
 *
 * Lifecycle: lazily initializes LiteRT GPU/NPU models on the first [detectSweep] call so OpenCL/EGL
 * contexts bind to the inference worker thread; release via [close].
 *
 * Threading: confined to the single-threaded inference executor.
 *
 * @param appContext Application context used to read `assets/` and locate `codeCacheDir` for
 *   serialized GPU/NPU shader caching.
 * @param confidenceThreshold Minimum probability threshold (`[0.0, 1.0]`) to retain a query box.
 * @param initialProfile Initial [LiveModelProfile] loaded when no `benchmark_override.json` is active.
 */
class RfDetrTwoGraphGpuDetector(
    private val appContext: Context,
    private val confidenceThreshold: Float = CocoStandInMetadata.DEFAULT_SCORE_THRESHOLD,
    initialProfile: LiveModelProfile = LiveModelProfile.RFDETR_NANO_384_BASELINE
) : StoneDetector {

    override val backend: DetectorBackend = DetectorBackend.RF_DETR_NANO_GPU

    @Volatile
    private var desiredLiveProfile: LiveModelProfile = initialProfile
    private var activeLiveProfile: LiveModelProfile = initialProfile

    /**
     * Optional listener invoked on the inference worker thread when asynchronous GPU shader
     * compilation/deserialization + 1-pass warmup starts (`isCompiling = true`) and finishes (`false`).
     */
    @Volatile
    var onCompilationStateChanged: ((isCompiling: Boolean, statusMessage: String) -> Unit)? = null

    private val gpuShaderCacheDir: String by lazy {
        File(appContext.codeCacheDir, GPU_SHADER_CACHE_SUBDIR).apply { mkdirs() }.absolutePath
    }

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
    private var part2ZeroCopyInputs: List<TensorBuffer> = emptyList()

    // Dedicated pre-warmed B=1 320x320 W8-Dyn Lock-On models kept warm alongside Patrol models
    private var lockOnInitializedOnThread: Boolean = false
    private var lockOnHardwareLoaded: Boolean = false
    private var lockOnPart1Model: CompiledModel? = null
    private var lockOnPart2Model: CompiledModel? = null
    private var lockOnPart1Inputs: List<TensorBuffer> = emptyList()
    private var lockOnPart1Outputs: List<TensorBuffer> = emptyList()
    private var lockOnPart2Inputs: List<TensorBuffer> = emptyList()
    private var lockOnPart2Outputs: List<TensorBuffer> = emptyList()
    private var lockOnPart2ZeroCopyInputs: List<TensorBuffer> = emptyList()
    private var lockOnMetadata: CocoStandInMetadata = CocoStandInMetadata(
        modelName = "rf_detr_nano_320_b1_w8dyn",
        inputResolution = CocoStandInMetadata.LOCK_ON_CROP_RESOLUTION_PX,
        batchSize = 1,
        encoderGridH = 20,
        encoderGridW = 20,
        numEncoderTokens = 400,
        numQueries = CocoStandInMetadata.LOCK_ON_TOP_K_QUERIES,
        numClasses = CocoStandInMetadata.DEFAULT_COCO_NUM_CLASSES,
        scoreThreshold = LOCK_ON_DEFAULT_SCORE_THRESHOLD
    )
    private var lockOnBoxesOutputIdx: Int = 0
    private var lockOnScoresOutputIdx: Int = 1
    private var lockOnUseZeroCopyEncMemory: Boolean = true
    private var lockOnBaseProposals: FloatArray = buildNormalizedProposalGrid(
        gridH = 20,
        gridW = 20,
        defaultWh = CocoStandInMetadata.DEFAULT_PROPOSAL_WH
    )
    private val lockOnSelectedPrenormFloats = FloatArray(
        CocoStandInMetadata.LOCK_ON_TOP_K_QUERIES * CocoStandInMetadata.DEFAULT_QUERY_HIDDEN_DIM
    )
    private val lockOnSelectedProposalsFloats = FloatArray(
        CocoStandInMetadata.LOCK_ON_TOP_K_QUERIES * 4
    )
    private val lockOnTopIndices = IntArray(CocoStandInMetadata.LOCK_ON_TOP_K_QUERIES)
    private val lockOnScratchHeapScores = FloatArray(CocoStandInMetadata.LOCK_ON_TOP_K_QUERIES)
    private val lockOnScratchHeapIndices = IntArray(CocoStandInMetadata.LOCK_ON_TOP_K_QUERIES)
    private val lockOnDummyInputFloats = FloatArray(
        CocoStandInMetadata.LOCK_ON_CROP_RESOLUTION_PX *
            CocoStandInMetadata.LOCK_ON_CROP_RESOLUTION_PX * 3
    )
    private var lockOnRollingLatencyMs: Float = 0f
    private var lockOnSweepCount: Long = 0L

    private var metadata: CocoStandInMetadata = CocoStandInMetadata(scoreThreshold = confidenceThreshold)
    private var activeTopology: String = CocoStandInMetadata.TOPOLOGY_TWO_GRAPH
    private var activeAcceleratorLabel: String = "GPU FP16"
    private var compiledBatchSize: Int = StripRoiGeometry.MAX_TILES
    private var activeInputResolution: Int = StripRoiGeometry.TILE_SIZE_PX
    private var activeTileFloatCount: Int = DEFAULT_TILE_FLOAT_COUNT
    private var singleBoxesOutputIdx: Int = 0
    private var singleScoresOutputIdx: Int = 1
    private var part2BoxesOutputIdx: Int = 0
    private var part2ScoresOutputIdx: Int = 1
    private var useZeroCopyEncMemory: Boolean = true
    private var interleaveEvenOdd: Boolean = false

    private var overrideActiveTiles: Int? = null
    private var overridePreferTextureWeights: Boolean? = null
    private var overrideConstantTensorSharing: Boolean? = null
    private var overrideAllowQuantizedFcConv: Boolean? = null
    private var overrideNumStepsCmdBuffer: Int? = null
    private var overrideBufferStorageType: CompiledModel.GpuOptions.BufferStorageType? = null
    private var overrideAcceleratorMode: String = "gpu"
    private var lastOverrideMtimeMs: Long = -1L

    private var initializedOnThread = false
    private var hardwareGraphsLoaded = false
    private var activeSlotFromLru = false
    private val warmPatrolLru = LinkedHashMap<LiveModelProfile, WarmPatrolSlot>(4, 0.75f, true)
    private var initMessage: String = initialProfile.statusBadge

    // Dynamic staging arrays sized once at model load time to match exact TensorBuffer byte capacities
    private var directInputByteBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(MAX_SUPPORTED_BATCH * DEFAULT_TILE_FLOAT_COUNT * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
    private var directInputFloatBuffer: FloatBuffer = directInputByteBuffer.asFloatBuffer()
    private var batchedInputFloats = FloatArray(StripRoiGeometry.MAX_TILES * DEFAULT_TILE_FLOAT_COUNT)
    private var singleTileInputFloats = FloatArray(DEFAULT_TILE_FLOAT_COUNT)

    // Precomputed 1D bilinear resampling tables for 384 -> activeInputResolution (320 or 256)
    private var resampleY0 = IntArray(StripRoiGeometry.TILE_SIZE_PX)
    private var resampleY1 = IntArray(StripRoiGeometry.TILE_SIZE_PX)
    private var resampleWy0 = FloatArray(StripRoiGeometry.TILE_SIZE_PX)
    private var resampleWy1 = FloatArray(StripRoiGeometry.TILE_SIZE_PX)
    private var resampleX0 = IntArray(StripRoiGeometry.TILE_SIZE_PX)
    private var resampleX1 = IntArray(StripRoiGeometry.TILE_SIZE_PX)
    private var resampleWx0 = FloatArray(StripRoiGeometry.TILE_SIZE_PX)
    private var resampleWx1 = FloatArray(StripRoiGeometry.TILE_SIZE_PX)

    // Precomputed 6x256 strip-to-tile X tables and reconstructed strip cache for B=6 256x256 mode
    private val reconstructedStripRgb = FloatArray(
        StripRoiGeometry.MAX_STRIP_WIDTH_PX * StripRoiGeometry.STRIP_HEIGHT_PX * 3
    )
    private val sixTileX0 = IntArray(256)
    private val sixTileX1 = IntArray(256)
    private val sixTileWx0 = FloatArray(256)
    private val sixTileWx1 = FloatArray(256)

    // Pre-allocated 2-graph Host Top-K bridge tables (resized on model initialization to match exact TensorBuffer sizes)
    private var baseProposalsTable: FloatArray = buildNormalizedProposalGrid(
        gridH = CocoStandInMetadata.DEFAULT_ENCODER_GRID_SIZE,
        gridW = CocoStandInMetadata.DEFAULT_ENCODER_GRID_SIZE,
        defaultWh = CocoStandInMetadata.DEFAULT_PROPOSAL_WH
    )
    private var batchedSelectedPrenormFloats = FloatArray(
        StripRoiGeometry.MAX_TILES *
            CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES *
            CocoStandInMetadata.DEFAULT_QUERY_HIDDEN_DIM
    )
    private var batchedSelectedProposalsFloats = FloatArray(
        StripRoiGeometry.MAX_TILES * CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES * 4
    )
    private var singleSelectedPrenormFloats = FloatArray(
        CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES *
            CocoStandInMetadata.DEFAULT_QUERY_HIDDEN_DIM
    )
    private var singleSelectedProposalsFloats = FloatArray(
        CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES * 4
    )
    private var batchedTopIndices = IntArray(
        StripRoiGeometry.MAX_TILES * CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES
    )
    private var scratchHeapScores = FloatArray(CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES)
    private var scratchHeapIndices = IntArray(CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES)

    // Track 4 Even/Odd Sub-Swath Interleaving state (caches per-tile detections for at most 1 sweep)
    private val interleavedBatchTileMap = IntArray(3)
    private val cachedTileDetections = Array(StripRoiGeometry.MAX_TILES) { ArrayList<TileDetection>(8) }
    private val cachedTileSweepStamp = LongArray(StripRoiGeometry.MAX_TILES) { -10L }
    private var priorityEvenTileIndex: Int = 2
    private val maxAlertableScoreScratch = FloatArray(1)

    // Legacy fallback profiler buffers used only when .tflite assets are absent in unit tests
    private val stubEncoderLogits = FloatArray(DEFAULT_ENCODER_ANCHORS * NUM_CLASSES)
    private val stubEncoderFeatures = FloatArray(DEFAULT_ENCODER_ANCHORS * QUERY_HIDDEN_DIM)
    private val selectedQueryFeaturesBuffer: FloatBuffer =
        FloatBuffer.allocate(TOP_K_QUERIES * QUERY_HIDDEN_DIM)
    private val selectedTopIndices = IntArray(TOP_K_QUERIES)

    private var rollingLatencyMs: Float = 0f
    private var sweepCount: Long = 0L

    init {
        buildResampleTables(StripRoiGeometry.TILE_SIZE_PX)
        buildSixTileXTables()
    }

    override val isHardwareReady: Boolean
        get() = hardwareGraphsLoaded

    /** True when the dedicated `B=1` `320x320` `W8-Dyn` Lock-On graphs are compiled and warm on GPU. */
    val isLockOnHardwareReady: Boolean
        get() = lockOnHardwareLoaded

    /** True when [desiredLiveProfile] is already compiled and warm on the inference thread. */
    val isDesiredProfileCompiled: Boolean
        get() = initializedOnThread && activeLiveProfile == desiredLiveProfile

    override val statusSummary: String
        get() = initMessage

    /** Currently selected [LiveModelProfile] when running without an external `benchmark_override.json`. */
    val currentLiveProfile: LiveModelProfile
        get() = desiredLiveProfile

    /**
     * Switches the active [LiveModelProfile] from the LCARS `SOURCE` drawer. The new LiteRT graphs
     * are hot-swapped asynchronously on the inference worker thread.
     */
    fun setLiveModelProfile(profile: LiveModelProfile) {
        if (desiredLiveProfile != profile) {
            desiredLiveProfile = profile
        }
    }

    /**
     * Eagerly compiles/deserializes and runs a 1-pass GPU warmup for both the active Patrol
     * [LiveModelProfile] and the `B=1` `320x320` Lock-On graphs on `GneissFind-Inference` so
     * profile switches and Lock-On transitions never block UI or stall on first frame.
     */
    fun ensureCompiledAndWarmOnInferenceThread() {
        ensureInitializedOnAnalyzerThread()
        ensureLockOn320ModelWarmOnAnalyzerThread()
    }

    /**
     * Checks whether `benchmark_override.json` or [desiredLiveProfile] changed so models can
     * hot-swap on the inference thread without restarting CameraX.
     */
    private fun checkBenchmarkOverrideHotSwap() {
        val overrideFile = resolveBenchmarkOverrideFile()
        val mtime = if (overrideFile != null && overrideFile.exists()) overrideFile.lastModified() else 0L
        val overrideChanged = mtime != lastOverrideMtimeMs
        val profileChanged = desiredLiveProfile != activeLiveProfile
        if (initializedOnThread && (overrideChanged || profileChanged)) {
            Log.i(
                TAG,
                "Hot-swapping LiteRT models (overrideMtime=$mtime, profile=${desiredLiveProfile.name})"
            )
            detachOrReleaseActiveModels()
            if (overrideChanged || overrideFile != null) {
                evictAllWarmPatrolSlots()
            }
            initializedOnThread = false
            sweepCount = 0L
            rollingLatencyMs = 0f
            resetInterleaveCache()
        }
    }

    /** Clears the 1-sweep Even/Odd tile detection cache. */
    private fun resetInterleaveCache() {
        for (i in 0 until StripRoiGeometry.MAX_TILES) {
            cachedTileDetections[i].clear()
            cachedTileSweepStamp[i] = -10L
        }
        priorityEvenTileIndex = 2
    }

    /**
     * Resolves an optional `benchmark_override.json` file in external or internal app storage,
     * returning `null` if neither exists.
     */
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
     * Precomputes 1D bilinear interpolation indices (`*3` for RGB stride on X) and weights mapping
     * `[0 until dstRes]` back into a `[384 x 384]` source tile.
     */
    private fun buildResampleTables(dstRes: Int) {
        val srcRes = StripRoiGeometry.TILE_SIZE_PX
        resampleY0 = IntArray(dstRes)
        resampleY1 = IntArray(dstRes)
        resampleWy0 = FloatArray(dstRes)
        resampleWy1 = FloatArray(dstRes)
        resampleX0 = IntArray(dstRes)
        resampleX1 = IntArray(dstRes)
        resampleWx0 = FloatArray(dstRes)
        resampleWx1 = FloatArray(dstRes)

        val scale = srcRes.toFloat() / dstRes.toFloat()
        for (i in 0 until dstRes) {
            val srcCoord = ((i + 0.5f) * scale - 0.5f).coerceIn(0f, (srcRes - 1).toFloat())
            val i0 = srcCoord.toInt().coerceIn(0, srcRes - 1)
            val i1 = (i0 + 1).coerceAtMost(srcRes - 1)
            val w1 = srcCoord - i0
            val w0 = 1.0f - w1

            resampleY0[i] = i0 * srcRes * 3
            resampleY1[i] = i1 * srcRes * 3
            resampleWy0[i] = w0
            resampleWy1[i] = w1

            resampleX0[i] = i0 * 3
            resampleX1[i] = i1 * 3
            resampleWx0[i] = w0
            resampleWx1[i] = w1
        }
    }

    /** Precomputes horizontal bilinear sampling tables mapping `256 px` into a `320 px` wide strip window. */
    private fun buildSixTileXTables() {
        val srcW = SIX_TILE_WINDOW_WIDTH_PX
        val dstW = 256
        val scale = srcW.toFloat() / dstW.toFloat()
        for (x in 0 until dstW) {
            val srcCoord = ((x + 0.5f) * scale - 0.5f).coerceIn(0f, (srcW - 1).toFloat())
            val x0 = srcCoord.toInt().coerceIn(0, srcW - 1)
            val x1 = (x0 + 1).coerceAtMost(srcW - 1)
            val w1 = srcCoord - x0
            sixTileX0[x] = x0
            sixTileX1[x] = x1
            sixTileWx0[x] = 1.0f - w1
            sixTileWx1[x] = w1
        }
    }

    /**
     * Lazily initializes LiteRT [Environment] and [CompiledModel] instances on the single-threaded
     * inference worker so GPU OpenCL/EGL and NPU contexts maintain strict thread affinity.
     */
    private fun ensureInitializedOnAnalyzerThread() {
        checkBenchmarkOverrideHotSwap()
        if (initializedOnThread) return
        initializedOnThread = true

        val overrideFile = resolveBenchmarkOverrideFile()
        lastOverrideMtimeMs = if (overrideFile != null && overrideFile.exists()) overrideFile.lastModified() else 0L
        activeLiveProfile = desiredLiveProfile

        if (overrideFile == null) {
            val cachedSlot = warmPatrolLru[activeLiveProfile]
            if (cachedSlot != null) {
                val swapStartNs = System.nanoTime()
                restoreWarmPatrolSlot(cachedSlot)
                val swapMs = (System.nanoTime() - swapStartNs) / 1_000_000.0
                onCompilationStateChanged?.invoke(false, initMessage)
                Log.i(
                    TAG,
                    String.format(
                        Locale.US,
                        "Warm LRU SRC switch to %s completed in %.2f ms (slots=%d/%d)",
                        initMessage,
                        swapMs,
                        warmPatrolLru.size,
                        MAX_WARM_PATROL_SLOTS
                    )
                )
                return
            }
        }

        try {
            val assetList = appContext.assets.list("")?.toSet().orEmpty()
            var baseMeta = if (CocoStandInMetadata.METADATA_ASSET_NAME in assetList) {
                val jsonText = appContext.assets.open(CocoStandInMetadata.METADATA_ASSET_NAME)
                    .bufferedReader()
                    .use { it.readText() }
                parseCocoMetadataJson(jsonText, confidenceThreshold)
            } else {
                CocoStandInMetadata(scoreThreshold = confidenceThreshold)
            }

            // Apply active LiveModelProfile defaults before optional benchmark_override.json
            val profile = activeLiveProfile
            val profileGridSide = profile.inputResolution / 16
            baseMeta = baseMeta.copy(
                modelName = profile.modelName,
                inputResolution = profile.inputResolution,
                batchSize = profile.batchSize,
                topology = profile.topology,
                singleGraphAsset = if (profile.singleGraphAsset in assetList) {
                    profile.singleGraphAsset
                } else {
                    baseMeta.singleGraphAsset
                },
                part1Asset = if (profile.part1Asset in assetList) {
                    profile.part1Asset
                } else {
                    baseMeta.part1Asset
                },
                part2Asset = if (profile.part2Asset in assetList) {
                    profile.part2Asset
                } else {
                    baseMeta.part2Asset
                },
                encoderGridH = profileGridSide,
                encoderGridW = profileGridSide,
                numEncoderTokens = profileGridSide * profileGridSide,
                numQueries = profile.numQueries,
                numClasses = profile.numClasses,
                cocoClassIdToStoneClassId = CocoStandInMetadata.buildDefaultCocoToStoneClassLut(profile.numClasses)
            )

            val overrideJson = overrideFile?.readText()?.let { JSONObject(it) }
            val desiredTopology = overrideJson?.optString("topology", baseMeta.topology) ?: baseMeta.topology
            val singleFileOverride = overrideJson?.optString("single_graph_path", "")?.takeIf { it.isNotEmpty() }
            val part1FileOverride = overrideJson?.optString("part1_path", "")?.takeIf { it.isNotEmpty() }
            val part2FileOverride = overrideJson?.optString("part2_path", "")?.takeIf { it.isNotEmpty() }

            val requestedRes = overrideJson?.optInt("input_resolution", baseMeta.inputResolution)
                ?: baseMeta.inputResolution
            val requestedBatch = overrideJson?.optInt("batch_size", baseMeta.batchSize)
                ?: baseMeta.batchSize
            val requestedQueries = overrideJson?.optInt("num_queries", baseMeta.numQueries)
                ?: baseMeta.numQueries
            val requestedClasses = overrideJson?.optInt("num_classes", baseMeta.numClasses)
                ?: baseMeta.numClasses

            overrideActiveTiles = if (overrideJson != null && overrideJson.has("active_tiles")) {
                overrideJson.optInt("active_tiles", StripRoiGeometry.MAX_TILES)
                    .coerceIn(1, MAX_SUPPORTED_BATCH)
            } else {
                null
            }
            overridePreferTextureWeights = if (overrideJson != null && overrideJson.has("prefer_texture_weights")) {
                overrideJson.optBoolean("prefer_texture_weights")
            } else {
                null
            }
            overrideConstantTensorSharing = if (overrideJson != null && overrideJson.has("constant_tensor_sharing")) {
                overrideJson.optBoolean("constant_tensor_sharing")
            } else {
                null
            }
            overrideAllowQuantizedFcConv = if (overrideJson != null && overrideJson.has("allow_quantized_fc_conv")) {
                overrideJson.optBoolean("allow_quantized_fc_conv")
            } else {
                true
            }
            overrideNumStepsCmdBuffer = if (overrideJson != null && overrideJson.has("num_steps_cmd_buffer")) {
                overrideJson.optInt("num_steps_cmd_buffer")
            } else {
                null
            }
            overrideBufferStorageType = when (overrideJson?.optString("buffer_storage_type", "")?.lowercase(Locale.US)) {
                "buffer" -> CompiledModel.GpuOptions.BufferStorageType.BUFFER
                "texture_2d" -> CompiledModel.GpuOptions.BufferStorageType.TEXTURE_2D
                "default" -> CompiledModel.GpuOptions.BufferStorageType.DEFAULT
                else -> null
            }
            overrideAcceleratorMode = overrideJson?.optString("accelerator", "gpu")?.lowercase(Locale.US) ?: "gpu"
            useZeroCopyEncMemory = overrideJson?.optBoolean("zero_copy_enc_memory", true) ?: true
            interleaveEvenOdd = overrideJson?.optBoolean("interleave_even_odd", profile.interleaveEvenOdd)
                ?: profile.interleaveEvenOdd

            val env = environment ?: Environment.create(appContext, enableCompilerCache = true).also {
                environment = it
            }

            val hasTwoGraphOverride = part1FileOverride != null && part2FileOverride != null &&
                File(part1FileOverride).exists() && File(part2FileOverride).exists()
            val hasSingleGraphOverride = singleFileOverride != null && File(singleFileOverride).exists()

            val hasTwoGraphAssets = baseMeta.part1Asset in assetList && baseMeta.part2Asset in assetList
            val hasSingleGraphAsset = baseMeta.singleGraphAsset in assetList

            val useTwoGraph = when {
                desiredTopology == CocoStandInMetadata.TOPOLOGY_SINGLE_GRAPH &&
                    (hasSingleGraphOverride || hasSingleGraphAsset) -> false
                desiredTopology == CocoStandInMetadata.TOPOLOGY_TWO_GRAPH &&
                    (hasTwoGraphOverride || hasTwoGraphAssets) -> true
                hasTwoGraphOverride || hasTwoGraphAssets -> true
                hasSingleGraphOverride || hasSingleGraphAsset -> false
                else -> false
            }

            onCompilationStateChanged?.invoke(true, profile.statusBadge)
            try {
                val compileStartNs = System.nanoTime()
                if (useTwoGraph && (hasTwoGraphOverride || hasTwoGraphAssets)) {
                    val p1CacheToken = if (hasTwoGraphOverride) null else "${profile.id}_p1_v1"
                    val p2CacheToken = if (hasTwoGraphOverride) null else "${profile.id}_p2_v1"
                    val (m1, m2, accelLabel) = loadTwoGraphModels(
                        env = env,
                        part1FilePath = part1FileOverride.takeIf { hasTwoGraphOverride },
                        part2FilePath = part2FileOverride.takeIf { hasTwoGraphOverride },
                        part1AssetName = baseMeta.part1Asset,
                        part2AssetName = baseMeta.part2Asset,
                        part1CacheKeyOverride = p1CacheToken,
                        part2CacheKeyOverride = p2CacheToken
                    )
                    val compileMs = (System.nanoTime() - compileStartNs) / 1_000_000.0
                    part1Model = m1
                    part2Model = m2
                    part1Inputs = m1.createInputBuffers()
                    part1Outputs = m1.createOutputBuffers()
                    part2Inputs = m2.createInputBuffers()
                    part2Outputs = m2.createOutputBuffers()
                    part2ZeroCopyInputs = listOf(part1Outputs[0], part2Inputs[1], part2Inputs[2])

                    // Auto-detect exact dimensions from compiled TensorBuffers
                    val p1InFloats = part1Inputs[0].readFloat().size
                    val p1MemFloats = part1Outputs[0].readFloat().size
                    val p1ScoresFloats = part1Outputs[2].readFloat().size
                    val p2PropFloats = part2Inputs[2].readFloat().size
                    val p2Out0Floats = part2Outputs[0].readFloat().size
                    val p2Out1Floats = part2Outputs[1].readFloat().size

                    activeInputResolution = requestedRes
                    activeTileFloatCount = activeInputResolution * activeInputResolution * 3
                    compiledBatchSize = if (activeTileFloatCount > 0 && p1InFloats >= activeTileFloatCount) {
                        (p1InFloats / activeTileFloatCount).coerceIn(1, MAX_SUPPORTED_BATCH)
                    } else {
                        requestedBatch.coerceIn(1, MAX_SUPPORTED_BATCH)
                    }

                    val detectedTokens = if (compiledBatchSize > 0 && p1ScoresFloats >= compiledBatchSize) {
                        p1ScoresFloats / compiledBatchSize
                    } else {
                        (activeInputResolution / 16) * (activeInputResolution / 16)
                    }
                    val detectedHiddenDim = if (p1ScoresFloats > 0 && p1MemFloats >= p1ScoresFloats) {
                        p1MemFloats / p1ScoresFloats
                    } else {
                        CocoStandInMetadata.DEFAULT_QUERY_HIDDEN_DIM
                    }
                    val detectedTopK = if (compiledBatchSize > 0 && p2PropFloats >= compiledBatchSize * 4) {
                        p2PropFloats / (compiledBatchSize * 4)
                    } else {
                        requestedQueries
                    }

                    part2BoxesOutputIdx = if (p2Out0Floats <= p2Out1Floats) 0 else 1
                    part2ScoresOutputIdx = 1 - part2BoxesOutputIdx
                    val scoresFloats = maxOf(p2Out0Floats, p2Out1Floats)
                    val detectedClasses = if (compiledBatchSize * detectedTopK > 0) {
                        scoresFloats / (compiledBatchSize * detectedTopK)
                    } else {
                        requestedClasses
                    }

                    val gridSide = sqrt(detectedTokens.toDouble()).roundToInt().coerceAtLeast(1)
                    metadata = baseMeta.copy(
                        inputResolution = activeInputResolution,
                        batchSize = compiledBatchSize,
                        topology = CocoStandInMetadata.TOPOLOGY_TWO_GRAPH,
                        encoderGridH = gridSide,
                        encoderGridW = detectedTokens / gridSide,
                        numEncoderTokens = detectedTokens,
                        hiddenDim = detectedHiddenDim,
                        numQueries = detectedTopK,
                        numClasses = detectedClasses,
                        cocoClassIdToStoneClassId = CocoStandInMetadata.buildDefaultCocoToStoneClassLut(detectedClasses)
                    )

                    allocateStagingBuffersForMetadata()
                    val warmupStartNs = System.nanoTime()
                    runTwoGraphDummyWarmupPass(m1, m2)
                    val warmupMs = (System.nanoTime() - warmupStartNs) / 1_000_000.0

                    activeTopology = CocoStandInMetadata.TOPOLOGY_TWO_GRAPH
                    activeAcceleratorLabel = accelLabel
                    hardwareGraphsLoaded = true
                    val intlvTag = if (interleaveEvenOdd && compiledBatchSize == 3) " INTLV" else ""
                    initMessage = if (overrideFile != null) {
                        "RF-DETR 2G ${activeInputResolution} B=${compiledBatchSize}${intlvTag} K=${metadata.numQueries} • $accelLabel"
                    } else {
                        profile.statusBadge
                    }
                    Log.i(
                        TAG,
                        String.format(
                            Locale.US,
                            "Initialized %s compile_ms=%.1f warmup_ms=%.1f total_init_ms=%.1f res=%d batch=%d tokens=%d topK=%d classes=%d zeroCopy=%b interleave=%b cacheDir=%s",
                            initMessage,
                            compileMs,
                            warmupMs,
                            compileMs + warmupMs,
                            activeInputResolution,
                            compiledBatchSize,
                            metadata.numEncoderTokens,
                            metadata.numQueries,
                            metadata.numClasses,
                            useZeroCopyEncMemory,
                            interleaveEvenOdd,
                            gpuShaderCacheDir
                        )
                    )
                } else if (hasSingleGraphOverride || hasSingleGraphAsset) {
                    val singleCacheToken = if (hasSingleGraphOverride) null else "${profile.id}_1g_v1"
                    val (sModel, accelLabel) = loadSingleGraphModel(
                        env = env,
                        singleFilePath = singleFileOverride.takeIf { hasSingleGraphOverride },
                        singleAssetName = baseMeta.singleGraphAsset,
                        singleCacheKeyOverride = singleCacheToken
                    )
                    val compileMs = (System.nanoTime() - compileStartNs) / 1_000_000.0
                    singleModel = sModel
                    singleInputs = sModel.createInputBuffers()
                    singleOutputs = sModel.createOutputBuffers()

                    val sInFloats = singleInputs[0].readFloat().size
                    val sOut0Floats = singleOutputs[0].readFloat().size
                    val sOut1Floats = singleOutputs[1].readFloat().size

                    activeInputResolution = requestedRes
                    activeTileFloatCount = activeInputResolution * activeInputResolution * 3
                    compiledBatchSize = if (activeTileFloatCount > 0 && sInFloats >= activeTileFloatCount) {
                        (sInFloats / activeTileFloatCount).coerceIn(1, MAX_SUPPORTED_BATCH)
                    } else {
                        requestedBatch.coerceIn(1, MAX_SUPPORTED_BATCH)
                    }

                    singleBoxesOutputIdx = if (sOut0Floats <= sOut1Floats) 0 else 1
                    singleScoresOutputIdx = 1 - singleBoxesOutputIdx
                    val boxesFloats = minOf(sOut0Floats, sOut1Floats)
                    val scoresFloats = maxOf(sOut0Floats, sOut1Floats)

                    val detectedQueries = if (compiledBatchSize > 0 && boxesFloats >= compiledBatchSize * 4) {
                        boxesFloats / (compiledBatchSize * 4)
                    } else {
                        requestedQueries
                    }
                    val detectedClasses = if (compiledBatchSize * detectedQueries > 0) {
                        scoresFloats / (compiledBatchSize * detectedQueries)
                    } else {
                        requestedClasses
                    }

                    metadata = baseMeta.copy(
                        inputResolution = activeInputResolution,
                        batchSize = compiledBatchSize,
                        topology = CocoStandInMetadata.TOPOLOGY_SINGLE_GRAPH,
                        numQueries = detectedQueries,
                        numClasses = detectedClasses,
                        cocoClassIdToStoneClassId = CocoStandInMetadata.buildDefaultCocoToStoneClassLut(detectedClasses)
                    )

                    allocateStagingBuffersForMetadata()
                    val warmupStartNs = System.nanoTime()
                    singleInputs[0].writeFloat(batchedInputFloats)
                    sModel.run(singleInputs, singleOutputs)
                    val warmupMs = (System.nanoTime() - warmupStartNs) / 1_000_000.0

                    activeTopology = CocoStandInMetadata.TOPOLOGY_SINGLE_GRAPH
                    activeAcceleratorLabel = accelLabel
                    hardwareGraphsLoaded = true
                    initMessage = if (overrideFile != null) {
                        "1G ${activeInputResolution} B=${compiledBatchSize} Q=${metadata.numQueries} C=${metadata.numClasses} • $accelLabel"
                    } else {
                        profile.statusBadge
                    }
                    Log.i(
                        TAG,
                        String.format(
                            Locale.US,
                            "Initialized %s compile_ms=%.1f warmup_ms=%.1f total_init_ms=%.1f res=%d batch=%d queries=%d classes=%d",
                            initMessage,
                            compileMs,
                            warmupMs,
                            compileMs + warmupMs,
                            activeInputResolution,
                            compiledBatchSize,
                            metadata.numQueries,
                            metadata.numClasses
                        )
                    )
                    if (overrideFile == null && hardwareGraphsLoaded) {
                        cacheCurrentAsWarmPatrolSlot(profile)
                    }
                } else {
                    hardwareGraphsLoaded = false
                    activeSlotFromLru = false
                    initMessage = "LiteRT GPU Stub (missing .tflite in assets/)"
                    Log.w(TAG, "$initMessage — falling back to host bridge profiler")
                }
                if (overrideFile == null && hardwareGraphsLoaded && !activeSlotFromLru) {
                    cacheCurrentAsWarmPatrolSlot(profile)
                }
            } finally {
                onCompilationStateChanged?.invoke(false, initMessage)
            }
        } catch (t: Throwable) {
            hardwareGraphsLoaded = false
            initMessage = "LiteRT GPU Stub (${t.javaClass.simpleName})"
            onCompilationStateChanged?.invoke(false, initMessage)
            Log.w(TAG, "LiteRT initialization fell back to stub profiler: ${t.message}", t)
        }
    }

    /** Executes one silent zero-input warmup sweep on [m1] + [m2] so GPU OpenCL kernels bind eagerly. */
    private fun runTwoGraphDummyWarmupPass(m1: CompiledModel, m2: CompiledModel) {
        if (compiledBatchSize > 1) {
            Arrays.fill(batchedInputFloats, 0f)
            part1Inputs[0].writeFloat(batchedInputFloats)
            m1.run(part1Inputs, part1Outputs)
            selectTopKProposalsFromMaxScores(
                encMaxScores = part1Outputs[2].readFloat(),
                outputMemoryPrenorm = part1Outputs[1].readFloat(),
                baseProposals = baseProposalsTable,
                batchSize = compiledBatchSize,
                activeBatchItems = compiledBatchSize,
                numTokens = metadata.numEncoderTokens,
                hiddenDim = metadata.hiddenDim,
                topK = metadata.numQueries,
                outSelectedPrenorm = batchedSelectedPrenormFloats,
                outSelectedProposals = batchedSelectedProposalsFloats,
                outSelectedIndices = batchedTopIndices,
                scratchHeapScores = scratchHeapScores,
                scratchHeapIndices = scratchHeapIndices
            )
            part2Inputs[1].writeFloat(batchedSelectedPrenormFloats)
            part2Inputs[2].writeFloat(batchedSelectedProposalsFloats)
            if (useZeroCopyEncMemory) {
                try {
                    m2.run(part2ZeroCopyInputs, part2Outputs)
                } catch (_: Throwable) {
                    useZeroCopyEncMemory = false
                    part2Inputs[0].writeFloat(part1Outputs[0].readFloat())
                    m2.run(part2Inputs, part2Outputs)
                }
            } else {
                part2Inputs[0].writeFloat(part1Outputs[0].readFloat())
                m2.run(part2Inputs, part2Outputs)
            }
        } else {
            Arrays.fill(singleTileInputFloats, 0f)
            part1Inputs[0].writeFloat(singleTileInputFloats)
            m1.run(part1Inputs, part1Outputs)
            selectTopKProposalsFromMaxScores(
                encMaxScores = part1Outputs[2].readFloat(),
                outputMemoryPrenorm = part1Outputs[1].readFloat(),
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
            part2Inputs[1].writeFloat(singleSelectedPrenormFloats)
            part2Inputs[2].writeFloat(singleSelectedProposalsFloats)
            if (useZeroCopyEncMemory) {
                try {
                    m2.run(part2ZeroCopyInputs, part2Outputs)
                } catch (_: Throwable) {
                    useZeroCopyEncMemory = false
                    part2Inputs[0].writeFloat(part1Outputs[0].readFloat())
                    m2.run(part2Inputs, part2Outputs)
                }
            } else {
                part2Inputs[0].writeFloat(part1Outputs[0].readFloat())
                m2.run(part2Inputs, part2Outputs)
            }
        }
    }

    /**
     * Pre-warms the dedicated `B=1` `320x320` `W8-Dyn` Lock-On 2-Graph models (`Part 1` + `Part 2`)
     * on `GneissFind-Inference` once per session and keeps them resident across `PATROL` profile
     * switches so tapping a target enters `LOCK-ON` in `< 50 ms` with zero shader pause.
     */
    private fun ensureLockOn320ModelWarmOnAnalyzerThread() {
        if (lockOnInitializedOnThread) return
        lockOnInitializedOnThread = true

        try {
            val assetList = appContext.assets.list("")?.toSet().orEmpty()
            val meta = if (CocoStandInMetadata.LOCK_ON_METADATA_ASSET_NAME in assetList) {
                val jsonText = appContext.assets.open(CocoStandInMetadata.LOCK_ON_METADATA_ASSET_NAME)
                    .bufferedReader()
                    .use { it.readText() }
                parseCocoMetadataJson(jsonText, LOCK_ON_DEFAULT_SCORE_THRESHOLD)
            } else {
                CocoStandInMetadata(
                    modelName = "rf_detr_nano_320_b1_w8dyn",
                    inputResolution = CocoStandInMetadata.LOCK_ON_CROP_RESOLUTION_PX,
                    batchSize = 1,
                    part1Asset = CocoStandInMetadata.LOCK_ON_PART1_ASSET_NAME,
                    part2Asset = CocoStandInMetadata.LOCK_ON_PART2_ASSET_NAME,
                    encoderGridH = 20,
                    encoderGridW = 20,
                    numEncoderTokens = 400,
                    numQueries = CocoStandInMetadata.LOCK_ON_TOP_K_QUERIES,
                    numClasses = CocoStandInMetadata.DEFAULT_COCO_NUM_CLASSES,
                    scoreThreshold = LOCK_ON_DEFAULT_SCORE_THRESHOLD
                )
            }

            if (meta.part1Asset !in assetList || meta.part2Asset !in assetList) {
                lockOnHardwareLoaded = false
                Log.i(TAG, "Lock-On B=1 320 assets (${meta.part1Asset}) not yet bundled; skipping pre-warm")
                return
            }

            val env = environment ?: Environment.create(appContext, enableCompilerCache = true).also {
                environment = it
            }
            val startNs = System.nanoTime()
            val (m1, m2, accelLabel) = loadTwoGraphModels(
                env = env,
                part1FilePath = null,
                part2FilePath = null,
                part1AssetName = meta.part1Asset,
                part2AssetName = meta.part2Asset,
                part1CacheKeyOverride = LOCK_ON_P1_CACHE_TOKEN,
                part2CacheKeyOverride = LOCK_ON_P2_CACHE_TOKEN
            )
            val compileMs = (System.nanoTime() - startNs) / 1_000_000.0

            lockOnPart1Model = m1
            lockOnPart2Model = m2
            lockOnPart1Inputs = m1.createInputBuffers()
            lockOnPart1Outputs = m1.createOutputBuffers()
            lockOnPart2Inputs = m2.createInputBuffers()
            lockOnPart2Outputs = m2.createOutputBuffers()
            lockOnPart2ZeroCopyInputs = listOf(lockOnPart1Outputs[0], lockOnPart2Inputs[1], lockOnPart2Inputs[2])

            val p2Out0 = lockOnPart2Outputs[0].readFloat().size
            val p2Out1 = lockOnPart2Outputs[1].readFloat().size
            lockOnBoxesOutputIdx = if (p2Out0 <= p2Out1) 0 else 1
            lockOnScoresOutputIdx = 1 - lockOnBoxesOutputIdx

            lockOnMetadata = meta
            lockOnBaseProposals = buildNormalizedProposalGrid(
                gridH = meta.encoderGridH,
                gridW = meta.encoderGridW,
                defaultWh = meta.proposalDefaultWh
            )

            // 1-pass warmup so OpenCL command buffers are hot before the user ever taps a target
            val warmStartNs = System.nanoTime()
            lockOnPart1Inputs[0].writeFloat(lockOnDummyInputFloats)
            m1.run(lockOnPart1Inputs, lockOnPart1Outputs)
            selectTopKProposalsFromMaxScores(
                encMaxScores = lockOnPart1Outputs[2].readFloat(),
                outputMemoryPrenorm = lockOnPart1Outputs[1].readFloat(),
                baseProposals = lockOnBaseProposals,
                batchSize = 1,
                activeBatchItems = 1,
                numTokens = meta.numEncoderTokens,
                hiddenDim = meta.hiddenDim,
                topK = meta.numQueries,
                outSelectedPrenorm = lockOnSelectedPrenormFloats,
                outSelectedProposals = lockOnSelectedProposalsFloats,
                outSelectedIndices = lockOnTopIndices,
                scratchHeapScores = lockOnScratchHeapScores,
                scratchHeapIndices = lockOnScratchHeapIndices
            )
            lockOnPart2Inputs[1].writeFloat(lockOnSelectedPrenormFloats)
            lockOnPart2Inputs[2].writeFloat(lockOnSelectedProposalsFloats)
            try {
                m2.run(lockOnPart2ZeroCopyInputs, lockOnPart2Outputs)
            } catch (_: Throwable) {
                lockOnUseZeroCopyEncMemory = false
                lockOnPart2Inputs[0].writeFloat(lockOnPart1Outputs[0].readFloat())
                m2.run(lockOnPart2Inputs, lockOnPart2Outputs)
            }
            val warmMs = (System.nanoTime() - warmStartNs) / 1_000_000.0
            lockOnHardwareLoaded = true
            Log.i(
                TAG,
                String.format(
                    Locale.US,
                    "Pre-warmed Lock-On B=1 320 (%s) compile_ms=%.1f warmup_ms=%.1f total_init_ms=%.1f",
                    accelLabel,
                    compileMs,
                    warmMs,
                    compileMs + warmMs
                )
            )
        } catch (t: Throwable) {
            lockOnHardwareLoaded = false
            Log.w(TAG, "Lock-On B=1 320 pre-warm failed: ${t.message}", t)
        }
    }

    /**
     * Runs foveated `B=1` `320x320` `W8-Dyn` `RF-DETR-Nano` inference on a single `320x320x3` RGB
     * `[0..1]` crop on `GneissFind-Inference` (`~16-22 FPS` on Tensor G5 GPU), returning detections
     * with coordinates in `[0.0, 320.0]` crop-local pixels (`tileIndex = 0`).
     *
     * @param cropRgb320Floats Flat `320 * 320 * 3` RGB `[0..1]` float array.
     * @param scoreThresholdOverride Minimum confidence score (`0.28f` default during Lock-On).
     * @return [SweepInferenceResult] containing crop-local [TileDetection] boxes in `[0..320]`.
     */
    fun detectLockOnCrop320(
        cropRgb320Floats: FloatArray,
        scoreThresholdOverride: Float = LOCK_ON_DEFAULT_SCORE_THRESHOLD
    ): SweepInferenceResult {
        ensureLockOn320ModelWarmOnAnalyzerThread()
        val m1 = lockOnPart1Model
        val m2 = lockOnPart2Model
        val detections = ArrayList<TileDetection>(8)
        if (!lockOnHardwareLoaded || m1 == null || m2 == null) {
            return SweepInferenceResult(
                tileDetections = detections,
                part1LatencyMs = 0f,
                hostTopKBridgeLatencyMs = 0f,
                part2LatencyMs = 0f,
                totalSweepLatencyMs = 0f,
                rollingAvgLatencyMs = lockOnRollingLatencyMs,
                isHardwareGraphActive = false
            )
        }

        val t0 = System.nanoTime()
        lockOnPart1Inputs[0].writeFloat(cropRgb320Floats)
        m1.run(lockOnPart1Inputs, lockOnPart1Outputs)
        val encPrenorm = lockOnPart1Outputs[1].readFloat()
        val encMaxScores = lockOnPart1Outputs[2].readFloat()
        val t1 = System.nanoTime()

        selectTopKProposalsFromMaxScores(
            encMaxScores = encMaxScores,
            outputMemoryPrenorm = encPrenorm,
            baseProposals = lockOnBaseProposals,
            batchSize = 1,
            activeBatchItems = 1,
            numTokens = lockOnMetadata.numEncoderTokens,
            hiddenDim = lockOnMetadata.hiddenDim,
            topK = lockOnMetadata.numQueries,
            outSelectedPrenorm = lockOnSelectedPrenormFloats,
            outSelectedProposals = lockOnSelectedProposalsFloats,
            outSelectedIndices = lockOnTopIndices,
            scratchHeapScores = lockOnScratchHeapScores,
            scratchHeapIndices = lockOnScratchHeapIndices
        )
        val t2 = System.nanoTime()

        lockOnPart2Inputs[1].writeFloat(lockOnSelectedPrenormFloats)
        lockOnPart2Inputs[2].writeFloat(lockOnSelectedProposalsFloats)
        if (lockOnUseZeroCopyEncMemory) {
            try {
                m2.run(lockOnPart2ZeroCopyInputs, lockOnPart2Outputs)
            } catch (_: Throwable) {
                lockOnUseZeroCopyEncMemory = false
                lockOnPart2Inputs[0].writeFloat(lockOnPart1Outputs[0].readFloat())
                m2.run(lockOnPart2Inputs, lockOnPart2Outputs)
            }
        } else {
            lockOnPart2Inputs[0].writeFloat(lockOnPart1Outputs[0].readFloat())
            m2.run(lockOnPart2Inputs, lockOnPart2Outputs)
        }
        val predBoxes = lockOnPart2Outputs[lockOnBoxesOutputIdx].readFloat()
        val predScores = lockOnPart2Outputs[lockOnScoresOutputIdx].readFloat()
        val t3 = System.nanoTime()

        decodeCocoSingleTilePredictions(
            tileIndex = 0,
            batchElementIndex = 0,
            boxesCxCyWhNorm = predBoxes,
            classScores = predScores,
            metadata = lockOnMetadata,
            confidenceThreshold = scoreThresholdOverride,
            outDetections = detections,
            targetBoxScalePx = CocoStandInMetadata.LOCK_ON_CROP_RESOLUTION_PX.toFloat()
        )

        val p1Ms = ((t1 - t0) / 1_000_000.0).toFloat()
        val bridgeMs = ((t2 - t1) / 1_000_000.0).toFloat()
        val p2Ms = ((t3 - t2) / 1_000_000.0).toFloat()
        val totalMs = ((t3 - t0) / 1_000_000.0).toFloat()
        lockOnSweepCount++
        lockOnRollingLatencyMs = if (lockOnSweepCount == 1L) {
            totalMs
        } else {
            lockOnRollingLatencyMs * 0.80f + totalMs * 0.20f
        }

        if (lockOnSweepCount <= 30L || lockOnSweepCount % 15L == 0L) {
            Log.i(
                TAG,
                String.format(
                    Locale.US,
                    "RFDETR_LOCKON_B1 sweep=%d total_ms=%.2f p1_ms=%.2f topk_ms=%.2f p2_ms=%.2f rolling_ms=%.2f dets=%d",
                    lockOnSweepCount,
                    totalMs,
                    p1Ms,
                    bridgeMs,
                    p2Ms,
                    lockOnRollingLatencyMs,
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
            rollingAvgLatencyMs = lockOnRollingLatencyMs,
            isHardwareGraphActive = true
        )
    }

    /**
     * Reallocates staging arrays once per model load to match the exact `TensorBuffer` dimensions
     * (`compiledBatchSize`, `activeInputResolution`, `numEncoderTokens`, `numQueries`).
     */
    private fun allocateStagingBuffersForMetadata() {
        buildResampleTables(activeInputResolution)
        baseProposalsTable = buildNormalizedProposalGrid(
            gridH = metadata.encoderGridH,
            gridW = metadata.encoderGridW,
            defaultWh = metadata.proposalDefaultWh
        )
        batchedInputFloats = FloatArray(compiledBatchSize * activeTileFloatCount)
        singleTileInputFloats = FloatArray(activeTileFloatCount)
        val directBytes = compiledBatchSize * activeTileFloatCount * Float.SIZE_BYTES
        if (directInputByteBuffer.capacity() < directBytes) {
            directInputByteBuffer = ByteBuffer.allocateDirect(directBytes).order(ByteOrder.nativeOrder())
            directInputFloatBuffer = directInputByteBuffer.asFloatBuffer()
        }
        val k = metadata.numQueries
        val d = metadata.hiddenDim
        batchedSelectedPrenormFloats = FloatArray(compiledBatchSize * k * d)
        batchedSelectedProposalsFloats = FloatArray(compiledBatchSize * k * 4)
        singleSelectedPrenormFloats = FloatArray(k * d)
        singleSelectedProposalsFloats = FloatArray(k * 4)
        batchedTopIndices = IntArray(compiledBatchSize * k)
        scratchHeapScores = FloatArray(k)
        scratchHeapIndices = IntArray(k)
    }

    /**
     * Builds LiteRT [CompiledModel.Options] targeting `Accelerator.GPU` (FP16 precision, serialized
     * OpenCL shader cache under `codeCacheDir/litert_gpu_cache`, optional quantized FC/Conv support)
     * with CPU fallback.
     */
    private fun buildGpuOptions(cacheKey: String): CompiledModel.Options {
        val xnnCache = File(gpuShaderCacheDir, "xnnpack_${cacheKey}.cache").absolutePath
        return CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
            gpuOptions = CompiledModel.GpuOptions(
                constantTensorSharing = overrideConstantTensorSharing ?: true,
                allowSrcQuantizedFcConvOps = overrideAllowQuantizedFcConv ?: true,
                precision = CompiledModel.GpuOptions.Precision.FP16,
                bufferStorageType = overrideBufferStorageType,
                preferTextureWeights = overridePreferTextureWeights,
                serializationDir = gpuShaderCacheDir,
                modelCacheKey = cacheKey,
                serializeProgramCache = true,
                priority = CompiledModel.GpuOptions.Priority.HIGH,
                numStepsOfCommandBufferPreparations = overrideNumStepsCmdBuffer
            )
            cpuOptions = CompiledModel.CpuOptions(
                numThreads = 4,
                xnnPackWeightCachePath = xnnCache
            )
        }
    }

    /**
     * Builds LiteRT [CompiledModel.Options] targeting Google Tensor G5 `Accelerator.NPU`
     * (either strict NPU+CPU or heterogeneous NPU+GPU+CPU).
     */
    private fun buildNpuOptions(cacheKey: String, allowGpuFallback: Boolean): CompiledModel.Options {
        val xnnCache = File(gpuShaderCacheDir, "xnnpack_npu_${cacheKey}.cache").absolutePath
        val accels = if (allowGpuFallback) {
            arrayOf(Accelerator.NPU, Accelerator.GPU, Accelerator.CPU)
        } else {
            arrayOf(Accelerator.NPU, Accelerator.CPU)
        }
        return CompiledModel.Options(*accels).apply {
            gpuOptions = CompiledModel.GpuOptions(
                constantTensorSharing = true,
                allowSrcQuantizedFcConvOps = true,
                precision = CompiledModel.GpuOptions.Precision.FP16,
                serializationDir = gpuShaderCacheDir,
                modelCacheKey = "npu_${cacheKey}",
                serializeProgramCache = true,
                priority = CompiledModel.GpuOptions.Priority.HIGH
            )
            cpuOptions = CompiledModel.CpuOptions(
                numThreads = 4,
                xnnPackWeightCachePath = xnnCache
            )
        }
    }

    /** Builds 4-thread XNNPACK CPU fallback [CompiledModel.Options] when GPU/NPU compilation fails. */
    private fun buildCpuFallbackOptions(): CompiledModel.Options {
        val cacheFile = File(gpuShaderCacheDir, "xnnpack_weights.cache").absolutePath
        return CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(
                numThreads = 4,
                xnnPackWeightCachePath = cacheFile
            )
        }
    }

    /**
     * Compiles Part 1 and Part 2 `.tflite` models for 2-Graph execution on the requested accelerator
     * (`gpu`, `npu`, `npu_strict`, `npu_p1_gpu_p2`, or `cpu`).
     */
    private fun loadTwoGraphModels(
        env: Environment,
        part1FilePath: String?,
        part2FilePath: String?,
        part1AssetName: String,
        part2AssetName: String,
        part1CacheKeyOverride: String? = null,
        part2CacheKeyOverride: String? = null
    ): Triple<CompiledModel, CompiledModel, String> {
        val p1Tag = part1CacheKeyOverride
            ?: (part1FilePath ?: part1AssetName).substringAfterLast('/').substringBeforeLast('.')
        val p2Tag = part2CacheKeyOverride
            ?: (part2FilePath ?: part2AssetName).substringAfterLast('/').substringBeforeLast('.')
        return try {
            val (opt1, opt2, label) = when (overrideAcceleratorMode) {
                "npu_strict" -> Triple(
                    buildNpuOptions(p1Tag, allowGpuFallback = false),
                    buildNpuOptions(p2Tag, allowGpuFallback = false),
                    "NPU STRICT"
                )
                "npu" -> Triple(
                    buildNpuOptions(p1Tag, allowGpuFallback = true),
                    buildNpuOptions(p2Tag, allowGpuFallback = true),
                    "NPU+GPU"
                )
                "npu_p1_gpu_p2" -> Triple(
                    buildNpuOptions(p1Tag, allowGpuFallback = false),
                    buildGpuOptions(p2Tag),
                    "NPU_P1+GPU_P2"
                )
                "cpu" -> Triple(
                    buildCpuFallbackOptions(),
                    buildCpuFallbackOptions(),
                    "CPU XNN"
                )
                else -> Triple(
                    buildGpuOptions(p1Tag),
                    buildGpuOptions(p2Tag),
                    "GPU FP16"
                )
            }
            val m1 = if (part1FilePath != null) {
                CompiledModel.create(part1FilePath, opt1, env)
            } else {
                CompiledModel.create(appContext.assets, part1AssetName, opt1, env)
            }
            val m2 = if (part2FilePath != null) {
                CompiledModel.create(part2FilePath, opt2, env)
            } else {
                CompiledModel.create(appContext.assets, part2AssetName, opt2, env)
            }
            Triple(m1, m2, label)
        } catch (accelErr: Throwable) {
            Log.w(
                TAG,
                "Accelerator '$overrideAcceleratorMode' failed for 2-graph (${accelErr.message}); falling back to GPU FP16 / CPU",
                accelErr
            )
            try {
                val opt1 = buildGpuOptions(p1Tag)
                val opt2 = buildGpuOptions(p2Tag)
                val m1 = if (part1FilePath != null) {
                    CompiledModel.create(part1FilePath, opt1, env)
                } else {
                    CompiledModel.create(appContext.assets, part1AssetName, opt1, env)
                }
                val m2 = if (part2FilePath != null) {
                    CompiledModel.create(part2FilePath, opt2, env)
                } else {
                    CompiledModel.create(appContext.assets, part2AssetName, opt2, env)
                }
                Triple(m1, m2, "GPU FP16 (FB)")
            } catch (gpuErr: Throwable) {
                Log.w(TAG, "GPU FP16 delegate unavailable (${gpuErr.message}); falling back to CPU XNNPACK")
                val cpuOpt = buildCpuFallbackOptions()
                val m1 = if (part1FilePath != null) {
                    CompiledModel.create(part1FilePath, cpuOpt, env)
                } else {
                    CompiledModel.create(appContext.assets, part1AssetName, cpuOpt, env)
                }
                val m2 = if (part2FilePath != null) {
                    CompiledModel.create(part2FilePath, cpuOpt, env)
                } else {
                    CompiledModel.create(appContext.assets, part2AssetName, cpuOpt, env)
                }
                Triple(m1, m2, "CPU XNN")
            }
        }
    }

    /**
     * Compiles the monolithic 1-Graph `.tflite` model on the requested accelerator (`gpu`, `npu`,
     * `npu_strict`, or `cpu`), falling back cleanly if compilation fails.
     */
    private fun loadSingleGraphModel(
        env: Environment,
        singleFilePath: String?,
        singleAssetName: String,
        singleCacheKeyOverride: String? = null
    ): Pair<CompiledModel, String> {
        val tag = singleCacheKeyOverride
            ?: (singleFilePath ?: singleAssetName).substringAfterLast('/').substringBeforeLast('.')
        return try {
            val (opt, label) = when (overrideAcceleratorMode) {
                "npu_strict" -> buildNpuOptions(tag, allowGpuFallback = false) to "NPU STRICT"
                "npu", "npu_p1_gpu_p2" -> buildNpuOptions(tag, allowGpuFallback = true) to "NPU+GPU"
                "cpu" -> buildCpuFallbackOptions() to "CPU XNN"
                else -> buildGpuOptions(tag) to "GPU FP16"
            }
            val m = if (singleFilePath != null) {
                CompiledModel.create(singleFilePath, opt, env)
            } else {
                CompiledModel.create(appContext.assets, singleAssetName, opt, env)
            }
            m to label
        } catch (accelErr: Throwable) {
            Log.w(
                TAG,
                "Accelerator '$overrideAcceleratorMode' failed for 1-graph (${accelErr.message}); falling back",
                accelErr
            )
            try {
                val opt = buildGpuOptions(tag)
                val m = if (singleFilePath != null) {
                    CompiledModel.create(singleFilePath, opt, env)
                } else {
                    CompiledModel.create(appContext.assets, singleAssetName, opt, env)
                }
                m to "GPU FP16 (FB)"
            } catch (gpuErr: Throwable) {
                Log.w(TAG, "GPU FP16 unavailable for 1-graph (${gpuErr.message}); falling back to CPU XNNPACK")
                val cpuOpt = buildCpuFallbackOptions()
                val m = if (singleFilePath != null) {
                    CompiledModel.create(singleFilePath, cpuOpt, env)
                } else {
                    CompiledModel.create(appContext.assets, singleAssetName, cpuOpt, env)
                }
                m to "CPU XNN"
            }
        }
    }

    /**
     * Bilinearly resamples a single `384x384x3` NHWC float tile into [dst] starting at [dstOffset]
     * with target resolution [activeInputResolution] using precomputed 1D index and weight tables.
     */
    private fun stageSingleResampledTile(
        srcTile384: FloatArray,
        dst: FloatArray,
        dstOffset: Int
    ) {
        if (activeInputResolution == StripRoiGeometry.TILE_SIZE_PX) {
            System.arraycopy(srcTile384, 0, dst, dstOffset, activeTileFloatCount)
            return
        }
        val res = activeInputResolution
        val y0Arr = resampleY0
        val y1Arr = resampleY1
        val wy0Arr = resampleWy0
        val wy1Arr = resampleWy1
        val x0Arr = resampleX0
        val x1Arr = resampleX1
        val wx0Arr = resampleWx0
        val wx1Arr = resampleWx1

        var outIdx = dstOffset
        for (y in 0 until res) {
            val row0 = y0Arr[y]
            val row1 = y1Arr[y]
            val wy0 = wy0Arr[y]
            val wy1 = wy1Arr[y]
            for (x in 0 until res) {
                val i00 = row0 + x0Arr[x]
                val i01 = row0 + x1Arr[x]
                val i10 = row1 + x0Arr[x]
                val i11 = row1 + x1Arr[x]
                val wx0 = wx0Arr[x]
                val wx1 = wx1Arr[x]

                val w00 = wy0 * wx0
                val w01 = wy0 * wx1
                val w10 = wy1 * wx0
                val w11 = wy1 * wx1

                dst[outIdx++] = srcTile384[i00] * w00 + srcTile384[i01] * w01 +
                    srcTile384[i10] * w10 + srcTile384[i11] * w11
                dst[outIdx++] = srcTile384[i00 + 1] * w00 + srcTile384[i01 + 1] * w01 +
                    srcTile384[i10 + 1] * w10 + srcTile384[i11 + 1] * w11
                dst[outIdx++] = srcTile384[i00 + 2] * w00 + srcTile384[i01 + 2] * w01 +
                    srcTile384[i10 + 2] * w10 + srcTile384[i11 + 2] * w11
            }
        }
    }

    /**
     * Reconstructs the `1792x384` strip from `sweepBuffers.tileRgbFloats[0..4]` and bilinearly
     * samples 6 overlapping `320x384 -> 256x256` tiles into [batchedInputFloats] for `B=6` mode.
     */
    private fun stageSixBy256TilesFromFiveTileStrip(sweepBuffers: PushBroomSweepBuffers) {
        val stripW = StripRoiGeometry.MAX_STRIP_WIDTH_PX
        val stripH = StripRoiGeometry.STRIP_HEIGHT_PX
        val tileW = StripRoiGeometry.TILE_SIZE_PX
        val stripOffsets = intArrayOf(0, 352, 704, 1056, 1408)

        for (t in 0 until StripRoiGeometry.MAX_TILES) {
            val tileSrc = sweepBuffers.tileRgbFloats[t]
            val xStart = if (t == 0) 0 else 16
            val copyWidth = if (t == 0) tileW else (tileW - 16)
            val dstXBase = stripOffsets[t] + xStart
            for (y in 0 until stripH) {
                val srcPos = (y * tileW + xStart) * 3
                val dstPos = (y * stripW + dstXBase) * 3
                System.arraycopy(tileSrc, srcPos, reconstructedStripRgb, dstPos, copyWidth * 3)
            }
        }

        val y0Arr = resampleY0
        val y1Arr = resampleY1
        val wy0Arr = resampleWy0
        val wy1Arr = resampleWy1
        val x0Arr = sixTileX0
        val x1Arr = sixTileX1
        val wx0Arr = sixTileWx0
        val wx1Arr = sixTileWx1

        for (b in 0 until 6) {
            val xOffset = SIX_TILE_256_X_OFFSETS[b]
            var outIdx = b * activeTileFloatCount
            for (y in 0 until 256) {
                val row0 = (y0Arr[y] / (tileW * 3)) * stripW * 3
                val row1 = (y1Arr[y] / (tileW * 3)) * stripW * 3
                val wy0 = wy0Arr[y]
                val wy1 = wy1Arr[y]
                for (x in 0 until 256) {
                    val sx0 = (xOffset + x0Arr[x]) * 3
                    val sx1 = (xOffset + x1Arr[x]) * 3
                    val i00 = row0 + sx0
                    val i01 = row0 + sx1
                    val i10 = row1 + sx0
                    val i11 = row1 + sx1
                    val w00 = wy0 * wx0Arr[x]
                    val w01 = wy0 * wx1Arr[x]
                    val w10 = wy1 * wx0Arr[x]
                    val w11 = wy1 * wx1Arr[x]

                    batchedInputFloats[outIdx++] = reconstructedStripRgb[i00] * w00 +
                        reconstructedStripRgb[i01] * w01 +
                        reconstructedStripRgb[i10] * w10 +
                        reconstructedStripRgb[i11] * w11
                    batchedInputFloats[outIdx++] = reconstructedStripRgb[i00 + 1] * w00 +
                        reconstructedStripRgb[i01 + 1] * w01 +
                        reconstructedStripRgb[i10 + 1] * w10 +
                        reconstructedStripRgb[i11 + 1] * w11
                    batchedInputFloats[outIdx++] = reconstructedStripRgb[i00 + 2] * w00 +
                        reconstructedStripRgb[i01 + 2] * w01 +
                        reconstructedStripRgb[i10 + 2] * w10 +
                        reconstructedStripRgb[i11 + 2] * w11
                }
            }
        }
    }

    /**
     * Stages tile float buffers into [batchedInputFloats] for static batched dispatch (`B=5`, `B=6`,
     * or `B=3`), applying bilinear resampling when `activeInputResolution != 384` and Even/Odd tile
     * selection when [useInterleaveThisSweep] is true.
     */
    private fun stageBatchedTileInputs(
        sweepBuffers: PushBroomSweepBuffers,
        activeBatchItems: Int,
        useInterleaveThisSweep: Boolean
    ) {
        if (compiledBatchSize == 6 && activeInputResolution == 256) {
            stageSixBy256TilesFromFiveTileStrip(sweepBuffers)
            return
        }
        for (batchSlot in 0 until compiledBatchSize) {
            val dstOffset = batchSlot * activeTileFloatCount
            if (batchSlot < activeBatchItems) {
                val srcTileIdx = if (useInterleaveThisSweep) {
                    interleavedBatchTileMap[batchSlot]
                } else {
                    batchSlot.coerceIn(0, StripRoiGeometry.MAX_TILES - 1)
                }
                stageSingleResampledTile(
                    srcTile384 = sweepBuffers.tileRgbFloats[srcTileIdx],
                    dst = batchedInputFloats,
                    dstOffset = dstOffset
                )
            } else {
                Arrays.fill(batchedInputFloats, dstOffset, dstOffset + activeTileFloatCount, 0f)
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
        maxAlertableScoreScratch[0] = 0f
        val requestedSwathTiles = sweepBuffers.activeTileCount.coerceIn(1, StripRoiGeometry.MAX_TILES)
        val useInterleaveThisSweep = interleaveEvenOdd &&
            compiledBatchSize == 3 &&
            requestedSwathTiles == StripRoiGeometry.MAX_TILES &&
            overrideActiveTiles == null

        if (useInterleaveThisSweep) {
            if (sweepCount % 2L == 0L) {
                interleavedBatchTileMap[0] = 0
                interleavedBatchTileMap[1] = 2
                interleavedBatchTileMap[2] = 4
            } else {
                interleavedBatchTileMap[0] = 1
                interleavedBatchTileMap[1] = 3
                interleavedBatchTileMap[2] = priorityEvenTileIndex.coerceIn(0, 4)
            }
        }

        val activeBatchItems = when {
            compiledBatchSize == 6 && activeInputResolution == 256 -> 6
            useInterleaveThisSweep -> 3
            else -> (overrideActiveTiles ?: requestedSwathTiles).coerceIn(1, compiledBatchSize.coerceAtLeast(1))
        }

        val m1 = part1Model
        val m2 = part2Model
        val sModel = singleModel

        if (hardwareGraphsLoaded && activeTopology == CocoStandInMetadata.TOPOLOGY_TWO_GRAPH && m1 != null && m2 != null) {
            if (compiledBatchSize > 1) {
                // Static batched (B=5, B=6, or B=3 Interleaved) 2-Graph execution
                val t0 = System.nanoTime()
                stageBatchedTileInputs(sweepBuffers, activeBatchItems, useInterleaveThisSweep)
                part1Inputs[0].writeFloat(batchedInputFloats)
                m1.run(part1Inputs, part1Outputs)
                val encPrenorm = part1Outputs[1].readFloat()
                val encMaxScores = part1Outputs[2].readFloat()
                val t1 = System.nanoTime()
                part1NsAcc += (t1 - t0)

                // Stage 2: Host Top-K query proposal gather across activeBatchItems
                selectTopKProposalsFromMaxScores(
                    encMaxScores = encMaxScores,
                    outputMemoryPrenorm = encPrenorm,
                    baseProposals = baseProposalsTable,
                    batchSize = compiledBatchSize,
                    activeBatchItems = activeBatchItems,
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

                // Stage 3: Decoder + Detection Head on GPU FP16 (zero-copy encMemory passthrough when supported)
                part2Inputs[1].writeFloat(batchedSelectedPrenormFloats)
                part2Inputs[2].writeFloat(batchedSelectedProposalsFloats)
                if (useZeroCopyEncMemory) {
                    try {
                        m2.run(part2ZeroCopyInputs, part2Outputs)
                    } catch (zeroCopyErr: Throwable) {
                        Log.w(TAG, "Zero-copy encMemory passthrough unsupported (${zeroCopyErr.message}); using CPU copy")
                        useZeroCopyEncMemory = false
                        part2Inputs[0].writeFloat(part1Outputs[0].readFloat())
                        m2.run(part2Inputs, part2Outputs)
                    }
                } else {
                    part2Inputs[0].writeFloat(part1Outputs[0].readFloat())
                    m2.run(part2Inputs, part2Outputs)
                }
                val predBoxes = part2Outputs[part2BoxesOutputIdx].readFloat()
                val predScores = part2Outputs[part2ScoresOutputIdx].readFloat()
                val t3 = System.nanoTime()
                part2NsAcc += (t3 - t2)

                decodeBatchedOutputsWithMode(
                    activeBatchItems = activeBatchItems,
                    useInterleaveThisSweep = useInterleaveThisSweep,
                    boxesCxCyWhNorm = predBoxes,
                    classScores = predScores,
                    outDetections = detections
                )
            } else {
                // Sequential N x B=1 2-Graph execution
                val seqTiles = (overrideActiveTiles ?: requestedSwathTiles).coerceIn(1, StripRoiGeometry.MAX_TILES)
                for (tileIdx in 0 until seqTiles) {
                    val t0 = System.nanoTime()
                    stageSingleResampledTile(sweepBuffers.tileRgbFloats[tileIdx], singleTileInputFloats, 0)
                    part1Inputs[0].writeFloat(singleTileInputFloats)
                    m1.run(part1Inputs, part1Outputs)
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

                    part2Inputs[1].writeFloat(singleSelectedPrenormFloats)
                    part2Inputs[2].writeFloat(singleSelectedProposalsFloats)
                    if (useZeroCopyEncMemory) {
                        try {
                            m2.run(part2ZeroCopyInputs, part2Outputs)
                        } catch (zeroCopyErr: Throwable) {
                            useZeroCopyEncMemory = false
                            part2Inputs[0].writeFloat(part1Outputs[0].readFloat())
                            m2.run(part2Inputs, part2Outputs)
                        }
                    } else {
                        part2Inputs[0].writeFloat(part1Outputs[0].readFloat())
                        m2.run(part2Inputs, part2Outputs)
                    }
                    val predBoxes = part2Outputs[part2BoxesOutputIdx].readFloat()
                    val predScores = part2Outputs[part2ScoresOutputIdx].readFloat()
                    val t3 = System.nanoTime()
                    part2NsAcc += (t3 - t2)

                    decodeCocoSingleTilePredictions(
                        tileIndex = tileIdx,
                        batchElementIndex = 0,
                        boxesCxCyWhNorm = predBoxes,
                        classScores = predScores,
                        metadata = metadata,
                        confidenceThreshold = confidenceThreshold,
                        outDetections = detections,
                        outMaxAlertableScore = maxAlertableScoreScratch
                    )
                }
            }
        } else if (hardwareGraphsLoaded && activeTopology == CocoStandInMetadata.TOPOLOGY_SINGLE_GRAPH && sModel != null) {
            if (compiledBatchSize > 1) {
                // Static batched 1-Graph execution (RF-DETR 1G or YOLO26n/s)
                val t0 = System.nanoTime()
                stageBatchedTileInputs(sweepBuffers, activeBatchItems, useInterleaveThisSweep)
                singleInputs[0].writeFloat(batchedInputFloats)
                sModel.run(singleInputs, singleOutputs)
                val predBoxes = singleOutputs[singleBoxesOutputIdx].readFloat()
                val predScores = singleOutputs[singleScoresOutputIdx].readFloat()
                val t1 = System.nanoTime()
                part1NsAcc += (t1 - t0)

                decodeBatchedOutputsWithMode(
                    activeBatchItems = activeBatchItems,
                    useInterleaveThisSweep = useInterleaveThisSweep,
                    boxesCxCyWhNorm = predBoxes,
                    classScores = predScores,
                    outDetections = detections
                )
            } else {
                val seqTiles = (overrideActiveTiles ?: requestedSwathTiles).coerceIn(1, StripRoiGeometry.MAX_TILES)
                for (tileIdx in 0 until seqTiles) {
                    val t0 = System.nanoTime()
                    stageSingleResampledTile(sweepBuffers.tileRgbFloats[tileIdx], singleTileInputFloats, 0)
                    singleInputs[0].writeFloat(singleTileInputFloats)
                    sModel.run(singleInputs, singleOutputs)
                    val predBoxes = singleOutputs[singleBoxesOutputIdx].readFloat()
                    val predScores = singleOutputs[singleScoresOutputIdx].readFloat()
                    val t1 = System.nanoTime()
                    part1NsAcc += (t1 - t0)

                    decodeCocoSingleTilePredictions(
                        tileIndex = tileIdx,
                        batchElementIndex = 0,
                        boxesCxCyWhNorm = predBoxes,
                        classScores = predScores,
                        metadata = metadata,
                        confidenceThreshold = confidenceThreshold,
                        outDetections = detections,
                        outMaxAlertableScore = maxAlertableScoreScratch
                    )
                }
            }
        } else {
            // Stubbed profiler fallback when .tflite assets are absent in unit tests
            for (tileIdx in 0 until requestedSwathTiles) {
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

        if (sweepCount <= 120L || sweepCount % 5L == 0L) {
            Log.i(
                TAG,
                String.format(
                    Locale.US,
                    "RFDETR_BENCH sweep=%d topology=%s res=%d batch=%d tiles=%d intlv=%b accel=%s total_ms=%.2f p1_ms=%.2f topk_ms=%.2f p2_ms=%.2f rolling_ms=%.2f dets=%d",
                    sweepCount,
                    activeTopology,
                    activeInputResolution,
                    compiledBatchSize,
                    activeBatchItems,
                    useInterleaveThisSweep,
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

        for (det in detections) {
            if (StoneClass.fromId(det.classId)?.isUserAlertable == true && det.confidence > maxAlertableScoreScratch[0]) {
                maxAlertableScoreScratch[0] = det.confidence
            }
        }

        return SweepInferenceResult(
            tileDetections = detections,
            part1LatencyMs = p1Ms,
            hostTopKBridgeLatencyMs = bridgeMs,
            part2LatencyMs = p2Ms,
            totalSweepLatencyMs = totalMs,
            rollingAvgLatencyMs = rollingLatencyMs,
            isHardwareGraphActive = hardwareGraphsLoaded,
            maxAlertableCandidateScore = maxAlertableScoreScratch[0]
        )
    }

    /**
     * Decodes batched predictions for standard `B=5`, `6x256` (`B=6`), or Track 4 `B=3` Even/Odd
     * Sub-Swath Interleaving (updating the 1-sweep opposite-parity cache so all 5 tiles are emitted).
     */
    private fun decodeBatchedOutputsWithMode(
        activeBatchItems: Int,
        useInterleaveThisSweep: Boolean,
        boxesCxCyWhNorm: FloatArray,
        classScores: FloatArray,
        outDetections: MutableList<TileDetection>
    ) {
        if (compiledBatchSize == 6 && activeInputResolution == 256) {
            val raw6 = ArrayList<TileDetection>(16)
            for (b in 0 until 6) {
                raw6.clear()
                decodeCocoSingleTilePredictions(
                    tileIndex = 0,
                    batchElementIndex = b,
                    boxesCxCyWhNorm = boxesCxCyWhNorm,
                    classScores = classScores,
                    metadata = metadata,
                    confidenceThreshold = confidenceThreshold,
                    outDetections = raw6,
                    outMaxAlertableScore = maxAlertableScoreScratch
                )
                val stripXOffset = SIX_TILE_256_X_OFFSETS[b].toFloat()
                val scaleX = SIX_TILE_WINDOW_WIDTH_PX.toFloat() / StripRoiGeometry.TILE_SIZE_PX.toFloat()
                val mappedTileIdx = ((b * 4) / 5).coerceIn(0, 4)
                val targetTileStripOffset = (mappedTileIdx * StripRoiGeometry.TILE_STRIDE_PX).toFloat()
                for (d in raw6) {
                    val stripL = stripXOffset + d.leftPx * scaleX
                    val stripR = stripXOffset + d.rightPx * scaleX
                    outDetections.add(
                        d.copy(
                            tileIndex = mappedTileIdx,
                            leftPx = (stripL - targetTileStripOffset).coerceIn(0f, 384f),
                            rightPx = (stripR - targetTileStripOffset).coerceIn(0f, 384f)
                        )
                    )
                }
            }
            return
        }

        if (useInterleaveThisSweep) {
            var bestEvenScore = -1f
            var bestEvenTile = priorityEvenTileIndex
            for (slot in 0 until activeBatchItems) {
                val tileIdx = interleavedBatchTileMap[slot]
                val cache = cachedTileDetections[tileIdx]
                cache.clear()
                decodeCocoSingleTilePredictions(
                    tileIndex = tileIdx,
                    batchElementIndex = slot,
                    boxesCxCyWhNorm = boxesCxCyWhNorm,
                    classScores = classScores,
                    metadata = metadata,
                    confidenceThreshold = confidenceThreshold,
                    outDetections = cache,
                    outMaxAlertableScore = maxAlertableScoreScratch
                )
                cachedTileSweepStamp[tileIdx] = sweepCount
                if (tileIdx % 2 == 0) {
                    for (det in cache) {
                        if (det.classId != StoneClass.HARD_CONFUSER.id && det.confidence > bestEvenScore) {
                            bestEvenScore = det.confidence
                            bestEvenTile = tileIdx
                        }
                    }
                }
            }
            priorityEvenTileIndex = bestEvenTile
            for (tileIdx in 0 until StripRoiGeometry.MAX_TILES) {
                if (sweepCount - cachedTileSweepStamp[tileIdx] <= 1L) {
                    outDetections.addAll(cachedTileDetections[tileIdx])
                }
            }
            return
        }

        decodeCocoBatchPredictions(
            activeTileCount = activeBatchItems,
            boxesCxCyWhNorm = boxesCxCyWhNorm,
            classScores = classScores,
            metadata = metadata,
            confidenceThreshold = confidenceThreshold,
            outDetections = outDetections,
            outMaxAlertableScore = maxAlertableScoreScratch
        )
    }

    /**
     * Retains a compiled Patrol [LiveModelProfile] and its GPU [TensorBuffer] bindings in the
     * bounded 3-slot LRU cache so warm `SRC` drawer switches complete in `< 5 ms` (`<= 500 ms`)
     * without re-unpacking `W8-Dyn` constant tensors onto GPU memory.
     */
    private class WarmPatrolSlot(
        val metadata: CocoStandInMetadata,
        val activeTopology: String,
        val activeAcceleratorLabel: String,
        val compiledBatchSize: Int,
        val activeInputResolution: Int,
        val activeTileFloatCount: Int,
        val singleModel: CompiledModel?,
        val part1Model: CompiledModel?,
        val part2Model: CompiledModel?,
        val singleInputs: List<TensorBuffer>,
        val singleOutputs: List<TensorBuffer>,
        val part1Inputs: List<TensorBuffer>,
        val part1Outputs: List<TensorBuffer>,
        val part2Inputs: List<TensorBuffer>,
        val part2Outputs: List<TensorBuffer>,
        val part2ZeroCopyInputs: List<TensorBuffer>,
        val singleBoxesOutputIdx: Int,
        val singleScoresOutputIdx: Int,
        val part2BoxesOutputIdx: Int,
        val part2ScoresOutputIdx: Int,
        var useZeroCopyEncMemory: Boolean,
        val interleaveEvenOdd: Boolean,
        val initMessage: String
    ) {
        fun close() {
            singleInputs.forEach { runCatching { it.close() } }
            singleOutputs.forEach { runCatching { it.close() } }
            part1Inputs.forEach { runCatching { it.close() } }
            part1Outputs.forEach { runCatching { it.close() } }
            part2Inputs.forEach { runCatching { it.close() } }
            part2Outputs.forEach { runCatching { it.close() } }
            runCatching { singleModel?.close() }
            runCatching { part1Model?.close() }
            runCatching { part2Model?.close() }
        }
    }

    private fun cacheCurrentAsWarmPatrolSlot(profile: LiveModelProfile) {
        warmPatrolLru[profile]?.takeIf {
            it.part1Model !== part1Model && it.singleModel !== singleModel
        }?.close()
        warmPatrolLru[profile] = WarmPatrolSlot(
            metadata = metadata,
            activeTopology = activeTopology,
            activeAcceleratorLabel = activeAcceleratorLabel,
            compiledBatchSize = compiledBatchSize,
            activeInputResolution = activeInputResolution,
            activeTileFloatCount = activeTileFloatCount,
            singleModel = singleModel,
            part1Model = part1Model,
            part2Model = part2Model,
            singleInputs = singleInputs,
            singleOutputs = singleOutputs,
            part1Inputs = part1Inputs,
            part1Outputs = part1Outputs,
            part2Inputs = part2Inputs,
            part2Outputs = part2Outputs,
            part2ZeroCopyInputs = part2ZeroCopyInputs,
            singleBoxesOutputIdx = singleBoxesOutputIdx,
            singleScoresOutputIdx = singleScoresOutputIdx,
            part2BoxesOutputIdx = part2BoxesOutputIdx,
            part2ScoresOutputIdx = part2ScoresOutputIdx,
            useZeroCopyEncMemory = useZeroCopyEncMemory,
            interleaveEvenOdd = interleaveEvenOdd,
            initMessage = initMessage
        )
        activeSlotFromLru = true
        while (warmPatrolLru.size > MAX_WARM_PATROL_SLOTS) {
            val iter = warmPatrolLru.entries.iterator()
            var evicted = false
            while (iter.hasNext()) {
                val entry = iter.next()
                if (entry.key != profile) {
                    entry.value.close()
                    iter.remove()
                    evicted = true
                    break
                }
            }
            if (!evicted) break
        }
    }

    private fun restoreWarmPatrolSlot(slot: WarmPatrolSlot) {
        metadata = slot.metadata
        activeTopology = slot.activeTopology
        activeAcceleratorLabel = slot.activeAcceleratorLabel
        compiledBatchSize = slot.compiledBatchSize
        activeInputResolution = slot.activeInputResolution
        activeTileFloatCount = slot.activeTileFloatCount
        singleModel = slot.singleModel
        part1Model = slot.part1Model
        part2Model = slot.part2Model
        singleInputs = slot.singleInputs
        singleOutputs = slot.singleOutputs
        part1Inputs = slot.part1Inputs
        part1Outputs = slot.part1Outputs
        part2Inputs = slot.part2Inputs
        part2Outputs = slot.part2Outputs
        part2ZeroCopyInputs = slot.part2ZeroCopyInputs
        singleBoxesOutputIdx = slot.singleBoxesOutputIdx
        singleScoresOutputIdx = slot.singleScoresOutputIdx
        part2BoxesOutputIdx = slot.part2BoxesOutputIdx
        part2ScoresOutputIdx = slot.part2ScoresOutputIdx
        useZeroCopyEncMemory = slot.useZeroCopyEncMemory
        interleaveEvenOdd = slot.interleaveEvenOdd
        initMessage = slot.initMessage
        allocateStagingBuffersForMetadata()
        hardwareGraphsLoaded = true
        activeSlotFromLru = true
    }

    private fun detachOrReleaseActiveModels() {
        if (activeSlotFromLru) {
            warmPatrolLru[activeLiveProfile]?.useZeroCopyEncMemory = useZeroCopyEncMemory
            activeSlotFromLru = false
            singleInputs = emptyList()
            singleOutputs = emptyList()
            part1Inputs = emptyList()
            part1Outputs = emptyList()
            part2Inputs = emptyList()
            part2Outputs = emptyList()
            part2ZeroCopyInputs = emptyList()
            singleModel = null
            part1Model = null
            part2Model = null
            hardwareGraphsLoaded = false
        } else {
            releaseModelsOnly()
        }
    }

    private fun evictAllWarmPatrolSlots() {
        warmPatrolLru.values.forEach { it.close() }
        warmPatrolLru.clear()
        activeSlotFromLru = false
    }

    /**
     * Closes an uncached active Patrol [TensorBuffer] and [CompiledModel] set while keeping
     * [environment] and the pre-warmed `B=1` `320` Lock-On models resident.
     */
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
        singleInputs = emptyList()
        singleOutputs = emptyList()
        part1Inputs = emptyList()
        part1Outputs = emptyList()
        part2Inputs = emptyList()
        part2Outputs = emptyList()
        part2ZeroCopyInputs = emptyList()
        singleModel = null
        part1Model = null
        part2Model = null
        hardwareGraphsLoaded = false
        activeSlotFromLru = false
    }

    /** Closes the pre-warmed `B=1` `320x320` Lock-On [TensorBuffer] and [CompiledModel] handles. */
    private fun releaseLockOnModels() {
        lockOnPart1Inputs.forEach { runCatching { it.close() } }
        lockOnPart1Outputs.forEach { runCatching { it.close() } }
        lockOnPart2Inputs.forEach { runCatching { it.close() } }
        lockOnPart2Outputs.forEach { runCatching { it.close() } }
        runCatching { lockOnPart1Model?.close() }
        runCatching { lockOnPart2Model?.close() }
        lockOnPart1Inputs = emptyList()
        lockOnPart1Outputs = emptyList()
        lockOnPart2Inputs = emptyList()
        lockOnPart2Outputs = emptyList()
        lockOnPart2ZeroCopyInputs = emptyList()
        lockOnPart1Model = null
        lockOnPart2Model = null
        lockOnHardwareLoaded = false
        lockOnInitializedOnThread = false
    }

    override fun close() {
        detachOrReleaseActiveModels()
        evictAllWarmPatrolSlots()
        releaseLockOnModels()
        runCatching { environment?.close() }
        environment = null
    }

    companion object {
        private const val TAG = "RfDetrTwoGraphGpu"
        private const val GPU_SHADER_CACHE_SUBDIR = "litert_gpu_cache"
        private const val MAX_WARM_PATROL_SLOTS = 3
        private const val LOCK_ON_P1_CACHE_TOKEN = "lockon_320_b1_p1_v1"
        private const val LOCK_ON_P2_CACHE_TOKEN = "lockon_320_b1_p2_v1"
        const val LOCK_ON_DEFAULT_SCORE_THRESHOLD = 0.28f
        const val PART1_ASSET_NAME = CocoStandInMetadata.PART1_ASSET_NAME
        const val PART2_ASSET_NAME = CocoStandInMetadata.PART2_ASSET_NAME
        const val SINGLE_GRAPH_ASSET_NAME = CocoStandInMetadata.SINGLE_GRAPH_ASSET_NAME

        private const val MAX_SUPPORTED_BATCH = 6
        private const val DEFAULT_TILE_FLOAT_COUNT =
            StripRoiGeometry.TILE_SIZE_PX * StripRoiGeometry.TILE_SIZE_PX * 3
        private const val SIX_TILE_WINDOW_WIDTH_PX = 320
        private val SIX_TILE_256_X_OFFSETS = intArrayOf(0, 294, 588, 882, 1176, 1472)
        private const val INTRA_TILE_NMS_IOU_THRESHOLD = 0.45f

        const val TOP_K_QUERIES = 100
        const val DEFAULT_ENCODER_ANCHORS = 900
        const val QUERY_HIDDEN_DIM = 256
        val NUM_CLASSES: Int = StoneClass.entries.size // 5 classes (0..3 targets + 4 hard_confuser)

        /**
         * Parses `rf_detr_coco_metadata.json` into a [CocoStandInMetadata] instance, building the
         * `O(1)` integer array lookup table mapping MS-COCO class indices to [StoneClass.id] and
         * all other indices to [StoneClass.HARD_CONFUSER].
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
         * grid matching `RFDETRNano`'s encoder grid ($24 \times 24$, $20 \times 20$, or $16 \times 16$).
         *
         * @param gridH Spatial height of the encoder grid (`24` default).
         * @param gridW Spatial width of the encoder grid (`24` default).
         * @param defaultWh Normalized initial proposal width and height (`0.05f` default).
         * @return Flat `FloatArray` of length `gridH * gridW * 4` in row-major `[cx, cy, w, h]` order.
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
         * Zero-allocation Host CPU Top-K ($K=100$ or $K=300$) proposal bridge between Part 1 and
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

        /** Restores the min-heap invariant at [startIdx] using [isWorseCandidate] tie-breaking. */
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
                        heapScores[worst],
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

        /** Returns `true` when candidate A has a strictly lower score than B, or equal score with a larger token index. */
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
            outDetections: MutableList<TileDetection>,
            outMaxAlertableScore: FloatArray? = null
        ) {
            for (tileIdx in 0 until activeTileCount) {
                decodeCocoSingleTilePredictions(
                    tileIndex = tileIdx,
                    batchElementIndex = tileIdx,
                    boxesCxCyWhNorm = boxesCxCyWhNorm,
                    classScores = classScores,
                    metadata = metadata,
                    confidenceThreshold = confidenceThreshold,
                    outDetections = outDetections,
                    outMaxAlertableScore = outMaxAlertableScore
                )
            }
        }

        /**
         * Decodes a single tile's `[numQueries, 4]` normalized `[cx, cy, w, h]` boxes and
         * `[numQueries, numClasses]` probabilities/logits at [batchElementIndex] into [outDetections].
         *
         * When `metadata.numQueries > 300` (dense anchor models like `YOLO26`), applies intra-tile
         * greedy NMS (`IoU >= 0.45`) so duplicate same-tile anchor predictions are suppressed before
         * cross-tile seam stitching.
         */
        fun decodeCocoSingleTilePredictions(
            tileIndex: Int,
            batchElementIndex: Int,
            boxesCxCyWhNorm: FloatArray,
            classScores: FloatArray,
            metadata: CocoStandInMetadata,
            confidenceThreshold: Float = metadata.scoreThreshold,
            outDetections: MutableList<TileDetection>,
            targetBoxScalePx: Float = StripRoiGeometry.TILE_SIZE_PX.toFloat(),
            outMaxAlertableScore: FloatArray? = null
        ) {
            val numQueries = metadata.numQueries
            val numClasses = metadata.numClasses
            val tileSize = targetBoxScalePx
            val lut = metadata.cocoClassIdToStoneClassId
            val scoresAreProbs = metadata.scoresAreProbabilities

            val batchBoxOffset = batchElementIndex * numQueries * 4
            val batchScoreOffset = batchElementIndex * numQueries * numClasses
            if (batchBoxOffset + numQueries * 4 > boxesCxCyWhNorm.size ||
                batchScoreOffset + numQueries * numClasses > classScores.size
            ) {
                return
            }

            val startIdx = outDetections.size
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
                val mappedStoneClassId = if (bestCocoClass in lut.indices) {
                    lut[bestCocoClass]
                } else if (metadata.unmappedConfidentToHardConfuser) {
                    StoneClass.HARD_CONFUSER.id
                } else {
                    -1
                }
                if (mappedStoneClassId < 0) continue

                if (outMaxAlertableScore != null &&
                    outMaxAlertableScore.isNotEmpty() &&
                    prob > outMaxAlertableScore[0] &&
                    StoneClass.fromId(mappedStoneClassId)?.isUserAlertable == true
                ) {
                    outMaxAlertableScore[0] = prob
                }

                if (prob < confidenceThreshold) continue

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

            // Apply intra-tile greedy NMS when decoding dense anchor models (e.g., YOLO26 with 3024 anchors)
            val addedCount = outDetections.size - startIdx
            if (numQueries > CocoStandInMetadata.DEFAULT_COCO_TOP_K_QUERIES && addedCount > 1) {
                val slice = ArrayList<TileDetection>(addedCount)
                for (i in startIdx until outDetections.size) {
                    slice.add(outDetections[i])
                }
                for (i in outDetections.size - 1 downTo startIdx) {
                    outDetections.removeAt(i)
                }
                slice.sortByDescending { it.confidence }
                for (cand in slice) {
                    var suppressed = false
                    for (k in startIdx until outDetections.size) {
                        val kept = outDetections[k]
                        if (kept.classId == cand.classId &&
                            tileDetectionIou(kept, cand) >= INTRA_TILE_NMS_IOU_THRESHOLD
                        ) {
                            suppressed = true
                            break
                        }
                    }
                    if (!suppressed) {
                        outDetections.add(cand)
                    }
                }
            }
        }

        /** Computes 2D Intersection-over-Union between two tile-local [TileDetection] boxes. */
        private fun tileDetectionIou(a: TileDetection, b: TileDetection): Float {
            val interL = max(a.leftPx, b.leftPx)
            val interT = max(a.topPx, b.topPx)
            val interR = min(a.rightPx, b.rightPx)
            val interB = min(a.bottomPx, b.bottomPx)
            val interW = (interR - interL).coerceAtLeast(0f)
            val interH = (interB - interT).coerceAtLeast(0f)
            val interArea = interW * interH
            if (interArea <= 0f) return 0f
            val areaA = (a.rightPx - a.leftPx).coerceAtLeast(0f) * (a.bottomPx - a.topPx).coerceAtLeast(0f)
            val areaB = (b.rightPx - b.leftPx).coerceAtLeast(0f) * (b.bottomPx - b.topPx).coerceAtLeast(0f)
            val union = areaA + areaB - interArea
            return if (union > 0f) interArea / union else 0f
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

        /** Restores the min-heap invariant at [startIdx] for [selectTopKQueriesOnHost]. */
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
