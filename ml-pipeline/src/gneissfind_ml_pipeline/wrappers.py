"""PyTorch export wrappers adapting ``RFDETRNano`` to the GneissFind LiteRT GPU tensor contract.

``PushBroomCropper`` on Android stages ``384x384`` RGB float32 tiles in ``NHWC`` layout with values
in ``[0.0, 1.0]`` (`.agents/rules/litert-inference.md` §2). This module wraps ``rfdetr``'s
``LWDETR`` detector into ``torch.export``-traceable modules for ``litert_torch.convert``:

- :class:`FusedRfDetrSingleGraph`: single-file graph fusing ``NHWC -> NCHW``, ImageNet
  normalization, backbone, two-stage query selection, decoder, and ``sigmoid`` class probabilities.
- :class:`RfDetrPart1BackboneEncoder` + :func:`host_select_topk_queries` +
  :class:`RfDetrPart2DecoderHead`: two-graph split that moves two-stage ``TopK`` query selection to
  the host CPU so both GPU subgraphs compile without ``TOPK_V2`` or ``GATHER_ND`` partitions.

Additionally rewrites DINOv2 windowed attention and deformable cross-attention tensor reshapes to
stay strictly within rank ``<= 4`` without ``BROADCAST_TO`` ops, matching the Android OpenCL GPU
delegate (`LITERT_CL`) rank-4 tensor constraint.
"""

from __future__ import annotations

import copy
import math
import os
import types
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rfdetr import RFDETRNano
from rfdetr.models import transformer as rfdetr_transformer
from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    BackboneOutput,
    WindowedDinov2WithRegistersBackbone,
    WindowedDinov2WithRegistersEmbeddings,
)
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn
from torch import Tensor, nn

IMAGENET_MEAN_RGB: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD_RGB: tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_INPUT_RESOLUTION: int = 384
ENCODER_GRID_HW: tuple[int, int] = (24, 24)
DEFAULT_PROPOSAL_WH: float = 0.05


def _clamp_unit_interval_via_relu(tensor: Tensor) -> Tensor:
    """Clamps ``tensor`` to ``[0.0, 1.0]`` via ``RELU(x) - RELU(x - 1)`` to avoid ``RELU_0_TO_1``."""
    return F.relu(tensor) - F.relu(tensor - 1.0)


def _gpu_friendly_sineembed_for_position(
    pos_tensor: Tensor,
    dim: int = 128,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    """Computes sine/cosine box coordinate embeddings without ``FLOOR_DIV`` or rank ``> 4`` ops."""
    batch_size, num_queries, coord_dim = pos_tensor.shape
    scale = 2.0 * math.pi
    half_dim = dim // 2
    denom = [10000.0 ** (2.0 * i / float(dim)) for i in range(half_dim)]
    denom_t = (
        pos_tensor.new_tensor(denom, dtype=torch.float32)
        .to(pos_tensor.dtype)
        .view(1, 1, half_dim)
    )
    if coord_dim == 2:
        order: tuple[int, ...] = (1, 0)
    elif coord_dim == 4:
        order = (1, 0, 2, 3)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1): {coord_dim}")

    parts = []
    for idx in order:
        ang = (pos_tensor[:, :, idx : idx + 1] * scale) / denom_t
        emb = torch.stack((ang.sin(), ang.cos()), dim=3).reshape(batch_size, num_queries, dim)
        parts.append(emb)
    pos = torch.cat(parts, dim=2)
    return pos if out_dtype is None else pos.to(out_dtype)


def pin_cuda_device_zero() -> str:
    """Pins ``CUDA_VISIBLE_DEVICES`` to GPU 0 and selects ``cuda:0`` if allocatable, else ``cpu``.

    Follows `.agents/rules/litert-inference.md` §1 (pinning workstation GPU jobs to GPU 0 while
    degrading cleanly to CPU when GPU 0 is absent or its VRAM is occupied).

    Returns:
        ``"cuda:0"`` when a test tensor can be allocated on device 0, otherwise ``"cpu"``.
        Mutates ``os.environ["CUDA_VISIBLE_DEVICES"]`` when not already set.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if not torch.cuda.is_available():
        return "cpu"
    try:
        probe = torch.zeros(1, device="cuda:0")
        del probe
        return "cuda:0"
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return "cpu"


def _repeat_along_dim0(tensor: Tensor, count: int) -> Tensor:
    """Repeats ``tensor`` ``count`` times along dimension 0 using ``torch.cat`` instead of ``BROADCAST_TO``."""
    if count <= 1:
        return tensor
    return torch.cat([tensor] * count, dim=0)


def bilinear_matmul_grid_sample(
    value: Tensor,
    grid: Tensor,
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Tensor:
    """Samples a 2D feature map bilinearly using rank-4 separable tent weights and ``torch.bmm``.

    ``litert_torch`` lowers ``torch.nn.functional.grid_sample`` to ``GATHER_ND`` plus boolean
    bounds masks (``LOGICAL_AND``, ``SELECT_V2``) that fall back to single-threaded CPU kernels,
    while ``rfdetr``'s ``pixel_row_grid_sample`` flattens ``(N, H+2, W+2)`` indices into float
    arithmetic that exceeds ``float16``'s exact integer limit (2048) once ``N * 676 > 2048``
    (see ``rfdetr/export/_litert/sampling.py``). Computing separable 1D bilinear tent weights
    ``relu(1 - |p - i|)`` over ``0..H-1`` and ``0..W-1`` uses integer coordinates ``<= 23`` (exact
    in both FP16 and FP32) and lowers exclusively to rank-4 GPU-native elementwise ops and
    ``BATCH_MATMUL``.

    Args:
        value: Feature map tensor of shape ``(N, C, H, W)`` in ``float32`` or ``float16``.
        grid: Normalized sampling coordinates ``(x, y)`` in ``[-1.0, 1.0]`` (with ``(-1, -1)`` at
            the top-left outer corner and ``(+1, +1)`` at the bottom-right outer corner), shape
            ``(N, Hg, Wg, 2)``. Out-of-bounds coordinates receive zero contribution.
        padding_mode: Must be ``"zeros"``.
        align_corners: Must be ``False``.

    Returns:
        Bilinearly sampled tensor of shape ``(N, C, Hg, Wg)`` matching ``F.grid_sample``.

    Raises:
        ValueError: If ``padding_mode != "zeros"`` or ``align_corners`` is ``True``.
    """
    if padding_mode != "zeros" or align_corners:
        raise ValueError(
            f"bilinear_matmul_grid_sample supports only padding_mode='zeros' and "
            f"align_corners=False; got padding_mode={padding_mode!r}, align_corners={align_corners}."
        )

    batch_heads, channels, height, width = value.shape
    _, grid_h, grid_w, _ = grid.shape
    num_points = grid_h * grid_w

    grid_flat = grid.reshape(batch_heads, num_points, 2)
    px = grid_flat[..., 0] * (width * 0.5) + ((width - 1.0) * 0.5)
    py = grid_flat[..., 1] * (height * 0.5) + ((height - 1.0) * 0.5)

    cols = torch.arange(width, dtype=value.dtype, device=value.device).view(1, 1, width)
    rows = torch.arange(height, dtype=value.dtype, device=value.device).view(1, 1, height)

    wx = F.relu(1.0 - (px.unsqueeze(-1) - cols).abs())
    wy = F.relu(1.0 - (py.unsqueeze(-1) - rows).abs())

    value_hcw = value.permute(0, 2, 1, 3).reshape(batch_heads, height, channels * width)
    sampled_y = torch.bmm(wy, value_hcw).reshape(batch_heads, num_points, channels, width)
    sampled_xy = (sampled_y * wx.unsqueeze(2)).sum(dim=-1)
    return sampled_xy.permute(0, 2, 1).reshape(batch_heads, channels, grid_h, grid_w)


def _gpu_rank4_embeddings_forward(
    self: WindowedDinov2WithRegistersEmbeddings,
    pixel_values: Tensor,
    bool_masked_pos: Tensor | None = None,
) -> Tensor:
    """Computes windowed DINOv2 patch embeddings using only rank ``<= 4`` tensors and no ``BROADCAST_TO``."""
    batch_size, _, height, width = pixel_values.shape
    target_dtype = self.patch_embeddings.projection.weight.dtype
    embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

    cls_tokens = _repeat_along_dim0(self.cls_token, batch_size)
    embeddings = torch.cat((cls_tokens, embeddings), dim=1)
    embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

    if self.config.num_windows > 1:
        num_h_patches = height // self.config.patch_size
        num_w_patches = width // self.config.patch_size
        cls_token_with_pos_embed = embeddings[:, :1]
        pixel_tokens_with_pos_embed = embeddings[:, 1:]
        channels = pixel_tokens_with_pos_embed.shape[-1]

        num_windows = self.config.num_windows
        num_h_win = num_h_patches // num_windows
        num_w_win = num_w_patches // num_windows

        # Fuse (num_w_win, channels) during the 2D window transpose so rank never exceeds 4.
        windowed_pixel_tokens = pixel_tokens_with_pos_embed.reshape(
            batch_size * num_windows,
            num_h_win,
            num_windows,
            num_w_win * channels,
        )
        windowed_pixel_tokens = windowed_pixel_tokens.permute(0, 2, 1, 3).reshape(
            batch_size * (num_windows**2),
            num_h_win * num_w_win,
            channels,
        )
        windowed_cls_token_with_pos_embed = _repeat_along_dim0(
            cls_token_with_pos_embed, num_windows**2
        )
        embeddings = torch.cat((windowed_cls_token_with_pos_embed, windowed_pixel_tokens), dim=1)

    if self.config.num_register_tokens > 0 and self.register_tokens is not None:
        reg_tokens = _repeat_along_dim0(self.register_tokens, embeddings.shape[0])
        embeddings = torch.cat((embeddings[:, :1], reg_tokens, embeddings[:, 1:]), dim=1)

    return self.dropout(embeddings)


def _gpu_rank4_backbone_forward(
    self: WindowedDinov2WithRegistersBackbone,
    pixel_values: Tensor,
    output_hidden_states: bool | None = None,
    output_attentions: bool | None = None,
    return_dict: bool | None = None,
) -> tuple[Any, ...] | BackboneOutput:
    """Runs DINOv2 windowed encoder and un-windows output maps using only rank ``<= 4`` tensors."""
    return_dict = return_dict if return_dict is not None else self.config.return_dict
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )

    embedding_output = self.embeddings(pixel_values)
    outputs = self.encoder(
        embedding_output,
        output_hidden_states=True,
        output_attentions=output_attentions,
        return_dict=return_dict,
    )
    hidden_states = outputs.hidden_states if return_dict else outputs[1]

    feature_maps: tuple[Tensor, ...] = ()
    for stage, hidden_state in zip(self.stage_names, hidden_states):
        if stage in self.out_features:
            if self.config.apply_layernorm:
                hidden_state = self.layernorm(hidden_state)
            if self.config.reshape_hidden_states:
                hidden_state = hidden_state[:, self.num_register_tokens + 1 :]
                batch_size, _, height, width = pixel_values.shape
                patch_size = self.config.patch_size
                num_h_patches = height // patch_size
                num_w_patches = width // patch_size

                if self.config.num_windows > 1:
                    num_windows = self.config.num_windows
                    num_windows_squared = num_windows**2
                    batch_windows, _, channels = hidden_state.shape
                    num_h_win = num_h_patches // num_windows
                    num_w_win = num_w_patches // num_windows
                    # Fuse (num_w_win, channels) so un-windowing stays rank 4.
                    hidden_state = hidden_state.reshape(
                        (batch_windows // num_windows_squared) * num_windows,
                        num_windows,
                        num_h_win,
                        num_w_win * channels,
                    )
                    hidden_state = hidden_state.permute(0, 2, 1, 3)

                hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, -1)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

            feature_maps += (hidden_state,)

    if not return_dict:
        if output_hidden_states:
            return (feature_maps,) + outputs[1:]
        return (feature_maps,) + outputs[2:]

    return BackboneOutput(
        feature_maps=feature_maps,  # type: ignore[arg-type]
        hidden_states=outputs.hidden_states if output_hidden_states else None,
        attentions=outputs.attentions if output_attentions else None,
    )


def _gpu_rank4_ms_deform_attn_forward(
    self: MSDeformAttn,
    query: Tensor,
    reference_points: Tensor,
    input_flatten: Tensor,
    input_spatial_shapes: Tensor,
    input_level_start_index: Tensor,
    input_padding_mask: Tensor | None = None,
    input_spatial_shapes_hw: list[tuple[int, int]] | None = None,
) -> Tensor:
    """Executes multi-scale deformable cross-attention using exclusively rank ``<= 4`` tensors."""
    batch_size, len_query, _ = query.shape
    _, len_input, _ = input_flatten.shape
    bq = batch_size * len_query
    head_dim = self.d_model // self.n_heads
    total_pts = self.n_levels * self.n_points

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))

    attention_weights = self.attention_weights(query).view(
        batch_size, len_query, self.n_heads, total_pts
    )
    attention_weights = F.softmax(attention_weights, -1)

    # Flatten (batch_size, len_query) -> bq so sampling_offsets and sampling_locations stay rank 4.
    sampling_offsets = self.sampling_offsets(query).view(bq, self.n_heads, total_pts, 2)

    n_ref_levels = reference_points.shape[2]
    ref_4d = reference_points.reshape(bq, 1, n_ref_levels, reference_points.shape[-1])
    if n_ref_levels == 1 and self.n_levels > 1:
        ref_4d = torch.cat([ref_4d] * self.n_levels, dim=2)

    # Repeat each level self.n_points times along dim=2 via torch.cat to avoid rank-5 repeat_interleave.
    ref_pts = torch.cat(
        [
            ref_4d[:, :, lvl : lvl + 1, :]
            for lvl in range(self.n_levels)
            for _ in range(self.n_points)
        ],
        dim=2,
    )

    if reference_points.shape[-1] == 4:
        sampling_locations = (
            ref_pts[:, :, :, :2]
            + sampling_offsets * (ref_pts[:, :, :, 2:] * (0.5 / float(self.n_points)))
        )
    else:
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )
        offset_normalizer = torch.cat(
            [
                offset_normalizer[lvl : lvl + 1, :]
                for lvl in range(self.n_levels)
                for _ in range(self.n_points)
            ],
            dim=0,
        ).view(1, 1, total_pts, 2)
        sampling_locations = ref_pts + sampling_offsets / offset_normalizer

    sampling_grids = 2.0 * sampling_locations - 1.0
    value = value.transpose(1, 2).contiguous().view(batch_size, self.n_heads, head_dim, len_input)

    shapes = (
        input_spatial_shapes_hw if input_spatial_shapes_hw is not None else input_spatial_shapes
    )
    num_levels = len(shapes)
    if num_levels == 1:
        value_list = [value]
    else:
        value_list = value.split([int(h) * int(w) for h, w in shapes], dim=3)  # type: ignore[no-untyped-call]

    sampling_value_list = []
    for level_index, (height, width) in enumerate(shapes):
        value_l = value_list[level_index].view(
            batch_size * self.n_heads, head_dim, int(height), int(width)
        )
        grid_l = sampling_grids[
            :, :, level_index * self.n_points : (level_index + 1) * self.n_points, :
        ]
        # Transpose (len_query, n_heads) in 4D by fusing (n_points, 2) -> n_points * 2.
        sampling_grid_l = (
            grid_l.reshape(batch_size, len_query, self.n_heads, self.n_points * 2)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, len_query, self.n_points, 2)
        )
        sampling_value_l = self.grid_sample(
            value_l, sampling_grid_l, padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l)

    attention_weights_4d = attention_weights.transpose(1, 2).reshape(
        batch_size * self.n_heads, 1, len_query, total_pts
    )
    if num_levels == 1:
        sampling_values = sampling_value_list[0]
    else:
        sampling_values = torch.cat(sampling_value_list, dim=-1)

    output = (sampling_values * attention_weights_4d).sum(-1).view(
        batch_size, self.n_heads * head_dim, len_query
    )
    return self.output_proj(output.transpose(1, 2).contiguous())


def flat_gather_top_rows(scores: Tensor, k: int) -> Callable[[Tensor], Tensor]:
    """Selects the ``k`` highest-scoring token rows per batch item via 1D ``index_select``.

    Replaces ``Transformer.select_top_rows``'s 3D ``torch.gather`` with a flattened 1D
    ``torch.index_select`` over ``(B * N, C)`` so the single-graph export avoids rank-3
    ``GATHER_ND`` over feature channels.

    Args:
        scores: Per-token ranking scores of shape ``(B, N)``.
        k: Number of top-scoring tokens to retain per batch element (``1 <= k <= N``).

    Returns:
        Closure mapping any ``(B, N, C)`` token tensor to its selected ``(B, k, C)`` subset in
        descending score order.
    """
    batch_size, num_tokens = scores.shape
    top_indices = torch.topk(scores, k, dim=1)[1]
    batch_offsets = (
        torch.arange(batch_size, device=scores.device, dtype=top_indices.dtype) * num_tokens
    ).unsqueeze(1)
    flat_indices = (top_indices + batch_offsets).reshape(-1)

    def _gather(rows: Tensor) -> Tensor:
        channels = rows.shape[-1]
        gathered = torch.index_select(rows.reshape(batch_size * num_tokens, channels), 0, flat_indices)
        return gathered.reshape(batch_size, k, channels)

    return _gather


def load_prepared_lwdetr(
    resolution: int = DEFAULT_INPUT_RESOLUTION,
    checkpoint_path: str | None = None,
    num_queries: int | None = None,
) -> tuple[nn.Module, RFDETRNano]:
    """Loads ``RFDETRNano`` on CPU and prepares a deep-copied ``LWDETR`` module for LiteRT export.

    Freezes DINOv2 position embeddings to ``(resolution, resolution)``, switches submodules to
    export mode, and patches DINOv2 windowed attention and :class:`MSDeformAttn` to use rank ``<= 4``
    tensors without ``BROADCAST_TO``.

    Args:
        resolution: Square input edge length in pixels (must be divisible by 32; 384 for Nano).
        checkpoint_path: Optional path to ``rf-detr-nano.pth``; ``None`` uses ``rfdetr``'s cached
            MD5-verified weights.
        num_queries: Optional decoder query count override (e.g., ``100`` vs default ``300``).

    Returns:
        Tuple ``(export_lwdetr, eager_detector)`` where ``export_lwdetr`` is an eval-mode CPU
        ``LWDETR`` ready for wrapping and ``eager_detector`` is the unmodified ``RFDETRNano``
        instance for reference parity checks.
    """
    init_kwargs: dict[str, Any] = {"resolution": DEFAULT_INPUT_RESOLUTION, "device": "cpu"}
    if checkpoint_path is not None:
        init_kwargs["pretrain_weights"] = checkpoint_path

    detector = RFDETRNano(**init_kwargs)
    lwdetr = copy.deepcopy(detector.model.model).cpu().eval()
    if num_queries is not None:
        lwdetr.num_queries = int(num_queries)
        lwdetr.transformer.num_queries = int(num_queries)

    # Freeze DINOv2 positional embeddings prior to tracing so bicubic antialiased upsampling is not
    # emitted into the exported graph (see rfdetr/export/prepare.py).
    for module in lwdetr.modules():
        if module.__class__.__name__ == "DinoV2" and hasattr(module, "shape"):
            module.shape = (resolution, resolution)
    lwdetr.export()

    for module in lwdetr.modules():
        if isinstance(module, WindowedDinov2WithRegistersEmbeddings):
            module.forward = types.MethodType(_gpu_rank4_embeddings_forward, module)
        elif isinstance(module, WindowedDinov2WithRegistersBackbone):
            module.forward = types.MethodType(_gpu_rank4_backbone_forward, module)
        elif isinstance(module, MSDeformAttn):
            module.grid_sample = bilinear_matmul_grid_sample
            module.forward = types.MethodType(_gpu_rank4_ms_deform_attn_forward, module)
    rfdetr_transformer.gen_sineembed_for_position = _gpu_friendly_sineembed_for_position
    lwdetr.transformer.select_top_rows = flat_gather_top_rows
    return lwdetr, detector


class FusedRfDetrSingleGraph(nn.Module):
    """Single-graph ``RF-DETR-Nano`` export wrapper with fused ``NHWC`` input normalization.

    Consumes float32 RGB tiles in ``[0.0, 1.0]`` directly from ``PushBroomSweepBuffers`` and emits
    normalized ``cxcywh`` boxes plus ``sigmoid`` class probabilities.

    Attributes:
        model: Export-mode ``LWDETR`` detector module.
        mean: Registered ``(1, 3, 1, 1)`` ImageNet RGB mean buffer.
        std: Registered ``(1, 3, 1, 1)`` ImageNet RGB standard-deviation buffer.
    """

    def __init__(self, export_lwdetr: nn.Module) -> None:
        """Initializes the single-graph wrapper around a prepared export-mode ``LWDETR`` module.

        Args:
            export_lwdetr: Module returned by :func:`load_prepared_lwdetr`.
        """
        super().__init__()
        self.model = export_lwdetr
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN_RGB, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD_RGB, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, nhwc_rgb_01: Tensor) -> tuple[Tensor, Tensor]:
        """Runs end-to-end single-graph object detection on a batch of ``NHWC`` tiles.

        Args:
            nhwc_rgb_01: Float32 RGB image batch of shape ``(B, 384, 384, 3)`` in ``[0.0, 1.0]``.

        Returns:
            Tuple ``(boxes_cxcywh, scores)``:
            - ``boxes_cxcywh``: Float32 normalized ``(cx, cy, w, h)`` boxes in ``[0.0, 1.0]``,
              shape ``(B, 300, 4)``.
            - ``scores``: Float32 independent per-class sigmoid probabilities in ``[0.0, 1.0]``,
              shape ``(B, 300, 91)``.
        """
        nchw = nhwc_rgb_01.permute(0, 3, 1, 2).contiguous()
        normalized = (nchw - self.mean) / self.std
        boxes_cxcywh, raw_logits = self.model(normalized)
        return _clamp_unit_interval_via_relu(boxes_cxcywh), raw_logits.sigmoid()


class RfDetrPart1BackboneEncoder(nn.Module):
    """Graph 1 of the two-graph LiteRT GPU split: DINOv2 backbone, projector, and proposal scorer.

    Stops immediately before two-stage ``TopK`` selection so the entire heavy backbone (92.6% of
    total FLOPs) compiles into a single GPU FP16 delegate partition with zero ``TOPK_V2`` or
    ``GATHER_ND`` ops. Omits ``PositionEmbeddingSine`` because the ``24x24`` sine embedding is
    constant across inputs and is baked into :class:`RfDetrPart2DecoderHead`.

    Attributes:
        encoder_backbone: Export-mode DINOv2 + single-scale ``P4`` projector module.
        enc_output: Two-stage encoder linear projection before LayerNorm.
        enc_output_norm: Token-wise LayerNorm preceding the encoder classification head.
        enc_out_class_embed: Encoder proposal classification linear head (91 classes).
    """

    def __init__(self, export_lwdetr: nn.Module) -> None:
        """Extracts Stage 1 submodules from a prepared export-mode ``LWDETR`` instance.

        Args:
            export_lwdetr: Module returned by :func:`load_prepared_lwdetr`.
        """
        super().__init__()
        self.encoder_backbone = export_lwdetr.backbone[0]
        self.enc_output = export_lwdetr.transformer.enc_output[0]
        self.enc_output_norm = export_lwdetr.transformer.enc_output_norm[0]
        self.enc_out_class_embed = export_lwdetr.transformer.enc_out_class_embed[0]
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN_RGB, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD_RGB, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, nhwc_rgb_01: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Projects ``NHWC`` tiles into flattened ``24x24`` encoder memory and per-token max logits.

        Args:
            nhwc_rgb_01: Float32 RGB image batch of shape ``(B, 384, 384, 3)`` in ``[0.0, 1.0]``.

        Returns:
            Tuple ``(memory, output_memory_prenorm, enc_max_scores)``:
            - ``memory``: Float32 encoder feature map flattened row-major over ``24x24``,
              shape ``(B, 576, 256)``.
            - ``output_memory_prenorm``: Float32 projected proposal features prior to
              ``enc_output_norm``, shape ``(B, 576, 256)``.
            - ``enc_max_scores``: Float32 maximum class logit per spatial token for host ``TopK``
              ranking, shape ``(B, 576)``.
        """
        nchw = nhwc_rgb_01.permute(0, 3, 1, 2).contiguous()
        normalized = (nchw - self.mean) / self.std
        srcs, _, _ = self.encoder_backbone(normalized)
        memory = srcs[0].flatten(2).transpose(1, 2).contiguous()
        output_memory_prenorm = self.enc_output(memory)
        output_memory_norm = self.enc_output_norm(output_memory_prenorm)
        enc_logits = self.enc_out_class_embed(output_memory_norm)
        enc_max_scores = enc_logits.max(dim=-1)[0]
        return memory, output_memory_prenorm, enc_max_scores


def build_base_proposals_np(
    grid_hw: tuple[int, int] = ENCODER_GRID_HW,
    default_wh: float = DEFAULT_PROPOSAL_WH,
) -> NDArray[np.float32]:
    """Builds the constant ``(H * W, 4)`` normalized ``(cx, cy, w, h)`` anchor grid for Stage 2.

    Matches ``rfdetr.models.transformer.gen_encoder_output_proposals`` with ``unsigmoid=False``
    (``bbox_reparam=True``).

    Args:
        grid_hw: ``(height, width)`` of the encoder feature map in tokens (``(24, 24)`` at 384 px).
        default_wh: Normalized initial proposal width and height in ``[0.0, 1.0]``.

    Returns:
        Float32 array of shape ``(H * W, 4)`` ordered row-major ``(y * W + x)``.
    """
    grid_h, grid_w = grid_hw
    ys, xs = np.meshgrid(
        np.arange(grid_h, dtype=np.float32),
        np.arange(grid_w, dtype=np.float32),
        indexing="ij",
    )
    cx = (xs + 0.5) / np.float32(grid_w)
    cy = (ys + 0.5) / np.float32(grid_h)
    wh = np.full_like(cx, np.float32(default_wh))
    return np.stack([cx, cy, wh, wh], axis=-1).reshape(grid_h * grid_w, 4)


def host_select_topk_queries(
    output_memory_prenorm: NDArray[np.float32],
    enc_max_scores: NDArray[np.float32],
    top_k: int = 300,
    base_proposals: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    """Reference host CPU ``TopK`` bridge between :class:`RfDetrPart1BackboneEncoder` and Part 2.

    Ranks the ``576`` spatial tokens per batch item by descending ``enc_max_scores`` (breaking ties
    by ascending token index, matching ``torch.topk(..., sorted=True)`` on CPU) and gathers both
    ``output_memory_prenorm`` and the static ``(cx, cy, w, h)`` anchor proposals.

    Args:
        output_memory_prenorm: Stage 1 pre-norm features of shape ``(B, N, C)``.
        enc_max_scores: Stage 1 per-token max class logits of shape ``(B, N)``.
        top_k: Number of object queries to select per tile (300 for ``RFDETRNano``).
        base_proposals: Optional precomputed ``(N, 4)`` proposal array from
            :func:`build_base_proposals_np`; computed on demand when ``None``.

    Returns:
        Tuple ``(selected_prenorm, selected_proposals, selected_indices)`` with shapes
        ``(B, top_k, C)``, ``(B, top_k, 4)``, and ``(B, top_k)``.
    """
    batch_size, num_tokens, _ = output_memory_prenorm.shape
    proposals_table = (
        base_proposals if base_proposals is not None else build_base_proposals_np()
    )
    token_order = np.arange(num_tokens, dtype=np.int64)

    selected_indices = np.empty((batch_size, top_k), dtype=np.int64)
    for b in range(batch_size):
        ranked = np.lexsort((token_order, -enc_max_scores[b]))[:top_k]
        selected_indices[b] = ranked

    batch_idx = np.arange(batch_size, dtype=np.int64)[:, None]
    selected_prenorm = np.ascontiguousarray(output_memory_prenorm[batch_idx, selected_indices])
    selected_proposals = np.ascontiguousarray(proposals_table[selected_indices])
    return selected_prenorm, selected_proposals, selected_indices


class RfDetrPart2DecoderHead(nn.Module):
    """Graph 2 of the two-graph LiteRT GPU split: proposal refinement, 2-layer decoder, and heads.

    Accepts host-gathered top-300 proposal features and anchor boxes alongside Stage 1's encoder
    memory so that Stage 2 also compiles without ``TOPK_V2`` or ``GATHER_ND`` ops.

    Attributes:
        enc_output_norm: Token-wise LayerNorm applied to gathered pre-norm query features.
        enc_out_bbox_embed: Pointwise MLP predicting initial decoder reference box deltas.
        decoder: Two-layer deformable cross-attention ``TransformerDecoder`` patched with
            :func:`bilinear_matmul_grid_sample`.
        bbox_embed: Final decoder bounding-box delta MLP.
        class_embed: Final 91-class linear classification head.
    """

    def __init__(
        self,
        export_lwdetr: nn.Module,
        resolution: int = DEFAULT_INPUT_RESOLUTION,
        grid_hw: tuple[int, int] = ENCODER_GRID_HW,
    ) -> None:
        """Initializes Stage 2 and bakes the constant ``(1, 576, 256)`` sine positional embedding.

        Args:
            export_lwdetr: Module returned by :func:`load_prepared_lwdetr`.
            resolution: Square input image size in pixels used to precompute ``pos_embed``.
            grid_hw: ``(height, width)`` of the encoder token grid (``(24, 24)`` at 384 px).
        """
        super().__init__()
        self.grid_hw = grid_hw
        self.enc_output_norm = export_lwdetr.transformer.enc_output_norm[0]
        self.enc_out_bbox_embed = export_lwdetr.transformer.enc_out_bbox_embed[0]
        self.decoder = export_lwdetr.transformer.decoder
        self.bbox_embed = export_lwdetr.bbox_embed
        self.class_embed = export_lwdetr.class_embed

        num_queries = int(export_lwdetr.num_queries)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, resolution, resolution, dtype=torch.float32)
            _, _, poss_list, _ = export_lwdetr.backbone(dummy)
            const_pos_embed = poss_list[0].flatten(2).transpose(1, 2).contiguous()

        self.register_buffer("query_feat", export_lwdetr.query_feat.weight[:num_queries].detach().clone())
        self.register_buffer(
            "refpoint_embed", export_lwdetr.refpoint_embed.weight[:num_queries].detach().clone()
        )
        self.register_buffer("pos_embed", const_pos_embed)
        self.register_buffer(
            "spatial_shapes", torch.tensor([list(grid_hw)], dtype=torch.long)
        )
        self.register_buffer("level_start_index", torch.tensor([0], dtype=torch.long))

    def forward(
        self,
        memory: Tensor,
        selected_prenorm: Tensor,
        selected_proposals: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Refines the host-selected 300 queries through the decoder and detection heads.

        Args:
            memory: Stage 1 flattened encoder feature map, shape ``(B, 576, 256)``.
            selected_prenorm: Host-gathered pre-norm query features, shape ``(B, 300, 256)``.
            selected_proposals: Host-gathered initial normalized ``(cx, cy, w, h)`` anchors,
                shape ``(B, 300, 4)``.

        Returns:
            Tuple ``(boxes_cxcywh, scores)`` matching :class:`FusedRfDetrSingleGraph`:
            - ``boxes_cxcywh``: Float32 normalized ``(cx, cy, w, h)`` boxes in ``[0.0, 1.0]``,
              shape ``(B, 300, 4)``.
            - ``scores``: Float32 per-class sigmoid probabilities in ``[0.0, 1.0]``,
              shape ``(B, 300, 91)``.
        """
        batch_size = memory.shape[0]

        # 1. Refine gathered encoder proposals (bbox_reparam=True multiplicative parameterization).
        tgt_ts = self.enc_output_norm(selected_prenorm)
        enc_delta = self.enc_out_bbox_embed(tgt_ts)
        ref_ts_cxcy = enc_delta[..., :2] * selected_proposals[..., 2:] + selected_proposals[..., :2]
        ref_ts_wh = enc_delta[..., 2:].exp() * selected_proposals[..., 2:]
        ref_ts = torch.cat([ref_ts_cxcy, ref_ts_wh], dim=-1)

        # 2. Combine learned decoder query embeddings with Stage 1 reference boxes using torch.cat
        #    along dim=0 so batch expansion never emits BROADCAST_TO on GPU.
        tgt = _repeat_along_dim0(self.query_feat.unsqueeze(0), batch_size)
        ref_init = _repeat_along_dim0(self.refpoint_embed.unsqueeze(0), batch_size)
        ref_cxcy = ref_init[..., :2] * ref_ts[..., 2:] + ref_ts[..., :2]
        ref_wh = ref_init[..., 2:].exp() * ref_ts[..., 2:]
        refpoints = torch.cat([ref_cxcy, ref_wh], dim=-1)

        # 3. Run 2-layer deformable cross-attention decoder and reparameterized output heads.
        pos = _repeat_along_dim0(self.pos_embed, batch_size)
        hs, ref_unsigmoid = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=None,
            pos=pos,
            refpoints_unsigmoid=refpoints,
            level_start_index=self.level_start_index,
            spatial_shapes=self.spatial_shapes,
            spatial_shapes_hw=[self.grid_hw],
            valid_ratios=None,
        )[:2]

        dec_delta = self.bbox_embed(hs)
        boxes_cxcy = dec_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
        boxes_wh = dec_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
        boxes = _clamp_unit_interval_via_relu(torch.cat([boxes_cxcy, boxes_wh], dim=-1))
        scores = self.class_embed(hs).sigmoid()

        if boxes.dim() == 4:
            boxes = boxes[-1]
            scores = scores[-1]
        return boxes, scores
