"""
Point-based UNet Model (PointUNet).

A UNet variant that processes B,C,H,W tensors point-wise along the C dimension.
Each (h,w) spatial position is treated as an independent "point" with a
C-dimensional feature vector. All spatial convolutions are replaced with
shared Linear layers (point-wise MLPs), making the model agnostic to spatial
structure.

Key properties
--------------
- **Point-independent**: no spatial convolutions; each point is processed by
  shared weights, so the model naturally handles irregular point sets.
- **Supports point dropping**: when ``allow_point_drop=True``,
  attention is disabled and any subset of points can be processed.
- **Optional attention**: when ``allow_point_drop=False``,
  multi-head self-attention across points is inserted at the resolutions
  specified by ``attention_resolutions`` (mirroring the original UNetModel).
- **Same interface as UNetModel**: constructor accepts (and ignores where
  appropriate) the same keyword arguments, so it can serve as a drop-in
  replacement.
"""

import math
from typing import List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialise all parameters of *module* and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class PointNorm(nn.Module):
    """LayerNorm that casts to float for numerical stability (like the
    GroupNorm wrappers in the original code-base)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.float()).type(x.dtype)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PointResBlock(nn.Module):
    """Residual block operating on ``(B, N, C)`` point features.

    Mirrors the structure of ``ResBlock`` in the original UNet but replaces
    every spatial convolution with a ``nn.Linear`` (shared across all points).
    Uses pre-norm layout (Norm → Activation → Linear).

    Parameters
    ----------
    channels : int
        Number of input feature channels.
    dropout : float
        Dropout probability.
    out_channels : int, optional
        Number of output channels.  Defaults to *channels*.
    use_checkpoint : bool
        If ``True``, use gradient checkpointing to save memory.
    """

    def __init__(
        self,
        channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        # Pre-norm: Norm → Act → Linear  (×2, with dropout before 2nd linear)
        self.norm1 = PointNorm(channels)
        self.act1 = nn.SiLU()
        self.linear1 = nn.Linear(channels, self.out_channels)

        self.norm2 = PointNorm(self.out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = zero_module(nn.Linear(self.out_channels, self.out_channels))

        # Skip / shortcut
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Linear(channels, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        h = self.linear1(self.act1(self.norm1(x)))
        h = self.linear2(self.dropout(self.act2(self.norm2(h))))
        return self.skip_connection(x) + h


class PointAttentionBlock(nn.Module):
    """Multi-head self-attention across points.

    Operates on ``(B, N, C)`` tensors where *N* is the number of points.
    Optionally supports *cross-view* self-attention by merging the view
    dimension into the point/token dimension before computing attention.

    Parameters
    ----------
    channels : int
        Feature dimensionality per point.
    num_heads : int
        Number of attention heads (used when ``num_head_channels == -1``).
    num_head_channels : int
        Per-head channel width (takes precedence over *num_heads*).
    use_checkpoint : bool
        Gradient checkpointing flag.
    num_frames : int
        Number of views for cross-view attention.
    use_cross_view_self_attn : bool
        If ``True``, views are merged into the token axis so that attention
        spans all views jointly.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        num_frames: int = 2,
        use_cross_view_self_attn: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        self.num_frames = num_frames
        self.use_cross_view_self_attn = use_cross_view_self_attn

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"channels={channels} not divisible by "
                f"num_head_channels={num_head_channels}"
            )
            self.num_heads = channels // num_head_channels

        self.head_dim = channels // self.num_heads

        self.norm = PointNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = zero_module(nn.Linear(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)  — or (B*V, N, C) when cross-view is used
        residual = x
        x_normed = self.norm(x)

        B, N, C = x_normed.shape

        # Optionally merge the view axis into the token axis
        if self.use_cross_view_self_attn:
            n_views = self.num_frames
            # (B*V, N, C) → (B, V*N, C)
            x_normed = rearrange(x_normed, "(b v) n c -> b (v n) c", v=n_views)
            B_eff = B // n_views
        else:
            B_eff = B

        N_eff = x_normed.shape[1]

        qkv = self.qkv(x_normed)                                       # (B_eff, N_eff, 3C)
        qkv = qkv.reshape(B_eff, N_eff, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                               # (3, B_eff, H, N_eff, D)
        q, k, v = qkv.unbind(0)                                         # each (B_eff, H, N_eff, D)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q * scale, k.transpose(-2, -1))             # (B_eff, H, N_eff, N_eff)
        attn = F.softmax(attn.float(), dim=-1).type(attn.dtype)

        out = torch.matmul(attn, v)                                     # (B_eff, H, N_eff, D)
        out = out.transpose(1, 2).reshape(B_eff, N_eff, C)              # (B_eff, N_eff, C)

        # Undo cross-view merge
        if self.use_cross_view_self_attn:
            out = rearrange(out, "b (v n) c -> (b v) n c", v=n_views)

        out = self.proj_out(out)
        return residual + out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PointUNet(nn.Module):
    """Point-based UNet.

    Accepts the **same constructor keyword arguments** as ``UNetModel`` so that
    it can be used as a drop-in replacement.  Spatial-convolution-specific
    parameters (``dims``, ``conv_resample``, …) are accepted but ignored.

    Additional parameter
    --------------------
    allow_point_drop : bool
        * ``False`` (default) — attention is used at the resolutions given by
          ``attention_resolutions``; the full set of points must be provided.
        * ``True`` — attention is disabled; each point is processed fully
          independently and an arbitrary subset of HxW points may be passed.
    """

    # Accept all UNetModel kwargs for API compatibility; spatial-only args
    # are silently ignored.
    def __init__(
        self,
        image_size,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Union[Set[int], List[int], Tuple[int, ...]],
        dropout: float = 0.0,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        conv_resample: bool = True,        # ignored (no spatial conv)
        dims: int = 2,                     # ignored (point-wise)
        middle_block_attn: bool = False,
        middle_block_no_identity: bool = False,
        postnorm: bool = False,            # ignored (always pre-norm)
        attn_prenorm: bool = False,        # ignored
        downsample_3ddim: bool = False,    # ignored
        zero_final_layer: bool = False,
        channels_per_group=None,           # ignored (use LayerNorm)
        num_classes=None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,   # ignored
        resblock_updown: bool = False,        # ignored
        use_new_attention_order: bool = False, # ignored
        use_spatial_transformer: bool = False, # ignored
        transformer_depth: int = 1,            # ignored
        context_dim=None,                      # ignored
        n_embed=None,                          # ignored
        legacy: bool = True,
        cross_attn_condition: bool = False,    # ignored
        tanh_gating: bool = False,             # ignored
        ffn_after_cross_attn: bool = False,    # ignored
        cross_attn_with_norm: bool = False,    # ignored
        condition_channels: int = 384,         # ignored
        condition_num_views: int = 3,          # ignored
        no_self_attn: bool = False,
        conv_kernel_size: int = 3,             # ignored
        concat_condition: bool = False,        # ignored
        concat_conv3x3: bool = False,          # ignored
        num_frames: int = 2,
        use_cross_view_self_attn: bool = False,
        downsample_factor=None,
        # ---- Point-UNet specific ----
        allow_point_drop: bool = False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1, (
                "Either num_heads or num_head_channels has to be set"
            )
        if num_head_channels == -1:
            assert num_heads != -1, (
                "Either num_heads or num_head_channels has to be set"
            )

        # ---- store hyper-parameters ----
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.allow_point_drop = allow_point_drop
        self.middle_block_attn = middle_block_attn
        self.downsample_factor = downsample_factor

        # Attention is used only when point dropping is *not* allowed
        use_attention = not allow_point_drop

        # ------------------------------------------------------------------ #
        #  Encoder (input blocks)                                             #
        # ------------------------------------------------------------------ #
        self.input_blocks = nn.ModuleList(
            [nn.ModuleList([nn.Linear(in_channels, model_channels)])]
        )
        input_block_chans: List[int] = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: List[nn.Module] = [
                    PointResBlock(
                        ch,
                        dropout,
                        out_channels=mult * model_channels,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = mult * model_channels

                if use_attention and ds in attention_resolutions:
                    if num_head_channels == -1:
                        _num_heads = num_heads
                    else:
                        _num_heads = ch // num_head_channels

                    if not no_self_attn:
                        layers.append(
                            PointAttentionBlock(
                                ch,
                                num_heads=_num_heads,
                                num_head_channels=(
                                    ch // _num_heads
                                    if num_head_channels == -1
                                    else num_head_channels
                                ),
                                use_checkpoint=use_checkpoint,
                                num_frames=num_frames,
                                use_cross_view_self_attn=use_cross_view_self_attn,
                            )
                        )

                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            # Spatial downsample position → identity placeholder (preserves
            # the skip-connection bookkeeping of the original UNet).
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([nn.Identity()]))
                input_block_chans.append(ch)
                ds *= 2

        # ------------------------------------------------------------------ #
        #  Middle block                                                       #
        # ------------------------------------------------------------------ #
        middle_layers: List[nn.Module] = [
            PointResBlock(ch, dropout, use_checkpoint=use_checkpoint),
        ]

        if use_attention and middle_block_attn:
            if num_head_channels == -1:
                _num_heads = num_heads
            else:
                _num_heads = ch // num_head_channels
            middle_layers.append(
                PointAttentionBlock(
                    ch,
                    num_heads=_num_heads,
                    num_head_channels=(
                        ch // _num_heads
                        if num_head_channels == -1
                        else num_head_channels
                    ),
                    use_checkpoint=use_checkpoint,
                    num_frames=num_frames,
                    use_cross_view_self_attn=use_cross_view_self_attn,
                )
            )

        middle_layers.append(
            PointResBlock(ch, dropout, use_checkpoint=use_checkpoint),
        )
        self.middle_block = nn.ModuleList(middle_layers)

        # ------------------------------------------------------------------ #
        #  Decoder (output blocks)                                            #
        # ------------------------------------------------------------------ #
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    PointResBlock(
                        ch + ich,
                        dropout,
                        out_channels=model_channels * mult,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = model_channels * mult

                if use_attention and ds in attention_resolutions:
                    if num_head_channels == -1:
                        _num_heads = num_heads_upsample
                    else:
                        _num_heads = ch // num_head_channels

                    if not no_self_attn:
                        layers.append(
                            PointAttentionBlock(
                                ch,
                                num_heads=_num_heads,
                                num_head_channels=(
                                    ch // _num_heads
                                    if num_head_channels == -1
                                    else num_head_channels
                                ),
                                use_checkpoint=use_checkpoint,
                                num_frames=num_frames,
                                use_cross_view_self_attn=use_cross_view_self_attn,
                            )
                        )

                # Spatial upsample position → identity placeholder
                if level and i == num_res_blocks:
                    layers.append(nn.Identity())
                    ds //= 2

                self.output_blocks.append(nn.ModuleList(layers))

        # Handle downsample_factor (output at lower resolution)
        if self.downsample_factor is not None:
            if self.downsample_factor == 2:
                del self.output_blocks[-3:]
            elif self.downsample_factor == 4:
                del self.output_blocks[-5:]
            else:
                raise NotImplementedError(
                    f"downsample_factor={self.downsample_factor} not supported"
                )

        # ------------------------------------------------------------------ #
        #  Output head                                                        #
        # ------------------------------------------------------------------ #
        if self.downsample_factor is not None:
            final_ch = (
                self.model_channels
                * self.channel_mult[self.downsample_factor // 2]
            )
        else:
            final_ch = ch   # == model_channels * channel_mult[0]

        self.out = nn.Sequential(
            PointNorm(final_ch),
            nn.SiLU(),
            zero_module(nn.Linear(final_ch, out_channels)),
        )

    # ---------------------------------------------------------------------- #
    #  Forward                                                                #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        x: torch.Tensor,
        num_views: Optional[int] = None,
        timesteps=None,
        context=None,
        y=None,
        **kwargs,
    ) -> torch.Tensor:
        """Process an input batch.

        Parameters
        ----------
        x : Tensor
            - **Grid mode** ``(B, C, H, W)`` — standard image-like layout.
            - **Flat-point mode** ``(B, C, N)`` — *N* arbitrary points.
              Useful when some points have been dropped.
        num_views, timesteps, context, y :
            Accepted for API parity with ``UNetModel``; currently unused.

        Returns
        -------
        Tensor
            Same spatial layout as the input:
            ``(B, D, H, W)`` in grid mode, ``(B, D, N)`` in flat-point mode.
        """
        is_grid = x.dim() == 4

        if is_grid:
            B, C, H, W = x.shape
            # (B, C, H, W) → (B, N, C)  with N = H*W
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:
            # (B, C, N) → (B, N, C)
            B = x.shape[0]
            x = x.permute(0, 2, 1)

        h = x.type(self.dtype)

        # ---- Encoder ----
        hs: List[torch.Tensor] = []
        for block in self.input_blocks:
            for layer in block:
                h = layer(h)
            hs.append(h)

        # ---- Middle ----
        for layer in self.middle_block:
            h = layer(h)

        # ---- Decoder ----
        for block in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=-1)   # concat along feature dim
            for layer in block:
                h = layer(h)

        h = h.type(x.dtype)

        # ---- Output head ----
        h = self.out(h)   # (B, N, out_channels)

        # ---- Reshape back ----
        if is_grid:
            h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, D, H, W)
        else:
            h = h.permute(0, 2, 1)  # (B, D, N)

        return h
