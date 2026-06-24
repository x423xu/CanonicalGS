from typing import Optional

from .encoder import Encoder
from .encoder_canonicalgs import EncoderCanonicalGS, EncoderCanonicalGSCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_canonicalgs import EncoderVisualizerCanonicalGS

ENCODERS = {
    "canonicalgs": (EncoderCanonicalGS, EncoderVisualizerCanonicalGS),
}

EncoderCfg = EncoderCanonicalGSCfg


def get_encoder(
    cfg: EncoderCfg,
    vggt_meta: bool,
    knn_down: bool = False,
    gaussian_merge: bool = False,
) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder_cls, visualizer_cls = ENCODERS[cfg.name]
    encoder = encoder_cls(cfg, True, vggt_meta, knn_down, gaussian_merge)
    visualizer = visualizer_cls(cfg.visualizer, encoder) if visualizer_cls is not None else None
    return encoder, visualizer
