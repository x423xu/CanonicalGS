from ..scene_field import GPDecoderHead, SceneFieldEncoder, SceneLattice

GSCubeInput = SceneLattice
GSCubeHead = GPDecoderHead
GSCubeEncoder = SceneFieldEncoder

__all__ = ["GSCubeInput", "GSCubeHead", "GSCubeEncoder"]
