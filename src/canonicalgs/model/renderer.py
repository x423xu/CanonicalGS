from __future__ import annotations

from math import isqrt

import torch

from .gaussian_readout import GaussianSet


def render_gaussian_views(
    gaussians: GaussianSet,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
    background_color: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("diff_gaussian_rasterization is required for rendering") from exc

    extrinsics = extrinsics.float()
    intrinsics = intrinsics.float()
    means = gaussians.means.float()
    covariances = gaussians.covariances.float()
    opacities = gaussians.opacities.float()
    colors = gaussians.appearance.float()

    num_views = extrinsics.shape[0]
    height, width = image_shape
    device = extrinsics.device
    dtype = means.dtype

    if background_color is None:
        background_color = torch.zeros((num_views, 3), dtype=dtype, device=device)
    elif background_color.ndim == 1:
        background_color = background_color.unsqueeze(0).expand(num_views, -1)

    if means.numel() == 0:
        images = background_color[:, :, None, None].expand(-1, -1, height, width).contiguous()
        coverage = torch.zeros(num_views, dtype=dtype, device=device)
        return images, coverage

    near, far = _estimate_near_far(means, extrinsics)
    fov_x, fov_y = _get_fov(intrinsics).unbind(dim=-1)
    projection_matrix = _get_projection_matrix(near, far, fov_x, fov_y).transpose(-1, -2)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()
    view_matrix = torch.linalg.inv(extrinsics)
    full_projection = view_matrix @ projection_matrix
    row, col = torch.triu_indices(3, 3, device=device)

    if colors.shape[-1] != 3:
        colors = colors[..., :3]
    colors = colors.clamp(0.0, 1.0)
    degree = isqrt(1) - 1

    images = []
    coverages = []
    for index in range(num_views):
        means2d = torch.zeros_like(means, requires_grad=True)
        settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tan_fov_x[index].item(),
            tanfovy=tan_fov_y[index].item(),
            bg=background_color[index],
            scale_modifier=1.0,
            viewmatrix=view_matrix[index],
            projmatrix=full_projection[index],
            sh_degree=degree,
            campos=extrinsics[index, :3, 3],
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)
        image, _ = rasterizer(
            means3D=means,
            means2D=means2d,
            shs=None,
            colors_precomp=colors,
            opacities=opacities[:, None],
            cov3D_precomp=covariances[:, row, col],
        )
        images.append(image)
        coverages.append((image.abs().sum(dim=0) > 1e-6).to(dtype).mean())

    return torch.stack(images, dim=0), torch.stack(coverages, dim=0)


def _estimate_near_far(means: torch.Tensor, extrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones_like(means[:, :1])
    means_h = torch.cat([means, ones], dim=-1)
    world_to_camera = torch.linalg.inv(extrinsics)
    camera_points = torch.einsum("vij,gj->vgi", world_to_camera, means_h)[..., :3]
    z = camera_points[..., 2]
    valid = z > 1e-4
    near = torch.where(valid, z, torch.full_like(z, float("inf"))).amin(dim=-1)
    far = torch.where(valid, z, torch.zeros_like(z)).amax(dim=-1)
    near = torch.where(torch.isfinite(near), near * 0.8, torch.full_like(near, 0.1)).clamp_min(1e-3)
    far = torch.maximum(far * 1.2, near + 1.0)
    return near, far


def _get_fov(intrinsics: torch.Tensor) -> torch.Tensor:
    fx = intrinsics[:, 0, 0].clamp_min(1e-6)
    fy = intrinsics[:, 1, 1].clamp_min(1e-6)
    fov_x = 2.0 * torch.atan(0.5 / fx)
    fov_y = 2.0 * torch.atan(0.5 / fy)
    return torch.stack((fov_x, fov_y), dim=-1)


def _get_projection_matrix(
    near: torch.Tensor,
    far: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
) -> torch.Tensor:
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()
    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    batch = near.shape[0]
    result = torch.zeros((batch, 4, 4), dtype=near.dtype, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result
