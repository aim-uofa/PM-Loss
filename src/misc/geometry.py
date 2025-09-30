# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# geometry utilitary functions
# --------------------------------------------------------
import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o+s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d+1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim-2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1]+1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):
    """
    Args:
        - depthmap (BxHxW array):
        - pseudo_focal: [B,H,W] ; [B,2,H,W] or [B,1,H,W]
    Returns:
        pointmap of absolute coordinates (BxHxWx3 array)
    """

    if len(depth.shape) == 4:
        B, H, W, n = depth.shape
    else:
        B, H, W = depth.shape
        n = None

    if len(pseudo_focal.shape) == 3:  # [B,H,W]
        pseudo_focalx = pseudo_focaly = pseudo_focal
    elif len(pseudo_focal.shape) == 4:  # [B,2,H,W] or [B,1,H,W]
        pseudo_focalx = pseudo_focal[:, 0]
        if pseudo_focal.shape[1] == 2:
            pseudo_focaly = pseudo_focal[:, 1]
        else:
            pseudo_focaly = pseudo_focalx
    else:
        raise NotImplementedError("Error, unknown input focal shape format.")

    assert pseudo_focalx.shape == depth.shape[:3]
    assert pseudo_focaly.shape == depth.shape[:3]
    grid_x, grid_y = xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]

    # set principal point
    if pp is None:
        grid_x = grid_x - (W-1)/2
        grid_y = grid_y - (H-1)/2
    else:
        grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
        grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

    if n is None:
        pts3d = torch.empty((B, H, W, 3), device=depth.device)
        pts3d[..., 0] = depth * grid_x / pseudo_focalx
        pts3d[..., 1] = depth * grid_y / pseudo_focaly
        pts3d[..., 2] = depth
    else:
        pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
        pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
        pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
        pts3d[..., 2, :] = depth
    return pts3d


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, image_width, image_height, pseudo_focal=None):
    """
    Modified for normalized intrinsics and OpenCV coordinate system
    
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: normalized 3x3 matrix (fx and fy divided by W/H)
        - image_width: original image width (to denormalize)
        - image_height: original image height (to denormalize)
    Returns:
        pointmap of camera coordinates (HxWx3 array) in OpenCV convention:
            +X right, +Y down, +Z forward (points into the screen)
        and valid pixel mask
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Verify no skew parameters
    assert camera_intrinsics[0, 1] == 0.0, "Skew not supported in normalized intrinsics"
    assert camera_intrinsics[1, 0] == 0.0, "Skew not supported in normalized intrinsics"

    # # Denormalize intrinsics
    # if pseudo_focal is None:
    #     # Original: fu = fx/W, fv = fy/H (from normalization)
    #     fu = camera_intrinsics[0, 0] * image_width  # Denormalize fx
    #     fv = camera_intrinsics[1, 1] * image_height  # Denormalize fy
    # else:
    #     raise NotImplementedError("Pseudo focal not supported with normalized intrinsics")

    # # Principal point was normalized to [0-1] range
    # cu = camera_intrinsics[0, 2] * image_width  # Denormalize cx
    # cv = camera_intrinsics[1, 2] * image_height  # Denormalize cy
    
    # for dtu dataset
    # fu = 361.54125 * 4
    # fv = 360.3975 * 4
    # cu = 82.900625 * 4
    # cv = 66.383875 * 4
    fu = 361.54125 * 4 * 0.7
    fv = 360.3975 * 4 * 0.5
    cu = 82.900625 * 4 * 0.7
    cv = 66.383875 * 4 * 0.5

    # Generate pixel grid with OpenCV convention (origin at top-left)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Convert to camera coordinates (OpenCV convention)
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv  # +Y down already handled by v indexing
    
    # Stack coordinates (already in OpenCV: +X right, +Y down, +Z forward)
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, image_width=640, image_height=512, **kw):
    """
    Modified for OpenCV-style camera-to-world transformation
    
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: normalized 3x3 matrix
        - camera_pose: 4x4 OpenCV-style cam2world matrix
            (+X right, +Y down, +Z forward in camera frame)
    Returns:
        pointmap in world coordinates (HxWx3 array) and valid mask
    """
    # Get camera coordinates (OpenCV convention)
    X_cam, valid_mask = depthmap_to_camera_coordinates(
        depthmap, camera_intrinsics, 
        image_width, image_height
    )

    # Verify camera pose format
    assert camera_pose.shape == (4, 4), "Camera pose must be 4x4 matrix"
    R_cam2world = camera_pose[:3, :3]
    t_cam2world = camera_pose[:3, 3]

    # Transform to world coordinates using Einstein summation
    # Equivalent to: X_world = (R @ X_cam.T).T + t
    X_world = np.einsum("ij,hwj->hwi", R_cam2world, X_cam) + t_cam2world

    return X_world, valid_mask