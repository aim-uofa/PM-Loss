import numpy as np
import torch
from plyfile import PlyData, PlyElement

# Refer to https://github.com/clementinboittiaux/umeyama-python
def umeyama(X, Y):
    """
    PyTorch implementation of Umeyama algorithm that preserves gradients.
    
    Parameters
    ----------
    X : torch.Tensor
        (m, n) shaped tensor. m is the dimension of the points,
        n is the number of points in the point set.
    Y : torch.Tensor
        (m, n) shaped tensor. Indexes should be consistent with `X`.
        
    Returns
    -------
    c : torch.Tensor
        Scale factor.
    R : torch.Tensor
        (3, 3) shaped rotation matrix.
    t : torch.Tensor
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(dim=1, keepdim=True)
    mu_y = Y.mean(dim=1, keepdim=True)

    n = X.shape[1]
    sigma_x = ((X - mu_x) @ (X - mu_x).transpose(-1, -2)) / n
    var_x = torch.trace(sigma_x)

    cov_xy = ((Y - mu_y) @ (X - mu_x).transpose(-1, -2)) / n
    U, D, VH = torch.linalg.svd(cov_xy)
    
    # Handle the reflection case
    S = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    if torch.det(U) * torch.det(VH) < 0:
        S[-1, -1] = -1
    
    c = torch.trace(torch.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    
    return c, R, t

def align_point_clouds_umeyama(source, target):
    """
    Align source point cloud to target using gradient-preserving Umeyama algorithm.
    Supports batched input.

    :param source: Source point cloud, torch.Tensor of shape [n, 3] or [b, n, 3]
    :param target: Target point cloud, torch.Tensor of shape [n, 3] or [b, n, 3]
    :return: Aligned source point cloud and transformation parameters
    """
    # Ensure input is torch.Tensor
    device = source.device
    
    if not isinstance(source, torch.Tensor):
        source = torch.tensor(source, device=device, dtype=torch.float32)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, device=device, dtype=torch.float32)
    

    is_batched = source.dim() == 3
    
    if not is_batched:
        return _align_single_point_cloud(source, target)
    else:
        batch_size = source.shape[0]
        aligned_sources = []
        transform_params = []
        
        for b in range(batch_size):
            aligned_source, params = _align_single_point_cloud(source[b], target[b])
            aligned_sources.append(aligned_source)
            transform_params.append(params)
        
        aligned_batch = torch.stack(aligned_sources, dim=0)
        
        batch_params = {
            'scale': torch.stack([p['scale'] for p in transform_params]),
            'rotation': torch.stack([p['rotation'] for p in transform_params]),
            'translation': torch.stack([p['translation'] for p in transform_params])
        }
        
        return aligned_batch, batch_params

def _align_single_point_cloud(source, target):
    """
    Perform Umeyama algorithm for a single point cloud pair.

    :param source: Source point cloud, torch.Tensor of shape [n, 3]
    :param target: Target point cloud, torch.Tensor of shape [n, 3]
    :return: Aligned source point cloud and transformation parameters
    """
    # Transpose to [3, n] to match umeyama function input
    source_t = source.transpose(0, 1)
    target_t = target.transpose(0, 1)

    # Call gradient-preserving umeyama function
    scale, rotation, translation = umeyama(source_t, target_t)
    
    aligned_source_t = scale * rotation @ source_t + translation
    
    aligned_source = aligned_source_t.transpose(0, 1)
    
    return aligned_source, {
        'scale': scale,
        'rotation': rotation,
        'translation': translation
    }

def get_point_from_pcd(path):
    """
    Load point cloud data from a PLY file.

    :param path: PLY file path
    :return: torch.Tensor of shape [n, 3], all points in the point cloud
    """
    with open(path, 'rb') as file:
        plydata = PlyData.read(file)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
    return xyz
