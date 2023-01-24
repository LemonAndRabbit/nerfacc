"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import Optional

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map

from nerfacc import OccupancyGrid, ray_marching, rendering
from nerfacc.vol_rendering import accumulate_along_rays

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccupancyGrid,
    rays: Rays,
    rays2: Rays,
    supersampling,
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    distortion_loss: bool = False,
    distortion_loss_llff: bool = False,
    sparsity_loss: bool = False,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
        if rays2 is not None:
            rays2 = namedtuple_map(
                lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays2
            )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field.query_density(positions, t)
        return radiance_field.query_density(positions)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices, requires_position=False):
        ray_indices = ray_indices.long()
        if supersampling:
            t_origins = chunk_rays2.origins[ray_indices]
            t_dirs = chunk_rays2.viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            t_coarse_dirs = chunk_rays.viewdirs[ray_indices]

            return radiance_field(positions, t_dirs, t_coarse_dirs, supersampling)
        else:
            t_origins = chunk_rays.origins[ray_indices]
            t_dirs = chunk_rays.viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0

            return radiance_field(positions, t_dirs), positions
        # if timestamps is not None:
        #     # dnerf
        #     t = (
        #         timestamps[ray_indices]
        #         if radiance_field.training
        #         else timestamps.expand_as(positions[:, :1])
        #     )
        #     return radiance_field(positions, t, t_dirs)
        # return radiance_field(positions, t_dirs)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    extra_loss = {}
    if distortion_loss or distortion_loss_llff:
        extra_loss['dis_loss'] = 0.
    if sparsity_loss:
        extra_loss['s_loss'] = 0.

    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        if supersampling:
            chunk_rays2 = namedtuple_map(lambda r: r[i : i + chunk], rays2)
        else:
            chunk_rays2 = None
        ray_indices, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=occupancy_grid,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
            requires_weight=distortion_loss or distortion_loss_llff,
            requires_sigma=sparsity_loss,
            requires_position=distortion_loss_llff,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)

        ray_indices = ray_indices.long()
        if distortion_loss:
            weights = extras['weight']
            with torch.no_grad():
                dis_mids = abs((t_starts+t_ends)/2.0 - (depth/(opacity+0.001))[ray_indices])

            extra_loss['dis_loss'] += (weights * dis_mids).sum()
            # print(extra_loss)
        elif distortion_loss_llff:
            weights = extras['weight']
            with torch.no_grad():
                # dis_mids = abs((t_starts+t_ends)/2.0 - (depth/(opacity+0.001))[ray_indices.long()])
                t_origins = chunk_rays.origins[ray_indices]
                t_dirs = chunk_rays.viewdirs[ray_indices]
                # print(t_origins.shape)
                # print(t_dirs.shape)
                # print(ray_indices.shape)
                # print(weight.shape)
                # exit()
                real_positions = extras['position'][...,2:3]
                real_positions = 1.0 / (real_positions-1)
                real_depths = accumulate_along_rays(
                    weights,
                    ray_indices,
                    values=real_positions,
                    n_rays=chunk_rays.origins.shape[0],
                )
                dis_mids = abs(real_positions - real_depths[ray_indices.long()])
                del real_positions
                del real_depths
            extra_loss['dis_loss'] += (weights * dis_mids).sum()

        if sparsity_loss:
            sigmas = extras['sigma']
            extra_loss['s_loss'] += torch.log(1.0 + 2*sigmas**2).sum()
    
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        extra_loss
    )


import plyfile
import skimage.measure
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)
