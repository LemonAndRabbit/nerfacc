"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import os
import argparse
import math
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPRadianceField

from examples.utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    render_image_with_occgrid_test,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator
from tqdm import tqdm

from utils import convert_sdf_samples_to_ply

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@torch.no_grad()
def export_mesh(radiance_field: NGPRadianceField, grid_size=[512, 512, 512], device="cuda:0", render_step_size=1e-2, save_path=None, level=0.05, aabb=None):
    samples = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, grid_size[0]),
        torch.linspace(0, 1, grid_size[1]),
        torch.linspace(0, 1, grid_size[2]),
    ), -1)
    if aabb is None:
        aabb = radiance_field.aabb
    else:
        aabb = torch.Tensor(aabb)
    dense_xyz = (1-samples)*aabb[:3].cpu() + samples*aabb[3:].cpu()

    alpha = torch.zeros_like(dense_xyz[...,0], device='cpu')
    for i in tqdm(range(grid_size[0]), desc="Extracting Alpha Field"):
        for j in range(grid_size[1]//128):
            for k in range(grid_size[2]//128):
                query_position = dense_xyz[i, j*128:(j+1)*128, k*128:(k+1)*128]
                density = radiance_field.query_density(query_position.to(device).view(-1,3))
                alphas = (1 - torch.exp(-density*render_step_size).view(query_position.shape[:-1]))
                alpha[i, j*128:(j+1)*128, k*128:(k+1)*128] = alphas.cpu()
    
    print("Mean Alpha: %f" % torch.mean(alpha).item())
    print("Max Alpha: %f" % torch.max(alpha).item())

    if save_path is not None:
        aabb = radiance_field.aabb.view((2,3)).cpu()
        convert_sdf_samples_to_ply(alpha, save_path, bbox=aabb, level=level)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path",
        type=str,
        # default=str(pathlib.Path.cwd() / "data/360_v2"),
        default=str(pathlib.Path.cwd() / "temp_ckpt"),
        help="the root dir of the ckpts",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
        help="which scene to use",
    )
    parser.add_argument(
        "--n_levels",
        type=int,
        default=16,
        help="number of hash levels",
    )
    parser.add_argument(
        "--hashmap_size",
        type=int,
        default=19,
        help="hashmap size for each level",
    )
    parser.add_argument(
        "--extract_aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default=None,
        help="delimited list input",
    )
    parser.add_argument(
        "--grid_size", 
        type=int, 
        default=512
    )
    parser.add_argument(
        "--mesh_level",
        type=float,
        default=0.2
    )

    args = parser.parse_args()

    device = "cuda:0"
    set_random_seed(42)

    if args.scene in MIPNERF360_UNBOUNDED_SCENES:
        # from datasets.nerf_360_v2 import SubjectLoader

        # # training parameters
        # max_steps = 20000
        # init_batch_size = 1024
        # target_sample_batch_size = 1 << 18
        # weight_decay = 0.0
        # scene parameters
        aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
        # near_plane = 0.2
        # far_plane = 1.0e10
        # # dataset parameters
        # train_dataset_kwargs = {"color_bkgd_aug": args.color_bkgd_aug, "factor": 4}
        # test_dataset_kwargs = {"factor": 4}
        # model parameters
        grid_resolution = 128
        grid_nlvl = 4
        # render parameters
        render_step_size = 1e-3
        # alpha_thre = 1e-2
        # cone_angle = 0.004

    else:
        # from datasets.nerf_synthetic import SubjectLoader

        # training parameters
        # max_steps = 20000
        # init_batch_size = 1024
        # target_sample_batch_size = 1 << 18
        # weight_decay = (
        #     1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
        # )
        # scene parameters
        aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
        # near_plane = 0.0
        # far_plane = 1.0e10
        # # dataset parameters
        # train_dataset_kwargs = {"color_bkgd_aug": args.color_bkgd_aug}
        # test_dataset_kwargs = {}
        # model parameters
        grid_resolution = 128
        grid_nlvl = 1
        # render parameters
        render_step_size = 5e-3
        # alpha_thre = 0.0
        # cone_angle = 0.0

    estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
    ).to(device)

    radiance_field = NGPRadianceField(
        aabb=estimator.aabbs[-1],
        n_levels=args.n_levels,
        log2_hashmap_size=args.hashmap_size
    ).to(device)

    # evaluating and only evaluating
    estimator_state_dict = torch.load(args.load_path + "/estimator_state_dict.pt")
    field_state_dict = torch.load(args.load_path + "/field_state_dict.pt")
    estimator.load_state_dict(estimator_state_dict) 
    radiance_field.load_state_dict(field_state_dict)

    print(radiance_field)

    radiance_field.eval()

    handle = logging.FileHandler(args.load_path + "/extract_mesh.log")
    handle.setLevel(logging.INFO)
    logger.addHandler(handle)

    logger.info("Extracting Mesh...")

    tic = time.time()

    # export mesh and only export mesh        
    export_mesh(
        radiance_field, 
        save_path=args.load_path+f"/export_stage1.ply", 
        level=args.mesh_level, 
        grid_size=[args.grid_size]*3, 
        aabb=args.extract_aabb, 
        render_step_size=render_step_size
    )

    elapsed_time = time.time() - tic
    logger.info(f"elapsed_time={elapsed_time:.2f}s")