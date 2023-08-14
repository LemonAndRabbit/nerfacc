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
from tqdm import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPRadianceField

from examples.utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    NSVF_SYNTHETIC_SCENES,
    TANKSANDTEMPLES_SCENES,
    render_image_with_occgrid,
    render_image_with_occgrid_test,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    # default=str(pathlib.Path.cwd() / "data/360_v2"),
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    # default=str(pathlib.Path.cwd() / "data/360_v2"),
    default=str(pathlib.Path.cwd() / "temp_ckpt"),
    help="the root dir of the ckpts",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=NERF_SYNTHETIC_SCENES + NSVF_SYNTHETIC_SCENES + TANKSANDTEMPLES_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--max_steps", 
    type=int, 
    default=20000,
)
parser.add_argument(
    "--color_bkgd_aug", 
    type=str, 
    default="white",
    choices=["random", "black", "white"],
)
parser.add_argument(
    "--centering_loss", 
    action="store_true", 
    help="punish floaters and background through centering loss"
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=None,
    help="weight_decay_factor",
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
    "--use_dataset_bbox",
    action="store_true",
    help="use dataset bbox",
)
parser.add_argument(
    "--adjust_step_size",
    action="store_true",
    help="adjust step size",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-2,
    help="learning rate",
)

args = parser.parse_args()

max_steps = args.max_steps

device = "cuda:0"
set_random_seed(42)

if args.scene in MIPNERF360_UNBOUNDED_SCENES:
    from datasets.nerf_360_v2 import SubjectLoader

    # training parameters
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 0.0
    # scene parameters
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    near_plane = 0.2
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": args.color_bkgd_aug, "factor": 4}
    test_dataset_kwargs = {"factor": 4}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 4
    # render parameters
    render_step_size = 1e-3
    alpha_thre = 1e-2
    cone_angle = 0.004

elif args.scene in NERF_SYNTHETIC_SCENES:
    from datasets.nerf_synthetic import SubjectLoader

    # training parameters
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = (
        1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    )
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": args.color_bkgd_aug}
    test_dataset_kwargs = {}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0
elif args.scene in NSVF_SYNTHETIC_SCENES:
    from datasets.nsvf_synthetic import SubjectLoader

    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 1e-5
    # scene parameters
    aabb = torch.tensor([-1, -1, -1, 1, 1, 1], device=device)
    near_plane = 0.0
    far_plane = 1.0e10
    # dataset parameters
    if args.scene in ["Lifestyle", "Spaceship", "Steamtrain"]:
        args.color_bkgd_aug = "white"
    if args.scene == "Steamtrain":
        args.lr = 1e-3
    train_dataset_kwargs = {"color_bkgd_aug": args.color_bkgd_aug}
    test_dataset_kwargs = {}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0
elif args.scene in TANKSANDTEMPLES_SCENES:
    from datasets.tanksandtemples import SubjectLoader
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 1e-5
    # scene parameters
    aabb = torch.tensor([-1, -1, -1, 1, 1, 1] , device=device) * 2.3
    near_plane = 0.0
    far_plane = 1.0e10
    # dataset parameters
    if args.scene != "Ignatius":
        args.color_bkgd_aug = "white"
    train_dataset_kwargs = {"color_bkgd_aug": args.color_bkgd_aug}
    test_dataset_kwargs = {}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0


if args.weight_decay is not None:
    weight_decay = args.weight_decay

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs,
)

if hasattr(train_dataset, "bbox") and args.use_dataset_bbox:
    print(train_dataset.bbox)
    aabb = torch.from_numpy(train_dataset.bbox).to(device=aabb.device)

if args.adjust_step_size:
    render_step_size = (
        (aabb[3:] - aabb[:3]).max()
        * math.sqrt(3)
        / init_batch_size
    ).item()
    print(aabb, (aabb[3:] - aabb[:3]).max() * math.sqrt(3))
    print("render_step_size:", render_step_size)

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# setup the radiance field we want to train.
grad_scaler = torch.cuda.amp.GradScaler(2**10)
radiance_field = NGPRadianceField(
    aabb=estimator.aabbs[-1],
    n_levels=args.n_levels,
    log2_hashmap_size=args.hashmap_size
).to(device)

optimizer = torch.optim.Adam(
    radiance_field.parameters(), lr=args.lr, eps=1e-15, weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

os.makedirs(args.ckpt_path, exist_ok=True)

file_handler = logging.FileHandler(args.ckpt_path + "/train.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

logger.info(f"args: {args}")

# training
tic = time.time()
for step in tqdm(range(max_steps + 1)):
    radiance_field.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    def occ_eval_fn(x):
        density = radiance_field.query_density(x)
        return density * render_step_size

    # update occupancy grid
    estimator.update_every_n_steps(
        step=step,
        occ_eval_fn=occ_eval_fn,
        occ_thre=1e-2,
    )

    # render
    rgb, acc, depth, n_rendering_samples, extra_output = render_image_with_occgrid(
        radiance_field,
        estimator,
        rays,
        # rendering options
        near_plane=near_plane,
        render_step_size=render_step_size,
        render_bkgd=render_bkgd,
        cone_angle=cone_angle,
        alpha_thre=alpha_thre,
        centering_loss=args.centering_loss,
    )
    if n_rendering_samples == 0:
        continue

    if target_sample_batch_size > 0:
        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)

    # compute loss
    loss = F.smooth_l1_loss(rgb, pixels)
    if args.centering_loss:
        loss += extra_output["centering_loss"] * 0.0000001

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    if step % 10000 == 0 or step == max_steps:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        # print(
        #     f"elapsed_time={elapsed_time:.2f}s | step={step} | "
        #     f"loss={loss:.5f} | psnr={psnr:.2f} | "
        #     f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
        #     f"max_depth={depth.max():.3f} | "
        # )
        logger.info(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )

    if step > 0 and step % max_steps == 0:
        # evaluation
        radiance_field.eval()
        estimator.eval()

        psnrs = []
        lpips = []
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # rendering
                rgb, acc, depth, _ = render_image_with_occgrid_test(
                    1024,
                    # scene
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                lpips.append(lpips_fn(rgb, pixels).item())
                # if i == 0:
                #     imageio.imwrite(
                #         "rgb_test.png",
                #         (rgb.cpu().numpy() * 255).astype(np.uint8),
                #     )
                #     imageio.imwrite(
                #         "rgb_error.png",
                #         (
                #             (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                #         ).astype(np.uint8),
                #     )
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        # print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")
        logger.info(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")

        torch.save(radiance_field.state_dict(), args.ckpt_path + f"/field_state_dict.pt")
        torch.save(estimator.state_dict(), args.ckpt_path + f"/estimator_state_dict.pt")
