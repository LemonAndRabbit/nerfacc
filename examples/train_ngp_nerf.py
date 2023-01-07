"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import os.path
import time

import imageio
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nerfacc import ContractionType, OccupancyGrid
from radiance_fields.ngp import NGPradianceField
from utils import render_image, set_random_seed, convert_sdf_samples_to_ply


@torch.no_grad()
def export_mesh(
    radiance_field: NGPradianceField,
    grid_size=[512, 512, 512],
    device="cuda:0",
    render_step_size=1e-2,
    save_path=None,
    level=0.05,
):
    samples = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, grid_size[0]),
            torch.linspace(0, 1, grid_size[1]),
            torch.linspace(0, 1, grid_size[2]),
        ),
        -1,
    )

    dense_xyz = (1 - samples) * radiance_field.aabb[
        :3
    ].cpu() + samples * radiance_field.aabb[3:].cpu()

    alpha = torch.zeros_like(dense_xyz[..., 0], device="cpu")
    for i in tqdm(range(grid_size[0]), desc="Extracting Alpha Field"):
        for j in range(grid_size[1] // 128):
            for k in range(grid_size[2] // 128):
                query_position = dense_xyz[
                    i, j * 128 : (j + 1) * 128, k * 128 : (k + 1) * 128
                ]
                density = radiance_field.query_density(
                    query_position.to(device).view(-1, 3)
                )
                alphas = 1 - torch.exp(-density * render_step_size).view(
                    query_position.shape[:-1]
                )
                alpha[
                    i, j * 128 : (j + 1) * 128, k * 128 : (k + 1) * 128
                ] = alphas.cpu()

    print("Mean Alpha: %f" % torch.mean(alpha).item())
    print("Max Alpha: %f" % torch.max(alpha).item())

    if save_path is not None:
        aabb = radiance_field.aabb.view((2, 3)).cpu()
        convert_sdf_samples_to_ply(alpha, save_path, bbox=aabb, level=level)


if __name__ == "__main__":

    device = "cuda:0"
    set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_split",
        type=str,
        default="trainval",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=[
            # nerf synthetic
            "chair",
            "drums",
            "ficus",
            "hotdog",
            "lego",
            "materials",
            "mic",
            "ship",
            # mipnerf360 unbounded
            "garden",
            "bicycle",
            "bonsai",
            "counter",
            "kitchen",
            "room",
            "stump",
        ],
        help="which scene to use",
    )
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
        help="delimited list input",
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="whether to use unbounded rendering",
    )
    parser.add_argument(
        "--auto_aabb",
        action="store_true",
        help="whether to automatically compute the aabb",
    )
    parser.add_argument("--grid_search", type=str, default=None)
    parser.add_argument("--cone_angle", type=float, default=0.0)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--get_initial_nerf", action="store_true")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--export_mesh", action="store_true")
    parser.add_argument("--export_image", action="store_true")
    parser.add_argument("--grid_size", type=int, default=512)
    parser.add_argument("--mesh_level", type=float, default=0.5)
    parser.add_argument(
        "--distortion_loss",
        action="store_true",
        help="punish floaters and background through distortion loss",
    )
    parser.add_argument("--color_bkgd_aug", type=str, default="white")
    parser.add_argument("--base_layer", type=int, default=1)
    parser.add_argument("--base_dim", type=int, default=64)
    parser.add_argument("--head_layer", type=int, default=2)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--geo_feat_dim", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=20000)
    args = parser.parse_args()

    render_n_samples = 1024

    # set up the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    if args.unbounded:
        from datasets.nerf_360_v2 import SubjectLoader

        data_root_fp = "/home/ruilongli/data/360_v2/"
        target_sample_batch_size = 1 << 20
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        grid_resolution = 256
    else:
        from datasets.nerf_synthetic import SubjectLoader

        data_root_fp = "/home/huaizhi_qu/workspace/nerf_synthetic"
        target_sample_batch_size = 1 << 18
        grid_resolution = 128
        train_dataset_kwargs = {"color_bkgd_aug": args.color_bkgd_aug}

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split=args.train_split,
        num_rays=target_sample_batch_size // render_n_samples,
        **train_dataset_kwargs,
    )

    train_dataset.images = train_dataset.images.to(device)
    train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    train_dataset.K = train_dataset.K.to(device)

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="test",
        num_rays=None,
        **test_dataset_kwargs,
    )
    test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)

    if args.auto_aabb:
        camera_locs = torch.cat([train_dataset.camtoworlds, test_dataset.camtoworlds])[
            :, :3, -1
        ]
        args.aabb = torch.cat(
            [camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]
        ).tolist()
        print("Using auto aabb", args.aabb)

    # set up the scene bounding box.
    if args.unbounded:
        print("Using unbounded rendering")
        contraction_type = ContractionType.UN_BOUNDED_SPHERE
        # contraction_type = ContractionType.UN_BOUNDED_TANH
        scene_aabb = None
        near_plane = 0.2
        far_plane = 1e4
        render_step_size = 1e-2
        alpha_thre = 1e-2
    else:
        contraction_type = ContractionType.AABB
        scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max() * math.sqrt(3) / render_n_samples
        ).item()
        alpha_thre = 0.0

    # evaluating and only evaluating
    if args.load_path is not None:
        radiance_field = torch.load(args.load_path + "/model.pt")
        occupancy_grid = torch.load(args.load_path + "/occgrid.pt")

        # print(radiance_field)

        radiance_field.eval()

        # export mesh and only export mesh
        if args.export_mesh:
            export_mesh(
                radiance_field,
                save_path=args.load_path + "/export.ply",
                level=args.mesh_level,
                grid_size=[args.grid_size] * 3,
            )
            exit()

        psnrs = []
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # rendering
                rgb, acc, depth, _, _ = render_image(
                    radiance_field,
                    occupancy_grid,
                    rays,
                    scene_aabb,
                    # rendering options
                    near_plane=near_plane,
                    far_plane=far_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=args.cone_angle,
                    alpha_thre=alpha_thre,
                    # test options
                    test_chunk_size=args.test_chunk_size,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())

                if args.export_image:
                    norm_depth = depth / (acc + 0.1)
                    norm_depth = norm_depth / norm_depth.max()
                    imageio.imwrite(
                        "/".join([args.save_path, "img%d_depth.jpg" % (i,)]),
                        (norm_depth * 255).detach().cpu().numpy().astype(np.uint8),
                    )
                    imageio.imwrite(
                        "/".join([args.save_path, "img%d_acc.jpg" % (i,)]),
                        (acc * 255).detach().cpu().numpy().astype(np.uint8),
                    )
                    imageio.imwrite(
                        "/".join([args.save_path, "img%d_orig.jpg" % (i,)]),
                        (pixels * 255).cpu().numpy().astype(np.uint8),
                    )
                    imageio.imwrite(
                        "/".join([args.save_path, "img%d_pred.jpg" % (i,)]),
                        (rgb * 255).detach().cpu().numpy().astype(np.uint8),
                    )
                    imageio.imwrite(
                        "/".join([args.save_path, "img%d_error.jpg" % (i,)]),
                        (abs(rgb - pixels) * 255)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8),
                    )

        psnr_avg = sum(psnrs) / len(psnrs)
        print(f"evaluation: {psnr_avg}")

        exit()

    # set up the radiance field we want to train.
    max_steps = args.max_steps
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    radiance_field = NGPradianceField(
        aabb=args.aabb,
        unbounded=args.unbounded,
        base_layer=args.base_layer,
        base_dim=args.base_dim,
        head_layer=args.head_layer,
        head_dim=args.head_dim,
        geo_feat_dim=args.geo_feat_dim,
    ).to(device)

    if args.get_initial_nerf:
        torch.save(radiance_field, "initial_nerf.pt")
        exit()

    optimizer = torch.optim.Adam(radiance_field.parameters(), lr=1e-2, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
        gamma=0.33,
    )

    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)

    # training
    step = 0
    torch.cuda.synchronize()
    train_start_time = time.time()
    for epoch in tqdm(range(10000)):
        for i in range(len(train_dataset)):
            radiance_field.train()
            data = train_dataset[i]

            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            def occ_eval_fn(x):
                if args.cone_angle > 0.0:
                    # randomly sample a camera for computing step size.
                    camera_ids = torch.randint(
                        0, len(train_dataset), (x.shape[0],), device=device
                    )
                    origins = train_dataset.camtoworlds[camera_ids, :3, -1]
                    t = (origins - x).norm(dim=-1, keepdim=True)
                    # compute actual step size used in marching, based on the distance to the camera.
                    step_size = torch.clamp(t * args.cone_angle, min=render_step_size)
                    # filter out the points that are not in the near far plane.
                    if (near_plane is not None) and (near_plane is not None):
                        step_size = torch.where(
                            (t > near_plane) & (t < far_plane),
                            step_size,
                            torch.zeros_like(step_size),
                        )
                else:
                    step_size = render_step_size
                # compute occupancy
                density = radiance_field.query_density(x)
                return density * step_size

            # update occupancy grid
            occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)

            # render
            rgb, acc, depth, n_rendering_samples, extra_loss = render_image(
                radiance_field,
                occupancy_grid,
                rays,
                scene_aabb,
                # rendering options
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=args.cone_angle,
                alpha_thre=alpha_thre,
                distortion_loss=args.distortion_loss,
            )
            if n_rendering_samples == 0:
                print("skipped!")
                continue

            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)
            alive_ray_mask = acc.squeeze(-1) > 0

            # compute loss
            loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

            if args.distortion_loss:
                loss += extra_loss * 0.0000001

            optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            optimizer.step()
            scheduler.step()

            # print for every 10000 steps
            if step % 10000 == 0 and step > 0:
                torch.cuda.synchronize()
                elapsed_time = time.time() - train_start_time
                loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                print(
                    f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                    f"loss={loss:.5f} | "
                    f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} |"
                )

            if step >= max_steps:
                torch.cuda.synchronize()
                train_end_time = time.time()
                print("training time is {}s".format(train_end_time - train_start_time))

            if step >= max_steps:
                # evaluation
                torch.cuda.synchronize()
                eval_start_time = time.time()
                radiance_field.eval()

                psnrs = []
                with torch.no_grad():
                    for i in tqdm(range(len(test_dataset))):
                        data = test_dataset[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]

                        # rendering
                        rgb, acc, depth, _, _ = render_image(
                            radiance_field,
                            occupancy_grid,
                            rays,
                            scene_aabb,
                            # rendering options
                            near_plane=near_plane,
                            far_plane=far_plane,
                            render_step_size=render_step_size,
                            render_bkgd=render_bkgd,
                            cone_angle=args.cone_angle,
                            alpha_thre=alpha_thre,
                            # test options
                            test_chunk_size=args.test_chunk_size,
                        )
                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                        # imageio.imwrite(
                        #     "acc_binary_test.png",
                        #     ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
                        # )
                        # imageio.imwrite(
                        #     "rgb_test.png",
                        #     (rgb.cpu().numpy() * 255).astype(np.uint8),
                        # )
                        # break
                psnr_avg = sum(psnrs) / len(psnrs)
                torch.cuda.synchronize()
                eval_end_time = time.time()
                print("evaluation time is {}s".format(eval_end_time - eval_start_time))
                print(f"evaluation: psnr_avg={psnr_avg}")
                train_dataset.training = True

            # early stops when step reaches max_steps
            if step >= max_steps:

                print("training stops")

                if args.save_path is not None:
                    print("Checkpoint saving to %s" % args.save_path)

                    torch.save(radiance_field, args.save_path + "/model.pt")
                    torch.save(occupancy_grid, args.save_path + "/occgrid.pt")

                running_time = radiance_field.running_time()
                print(
                    "time_base: {}s | avg_time_base: {}s | time_head: {}s | avg_time_head: {}s\n".format(
                        running_time[0],
                        running_time[1],
                        running_time[2],
                        running_time[3],
                    )
                )
                search_result = "./grid_search_{}_{}_{}.csv".format(
                    args.grid_search,
                    args.head_dim if args.grid_search == "base" else args.base_dim,
                    args.head_layer if args.grid_search == "base" else args.base_layer,
                )
                if args.grid_search in ["base", "head"]:
                    pd.DataFrame.from_dict(
                        {
                            f"{args.grid_search}_dim": [args.head_dim],
                            f"{args.grid_search}_layer": [args.head_layer],
                            "time_base": [running_time[0]],
                            "avg_time_base": [running_time[1]],
                            "time_head": [running_time[2]],
                            "avg_time_head": [running_time[3]],
                            "train_time": [train_end_time - train_start_time],
                            "eval_time": [eval_end_time - eval_start_time],
                            "psnr_avg": [psnr_avg],
                        }
                    ).to_csv(
                        search_result,
                        mode="a",
                        header=False if os.path.exists(search_result) else True,
                        index=False,
                    )
                exit()

            # each step involves 1 optimization
            step += 1
