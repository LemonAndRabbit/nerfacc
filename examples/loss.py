import torch
import numpy as np

# def total_variation_loss(embeddings, min_resolution, max_resolution, level, log2_hashmap_size, n_levels=16):
#     # Get resolution
#     b = np.exp((np.log(max_resolution)-np.log(min_resolution))/(n_levels-1))
#     resolution = torch.tensor(np.floor(min_resolution * b**level))

#     # Cube size to apply TV loss
#     min_cube_size = min_resolution - 1
#     max_cube_size = 50 # can be tuned
#     if min_cube_size > max_cube_size:
#         assert False, "ALERT! min cuboid size greater than max!"
#     cube_size = torch.floor(torch.clip(resolution/10.0, min_cube_size, max_cube_size)).int()

#     # Sample cuboid
#     min_vertex = torch.randint(0, resolution-cube_size, (3,))
#     idx = min_vertex + torch.stack([torch.arange(cube_size+1) for _ in range(3)], dim=-1)
#     cube_indices = torch.stack(torch.meshgrid(idx[:,0], idx[:,1], idx[:,2]), dim=-1)

#     hashed_indices = hash(cube_indices, log2_hashmap_size)
#     cube_embeddings = embeddings(hashed_indices)
#     tv_x = torch.pow(cube_embeddings[1:,:,:,:]-cube_embeddings[:-1,:,:,:], 2).sum()
#     tv_y = torch.pow(cube_embeddings[:,1:,:,:]-cube_embeddings[:,:-1,:,:], 2).sum()
#     tv_z = torch.pow(cube_embeddings[:,:,1:,:]-cube_embeddings[:,:,:-1,:], 2).sum()

#     return (tv_x + tv_y + tv_z)/cube_size

def total_variation_loss(embedding_func, base_resolution, per_level_scale, level):
    # Get resolution
    resolution = torch.tensor(np.floor(base_resolution * per_level_scale**level))

    # Cube size to apply TV loss
    min_cube_size = base_resolution - 1
    max_cube_size = 100 # can be tuned
    if min_cube_size > max_cube_size:
        assert False, "ALERT! min cuboid size greater than max!"
    cube_size = torch.floor(torch.clip(resolution/5.0, min_cube_size, max_cube_size)).int()

    # Sample cuboid
    min_vertex = torch.empty((3,)).uniform_(0, resolution-cube_size)
    idx = (min_vertex + torch.stack([torch.arange(cube_size+1) for _ in range(3)], dim=-1))/resolution
    cube_indices = torch.stack(torch.meshgrid(idx[:,0], idx[:,1], idx[:,2]), dim=-1)
    rgb, density = embedding_func(cube_indices)
    r_tv_x = torch.pow(rgb[1:,:,:,:]-rgb[:-1,:,:,:], 2).sum()
    r_tv_y = torch.pow(rgb[:,1:,:,:]-rgb[:,:-1,:,:], 2).sum()
    r_tv_z = torch.pow(rgb[:,:,1:,:]-rgb[:,:,:-1,:], 2).sum()
    r_tv = (r_tv_x + r_tv_y + r_tv_z)/cube_size

    d_tv_x = torch.pow(density[1:,:,:,:]-density[:-1,:,:,:], 2).sum()
    d_tv_y = torch.pow(density[:,1:,:,:]-density[:,:-1,:,:], 2).sum()
    d_tv_z = torch.pow(density[:,:,1:,:]-density[:,:,:-1,:], 2).sum()
    d_tv = (d_tv_x + d_tv_y + d_tv_z)/cube_size

    return r_tv, d_tv
