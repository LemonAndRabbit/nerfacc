import torch

model = torch.load("/home/huaizhi_qu/workspace/nerfacc/mesh_0.2_512_train_bkgd_aug_35k/chair/model.pt")
print(model.base_dim)
print(model.base_layer)