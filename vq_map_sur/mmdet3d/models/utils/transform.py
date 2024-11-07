import torch

def get_reference_points(num_token, num_query_each_token, Z=8):
    H, W = int(num_token ** 0.5), int(num_token ** 0.5)
    assert H * W == num_token

    xs = torch.linspace(0.5, W - 0.5, W).view(1, W, 1).expand(num_query_each_token, W, H) / W
    ys = torch.linspace(0.5, H - 0.5, H).view(1, 1, H).expand(num_query_each_token, W, H) / H
    zs = torch.linspace(0.5, Z - 0.5, num_query_each_token).view(
        num_query_each_token, 1, 1).expand(num_query_each_token, H, W) / Z
    
    ref_3d = torch.stack((xs, ys, zs), -1)
    ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
    return ref_3d[None] # (1, num_query_each_token, num_token, 3)