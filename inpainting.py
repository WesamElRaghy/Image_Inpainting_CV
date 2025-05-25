import numpy as np
import torch
from PIL import Image
from models.skip import skip
from utils.inpainting_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def inpaint_image(img_path, mask_path, num_iter=3000, lr=0.01, imsize=-1, dim_div_by=64):
    # Load and preprocess image and mask
    img_pil, img_np = get_image(img_path, imsize)
    img_mask_pil, img_mask_np = get_image(mask_path, imsize)

    # Center crop to ensure dimensions are divisible by dim_div_by
    img_mask_pil = crop_image(img_mask_pil, dim_div_by)
    img_pil = crop_image(img_pil, dim_div_by)
    img_np = pil_to_np(img_pil)
    img_mask_np = pil_to_np(img_mask_pil)

    # Convert to torch tensors
    img_var = np_to_torch(img_np).type(dtype)
    mask_var = np_to_torch(img_mask_np).type(dtype)

    # Setup network
    pad = 'reflection'
    input_depth = 32
    reg_noise_std = 0.03
    param_noise = False

    net = skip(
        input_depth, img_np.shape[0],
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[128] * 5,
        filter_size_up=3, filter_size_down=3, filter_skip_size=1,
        upsample_mode='nearest', need_sigmoid=True, need_bias=True,
        pad=pad, act_fun='LeakyReLU'
    ).type(dtype)

    # Initialize input
    net_input = get_noise(input_depth, 'noise', img_np.shape[1:]).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    # Loss and optimizer
    mse = torch.nn.MSELoss().type(dtype)
    p = get_params('net', net, net_input)
    optimizer = torch.optim.Adam(p, lr=lr)

    # Optimization loop
    for i in range(num_iter):
        optimizer.zero_grad()
        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n.data += n.data.detach().clone().normal_() * n.std() / 50
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        
        out = net(net_input)
        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Iteration {i:05d} Loss {total_loss.item():.4f}')

    # Convert output to numpy and save
    out_np = torch_to_np(out)
    out_np = np.clip(out_np, 0, 1)
    out_pil = np_to_pil(out_np)
    return out_pil

if __name__ == "__main__":
    # Example usage
    img_path = 'data/inpainting/kate.png'
    mask_path = 'data/inpainting/kate_mask.png'
    out_pil = inpaint_image(img_path, mask_path, num_iter=1000, lr=0.01)
    out_pil.save('output.png')