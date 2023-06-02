import torch
from tqdm import tqdm
from utils import inverse_transform, get, frames2vid
import torchvision
import torch.nn.functional as TF
from PIL import Image
from config import *
from model import UNet
import os
from pathlib import Path
from datetime import datetime

__all__ = ['reverse_diffusion']

@torch.inference_mode()
def reverse_diffusion(model, 
                      sd: SimpleDiffusion, 
                      timesteps=1000, 
                      img_shape=(3, 64, 64), 
                      num_images=5, 
                      nrow=8, 
                      device="cpu", 
                      save_path=None,
                      **kwargs):

    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()

    if kwargs.get("generate_video", False):
        outs = []

    for time_step in tqdm(iterable=reversed(range(1, timesteps)), total=timesteps-1, dynamic_ncols=False, desc="Sampling :: ", position=0):
        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise            = model(x, ts)
        beta_t                     = get(sd.beta, ts)
        one_by_sqrt_alpha_t        = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_bar_t = get(sd.sqrt_one_minus_alpha_bar, ts) 

        # sample x_{t-1} by given x_t and timestep t
        x = (one_by_sqrt_alpha_t
             * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)
             + torch.sqrt(beta_t) * z)

        if kwargs.get("generate_video", False):
            x_inv = inverse_transform(x).type(torch.uint8)
            grid  = torchvision.utils.make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            outs.append(ndarr)

    if kwargs.get("generate_video", False): # Generate and save video of the entire reverse process. 
        frames2vid(outs, save_path)
        return None
    else: # Display and save the image at the final timestep of the reverse process. 
        x = inverse_transform(x).type(torch.uint8)
        grid = torchvision.utils.make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        pil_image = TF.functional.to_pil_image(grid)
        pil_image.save(save_path, format=save_path[-3:].upper())
        return None
    

def validate():
    model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
    model.load_state_dict(torch.load("mnist_ckpt.tar", map_location='cpu')['model'])
    model.to(BaseConfig.DEVICE)

    sd = SimpleDiffusion(
        num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
        img_shape               = TrainingConfig.IMG_SHAPE,
        device                  = BaseConfig.DEVICE,
    )

    log_dir = "inference_results"
    os.makedirs(log_dir, exist_ok=True)
    generate_video = True

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"
    save_path = os.path.join(log_dir, filename)
    reverse_diffusion(
        model,
        sd,
        num_images=64,
        generate_video=generate_video,
        save_path=save_path,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=8,
    )
    print(save_path)