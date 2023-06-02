import torch
from config import *
from utils import *
from tqdm import tqdm
from torch.cuda import amp
from torchmetrics import MeanMetric
from model import UNet
import torch.nn as nn
from data import get_dataloader
import os
import gc
from validate import reverse_diffusion

        
def forward_diffusion(sd: SimpleDiffusion, x0: torch.Tensor, timesteps: torch.Tensor):
    """
    Inputs:
        sd: 
        x0: (b, c, h, w)
        timesteps: (b,) / timestep 取值范围为[1, T-1]
    Outputs:
        sample: x_t by given x_0 and timestep t
        eps: epsilon_0
    """
    eps     = torch.randn_like(x0)                           # Noise
    mean    = get(sd.sqrt_alpha_bar, t=timesteps) * x0       # Image scaled
    std_dev = get(sd.sqrt_one_minus_alpha_bar, t=timesteps)  # Noise scaled
    sample  = mean + std_dev * eps                           # scaled inputs * scaled noise / get x_t by given x_0 and timestep t

    return sample, eps  # return ... , gt noise --> model predicts this


def train_one_epoch(model, 
                    sd, 
                    loader, 
                    optimizer, 
                    scaler, 
                    loss_fn, 
                    epoch=800, 
                    base_config=BaseConfig(), 
                    training_config=TrainingConfig()):
    
    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")
         
        for x0s, _ in loader:
            tq.update(1)
            
            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=base_config.DEVICE)
            xts, gt_noise = forward_diffusion(sd, x0s, ts)

            with amp.autocast():
                pred_noise = model(xts, ts)  # predict epsilon by given x_t and timestep t 
                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        mean_loss = loss_record.compute().item()
    
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")
    
    return mean_loss 


def train():

    model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
    model.to(BaseConfig.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)

    dataloader = get_dataloader(
        dataset_name = BaseConfig.DATASET,
        batch_size   = TrainingConfig.BATCH_SIZE,
        device       = BaseConfig.DEVICE,
        pin_memory   = True,
        num_workers  = TrainingConfig.NUM_WORKERS,
    )

    loss_fn = nn.MSELoss()

    sd = SimpleDiffusion(
        num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
        img_shape               = TrainingConfig.IMG_SHAPE,
        device                  = BaseConfig.DEVICE,
    )

    scaler = amp.GradScaler()

    total_epochs = TrainingConfig.NUM_EPOCHS + 1
    log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())

    generate_video = True
    ext = ".mp4" if generate_video else ".png"

    for epoch in range(1, total_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        
        # Algorithm 1: Training
        train_one_epoch(model, sd, dataloader, optimizer, scaler, loss_fn, epoch=epoch)

        if epoch % 5 == 0:
            save_path = os.path.join(log_dir, f"{epoch}{ext}")
            
            # Algorithm 2: Sampling
            reverse_diffusion(model, sd, timesteps=TrainingConfig.TIMESTEPS, num_images=32, generate_video=generate_video,
                              save_path=save_path, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE)

            # clear_output()
            checkpoint_dict = {"opt": optimizer.state_dict(), "scaler": scaler.state_dict(), "model": model.state_dict()}
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
            del checkpoint_dict


if __name__ == "__main__":
    train()