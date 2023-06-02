from dataclasses import dataclass
import torch
import os


__all__ = ['BaseConfig', 'TrainingConfig', 'SimpleDiffusion', 'ModelConfig']

def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = "MNIST" #  "MNIST", "Cifar-10", "Cifar-100", "Flowers"
    
    # For logging inferece images and saving checkpoints.
    root_log_dir = os.path.join("Logs_Checkpoints", "Inference")
    root_checkpoint_dir = os.path.join("Logs_Checkpoints", "checkpoints")

    # Current log and checkpoint directory.
    log_dir = "version_0"
    checkpoint_dir = "version_0"

@dataclass
class TrainingConfig:
    TIMESTEPS = 1000 # Define number of diffusion timesteps
    IMG_SHAPE = (1, 32, 32) if BaseConfig.DATASET == "MNIST" else (3, 32, 32) 
    NUM_EPOCHS = 30
    BATCH_SIZE = 128
    LR = 2e-4
    NUM_WORKERS = 2


@dataclass
class ModelConfig:
    BASE_CH = 64  # 64, 128, 256, 512
    BASE_CH_MULT = (1, 2, 4, 8) # 32, 16, 8, 4 
    APPLY_ATTENTION = (False, False, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 2 # 128


class SimpleDiffusion:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        img_shape=(3, 64, 64),
        device="cpu",
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device

        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta  = self.get_betas()  # (timesteps,)
        self.alpha = 1 - self.beta     # (timesteps,)
        
        self_sqrt_beta                = torch.sqrt(self.beta)  # (timesteps,)
        self.alpha_cumulative         = torch.cumprod(self.alpha, dim=0)  # (timesteps,)
        self.sqrt_alpha_bar           = torch.sqrt(self.alpha_cumulative)  # (timesteps,)
        self.one_by_sqrt_alpha        = 1. / torch.sqrt(self.alpha)  # (timesteps,)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_cumulative)  # (timesteps,)
         
    def get_betas(self):
        """
        linear schedule, proposed in original ddpm paper

        Output:
            beta: (1000,) / 取值范围为[0.0001, 0.02]
        """
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )