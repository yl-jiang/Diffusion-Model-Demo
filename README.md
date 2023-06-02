# Diffusion-Model-Demo

## UNet in diffusion model

### Macro model structure 

<div align=center><img src="./imgs/StableDiffusion-UNet.drawio.svg"></div>

### ResnetBlock

<div align=center><img src="./imgs/res.svg"></div>

### ResnetBlock with Attention

<div align=center><img src="./imgs/resa.svg"></div>

### Final layers

<div align=center><img src="./imgs/final.svg"></div>

## Training

```shell
$ git clone https://github.com/yl-jiang/Diffusion-Model-Demo.git
$ cd Diffusion-Model-Demo
$ conda activate your_pytorch_environment
$ python train.py
```

if everything is ok, then you will see something like this:

```shell
Logging at: Logs_Checkpoints/Inference/version_0
Model Checkpoint at: Logs_Checkpoints/checkpoints/version_0
Train :: Epoch: 1/30:   9%|█▏          | 42/469 [00:06<00:49,  8.61it/s, Loss: 0.0507]

```
