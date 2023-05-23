# DDPM

Pytorch implementation pipeline for denoising defusion probabilistic models

[Math-statistics walkthrough](https://scrawny-toad-78d.notion.site/Hopefully-a-deeper-understanding-cbf1ab8b6dea4df29cc0284f17223986)

In-progress:

simple unet performance - still in progress

## SDDPM - Simple Denoising Diffusion Probabilistic Model
- Time steps = 256
- Noise Samples = 1000
- Batch size = 16
- Learning rate = 3e-4
- Optimizer = Adam
- Loss = MSE
- Epochs = 20
- Dataset = FashionMNIST
- Model = Simple Unet

## Obtained Loss
after 20 epochs
```bash
++++++++++++++++++++++++++++++++++++++++++++++++++++
at epoch 19 - loss improved to 0.047036067663133146
++++++++++++++++++++++++++++++++++++++++++++++++++++
```
## Results
- Denoise back a generated noise over 999 steps
- this gif is slightly manipulated to show the denoising process faster, as noise occupy most of the frames duo to `linear beta schedule`


<img src="SDDPM_results.gif" width="480" height="480" />

## Milestones

- [x] simple unet
- [x] training pipeline
- [ ] Cosine beta scheduler
- [ ] better performance
- [ ] attention based U-net
- [ ] better sampling pipeline
