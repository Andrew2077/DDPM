# DDPM

Pytorch implementation pipeline for denoising defusion probabilistic models


[My perosnal notes](https://scrawny-toad-78d.notion.site/Hopefully-a-deeper-understanding-cbf1ab8b6dea4df29cc0284f17223986) on the Math-statistics behind Denoising Diffusion Probabilistic Models

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
- Apply 999 denoising steps to the image
- this gif is slightly manipulated to show the denoising process faster, as noise occupy most of the frames duo to `linear beta schedule`


<img src="SDDPM_results.gif" width="640" height="640" />

## Milestones

- [x] simple unet
- [ ] attention based U-net - partially done
- [ ] Conditional DDPM
- [x] training pipeline
- [ ] CLI interface - partially done
- [x] linear beta scheduler
- [ ] Cosine beta scheduler
- [ ] better sampling pipeline
