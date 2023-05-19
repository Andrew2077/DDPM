from DDUN.Forward import Forward_pass
from torch.nn.functional import l1_loss


def get_loss(
    model,
    image,
    t,
    sqrt_alphas_cumpord,
    sqrt_one_minus_alphas_cumpord,
    device="cpu",
    torch_seed=42,
):
    noisy_img, noise = Forward_pass(
        image, t, sqrt_alphas_cumpord, sqrt_one_minus_alphas_cumpord, device, torch_seed
    )

    noise_pred = model(noisy_img, t)
    return l1_loss(noise_pred, noise)
