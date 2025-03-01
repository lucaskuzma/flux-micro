import torch


def make_diffusion_schedule(T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
    """Creates a linear beta schedule and pre-computes useful coefficients."""
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    # Precompute often-used terms
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
    return (
        betas,
        alphas,
        alpha_cumprod,
        sqrt_alpha_cumprod,
        sqrt_one_minus_alpha_cumprod,
    )


def add_noise(x0, t, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod):
    """
    Adds noise to batch of images x0 at timestep t.
    :param x0: Original images (batch, C, H, W) in [-1,1] range.
    :param t: Tensor of shape (batch,) with time-steps.
    :param sqrt_alpha_cumprod: Precomputed sqrt(alpha_cumprod) for all t.
    :param sqrt_one_minus_alpha_cumprod: Precomputed sqrt(1 - alpha_cumprod) for all t.
    :return: Noisy images x_t and the added noise.
    """
    # gather the appropriate coefficients for each sample in the batch
    # sqrt_alpha_cumprod and sqrt_one_minus_alpha_cumprod are arrays of length T
    device = x0.device
    batch_size = x0.size(0)
    # Gather coefficients for given t
    # We unsqueeze to match dimensions for broadcasting: (batch, 1, 1, 1)
    sqrt_ac = sqrt_alpha_cumprod[t].view(batch_size, 1, 1, 1).to(device)
    sqrt_omc = sqrt_one_minus_alpha_cumprod[t].view(batch_size, 1, 1, 1).to(device)
    noise = torch.randn_like(x0)
    x_t = sqrt_ac * x0 + sqrt_omc * noise  # apply forward diffusion formula
    return x_t, noise
