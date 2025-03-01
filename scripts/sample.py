import os
import torch
from flux_micro.model import FluxMicroModel
from flux_micro.diffusion_utils import make_diffusion_schedule

# Load the trained model
device = torch.device("cpu")
model = FluxMicroModel(
    image_size=32, channels=3, embed_dim=128, num_layers=8, num_heads=4, ff_dim=512
)

# Define checkpoint directory relative to project root
checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

# Get all saved checkpoint files
checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")])

if not checkpoint_files:
    raise FileNotFoundError("No checkpoints found. Train the model first.")

latest_checkpoint = os.path.join(
    checkpoint_dir, checkpoint_files[-1]
)  # Load the most recent checkpoint
print(f"Loading checkpoint: {latest_checkpoint}")

model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
model = model.to(device)
model.eval()

# Diffusion schedule (must match the one used in training)
T = 1000
betas, alphas, alpha_cumprod, sqrt_ac, sqrt_omc = make_diffusion_schedule(
    T=T, device=device
)


# Sampling function: given a trained model, generate an image starting from noise
def sample_image(model, T, betas, alpha_cumprod):
    print("Sampling image...")
    # Start from pure noise x_T
    x_t = torch.randn(1, 3, 32, 32, device=device)  # batch of 1
    # Iteratively refine the image from t=T down to 1
    for t in reversed(
        range(1, T)
    ):  # from T-1 down to 1 (we'll handle t and t-1 indices)
        print(f"Sampling step {t} of {T}")
        t_tensor = torch.tensor([t], device=device)
        # Predict noise at this step
        pred_noise = model(x_t, t_tensor)  # shape (1,3,32,32)
        # Compute the denoised mean (mu) for x_{t-1} using the predicted noise
        alpha_t = 1.0 - betas[t]  # = alphas[t]
        alpha_cum_t = alpha_cumprod[t]  # \bar{\alpha}_t
        alpha_cum_prev = (
            alpha_cumprod[t - 1] if t > 1 else alpha_cumprod[0] * 0 + 1.0
        )  # \bar{\alpha}_{t-1}, define \bar{\alpha}_0 = 1
        # Corresponding coefficients for prediction
        # (These formulas are from the DDPM paper)
        coef1 = torch.sqrt(1.0 / alpha_t)  # 1/sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_cum_t)
        # Predicted x0 (denoised image at step t, using predicted noise)
        x0_pred = (x_t - coef2 * pred_noise) * coef1
        # Compute the mean for x_{t-1}
        mean = (
            torch.sqrt(alpha_cum_prev / alpha_cum_t) * x0_pred
            + torch.sqrt(1 - alpha_cum_prev) * pred_noise
        )
        # Sample x_{t-1} from a normal distribution centered at this mean (with variance beta_t)
        noise = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
        x_t = mean + torch.sqrt(betas[t]) * noise  # add randomness for sampling
    # After loop, x_t at t=0 is the final denoised image
    x_0 = x_t
    return x_0


# Generate an image
with torch.no_grad():
    generated = sample_image(model, T, betas, alpha_cumprod)  # shape (1,3,32,32)
# Convert the generated image from [-1,1] to [0,1] for viewing
generated_image = (generated.clamp(-1, 1) + 1) / 2  # shape (1,3,32,32), values in [0,1]
# Save or display the image (example uses PIL)
import torchvision.utils as vutils
from PIL import Image

vutils.save_image(generated_image, "sample_output.png")
Image.open("sample_output.png").show()
