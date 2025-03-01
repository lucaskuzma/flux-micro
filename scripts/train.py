import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Import our model and diffusion utilities
from flux_micro.model import FluxMicroModel
from flux_micro.diffusion_utils import make_diffusion_schedule, add_noise

checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Hyperparameters
image_size = 32
batch_size = 4  # small batch size for CPU
epochs = 1  # (For demonstration, use 1 epoch. Increase for actual training)
T = 1000  # number of diffusion steps
lr = 1e-4  # learning rate for optimizer

# Set up dataset (CIFAR-10)
transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: (x * 2) - 1),  # scale images from [0,1] to [-1,1]
    ]
)
trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
# Optionally create a subset of 1000 images
subset_size = 1000
subset_indices = torch.randperm(len(trainset))[:subset_size]
trainset = Subset(trainset, subset_indices)

# Create a DataLoader for the subset
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
device = torch.device("cpu")
model = FluxMicroModel(
    image_size=image_size,
    channels=3,
    embed_dim=128,
    num_layers=8,
    num_heads=4,
    ff_dim=512,
)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse_loss = torch.nn.MSELoss()

# Prepare diffusion schedule coefficients
betas, alphas, alpha_cumprod, sqrt_ac, sqrt_omc = make_diffusion_schedule(
    T=T, device=device
)

iterations = epochs * len(trainloader)
print(
    f"Training on {len(trainset)} images with batch size {batch_size} for a total of {iterations} iterations"
)

# Training loop
model.train()
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")

    for batch_idx, (real_images, _) in enumerate(trainloader):
        print(f"Batch {batch_idx+1} of {len(trainloader)}")

        real_images = real_images.to(device)  # shape: (B, 3, 32, 32), values in [-1,1]
        # Sample a random timestep for each image in the batch
        t = torch.randint(0, T, (real_images.size(0),), device=device).long()
        # Generate noisy images x_t and the added noise
        x_t, noise = add_noise(real_images, t, sqrt_ac, sqrt_omc)
        # Predict the noise using the model
        pred_noise = model(x_t, t)
        # Compute MSE loss between the predicted noise and true noise
        loss = mse_loss(pred_noise, noise)
        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Step {batch_idx}, Loss: {loss.item():.4f}")

    # Save checkpoint at the end of each epoch
    checkpoint_path = os.path.join(checkpoint_dir, f"flux_micro_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
