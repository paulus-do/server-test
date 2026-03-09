import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

# ── 1. System check ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 2. Simple MLP (actor-like network) ───────────────────────────
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=2, act_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
            nn.Linear(256, act_dim)
        )
    def forward(self, x):
        return self.net(x)

OBS_DIM = 2
ACT_DIM = 1

model = PolicyNet(obs_dim=OBS_DIM, act_dim=ACT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()

# ── 3. Fake rollout training loop ────────────────────────────────
N_STEPS = 1000
BATCH   = 256

print(f"\nTraining for {N_STEPS} steps, batch={BATCH}...")
start = time.time()

for step in range(N_STEPS):
    # sample (x, y) uniformly in [-1, 1]^2
    obs = (torch.rand(BATCH, OBS_DIM, device=device) * 2.0) - 1.0

    # target is sqrt(x^2 + y^2)
    x = obs[:, 0]
    y = obs[:, 1]
    targets = torch.sqrt(x * x + y * y).unsqueeze(1)

    preds = model(obs)
    loss  = loss_fn(preds, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"  step {step:4d} | loss {loss.item():.4f}")

elapsed = time.time() - start
print(f"\nDone in {elapsed:.2f}s  ({N_STEPS/elapsed:.0f} steps/sec)")
print("✓ Server smoke test passed")

# ── 4. Visualize learned function on [-2, 2]^2 ─────────────────────
model.eval()
with torch.no_grad():
    grid_size = 100
    x_lin = torch.linspace(-2.0, 2.0, grid_size)
    y_lin = torch.linspace(-2.0, 2.0, grid_size)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")

    # Flatten grid and run through model
    grid_points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)
    preds = model(grid_points).cpu().reshape(grid_size, grid_size)

    # True function for comparison
    true_vals = torch.sqrt(X * X + Y * Y)

    # Plot prediction heatmap
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Model prediction: sqrt(x^2 + y^2)")
    im1 = plt.imshow(
        preds.numpy(),
        extent=[-2, 2, -2, 2],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(1, 2, 2)
    plt.title("True sqrt(x^2 + y^2)")
    im2 = plt.imshow(
        true_vals.numpy(),
        extent=[-2, 2, -2, 2],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()



