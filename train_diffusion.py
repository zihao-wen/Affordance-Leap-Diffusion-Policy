"""
Action Diffusion Policy for 16-DoF Leap Hand Manipulation.
Reference: Chi et al., "Diffusion Policy" (RSS 2023).
"""

import math
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# 1. Sinusoidal Timestep Embedding (Chi et al., Section 4.1)
# -----------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    """Encodes discrete diffusion steps into continuous frequency vectors."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# -----------------------------------------------------------------------------
# 2. 1D Temporal Residual Convolutional Block
# -----------------------------------------------------------------------------
class Conv1dBlock(nn.Module):
    """Temporal 1D Conv module with Batch Normalization and Mish activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# -----------------------------------------------------------------------------
# 3. Conditional 1D Temporal U-Net Denoiser (Chi et al., Figure 2(b))
# -----------------------------------------------------------------------------
class ConditionalUnet1D(nn.Module):
    """
    Predicts added Gaussian noise for a multi-step action trajectory
    conditioned on diffusion step k and current low-dim robot observation.
    """

    def __init__(
        self,
        action_dim: int = 16,
        obs_dim: int = 32,
        time_dim: int = 64,
        diffusion_step_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim

        # Timestep condition encoder
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, diffusion_step_dim),
            nn.Mish(),
            nn.Linear(diffusion_step_dim, diffusion_step_dim),
        )

        # Observation state condition encoder
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim, diffusion_step_dim),
            nn.Mish(),
            nn.Linear(diffusion_step_dim, diffusion_step_dim),
        )

        # Shared condition projection
        cond_dim = diffusion_step_dim * 2

        # U-Net Encoder
        self.in_proj = nn.Conv1d(action_dim, 64, kernel_size=1)
        self.down1 = Conv1dBlock(64, 128)
        self.down2 = Conv1dBlock(128, 256)

        # Bottleneck
        self.mid = Conv1dBlock(256, 256)

        # U-Net Decoder (Skip connections concatenation)
        self.up1 = Conv1dBlock(256 + 128, 128)
        self.up2 = Conv1dBlock(128 + 64, 64)
        self.out_proj = nn.Conv1d(64, action_dim, kernel_size=1)

        # Conditioning adapters for each level
        self.cond_proj1 = nn.Linear(cond_dim, 128)
        self.cond_proj2 = nn.Linear(cond_dim, 256)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: (B, Action_Dim, Pred_Horizon)
            timesteps: (B,)
            obs_cond: (B, Obs_Dim)
        Returns:
            predicted_noise: (B, Action_Dim, Pred_Horizon)
        """
        # Joint condition vector: timestep embedding + observation embedding
        t_emb = self.time_mlp(timesteps)
        o_emb = self.obs_mlp(obs_cond)
        global_cond = torch.cat([t_emb, o_emb], dim=-1)

        # Encoder path
        x0 = self.in_proj(noisy_actions)
        x1 = self.down1(x0) + self.cond_proj1(global_cond).unsqueeze(-1)
        x2 = self.down2(x1) + self.cond_proj2(global_cond).unsqueeze(-1)

        # Latent bottleneck
        mid = self.mid(x2)

        # Decoder path with skip connections
        u1 = self.up1(torch.cat([mid, x1], dim=1))
        u2 = self.up2(torch.cat([u1, x0], dim=1))
        pred_noise = self.out_proj(u2)
        return pred_noise


# -----------------------------------------------------------------------------
# 4. Trajectory Dataset with Action Chunking (Chi et al., Section 4.2)
# -----------------------------------------------------------------------------
class LeapHandTrajectoryDataset(Dataset):
    """
    Slices demonstration data into pairs of:
      - Observation history (To = 2)
      - Action prediction trajectory (Tp = 16)
    Applies Min-Max normalization to the [-1, 1] interval.
    """

    def __init__(
        self,
        npy_path: str = "leap_hand_qpos.npy",
        obs_horizon: int = 2,
        pred_horizon: int = 16,
    ):
        raw_data = np.load(npy_path).astype(np.float32)  # (T, 16)

        # Compute dataset stats for normalization
        self.stats = {
            "min": np.min(raw_data, axis=0),
            "max": np.max(raw_data, axis=0) + 1e-4,
        }
        normalized_data = (
            2.0 * (raw_data - self.stats["min"]) / (self.stats["max"] - self.stats["min"])
            - 1.0
        )

        self.samples = []
        total_len = len(normalized_data)
        window_size = obs_horizon + pred_horizon

        for i in range(total_len - window_size + 1):
            obs_chunk = normalized_data[i : i + obs_horizon]  # (To, 16)
            act_chunk = normalized_data[
                i + obs_horizon : i + window_size
            ]  # (Tp, 16)

            self.samples.append(
                (
                    obs_chunk.reshape(-1),  # Flattened (To * 16,) -> 32
                    act_chunk.T,  # (16, Tp) for 1D Temporal Conv
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, act = self.samples[idx]
        return torch.from_numpy(obs), torch.from_numpy(act)


# -----------------------------------------------------------------------------
# 5. DDPM Gaussian Diffusion Scheduler (Chi et al., Section 4.1)
# -----------------------------------------------------------------------------
class DDPMScheduler:
    """Standard Linear Variance DDPM Scheduler."""

    def __init__(self, num_diffusion_steps: int = 100, device: str = "cuda"):
        self.num_steps = num_diffusion_steps
        self.device = device

        self.betas = torch.linspace(1e-4, 0.02, num_diffusion_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self, original_actions: torch.Tensor, timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(original_actions)
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[timesteps]).view(-1, 1, 1)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - self.alphas_cumprod[timesteps]).view(
            -1, 1, 1
        )
        noisy_actions = sqrt_alpha * original_actions + sqrt_one_minus_alpha * noise
        return noisy_actions, noise


# -----------------------------------------------------------------------------
# 6. Training Pipeline (Exporting diffusion_policy_best.ckpt)
# -----------------------------------------------------------------------------
def train_policy():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing training backend on device: {device}")

    # Setup Dataset
    dataset = LeapHandTrajectoryDataset(
        npy_path="leap_hand_qpos.npy", obs_horizon=2, pred_horizon=16
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Instantiate Model & Scheduler
    model = ConditionalUnet1D(action_dim=16, obs_dim=32, time_dim=64).to(device)
    scheduler = DDPMScheduler(num_diffusion_steps=100, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    epochs = 120
    best_loss = float("inf")
    output_ckpt = "diffusion_policy_best.ckpt"

    print("[INFO] Starting Action Diffusion Policy training loop...")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for obs, actions in dataloader:
            obs = obs.to(device)  # (Batch, 32)
            actions = actions.to(device)  # (Batch, 16, 16)
            batch_size = actions.shape[0]

            # Sample random diffusion steps
            t = torch.randint(0, scheduler.num_steps, (batch_size,), device=device).long()

            # Forward diffusion: inject noise
            noisy_act, ground_truth_noise = scheduler.add_noise(actions, t)

            # Predict noise and calculate MSE loss
            pred_noise = model(noisy_act, t, obs)
            loss = loss_fn(pred_noise, ground_truth_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"[TRAIN] Epoch {epoch:03d}/{epochs} | Diffusion MSE Loss: {avg_loss:.6f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save checkpoint state and dataset normalization parameters
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "norm_stats": dataset.stats,
                    "loss": best_loss,
                    "config": {
                        "action_dim": 16,
                        "obs_horizon": 2,
                        "pred_horizon": 16,
                        "diffusion_steps": 100,
                    },
                },
                output_ckpt,
            )

    print(f"\n[INFO] Optimization successfully converged. Best Loss: {best_loss:.6f}")
    print(f"[INFO] Deliverable 3 saved to: '{output_ckpt}'")


if __name__ == "__main__":
    train_policy()