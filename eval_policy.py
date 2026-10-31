"""
Reverse Diffusion Sampling and Policy Inference for 16-DoF Leap Hand Manipulation.
Reference: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023).
"""

import numpy as np
import torch
import torch.nn as nn

# Import the identical network and scheduler architectures directly from train_diffusion
from train_diffusion import ConditionalUnet1D, DDPMScheduler


# -----------------------------------------------------------------------------
# 1. Reverse Diffusion Policy Sampler (DDPM Sampling Loop)
# -----------------------------------------------------------------------------
class ActionDiffusionSampler:
    """
    Loads trained diffusion weights and executes reverse Langevin dynamics
    to infer collision-free Leap Hand joint trajectories.
    """

    def __init__(self, checkpoint_path: str = "diffusion_policy_best.ckpt", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing inference engine on device: {self.device}")

        # Load checkpoint payload
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint.get("config", {})
        self.action_dim = self.config.get("action_dim", 16)
        self.pred_horizon = self.config.get("pred_horizon", 16)
        self.obs_horizon = self.config.get("obs_horizon", 2)
        self.num_diffusion_steps = self.config.get("diffusion_steps", 100)

        # Retrieve normalization statistics strictly matching LeapHandTrajectoryDataset
        norm_stats = checkpoint["norm_stats"]
        self.stat_min = torch.tensor(norm_stats["min"], dtype=torch.float32, device=self.device)
        self.stat_max = torch.tensor(norm_stats["max"], dtype=torch.float32, device=self.device)

        # Instantiate model architecture with identical parameters
        self.model = ConditionalUnet1D(
            action_dim=self.action_dim,
            obs_dim=self.obs_horizon * self.action_dim,  # 2 * 16 = 32
            time_dim=64,
            diffusion_step_dim=128
        ).to(self.device)

        # Load trained state dictionary
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Initialize DDPM diffusion parameters
        self.scheduler = DDPMScheduler(num_diffusion_steps=self.num_diffusion_steps, device=self.device)
        self.betas = self.scheduler.betas
        self.alphas = self.scheduler.alphas
        self.alphas_cumprod = self.scheduler.alphas_cumprod

        print(f"[INFO] Loaded checkpoint trained at epoch {checkpoint.get('epoch', 'N/A')} with loss {checkpoint.get('loss', 0.0):.6f}")

    @torch.no_grad()
    def sample_trajectory(self, obs_cond: torch.Tensor) -> np.ndarray:
        """
        Executes reverse DDPM sampling: transforms standard Gaussian noise
        into continuous, dynamically feasible joint position trajectories.

        Args:
            obs_cond: Flattened observation history tensor of shape (Batch_Size, 32).

        Returns:
            unnormalized_actions: Numpy array of shape (Batch_Size, Pred_Horizon, 16) in physical radians.
        """
        batch_size = obs_cond.shape[0]
        obs_cond = obs_cond.to(self.device)

        # 1. Initialize action sequence from pure standard Gaussian noise
        # Shape: (Batch_Size, Action_Dim=16, Pred_Horizon=16) matching 1D temporal CNN format
        actions = torch.randn(
            (batch_size, self.action_dim, self.pred_horizon),
            dtype=torch.float32,
            device=self.device
        )

        # 2. Reverse Diffusion Denoising Loop (from T-1 down to 0)
        for t in reversed(range(self.num_diffusion_steps)):
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict injected noise gradient using conditional 1D U-Net
            predicted_noise = self.model(actions, timesteps, obs_cond)

            # Retrieve scheduler scalar parameters for step t
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            # DDPM reverse step mean formulation:
            # mu = (1 / sqrt(alpha_t)) * (actions - (beta_t / sqrt(1 - alpha_cumprod_t)) * predicted_noise)
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                actions - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise
            )

            # Add stochastic Langevin dynamics noise if t > 0
            if t > 0:
                noise = torch.randn_like(actions)
                actions = mean + torch.sqrt(beta_t) * noise
            else:
                actions = mean

        # 3. Clip normalized actions to valid bounded manifold [-1, 1]
        actions_clipped = torch.clamp(actions, -1.0, 1.0)

        # 4. Invert Min-Max normalization to recover physical motor angles:
        # raw = ((normalized + 1.0) / 2.0) * (max - min) + min
        actions_clipped = actions_clipped.permute(0, 2, 1)  # (Batch, Pred_Horizon, Action_Dim)
        unnormalized_actions = (
            (actions_clipped + 1.0) / 2.0 * (self.stat_max - self.stat_min) + self.stat_min
        )

        return unnormalized_actions.cpu().numpy()


# -----------------------------------------------------------------------------
# 2. Main Execution Entry Point
# -----------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_file = "diffusion_policy_best.ckpt"

    # Instantiate sampler
    sampler = ActionDiffusionSampler(checkpoint_path=checkpoint_file, device=device)

    # Simulate an incoming observation window: To=2 frames flattened to (1, 32)
    dummy_observation = torch.randn((1, 32), device=device)

    # Perform reverse diffusion trajectory rollout
    print("[INFO] Executing 100-step DDPM reverse denoising rollout...")
    sampled_trajectory = sampler.sample_trajectory(dummy_observation)

    print(f"[SUCCESS] Trajectory generation complete! Tensor shape: {sampled_trajectory.shape}")
    print(f"[INFO] Motor 0 (Thumb Root) predicted sequence (first 5 steps):")
    print(f"       {sampled_trajectory[0, :5, 0]}")

    # Export sampled trajectory asset for MuJoCo simulation rollout
    output_filename = "sampled_action_traj.npy"
    np.save(output_filename, sampled_trajectory)
    print(f"[INFO] Deliverable exported successfully: '{output_filename}'")


if __name__ == "__main__":
    main()
