import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_ddim_timestep_embedding(timesteps: torch.Tensor, dim: int = 512) -> torch.Tensor:
    """Standard DDIM sinusoidal timestep embedding matching AffHandGen official checkpoints."""
    timesteps = timesteps.reshape(-1).float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / (half - 1)
    )
    args = timesteps[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class LinearDiffusionSchedule:
    """
    Linear beta schedule matching narusuzuki/AffHandGen `hog/train.yml`:
      beta_start=1e-4, beta_end=1e-2, num_diffusion_timesteps=1000.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 1e-2,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps."""
        t = t.to(device=self.device, dtype=torch.long).clamp(0, self.num_timesteps - 1)
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        b = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return a * x0 + b * noise

    def ddim_step(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        t: int,
        t_prev: int,
    ) -> torch.Tensor:
        """
        Deterministic DDIM reverse update (Algorithm 1, Lines 6-7):
          eps_t    = (x_t - sqrt(alpha_bar_t) * x0_hat) / sqrt(1 - alpha_bar_t)
          x_{prev} = sqrt(alpha_bar_{prev}) * x0_hat + sqrt(1 - alpha_bar_{prev}) * eps_t
        """
        if t_prev < 0:
            return x0_hat
        t_idx = max(0, min(int(t), self.num_timesteps - 1))
        prev_idx = max(0, min(int(t_prev), self.num_timesteps - 1))

        sqrt_ab_t = self.sqrt_alphas_cumprod[t_idx]
        sqrt_1m_ab_t = self.sqrt_one_minus_alphas_cumprod[t_idx].clamp(min=1e-6)
        eps_t = (x_t - sqrt_ab_t * x0_hat) / sqrt_1m_ab_t

        sqrt_ab_prev = self.sqrt_alphas_cumprod[prev_idx]
        sqrt_1m_ab_prev = self.sqrt_one_minus_alphas_cumprod[prev_idx]
        return sqrt_ab_prev * x0_hat + sqrt_1m_ab_prev * eps_t


class DenseMLP(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        super().__init__()
        self.dense = nn.ModuleList([
            nn.Linear(in_dim, hid_dim),
            nn.Linear(hid_dim, out_dim),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense[1](F.silu(self.dense[0](x)))


class TextCondMLP(nn.Module):
    def __init__(self, hid_dim: int = 2056, out_dim: int = 512):
        super().__init__()
        # Legacy 512-D CLIP branch kept solely for strict=True checkpoint compatibility
        self.dense = nn.ModuleList([
            nn.Linear(512, hid_dim),
            nn.Linear(hid_dim, out_dim),
        ])
        # Active 768-D DistilBERT projection branch
        self.dense_distilbert = nn.ModuleList([
            nn.Linear(768, hid_dim),
            nn.Linear(hid_dim, out_dim),
        ])

    def forward(self, txt_tokens: torch.Tensor) -> torch.Tensor:
        return self.dense_distilbert[1](F.silu(self.dense_distilbert[0](txt_tokens)))


class AttnWrap(nn.Module):
    def __init__(self, dim: int = 512, n_head: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)


class FFWrap(nn.Module):
    def __init__(self, dim: int = 512, ff_dim: int = 1024, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )


class CrossAttnBlock(nn.Module):
    """Post-LN Attention + Pre-LN FFN Transformer Decoder block."""

    def __init__(self, dim: int = 512, n_head: int = 4, ff_dim: int = 1024, dropout: float = 0.25):
        super().__init__()
        self.attn1 = AttnWrap(dim, n_head)
        self.attn2 = AttnWrap(dim, n_head)
        self.ff = FFWrap(dim, ff_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attn1.norm(x + self.attn1.attn(x, x, x, need_weights=False)[0])
        x = self.attn2.norm(x + self.attn2.attn(x, ctx, ctx, key_padding_mask=key_padding_mask, need_weights=False)[0])
        x = x + self.ff.net(x)
        return x


class PosEnc(nn.Module):
    def __init__(self, max_len: int = 5000, dim: int = 512):
        super().__init__()
        self.register_buffer("pe", torch.zeros(1, max_len, dim))


class AffHandGen(nn.Module):
    """
    Affordance-Guided 3D Hand Pose Diffusion Prior (Suzuki et al., ECCV 2026).
    Exact 63-tensor architecture compatible with official `narusuzuki/AffHandGen` checkpoints.
      Input:  noisy MANO hand pose x_t (B, 45), diffusion step t (B,),
              DistilBERT text tokens c_D (B, 64, 768), HaMeR ViT image tokens c_I (B, 192, 1280)
      Output: denoised MANO hand pose x_hat_0 (B, 45) in raw axis-angle space
    """

    def __init__(self, emb_dim: int = 512, hid_dim: int = 2056, n_head: int = 4, dropout: float = 0.25):
        super().__init__()
        self.emb_dim = emb_dim
        self.register_buffer("keypoint_ids", torch.zeros(15, dtype=torch.long))
        self.register_buffer("group_ids", torch.zeros(15, dtype=torch.long))

        self.temb = DenseMLP(emb_dim, hid_dim, emb_dim)
        self.iemb = DenseMLP(45, hid_dim, emb_dim)
        self.cemb_txt = TextCondMLP(hid_dim, emb_dim)
        self.project_hamer_feat = nn.Linear(1280, emb_dim)

        # Kept solely for strict=True compatibility with checkpoint state_dict
        self.keypoint_pe = nn.Embedding(15, emb_dim)
        self.group_pe = nn.Embedding(3, emb_dim)

        self.mask_tokens = nn.ParameterDict({
            "image": nn.Parameter(torch.zeros(emb_dim)),
            "text": nn.Parameter(torch.zeros(emb_dim)),
        })
        self.attention_text = nn.ModuleList([CrossAttnBlock(emb_dim, n_head, 1024, dropout)])
        self.attention_image = nn.ModuleList([CrossAttnBlock(emb_dim, n_head, 1024, dropout)])
        self.output_process = nn.Linear(emb_dim * 2, 45)
        self.pos_enc = PosEnc(5000, emb_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
        use_img_pe: bool = True,
    ) -> torch.Tensor:
        B = x_t.shape[0]
        te = self.temb(get_ddim_timestep_embedding(t, self.emb_dim))  # (B, 512)
        ie = self.iemb(x_t)                                           # (B, 512)

        if text_tokens is None:
            ce_t = self.mask_tokens["text"].view(1, 1, self.emb_dim).expand(B, 1, self.emb_dim)
            kpm = None
        else:
            ce_t = self.cemb_txt(text_tokens)                         # (B, L_t, 512)
            kpm = (text_attention_mask == 0).bool() if text_attention_mask is not None else None

        if image_tokens is None:
            ce_i = self.mask_tokens["image"].view(1, 1, self.emb_dim).expand(B, 1, self.emb_dim)
        else:
            ce_i = self.project_hamer_feat(image_tokens)              # (B, 192, 512)
            if use_img_pe:
                ce_i = ce_i + self.pos_enc.pe[:, : ce_i.shape[1], :]

        h = torch.stack([ie, te], dim=1)                              # (B, 2, 512)
        for blk in self.attention_text:
            h = blk(h, ce_t, key_padding_mask=kpm)
        for blk in self.attention_image:
            h = blk(h, ce_i, key_padding_mask=None)

        return self.output_process(h.reshape(B, -1))                  # (B, 45)


def load_affhandgen(
    ckpt_path: str = "weights/affhandgen/hog_ckpt_best.pth",
    device: Optional[torch.device] = None,
) -> Tuple[AffHandGen, LinearDiffusionSchedule]:
    """Load official AffHandGen checkpoint with strict=True verification."""
    assert os.path.exists(ckpt_path), f"Missing AffHandGen checkpoint: {ckpt_path}"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, (list, tuple)):
        sd = ckpt[0]
    elif isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    model = AffHandGen().to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.requires_grad_(False)

    schedule = LinearDiffusionSchedule(
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=1e-2,
        device=device,
    )
    return model, schedule


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sched = load_affhandgen("weights/affhandgen/hog_ckpt_best.pth", device=dev)
    with torch.no_grad():
        dummy_x = torch.zeros(2, 45, device=dev)
        dummy_t = torch.tensor([10, 500], dtype=torch.long, device=dev)
        dummy_txt = torch.zeros(2, 64, 768, device=dev)
        dummy_msk = torch.ones(2, 64, dtype=torch.uint8, device=dev)
        dummy_img = torch.zeros(2, 192, 1280, device=dev)

        te = get_ddim_timestep_embedding(dummy_t, 512)
        te_diff = float((te[0] - te[1]).abs().mean().item())
        assert te_diff > 0.1, f"Timestep embedding collapsed! diff={te_diff}"

        out = model(dummy_x, dummy_t, dummy_txt, dummy_msk, dummy_img)
        x_prev = sched.ddim_step(dummy_x[:1], out[:1], t=500, t_prev=480)

    print(
        f"[OK] AffHandGen strict=True verified | output={tuple(out.shape)} | "
        f"te_diff={te_diff:.4f} | ddim_step={tuple(x_prev.shape)} | finite={bool(torch.isfinite(out).all())}"
    )