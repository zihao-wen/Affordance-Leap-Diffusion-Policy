import os
import sys
import glob
import json
from functools import partial
from unittest.mock import MagicMock

# Headless rendering protection & HuggingFace mirror cache
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["HF_HOME"] = "/root/autodl-tmp/cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.makedirs("/root/autodl-tmp/cache/huggingface", exist_ok=True)
sys.modules["pyrender"] = MagicMock()

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, DistilBertModel

# Reuse HaMeR-compatible ViTPose building blocks & ViTDetDataset from local WiLoR repo
WILOR_ROOT = os.path.abspath("./WiLoR")
if WILOR_ROOT not in sys.path:
    sys.path.append(WILOR_ROOT)

from wilor.models.backbones.vit import PatchEmbed, Block
from wilor.datasets.vitdet_dataset import ViTDetDataset


def recursive_to(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: recursive_to(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_to(v, device) for v in x]
    return x


class HaMeRViTBackbone(nn.Module):
    """
    Pure HaMeR ViT-Huge/16 backbone matching the exact 389 `backbone.*` tensors in hamer.ckpt.
    Input:  (B, 3, 256, 192)
    Output: (B, 192, 1280) localized spatial patch tokens (16 x 12 grid)
    """

    def __init__(self):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(
            img_size=(256, 192),
            patch_size=16,
            in_chans=3,
            embed_dim=1280,
            ratio=1,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 193, 1280))
        self.blocks = nn.ModuleList([
            Block(
                dim=1280,
                num_heads=16,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for _ in range(32)
        ])
        self.last_norm = norm_layer(1280)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (hp, wp) = self.patch_embed(x)
        assert hp == 16 and wp == 12, f"Expected 16x12 patch grid, got ({hp}, {wp})"
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)
        return x


def load_distilbert(model_id: str, device: torch.device):
    """Load frozen DistilBERT from local cache first, falling back to hf-mirror."""
    try:
        print(f"[INFO] Loading {model_id} from local cache...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = DistilBertModel.from_pretrained(model_id, local_files_only=True)
    except OSError:
        print(f"[INFO] Local cache not found; downloading {model_id} via hf-mirror...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = DistilBertModel.from_pretrained(model_id)

    model = model.to(device).eval()
    model.requires_grad_(False)
    return tokenizer, model


def extract_text_tokens(descriptions: list, device: torch.device, max_length: int = 64) -> tuple:
    """
    Encode per-frame affordance descriptions into fixed-length token sequences
    using frozen DistilBERT.
      Returns:
        text_tokens:         (N, 64, 768) float32
        text_attention_mask: (N, 64)      uint8
    """
    tokenizer, text_encoder = load_distilbert("distilbert-base-uncased", device)

    encoded = tokenizer(
        descriptions,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = outputs.last_hidden_state.detach().cpu().numpy().astype(np.float32)

    mask_np = attention_mask.detach().cpu().numpy().astype(np.uint8)
    assert text_tokens.shape == (len(descriptions), max_length, 768), f"Unexpected text_tokens shape: {text_tokens.shape}"
    assert mask_np.shape == (len(descriptions), max_length), f"Unexpected attention_mask shape: {mask_np.shape}"
    assert np.all(np.isfinite(text_tokens)), "Non-finite values detected in text_tokens!"

    del text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return text_tokens, mask_np


def load_hamer_vit_backbone(ckpt_path: str, device: torch.device) -> nn.Module:
    """Strictly load all 389 frozen HaMeR ViT-Huge/16 backbone tensors."""
    assert os.path.exists(ckpt_path), f"Missing HaMeR checkpoint: {ckpt_path}"
    print(f"[INFO] Loading frozen HaMeR ViT-Huge/16 backbone from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    backbone_sd = {
        k[len("backbone."):]: v
        for k, v in sd.items()
        if k.startswith("backbone.")
    }
    assert len(backbone_sd) == 389, f"Expected 389 HaMeR backbone tensors, got {len(backbone_sd)}"

    vit_backbone = HaMeRViTBackbone()
    vit_backbone.load_state_dict(backbone_sd, strict=True)
    vit_backbone = vit_backbone.to(device).eval()
    vit_backbone.requires_grad_(False)
    print("[INFO] Successfully verified strict=True loading of all 389 HaMeR ViT tensors.")
    return vit_backbone


def load_preprocess_cfg(cfg_path: str):
    assert os.path.exists(cfg_path), f"Missing ViTDet preprocess config: {cfg_path}"
    try:
        from wilor.configs import get_config
        return get_config(cfg_path)
    except Exception:
        from omegaconf import OmegaConf
        return OmegaConf.load(cfg_path)


def extract_image_tokens(
    frame_paths: list,
    hand_bboxes: list,
    is_right_flags: list,
    device: torch.device
) -> np.ndarray:
    """
    Extract 192 localized spatial visual tokens per frame using the frozen HaMeR ViT-Huge/16 backbone.
      Input crop from ViTDetDataset: (1, 3, 256, 256) -> sliced to (1, 3, 256, 192)
      HaMeR ViT output:              (1, 192, 1280)
      Returns:
        image_tokens: (N, 192, 1280) float32
    """
    hamer_ckpt_path = "weights/hamer/hamer.ckpt"
    cfg_path = os.path.join(WILOR_ROOT, "pretrained_models/model_config.yaml")

    model_cfg = load_preprocess_cfg(cfg_path)
    hamer_vit = load_hamer_vit_backbone(hamer_ckpt_path, device)

    image_tokens_list = []

    with torch.no_grad():
        for t, img_path in enumerate(frame_paths):
            img_cv2 = cv2.imread(img_path)
            assert img_cv2 is not None, f"Failed to read frame: {img_path}"

            box = hand_bboxes[t]
            assert box is not None and len(box) == 4, f"Invalid hand bbox at frame {t}: {box}"
            is_right_val = int(bool(is_right_flags[t]))

            boxes_np = np.array([box], dtype=np.float32)
            right_np = np.array([is_right_val], dtype=np.int64)

            dataset = ViTDetDataset(model_cfg, img_cv2, boxes_np, right_np, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

            batch = next(iter(dataloader))
            batch = recursive_to(batch, device)
            x = batch["img"]
            assert x.ndim == 4 and x.shape[2] == 256 and x.shape[3] == 256, (
                f"Unexpected ViTDetDataset image shape at frame {t}: {tuple(x.shape)}"
            )

            # Match HaMeR native backbone aspect ratio: 256 x 192 -> 16 x 12 = 192 patch tokens
            tokens_t = hamer_vit(x[:, :, :, 32:-32])[0].detach().cpu().numpy().astype(np.float32)
            assert tokens_t.shape == (192, 1280), f"Unexpected token shape at frame {t}: {tokens_t.shape}"
            image_tokens_list.append(tokens_t)

            if (t + 1) % 10 == 0 or t == len(frame_paths) - 1:
                print(f"  --> Extracted HaMeR ViT tokens [{t + 1:02d}/{len(frame_paths):02d}] | shape={tokens_t.shape}")

    image_tokens = np.stack(image_tokens_list, axis=0)
    assert np.all(np.isfinite(image_tokens)), "Non-finite values detected in image_tokens!"
    return image_tokens


def main():
    print("=" * 70)
    print("[Module 4] Multimodal Condition Feature Extraction (DistilBERT + HaMeR ViT)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Hardware platform: {device}")

    # 1. Load Upstream Data Contracts
    desc_path = "results/affordance_descriptions.json"
    hand_path = "results/detected_hand_bboxes.json"
    assert os.path.exists(desc_path), f"Missing affordance descriptions: {desc_path}"
    assert os.path.exists(hand_path), f"Missing hand bounding boxes: {hand_path}"

    with open(desc_path, "r") as f:
        desc_data = json.load(f)
    per_frame_desc = sorted(desc_data["per_frame_descriptions"], key=lambda item: item["frame_idx"])
    descriptions = [item["affordance_description"] for item in per_frame_desc]
    expected_is_right = (str(desc_data.get("handedness", "left")).strip().lower() == "right")

    with open(hand_path, "r") as f:
        hand_data = json.load(f)
    if isinstance(hand_data, dict):
        hand_bboxes = hand_data["bboxes"]
        raw_right = hand_data.get(
            "per_frame_is_right",
            hand_data.get("is_right_per_frame", hand_data.get("is_right", None))
        )
        if isinstance(raw_right, list):
            is_right_flags = [bool(v) for v in raw_right]
        elif raw_right is not None:
            is_right_flags = [bool(raw_right)] * len(hand_bboxes)
        else:
            is_right_flags = [expected_is_right] * len(hand_bboxes)
    else:
        hand_bboxes = hand_data
        is_right_flags = [expected_is_right] * len(hand_bboxes)

    frame_paths = sorted(glob.glob("./data/dexycb_025_mug/frame_*.jpg"))
    num_frames = len(frame_paths)
    assert num_frames == len(descriptions), f"Frame/description mismatch: {num_frames} vs {len(descriptions)}"
    assert num_frames == len(hand_bboxes), f"Frame/bbox mismatch: {num_frames} vs {len(hand_bboxes)}"
    assert num_frames == len(is_right_flags), f"Frame/is_right mismatch: {num_frames} vs {len(is_right_flags)}"
    assert [item["frame_idx"] for item in per_frame_desc] == list(range(num_frames)), "Non-contiguous frame indices!"

    # Cross-check handedness consistency to prevent silent left/right mirror bugs
    right_count = sum(is_right_flags)
    majority_is_right = (right_count > num_frames / 2)
    assert majority_is_right == expected_is_right, (
        f"Handedness mismatch! descriptions says is_right={expected_is_right}, "
        f"but hand_bboxes has {right_count}/{num_frames} right-hand frames."
    )
    print(
        f"[INFO] Verified alignment across {num_frames} frames | "
        f"handedness={'right' if expected_is_right else 'left'} (right_frames={right_count}/{num_frames})"
    )

    # 2. Extract Frozen Text Tokens (c_D: 64 x 768)
    text_tokens, text_attention_mask = extract_text_tokens(descriptions, device=device, max_length=64)
    for kf in [0, 10, 15, 20, 29]:
        active_len = int(text_attention_mask[kf].sum())
        print(f"  --> Keyframe {kf:02d} | Active text tokens: {active_len:2d}/64 | Text: {descriptions[kf]}")

    # 3. Extract Frozen HaMeR ViT Image Tokens (c_I: 192 x 1280)
    image_tokens = extract_image_tokens(frame_paths, hand_bboxes, is_right_flags, device=device)

    # 4. Save Unified Multimodal Condition Cache
    os.makedirs("results", exist_ok=True)
    out_path = "results/condition_features.npz"
    np.savez_compressed(
        out_path,
        text_tokens=text_tokens,
        text_attention_mask=text_attention_mask,
        image_tokens=image_tokens,
        vision_encoder=np.array("hamer_vit_huge_16"),
    )
    file_mb = os.path.getsize(out_path) / (1024 * 1024)

    print("\n" + "=" * 70)
    print(f"★ Condition Features Saved: {out_path} ({file_mb:.2f} MB)")
    print(f"  - text_tokens (c_D):         {text_tokens.shape} | {text_tokens.dtype}")
    print(f"  - text_attention_mask:       {text_attention_mask.shape} | {text_attention_mask.dtype}")
    print(f"  - image_tokens (c_I):        {image_tokens.shape} | {image_tokens.dtype} (HaMeR ViT-Huge/16)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()