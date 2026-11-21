import argparse
import glob
import json
import os
import sys
from unittest.mock import MagicMock

# Headless rendering protection
os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.modules["pyrender"] = MagicMock()

import cv2
import numpy as np
import torch
import torch.nn.functional as F

WILOR_ROOT = os.path.abspath("./WiLoR")
PERCEPTION_ROOT = os.path.abspath("./perception")
for p in (WILOR_ROOT, PERCEPTION_ROOT):
    if p not in sys.path:
        sys.path.append(p)

from wilor.models import load_wilor
from wilor.datasets.vitdet_dataset import ViTDetDataset
from affhandgen_model import load_affhandgen


def recursive_to(x, device):
    """Move nested tensors, dicts, or lists to the target compute device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: recursive_to(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_to(v, device) for v in x]
    return x


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0):
    """Convert weak-perspective crop camera parameters into full-frame 3D translation."""
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def compute_pa_mpjpe(predicted_joints: np.ndarray, ground_truth_joints: np.ndarray) -> np.ndarray:
    """Compute per-frame Procrustes-Aligned MPJPE in millimeters (same protocol as baseline)."""
    num_frames = predicted_joints.shape[0]
    errors = np.zeros(num_frames, dtype=np.float32)
    for t in range(num_frames):
        pred = predicted_joints[t]
        gt = ground_truth_joints[t]
        pred_c = pred - pred.mean(axis=0, keepdims=True)
        gt_c = gt - gt.mean(axis=0, keepdims=True)

        cov = np.dot(pred_c.T, gt_c)
        u, s, vt = np.linalg.svd(cov)
        rot = np.dot(vt.T, u.T)
        if np.linalg.det(rot) < 0:
            vt[-1, :] *= -1
            rot = np.dot(vt.T, u.T)

        scale = np.sum(s) / np.sum(pred_c ** 2)
        aligned = scale * np.dot(pred_c, rot.T)
        errors[t] = float(np.mean(np.linalg.norm(aligned - gt_c, axis=-1) * 1000.0))
    return errors


def axis_angle_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues conversion from axis-angle (..., 3) to rotation matrix (..., 3, 3)."""
    orig_shape = aa.shape[:-1]
    aa_flat = aa.reshape(-1, 3)
    theta2 = (aa_flat ** 2).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp(min=1e-12))

    # Use Taylor expansion near zero angle to keep gradients smooth and finite
    small = theta2 < 1e-6
    A = torch.where(small, 1.0 - theta2 / 6.0 + (theta2 ** 2) / 120.0, torch.sin(theta) / theta)
    B = torch.where(
        small,
        0.5 - theta2 / 24.0 + (theta2 ** 2) / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp(min=1e-12),
    )

    wx, wy, wz = aa_flat[:, 0], aa_flat[:, 1], aa_flat[:, 2]
    zero = torch.zeros_like(wx)
    K = torch.stack([
        torch.stack([zero, -wz, wy], dim=-1),
        torch.stack([wz, zero, -wx], dim=-1),
        torch.stack([-wy, wx, zero], dim=-1),
    ], dim=-2)

    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0)
    R = eye + A.unsqueeze(-1) * K + B.unsqueeze(-1) * torch.bmm(K, K)
    return R.reshape(*orig_shape, 3, 3)


def rotmat_to_axis_angle_np(rotmats: np.ndarray) -> np.ndarray:
    """Convert (N, 15, 3, 3) MANO rotation matrices to (N, 45) axis-angle vectors."""
    N = rotmats.shape[0]
    aa = np.zeros((N, 45), dtype=np.float32)
    for i in range(N):
        for j in range(15):
            rvec, _ = cv2.Rodrigues(rotmats[i, j].astype(np.float64))
            aa[i, 3 * j : 3 * j + 3] = rvec.reshape(3)
    return aa


def forward_mano_and_project(
    mano_layer,
    hand_pose_aa: torch.Tensor,      # (1, 45)
    global_orient: torch.Tensor,     # (1, 1, 3, 3)
    betas: torch.Tensor,             # (1, 10)
    cam_t: torch.Tensor,             # (1, 3)
    is_right: int,
    scaled_focal: float,
    img_w: float,
    img_h: float,
):
    """Run differentiable MANO forward kinematics and project 3D joints onto the 2D image plane."""
    hand_pose_rm = axis_angle_to_rotmat(hand_pose_aa.view(1, 15, 3))
    mano_out = mano_layer(
        global_orient=global_orient,
        hand_pose=hand_pose_rm,
        betas=betas,
        pose2rot=False,
    )

    # Mirror X-axis back to left-hand camera space when is_right == 0
    sign_x = 1.0 if int(is_right) == 1 else -1.0
    jl = mano_out.joints
    vl = mano_out.vertices
    joints_local = torch.stack([jl[..., 0] * sign_x, jl[..., 1], jl[..., 2]], dim=-1)
    verts_local = torch.stack([vl[..., 0] * sign_x, vl[..., 1], vl[..., 2]], dim=-1)

    joints_global = joints_local + cam_t[:, None, :]
    verts_global = verts_local + cam_t[:, None, :]

    # Pinhole perspective projection to pixel coordinates
    z = joints_global[..., 2].clamp(min=1e-4)
    u = scaled_focal * (joints_global[..., 0] / z) + (img_w * 0.5)
    v = scaled_focal * (joints_global[..., 1] / z) + (img_h * 0.5)
    joints_2d = torch.stack([u, v], dim=-1)
    return joints_global, verts_global, joints_2d


def recover_baseline_mano_context(frame_paths, hand_bboxes, is_right_flags, device):
    """
    Re-extract fixed per-frame MANO wrist orientation (global_orient), hand shape (betas),
    camera translation (cam_t), and the differentiable MANO layer from WiLoR using cached bboxes.
    """
    ckpt_path = os.path.join(WILOR_ROOT, "pretrained_models/wilor_final.ckpt")
    cfg_path = os.path.join(WILOR_ROOT, "pretrained_models/model_config.yaml")
    wilor_model, model_cfg = load_wilor(cfg_path=cfg_path, checkpoint_path=ckpt_path)
    wilor_model = wilor_model.to(device).eval()
    wilor_model.requires_grad_(False)

    mano_layer = wilor_model.mano
    mano_faces_right = mano_layer.faces.astype(np.int32)
    mano_faces_left = mano_faces_right[:, [0, 2, 1]]

    global_orients, betas_list, hand_poses_rm, cam_ts = [], [], [], []
    scaled_focal_val = None
    img_w, img_h = None, None

    with torch.no_grad():
        for idx, img_path in enumerate(frame_paths):
            img_cv2 = cv2.imread(img_path)
            assert img_cv2 is not None, f"Failed to read frame: {img_path}"
            img_h, img_w = img_cv2.shape[:2]

            boxes_np = np.array([hand_bboxes[idx]], dtype=np.float32)
            right_np = np.array([int(bool(is_right_flags[idx]))], dtype=np.int64)
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes_np, right_np, rescale_factor=2.0)
            batch = recursive_to(next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False))), device)

            out = wilor_model(batch)
            multiplier = 2 * batch["right"] - 1
            pred_cam = out["pred_cam"].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            img_size = batch["img_size"].float()
            scaled_focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            scaled_focal_val = float(scaled_focal.item())

            cam_t = cam_crop_to_full(
                pred_cam, batch["box_center"].float(), batch["box_size"].float(), img_size, scaled_focal
            )
            global_orients.append(out["pred_mano_params"]["global_orient"].detach())
            betas_list.append(out["pred_mano_params"]["betas"].detach())
            hand_poses_rm.append(out["pred_mano_params"]["hand_pose"].detach())
            cam_ts.append(cam_t.detach())

    # Free WiLoR backbone memory and retain only the MANO kinematics layer
    del wilor_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "mano_layer": mano_layer,
        "faces_left": mano_faces_left,
        "faces_right": mano_faces_right,
        "global_orient": torch.cat(global_orients, dim=0),   # (N, 1, 3, 3)
        "betas": torch.cat(betas_list, dim=0),               # (N, 10)
        "hand_pose_rm": torch.cat(hand_poses_rm, dim=0),     # (N, 15, 3, 3)
        "cam_t": torch.cat(cam_ts, dim=0),                   # (N, 3)
        "scaled_focal": scaled_focal_val,
        "img_w": float(img_w),
        "img_h": float(img_h),
    }


def optimize_single_frame(
    aff_model,
    schedule,
    mano_layer,
    x0_init: torch.Tensor,          # (1, 45) raw MANO axis-angle
    hand_mean: torch.Tensor,        # (1, 45) MANO natural rest-hand mean
    pose_sigma: float,              # isotropic Z-score scale (~0.13)
    global_orient: torch.Tensor,    # (1, 1, 3, 3)
    betas: torch.Tensor,            # (1, 10)
    cam_t: torch.Tensor,            # (1, 3)
    is_right: int,
    scaled_focal: float,
    img_w: float,
    img_h: float,
    target_j2d: torch.Tensor,       # (1, 21, 2)
    vis_mask: torch.Tensor,         # (1, 21) 1=visible, 0=occluded
    txt_tok: torch.Tensor,          # (1, 64, 768)
    txt_mask: torch.Tensor,         # (1, 64)
    img_tok: torch.Tensor,          # (1, 192, 1280)
    occ_thresh: int,
    n_min: int,
    n_max: int,
    step_m: int,
    lambda_2d: float,
    post_steps: int,
    post_lr: float,
    post_reg: float,
    use_img_pe: bool,
    generator: torch.Generator,
):
    """
    Occlusion-aware two-stage MAP pose optimization in normalized MANO offset space:
      z = (x - hand_mean) / pose_sigma
    1. Occlusion gating: preserve clean unoccluded frames (num_occ < occ_thresh) untouched.
    2. Denoise in Z-space using frozen AffHandGen (Post-Attn + Pre-FF) conditioned on text + image.
    3. Project visible joints back onto 2D observations while softly anchoring occluded joints to the prior.
    """
    num_occ = int((vis_mask[0] < 0.5).sum().item())
    has_vis = bool(vis_mask.sum().item() > 0.5)

    # Occlusion-aware gating: do not inject noise into clean unoccluded frames
    if num_occ < occ_thresh:
        x0_final = x0_init.detach().clone()
        with torch.no_grad():
            j3d_out, verts_out, j2d_out = forward_mano_and_project(
                mano_layer, x0_final, global_orient, betas, cam_t, is_right, scaled_focal, img_w, img_h
            )
            vis_px_err = float(
                (vis_mask * torch.norm(j2d_out - target_j2d, dim=-1)).sum().item()
                / vis_mask.sum().clamp(min=1.0).item()
            )
        return x0_final, j3d_out, verts_out, j2d_out, 0, 0, vis_px_err, 0.0

    # Adaptive starting diffusion step proportional to occluded joint ratio
    n_start = int(round(n_min + (n_max - n_min) * (num_occ / 21.0)))
    n_start = max(1, min(n_start, schedule.num_timesteps - 1))

    # Stage 1: Diffusion prior denoising in normalized MANO offset space
    z0_init = (x0_init - hand_mean) / pose_sigma
    eps = torch.randn(z0_init.shape, generator=generator, device=z0_init.device, dtype=z0_init.dtype)
    t_n = torch.tensor([n_start], dtype=torch.long, device=z0_init.device)
    z_t = schedule.q_sample(z0_init, t_n, noise=eps)

    steps = list(range(n_start, 0, -step_m)) or [n_start]
    z0_hat = z0_init.clone()
    guid_deltas = []

    for i, t_cur in enumerate(steps):
        t_prev = steps[i + 1] if (i + 1) < len(steps) else -1
        t_tensor = torch.tensor([t_cur], dtype=torch.long, device=z0_init.device)

        with torch.no_grad():
            z0_hat = aff_model(
                z_t,
                t_tensor,
                text_tokens=txt_tok,
                text_attention_mask=txt_mask,
                image_tokens=img_tok,
                use_img_pe=use_img_pe,
            )

        # Optional in-loop 2D guidance (default lambda_2d=0.0 for clean two-stage MAP)
        if lambda_2d > 0.0 and has_vis:
            x0_opt = (z0_hat * pose_sigma + hand_mean).detach().clone().requires_grad_(True)
            _, _, j2d_hat = forward_mano_and_project(
                mano_layer, x0_opt, global_orient, betas, cam_t, is_right, scaled_focal, img_w, img_h
            )
            sq_err_px = ((j2d_hat - target_j2d) ** 2).sum(dim=-1)
            loss_2d = (vis_mask * sq_err_px).sum() / vis_mask.sum().clamp(min=1.0)
            grad = torch.autograd.grad(loss_2d, x0_opt)[0]
            delta = (lambda_2d * grad).clamp(-0.08, 0.08)
            guid_deltas.append(float(delta.abs().mean().item()))
            x0_upd = (x0_opt - delta).detach()
            z0_hat = (x0_upd - hand_mean) / pose_sigma

        z_t = schedule.ddim_step(z_t, z0_hat, t=t_cur, t_prev=t_prev).detach()

    # Map denoised prior back to raw MANO axis-angle space
    x0_final = (z0_hat * pose_sigma + hand_mean).detach().clone()

    # Stage 2: MAP post-optimization fitting visible 2D keypoints with L2 prior regularization
    if post_steps > 0 and has_vis:
        x0_prior = x0_final.detach().clone()
        x0_param = x0_final.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([x0_param], lr=post_lr)
        for _ in range(post_steps):
            opt.zero_grad()
            _, _, j2d_hat = forward_mano_and_project(
                mano_layer, x0_param, global_orient, betas, cam_t, is_right, scaled_focal, img_w, img_h
            )
            sq_err_px = ((j2d_hat - target_j2d) ** 2).sum(dim=-1)
            loss_2d = (vis_mask * sq_err_px).sum() / vis_mask.sum().clamp(min=1.0)
            loss_reg = post_reg * F.mse_loss(x0_param, x0_prior)
            (loss_2d + loss_reg).backward()
            opt.step()
        x0_final = x0_param.detach()

    with torch.no_grad():
        j3d_out, verts_out, j2d_out = forward_mano_and_project(
            mano_layer, x0_final, global_orient, betas, cam_t, is_right, scaled_focal, img_w, img_h
        )
        vis_px_err = float(
            (vis_mask * torch.norm(j2d_out - target_j2d, dim=-1)).sum().item()
            / vis_mask.sum().clamp(min=1.0).item()
        )

    mean_guid_delta = float(np.mean(guid_deltas)) if guid_deltas else 0.0
    return x0_final, j3d_out, verts_out, j2d_out, n_start, len(steps), vis_px_err, mean_guid_delta


def main():
    parser = argparse.ArgumentParser(description="Occlusion-Aware Affordance-Guided 3D Hand Pose Optimization (Ours)")
    parser.add_argument("--data_dir", type=str, default="./data/dexycb_025_mug", help="Sequence frame directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Upstream/downstream results directory")
    parser.add_argument(
        "--mode",
        choices=["standard", "heavy_occ"],
        default="standard",
        help="Denoising schedule mode: standard (low-noise MAP) or heavy_occ (deeper prior recovery)",
    )
    parser.add_argument("--occ_thresh", type=int, default=1, help="Minimum occluded joints to trigger diffusion")
    parser.add_argument("--pose_sigma", type=float, default=0.13, help="MANO offset Z-score scaling sigma")
    parser.add_argument("--n_min", type=int, default=None, help="Minimum starting diffusion step")
    parser.add_argument("--n_max", type=int, default=None, help="Maximum starting diffusion step")
    parser.add_argument("--step_m", type=int, default=None, help="Reverse diffusion stride")
    parser.add_argument("--lambda_2d", type=float, default=None, help="In-loop 2D guidance weight (0.0 = two-stage MAP)")
    parser.add_argument("--post_steps", type=int, default=100, help="Post-processing visible 2D fitting iterations")
    parser.add_argument("--post_lr", type=float, default=5e-3, help="Post-processing Adam learning rate")
    parser.add_argument("--post_reg", type=float, default=1e-2, help="Prior anchor weight in Stage 2 MAP")
    parser.add_argument("--use_img_pe", action="store_true", help="Add positional encoding to HaMeR visual tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible noise sampling")
    parser.add_argument("--ckpt", type=str, default="weights/affhandgen/hog_ckpt_best.pth", help="Diffusion prior checkpoint")
    args = parser.parse_args()

    if args.mode == "standard":
        n_min = 0 if args.n_min is None else args.n_min
        n_max = 20 if args.n_max is None else args.n_max
        step_m = 3 if args.step_m is None else args.step_m
        lambda_2d = 0.0 if args.lambda_2d is None else args.lambda_2d
    else:
        n_min = 20 if args.n_min is None else args.n_min
        n_max = 120 if args.n_max is None else args.n_max
        step_m = 10 if args.step_m is None else args.step_m
        lambda_2d = 0.0 if args.lambda_2d is None else args.lambda_2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("[Module 5] Occlusion-Aware Affordance-Guided Pose Optimization (Baseline vs. Ours)")
    print(
        f"  Mode={args.mode} | sigma={args.pose_sigma} | occ_thresh={args.occ_thresh} | "
        f"n=[{n_min}, {n_max}] | stride={step_m} | lambda_2d={lambda_2d} | post={args.post_steps}"
    )
    print("=" * 78)

    # 1. Load sequence frames, baseline cache, occlusion masks, and multimodal condition tokens
    data_dir = args.data_dir
    res_dir = args.results_dir
    frame_paths = sorted(glob.glob(os.path.join(data_dir, "frame_*.jpg")))
    num_frames = len(frame_paths)
    assert num_frames > 0, f"No frames found in {data_dir}"

    with open(os.path.join(res_dir, "detected_hand_bboxes.json"), "r") as f:
        hand_meta = json.load(f)
    hand_bboxes = hand_meta["bboxes"]
    raw_right = hand_meta.get(
        "per_frame_is_right",
        hand_meta.get("is_right_per_frame", hand_meta.get("is_right", False)),
    )
    is_right_flags = [bool(v) for v in raw_right] if isinstance(raw_right, list) else [bool(raw_right)] * num_frames
    majority_is_right = sum(is_right_flags) > (num_frames / 2.0)

    baseline_j3d_cached = np.load(os.path.join(res_dir, "baseline_joints_3d.npy"))
    baseline_j2d_cached = np.load(os.path.join(res_dir, "baseline_joints_2d.npy"))
    vis_mask_np = np.load(os.path.join(res_dir, "joint_visibility_mask.npy")).astype(np.float32)
    cond = np.load(os.path.join(res_dir, "condition_features.npz"))
    eval_gt = torch.load(os.path.join(data_dir, "eval_gt.pt"), map_location="cpu", weights_only=False)
    gt_joints_3d = eval_gt["joint_3d"].numpy()

    assert str(cond["vision_encoder"].item()) == "hamer_vit_huge_16", (
        f"Expected HaMeR ViT features, found {cond['vision_encoder']}"
    )

    # 2. Recover differentiable MANO context and verify exact alignment with cached baseline
    ctx = recover_baseline_mano_context(frame_paths, hand_bboxes, is_right_flags, device)
    x0_init_np = rotmat_to_axis_angle_np(ctx["hand_pose_rm"].detach().cpu().numpy())
    x0_init_all = torch.from_numpy(x0_init_np).to(device)
    hand_mean = ctx["mano_layer"].hand_mean.detach().to(device).view(1, 45)

    chk_j3d_list, chk_j2d_list = [], []
    with torch.no_grad():
        for i in range(num_frames):
            j3d_i, _, j2d_i = forward_mano_and_project(
                ctx["mano_layer"],
                x0_init_all[i : i + 1],
                ctx["global_orient"][i : i + 1],
                ctx["betas"][i : i + 1],
                ctx["cam_t"][i : i + 1],
                int(is_right_flags[i]),
                ctx["scaled_focal"],
                ctx["img_w"],
                ctx["img_h"],
            )
            chk_j3d_list.append(j3d_i[0].cpu().numpy())
            chk_j2d_list.append(j2d_i[0].cpu().numpy())

    chk_j3d_all = np.stack(chk_j3d_list, axis=0)
    chk_j2d_all = np.stack(chk_j2d_list, axis=0)

    diff_3d_mm = float(np.abs(chk_j3d_all - baseline_j3d_cached).max() * 1000.0)
    diff_2d_px = float(np.abs(chk_j2d_all - baseline_j2d_cached).max())
    print(
        f"[CHECK] Differentiable MANO vs Baseline cache (all {num_frames} frames): "
        f"max_3d_diff={diff_3d_mm:.4f} mm | max_2d_diff={diff_2d_px:.4f} px"
    )
    assert diff_3d_mm < 0.2, f"3D MANO alignment mismatch ({diff_3d_mm:.4f} mm)!"

    # 3. Load frozen diffusion prior and move conditioning tensors to GPU
    aff_model, schedule = load_affhandgen(args.ckpt, device=device)
    txt_tokens_all = torch.from_numpy(cond["text_tokens"]).to(device)
    txt_mask_all = torch.from_numpy(cond["text_attention_mask"]).to(device)
    img_tokens_all = torch.from_numpy(cond["image_tokens"]).to(device)
    vis_mask_all = torch.from_numpy(vis_mask_np).to(device)

    # 4. Run frame-by-frame pose optimization (ground truth is never accessed here)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    ours_poses_aa, ours_poses_rm, ours_j3d_list, ours_verts_list, ours_j2d_list = [], [], [], [], []
    step_counts, vis_px_errs, guid_delta_list = [], [], []

    for t_idx in range(num_frames):
        target_j2d = torch.from_numpy(chk_j2d_all[t_idx : t_idx + 1]).to(device)

        x0_ours, j3d_ours, verts_ours, j2d_ours, n_start, n_steps, vis_px, mean_dtheta = optimize_single_frame(
            aff_model=aff_model,
            schedule=schedule,
            mano_layer=ctx["mano_layer"],
            x0_init=x0_init_all[t_idx : t_idx + 1],
            hand_mean=hand_mean,
            pose_sigma=args.pose_sigma,
            global_orient=ctx["global_orient"][t_idx : t_idx + 1],
            betas=ctx["betas"][t_idx : t_idx + 1],
            cam_t=ctx["cam_t"][t_idx : t_idx + 1],
            is_right=int(is_right_flags[t_idx]),
            scaled_focal=ctx["scaled_focal"],
            img_w=ctx["img_w"],
            img_h=ctx["img_h"],
            target_j2d=target_j2d,
            vis_mask=vis_mask_all[t_idx : t_idx + 1],
            txt_tok=txt_tokens_all[t_idx : t_idx + 1],
            txt_mask=txt_mask_all[t_idx : t_idx + 1],
            img_tok=img_tokens_all[t_idx : t_idx + 1],
            occ_thresh=args.occ_thresh,
            n_min=n_min,
            n_max=n_max,
            step_m=step_m,
            lambda_2d=lambda_2d,
            post_steps=args.post_steps,
            post_lr=args.post_lr,
            post_reg=args.post_reg,
            use_img_pe=args.use_img_pe,
            generator=gen,
        )

        with torch.no_grad():
            pose_rm_flat = axis_angle_to_rotmat(x0_ours.view(1, 15, 3))[0].cpu().numpy().reshape(-1)

        ours_poses_aa.append(x0_ours[0].cpu().numpy())
        ours_poses_rm.append(pose_rm_flat)
        ours_j3d_list.append(j3d_ours[0].cpu().numpy())
        ours_verts_list.append(verts_ours[0].cpu().numpy())
        ours_j2d_list.append(j2d_ours[0].cpu().numpy())
        step_counts.append(n_steps)
        vis_px_errs.append(vis_px)
        guid_delta_list.append(mean_dtheta)

    ours_poses_aa = np.stack(ours_poses_aa, axis=0).astype(np.float32)
    ours_poses_rm = np.stack(ours_poses_rm, axis=0).astype(np.float32)
    ours_j3d = np.stack(ours_j3d_list, axis=0).astype(np.float32)
    ours_verts = np.stack(ours_verts_list, axis=0).astype(np.float32)
    ours_j2d = np.stack(ours_j2d_list, axis=0).astype(np.float32)

    # 5. Evaluate 3D PA-MPJPE comparison: Baseline (WiLoR) vs. Ours
    base_errs = compute_pa_mpjpe(baseline_j3d_cached, gt_joints_3d)
    ours_errs = compute_pa_mpjpe(ours_j3d, gt_joints_3d)

    base_mean, ours_mean = float(base_errs.mean()), float(ours_errs.mean())
    hi_slice = slice(13, num_frames) if num_frames > 13 else slice(0, num_frames)
    ct_slice = slice(10, min(17, num_frames)) if num_frames > 10 else slice(0, num_frames)
    base_hi, ours_hi = float(base_errs[hi_slice].mean()), float(ours_errs[hi_slice].mean())
    base_contact, ours_contact = float(base_errs[ct_slice].mean()), float(ours_errs[ct_slice].mean())
    delta_mean = ours_mean - base_mean
    delta_hi = ours_hi - base_hi

    print("\n=== Per-Frame PA-MPJPE Comparison: Baseline vs. Ours (mm) ===")
    print(
        f"{'Frame':>5s} {'Occ':>4s} {'Steps':>5s} | {'Baseline':>8s} {'Ours':>8s} {'Delta':>8s} | "
        f"{'Vis2D(px)':>9s} {'Guid(|dθ|)':>10s}"
    )
    print("-" * 72)
    for t_idx in range(num_frames):
        occ = int((vis_mask_np[t_idx] < 0.5).sum())
        d = ours_errs[t_idx] - base_errs[t_idx]
        print(
            f"{t_idx:5d} {occ:4d} {step_counts[t_idx]:5d} | "
            f"{base_errs[t_idx]:8.2f} {ours_errs[t_idx]:8.2f} {d:+8.2f} | "
            f"{vis_px_errs[t_idx]:9.2f} {guid_delta_list[t_idx]:10.4f}"
        )

    print("\n" + "=" * 78)
    print(f"★ Mean PA-MPJPE (0-{num_frames - 1}):        Baseline = {base_mean:.2f} mm  ->  Ours = {ours_mean:.2f} mm  ({delta_mean:+.2f} mm)")
    print(f"★ High-Occ PA-MPJPE (13-{num_frames - 1}):   Baseline = {base_hi:.2f} mm  ->  Ours = {ours_hi:.2f} mm  ({delta_hi:+.2f} mm)")
    print(f"★ Peak-Contact (10-16):        Baseline = {base_contact:.2f} mm  ->  Ours = {ours_contact:.2f} mm  ({ours_contact - base_contact:+.2f} mm)")
    print(
        f"★ Mean Visible 2D Error:       {np.mean(vis_px_errs):.2f} px | "
        f"Avg Guid |dθ|: {np.mean(guid_delta_list):.4f} rad | Avg steps: {np.mean(step_counts):.1f}"
    )
    print("=" * 78)

    # 6. Save Ours outputs symmetrically with baseline_* files
    os.makedirs(res_dir, exist_ok=True)
    np.save(os.path.join(res_dir, "ours_joints_3d.npy"), ours_j3d)
    np.save(os.path.join(res_dir, "ours_joints_2d.npy"), ours_j2d)
    np.save(os.path.join(res_dir, "ours_human_pose_mano.npy"), {
        "pose": ours_poses_rm,
        "pose_aa": ours_poses_aa,
        "global_orient": ctx["global_orient"].cpu().numpy(),
        "betas": ctx["betas"].cpu().numpy(),
        "joints3d": ours_j3d,
        "vertices": ours_verts,
        "faces": ctx["faces_right"] if majority_is_right else ctx["faces_left"],
        "cam_t": ctx["cam_t"].cpu().numpy(),
        "scaled_focal": ctx["scaled_focal"],
    })

    metrics = {
        "metric_name": "PA-MPJPE",
        "unit": "mm",
        "config": {
            "mode": args.mode,
            "ckpt": args.ckpt,
            "pose_sigma": args.pose_sigma,
            "occ_thresh": args.occ_thresh,
            "n_min": n_min,
            "n_max": n_max,
            "step_m": step_m,
            "lambda_2d": lambda_2d,
            "post_steps": args.post_steps,
            "post_lr": args.post_lr,
            "post_reg": args.post_reg,
            "use_img_pe": args.use_img_pe,
            "seed": args.seed,
        },
        "baseline_mean_pa_mpjpe_mm": round(base_mean, 2),
        "ours_mean_pa_mpjpe_mm": round(ours_mean, 2),
        "delta_mean_mm": round(delta_mean, 2),
        "baseline_high_occlusion_pa_mpjpe_mm": round(base_hi, 2),
        "ours_high_occlusion_pa_mpjpe_mm": round(ours_hi, 2),
        "delta_high_occlusion_mm": round(delta_hi, 2),
        "baseline_peak_contact_pa_mpjpe_mm": round(base_contact, 2),
        "ours_peak_contact_pa_mpjpe_mm": round(ours_contact, 2),
        "mean_visible_2d_error_px": round(float(np.mean(vis_px_errs)), 2),
        "instantaneous_baseline_mm": [round(float(e), 2) for e in base_errs],
        "instantaneous_ours_mm": [round(float(e), 2) for e in ours_errs],
    }
    with open(os.path.join(res_dir, "ours_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[SUCCESS] Saved {res_dir}/ours_joints_3d.npy, ours_human_pose_mano.npy, ours_metrics.json")


if __name__ == "__main__":
    main()