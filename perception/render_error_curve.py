import argparse
import glob
import os
import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_pa_mpjpe_seq(pred_j3d: np.ndarray, gt_j3d: np.ndarray) -> np.ndarray:
    """Compute per-frame Procrustes-aligned MPJPE in mm."""
    scale_unit = 1000.0 if np.abs(gt_j3d).max() < 10.0 else 1.0
    errs = np.zeros(len(pred_j3d), dtype=np.float64)
    for i in range(len(pred_j3d)):
        X = pred_j3d[i].astype(np.float64)
        Y = gt_j3d[i].astype(np.float64)
        X0 = X - X.mean(axis=0, keepdims=True)
        Y0 = Y - Y.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(X0.T @ Y0)
        R = u @ vt
        if np.linalg.det(R) < 0:
            vt[-1, :] *= -1
            s[-1] *= -1
            R = u @ vt
        scale = s.sum() / (np.sum(X0 ** 2) + 1e-12)
        aligned = scale * (X0 @ R)
        errs[i] = np.linalg.norm(aligned - Y0, axis=-1).mean() * scale_unit
    return errs


def render_single_curve_frame(
    frame_idx: int,
    base_errs: np.ndarray,
    ours_errs: np.ndarray,
    fps: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """
    Render one frame of the real-time PA-MPJPE comparison chart (NO top title).
    4 curves:
      - Baseline real-time (solid red)   & Baseline 30-frame mean (dashed red)
      - Ours real-time     (solid green) & Ours 30-frame mean     (dashed green)
    With moving vertical timestamp tracking cursor.
    """
    n_frames = len(base_errs)
    frames = np.arange(n_frames)
    t_sec = frame_idx / float(fps)
    base_mean = float(np.mean(base_errs))
    ours_mean = float(np.mean(ours_errs))

    color_base = "#FF4B4B"  # Red for Baseline
    color_ours = "#00E676"  # Green for Ours
    bg_color = "#101218"
    panel_color = "#161A24"
    grid_color = "#2A3040"

    dpi = 150
    fig, ax = plt.subplots(figsize=(out_w / dpi, out_h / dpi), dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(panel_color)

    # Subtle full-sequence guide paths behind live trace
    ax.plot(frames, base_errs, color=color_base, lw=1.2, alpha=0.22)
    ax.plot(frames, ours_errs, color=color_ours, lw=1.2, alpha=0.22)

    # 1 & 2: 30-frame mean horizontal dashed lines
    ax.axhline(
        base_mean, color=color_base, linestyle="--", linewidth=1.6, alpha=0.90,
        label=f"Baseline Mean ({base_mean:.2f} mm)"
    )
    ax.axhline(
        ours_mean, color=color_ours, linestyle="--", linewidth=1.6, alpha=0.90,
        label=f"Ours Mean ({ours_mean:.2f} mm)"
    )

    # 3 & 4: Real-time solid curves traced up to current frame
    ax.plot(
        frames[: frame_idx + 1], base_errs[: frame_idx + 1],
        color=color_base, linestyle="-", linewidth=2.4,
        label="Baseline (Per-Frame)"
    )
    ax.plot(
        frames[: frame_idx + 1], ours_errs[: frame_idx + 1],
        color=color_ours, linestyle="-", linewidth=2.4,
        label="Ours (Per-Frame)"
    )

    # Moving vertical timestamp tracking cursor & live markers
    ax.axvline(frame_idx, color="#F8FAFC", linestyle=":", linewidth=1.5, alpha=0.90)
    ax.scatter([frame_idx], [base_errs[frame_idx]], color=color_base, edgecolor="white", s=55, zorder=6, linewidth=1.2)
    ax.scatter([frame_idx], [ours_errs[frame_idx]], color=color_ours, edgecolor="white", s=55, zorder=6, linewidth=1.2)

    y_max = max(10.0, float(max(base_errs.max(), ours_errs.max())) * 1.25)
    ax.set_xlim(-0.5, n_frames - 0.5)
    ax.set_ylim(0.0, y_max)

    # Live timestamp callout following the cursor
    ha = "left" if frame_idx < n_frames * 0.58 else "right"
    tx = frame_idx + (0.6 if ha == "left" else -0.6)
    ax.text(
        tx, y_max * 0.95,
        f"Frame {frame_idx:02d}/{n_frames - 1:02d}  |  t = {t_sec:.2f} s\n"
        f"Base: {base_errs[frame_idx]:.2f} mm   Ours: {ours_errs[frame_idx]:.2f} mm",
        color="#FFFFFF", fontsize=8.2, fontweight="bold", ha=ha, va="top",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#1E2433", edgecolor="#505A70", alpha=0.92),
        zorder=7,
    )

    ax.set_xlabel("Frame Index (30 FPS = 1.00 s)", color="#CBD5E1", fontsize=10.0)
    ax.set_ylabel("PA-MPJPE (mm)", color="#CBD5E1", fontsize=10.0)
    ax.set_xticks(np.arange(0, n_frames, max(1, n_frames // 6)))
    ax.tick_params(colors="#CBD5E1", labelsize=9.5)
    ax.grid(True, color=grid_color, linestyle="-", linewidth=0.6, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color("#3B4254")

    leg = ax.legend(
        loc="lower left", ncol=2, frameon=True, facecolor="#161A24",
        edgecolor="#3B4254", fontsize=9.0, labelcolor="#F1F5F9"
    )
    leg.get_frame().set_alpha(0.92)

    fig.tight_layout(pad=0.9)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    plt.close(fig)

    if bgr.shape[1] != out_w or bgr.shape[0] != out_h:
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return bgr


def main():
    parser = argparse.ArgumentParser(description="Render Standalone Real-Time PA-MPJPE Comparison Curve (No Title)")
    parser.add_argument("--data_dir", type=str, default="./data/dexycb_025_mug", help="Sequence directory with eval_gt.pt")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory with baseline/ours_joints_3d.npy")
    parser.add_argument("--out_path", type=str, default="visualizations/pa_mpjpe_curve.mp4", help="Output mp4 path")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS (30 frames @ 30 FPS = 1.00 s)")
    args = parser.parse_args()

    base_j3d = np.load(os.path.join(args.results_dir, "baseline_joints_3d.npy"))
    ours_j3d = np.load(os.path.join(args.results_dir, "ours_joints_3d.npy"))
    gt_j3d = torch.load(os.path.join(args.data_dir, "eval_gt.pt"), map_location="cpu", weights_only=False)["joint_3d"].numpy()

    base_errs = compute_pa_mpjpe_seq(base_j3d, gt_j3d)
    ours_errs = compute_pa_mpjpe_seq(ours_j3d, gt_j3d)
    num_frames = len(base_errs)

    # Match exact aspect ratio of raw sequence frames at crisp resolution
    frame_paths = sorted(glob.glob(os.path.join(args.data_dir, "frame_*.jpg")))
    if frame_paths:
        sample = cv2.imread(frame_paths[0])
        h, w = sample.shape[:2]
        out_w, out_h = w * 2, h * 2 if w <= 640 else (w, h)
        if w > 640:
            out_w, out_h = w, h
    else:
        out_w, out_h = 960, 720

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_path, fourcc, args.fps, (out_w, out_h))

    print("=" * 70)
    print(f"[Render Curve] {num_frames} frames @ {args.fps:.0f} FPS ({num_frames/args.fps:.2f} s) | {out_w}x{out_h}")
    print(f"  Baseline Mean (Red Dashed):   {base_errs.mean():.2f} mm")
    print(f"  Ours Mean     (Green Dashed): {ours_errs.mean():.2f} mm")
    print("=" * 70)

    for i in range(num_frames):
        frame_bgr = render_single_curve_frame(i, base_errs, ours_errs, fps=args.fps, out_w=out_w, out_h=out_h)
        writer.write(frame_bgr)
        if (i + 1) % 10 == 0 or i == num_frames - 1:
            print(f"  --> Rendered Curve Frame [{i + 1:02d}/{num_frames:02d}] | t={i/args.fps:.2f}s")

    writer.release()
    print(f"\n[SUCCESS] Real-time error curve video saved: {args.out_path}")


if __name__ == "__main__":
    main()