import argparse
import glob
import os
import cv2
import numpy as np


def read_video_frames(video_path: str) -> list:
    assert os.path.exists(video_path), f"Missing video file: {video_path}"
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) > 0, f"No frames decoded from: {video_path}"
    return frames


def load_raw_frames(raw_video: str, data_dir: str) -> list:
    if raw_video and os.path.exists(raw_video):
        return read_video_frames(raw_video)
    paths = sorted(glob.glob(os.path.join(data_dir, "frame_*.jpg")))
    assert len(paths) > 0, f"No frame_*.jpg found in {data_dir}"
    frames = [cv2.imread(p) for p in paths]
    assert all(f is not None for f in frames), "Failed to read one or more raw frames"
    return frames


def make_titled_panel(
    frame: np.ndarray,
    title: str,
    panel_w: int,
    panel_h: int,
    header_h: int = 46,
    accent_bgr: tuple = (210, 215, 225),
) -> np.ndarray:
    """Resize panel cleanly and attach a uniform top header bar (zero overlap with content)."""
    resized = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    header = np.full((header_h, panel_w, 3), (18, 21, 28), dtype=np.uint8)
    # Uniform accent bottom line
    cv2.line(header, (0, header_h - 2), (panel_w, header_h - 2), accent_bgr, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.66
    thickness = 2
    (tw, th), _ = cv2.getTextSize(title, font, font_scale, thickness)
    tx = max(12, (panel_w - tw) // 2)
    ty = (header_h + th) // 2 - 2
    cv2.putText(header, title, (tx, ty), font, font_scale, (242, 245, 250), thickness, cv2.LINE_AA)

    return np.vstack([header, resized])


def main():
    parser = argparse.ArgumentParser(description="Compose 2x2 Quad Comparison Demo with Unified Titles")
    parser.add_argument("--data_dir", type=str, default="./data/dexycb_025_mug", help="Raw 30-frame image directory")
    parser.add_argument("--raw_video", type=str, default="", help="Optional raw mp4 path (overrides data_dir frames)")
    parser.add_argument("--ours_video", type=str, default="visualizations/ours_hand_mesh_3d.mp4")
    parser.add_argument("--baseline_video", type=str, default="visualizations/baseline_hand_mesh_3d.mp4")
    parser.add_argument("--curve_video", type=str, default="visualizations/pa_mpjpe_curve.mp4")
    parser.add_argument("--out_path", type=str, default="visualizations/quad_comparison_demo.mp4")
    parser.add_argument("--panel_w", type=int, default=640, help="Width of each quadrant content")
    parser.add_argument("--panel_h", type=int, default=480, help="Height of each quadrant content")
    parser.add_argument("--header_h", type=int, default=46, help="Uniform title bar height")
    parser.add_argument("--fps", type=float, default=30.0, help="Output FPS (30 frames @ 30 FPS = 1.00 s)")
    args = parser.parse_args()

    raw_frames = load_raw_frames(args.raw_video, args.data_dir)
    ours_frames = read_video_frames(args.ours_video)
    base_frames = read_video_frames(args.baseline_video)
    curve_frames = read_video_frames(args.curve_video)

    n_frames = min(len(raw_frames), len(ours_frames), len(base_frames), len(curve_frames))
    assert len(raw_frames) == len(ours_frames) == len(base_frames) == len(curve_frames), (
        f"Frame count mismatch! raw={len(raw_frames)}, ours={len(ours_frames)}, "
        f"baseline={len(base_frames)}, curve={len(curve_frames)}"
    )

    divider = 4
    quad_w = args.panel_w * 2 + divider
    quad_h = (args.panel_h + args.header_h) * 2 + divider
    v_div = np.full((args.panel_h + args.header_h, divider, 3), (45, 50, 62), dtype=np.uint8)
    h_div = np.full((divider, quad_w, 3), (45, 50, 62), dtype=np.uint8)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_path, fourcc, args.fps, (quad_w, quad_h))

    print("=" * 72)
    print(f"[Compose Quad] {n_frames} frames @ {args.fps:.0f} FPS ({n_frames/args.fps:.2f} s) | {quad_w}x{quad_h}")
    print("  Top-Left:     Input Monocular RGB")
    print("  Top-Right:    Ours (AffHandGen) 3D Error Heatmap")
    print("  Bottom-Left:  Real-Time PA-MPJPE Comparison Curve")
    print("  Bottom-Right: Baseline (WiLoR) 3D Error Heatmap")
    print("=" * 72)

    for i in range(n_frames):
        tl = make_titled_panel(
            raw_frames[i],
            "Input Monocular RGB",
            args.panel_w, args.panel_h, args.header_h,
            accent_bgr=(200, 205, 215),  # Neutral Silver
        )
        tr = make_titled_panel(
            ours_frames[i],
            "Ours (AffHandGen) 3D Error Heatmap",
            args.panel_w, args.panel_h, args.header_h,
            accent_bgr=(118, 230, 0),    # Emerald Green (BGR)
        )
        bl = make_titled_panel(
            curve_frames[i],
            "Per-Frame PA-MPJPE (Baseline vs. Ours)",
            args.panel_w, args.panel_h, args.header_h,
            accent_bgr=(230, 190, 90),   # Clean Steel Blue (BGR)
        )
        br = make_titled_panel(
            base_frames[i],
            "Baseline (WiLoR) 3D Error Heatmap",
            args.panel_w, args.panel_h, args.header_h,
            accent_bgr=(75, 75, 255),    # Coral Red (BGR)
        )

        top_row = np.hstack([tl, v_div, tr])
        bot_row = np.hstack([bl, v_div, br])
        full_frame = np.vstack([top_row, h_div, bot_row])
        writer.write(full_frame)

    writer.release()
    print(f"\n[SUCCESS] Quad comparison demo saved: {args.out_path}")


if __name__ == "__main__":
    main()