import argparse
import glob
import os
import cv2
import torch
import numpy as np


def subdivide_mesh_with_error(vertices: np.ndarray, faces: np.ndarray, vert_errors: np.ndarray):
    """Subdivide 3D triangles (1538 -> 6152 faces) and interpolate per-vertex error values."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    e0, e1, e2 = vert_errors[faces[:, 0]], vert_errors[faces[:, 1]], vert_errors[faces[:, 2]]

    m01 = 0.5 * (v0 + v1)
    m12 = 0.5 * (v1 + v2)
    m20 = 0.5 * (v2 + v0)

    em01 = 0.5 * (e0 + e1)
    em12 = 0.5 * (e1 + e2)
    em20 = 0.5 * (e2 + e0)

    n_verts = len(vertices)
    n_faces = len(faces)

    idx_m01 = np.arange(n_verts, n_verts + n_faces)
    idx_m12 = np.arange(n_verts + n_faces, n_verts + 2 * n_faces)
    idx_m20 = np.arange(n_verts + 2 * n_faces, n_verts + 3 * n_faces)

    new_vertices = np.vstack([vertices, m01, m12, m20])
    new_errors = np.concatenate([vert_errors, em01, em12, em20])

    f0 = np.stack([faces[:, 0], idx_m01, idx_m20], axis=-1)
    f1 = np.stack([idx_m01, faces[:, 1], idx_m12], axis=-1)
    f2 = np.stack([idx_m20, idx_m12, faces[:, 2]], axis=-1)
    f3 = np.stack([idx_m01, idx_m12, idx_m20], axis=-1)

    new_faces = np.vstack([f0, f1, f2, f3])
    return new_vertices, new_faces, new_errors


def error_to_jet_color(errors: np.ndarray, max_err: float = 10.0) -> np.ndarray:
    """Map Euclidean error (0 to max_err mm) to high-sensitivity Jet colormap."""
    norm = np.clip(errors / max_err, 0.0, 1.0)
    colors = np.zeros((len(errors), 3), dtype=np.float32)

    r1 = norm < 0.25
    colors[r1, 0] = 0.0
    colors[r1, 1] = 4.0 * norm[r1]
    colors[r1, 2] = 1.0

    r2 = (norm >= 0.25) & (norm < 0.5)
    colors[r2, 0] = 0.0
    colors[r2, 1] = 1.0
    colors[r2, 2] = 1.0 - 4.0 * (norm[r2] - 0.25)

    r3 = (norm >= 0.5) & (norm < 0.75)
    colors[r3, 0] = 4.0 * (norm[r3] - 0.5)
    colors[r3, 1] = 1.0
    colors[r3, 2] = 0.0

    r4 = norm >= 0.75
    colors[r4, 0] = 1.0
    colors[r4, 1] = 1.0 - 4.0 * (norm[r4] - 0.75)
    colors[r4, 2] = 0.0

    return colors * 255.0


def render_shaded_error_mesh(
    canvas: np.ndarray,
    vertices_3d: np.ndarray,
    faces: np.ndarray,
    vert_errors: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    max_err: float = 10.0,
) -> np.ndarray:
    """Rasterize hand mesh with per-vertex error colors and Blinn-Phong lighting."""
    h, w, _ = canvas.shape
    ss = 2
    ss_h, ss_w = h * ss, w * ss
    ss_canvas = cv2.resize(canvas, (ss_w, ss_h), interpolation=cv2.INTER_LINEAR)

    verts, sub_faces, sub_errs = subdivide_mesh_with_error(vertices_3d, faces, vert_errors)

    z = np.clip(verts[:, 2], 1e-4, None)
    u = (focal * (verts[:, 0] / z) + cx) * ss
    v = (focal * (verts[:, 1] / z) + cy) * ss
    pts_2d = np.stack([u, v], axis=-1).astype(np.int32)

    tri_v0 = verts[sub_faces[:, 0]]
    tri_v1 = verts[sub_faces[:, 1]]
    tri_v2 = verts[sub_faces[:, 2]]

    normals = np.cross(tri_v1 - tri_v0, tri_v2 - tri_v0)
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)

    tri_centers = (tri_v0 + tri_v1 + tri_v2) / 3.0
    view_dirs = -tri_centers / (np.linalg.norm(tri_centers, axis=-1, keepdims=True) + 1e-8)

    key_light = np.array([-0.35, -0.45, -0.82])
    key_light /= np.linalg.norm(key_light)
    fill_light = np.array([0.55, 0.25, -0.79])
    fill_light /= np.linalg.norm(fill_light)

    diff_key = np.clip(np.sum(normals * key_light[None, :], axis=-1), 0.0, 1.0)
    diff_fill = np.clip(np.sum(normals * fill_light[None, :], axis=-1), 0.0, 1.0)
    rim = (1.0 - np.clip(np.sum(normals * view_dirs, axis=-1), 0.0, 1.0)) ** 2.2

    face_errs = (sub_errs[sub_faces[:, 0]] + sub_errs[sub_faces[:, 1]] + sub_errs[sub_faces[:, 2]]) / 3.0
    face_base_rgb = error_to_jet_color(face_errs, max_err=max_err)

    intensity = 0.35 + 0.55 * diff_key + 0.15 * diff_fill + 0.10 * rim
    face_rgb = face_base_rgb * intensity[:, None]
    face_bgr = np.clip(face_rgb[:, ::-1], 0, 255).astype(np.uint8)

    sort_order = np.argsort(tri_centers[:, 2])[::-1]
    tri_pts_2d = pts_2d[sub_faces]

    for f_idx in sort_order:
        poly = tri_pts_2d[f_idx]
        color = tuple(int(c) for c in face_bgr[f_idx])
        cv2.fillConvexPoly(ss_canvas, poly, color, lineType=cv2.LINE_AA)

    return cv2.resize(ss_canvas, (w, h), interpolation=cv2.INTER_AREA)


def draw_scientific_colorbar(canvas: np.ndarray, x: int, y: int, bar_w: int, bar_h: int, max_err: float = 10.0):
    """Draw vertical Jet colorbar calibrated to [0, max_err] mm (matching render_baseline.py)."""
    for i in range(bar_h):
        norm_val = 1.0 - (i / float(bar_h))
        color_rgb = error_to_jet_color(np.array([norm_val * max_err]), max_err=max_err)[0]
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        cv2.line(canvas, (x, y + i), (x + bar_w, y + i), color_bgr, 1)

    cv2.rectangle(canvas, (x - 1, y - 1), (x + bar_w + 1, y + bar_h + 1), (220, 220, 220), 1)

    ticks = [
        (0.0, f"{max_err:.0f} mm"),
        (0.5, f"{max_err / 2.0:.0f} mm"),
        (1.0, "0 mm"),
    ]
    for ratio, label in ticks:
        tick_y = int(y + ratio * bar_h)
        cv2.line(canvas, (x + bar_w + 1, tick_y), (x + bar_w + 8, tick_y), (220, 220, 220), 1)
        cv2.putText(
            canvas, label, (x + bar_w + 14, tick_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA
        )

    cv2.putText(canvas, "Vertex", (x - 8, y - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Error", (x - 4, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)


def draw_studio_background(w: int, h: int) -> np.ndarray:
    """Create sleek dark studio background with subtle spatial grid."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    y_grad = np.linspace(24, 12, h).astype(np.uint8)
    bg[:, :, 0] = y_grad[:, None] + 4
    bg[:, :, 1] = y_grad[:, None] + 2
    bg[:, :, 2] = y_grad[:, None]

    grid_step = 80
    for gx in range(0, w, grid_step):
        cv2.line(bg, (gx, 0), (gx, h), (32, 30, 28), 1)
    for gy in range(0, h, grid_step):
        cv2.line(bg, (0, gy), (w, gy), (32, 30, 28), 1)
    return bg


def main():
    parser = argparse.ArgumentParser(description="Render Ours 3D Vertex Error Heatmap (matching render_baseline.py)")
    parser.add_argument("--data_dir", type=str, default="./data/dexycb_025_mug", help="Sequence frame directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing ours_human_pose_mano.npy")
    parser.add_argument("--out_path", type=str, default="visualizations/ours_hand_mesh_3d.mp4", help="Output mp4 path")
    parser.add_argument("--max_err", type=float, default=10.0, help="Colorbar upper bound in mm")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS")
    args = parser.parse_args()

    print("=" * 70)
    print(f"[Render Ours] 3D Vertex Error Heatmap (Calibrated: 0 - {args.max_err:.0f} mm)")
    print("=" * 70)

    mano_path = os.path.join(args.results_dir, "ours_human_pose_mano.npy")
    gt_path = os.path.join(args.data_dir, "eval_gt.pt")
    assert os.path.exists(mano_path), f"Missing MANO file: {mano_path}"
    assert os.path.exists(gt_path), f"Missing GT file: {gt_path}"

    mano_data = np.load(mano_path, allow_pickle=True).item()
    verts_seq = mano_data["vertices"]
    joints_seq = mano_data["joints3d"]
    faces = mano_data["faces"]
    scaled_focal = float(mano_data["scaled_focal"])

    gt_data = torch.load(gt_path, map_location="cpu", weights_only=False)
    gt_joints_3d = gt_data["joint_3d"].numpy()
    scale_unit = 1000.0 if np.abs(gt_joints_3d).max() < 10.0 else 1.0

    frame_paths = sorted(glob.glob(os.path.join(args.data_dir, "frame_*.jpg")))
    num_frames = len(frame_paths)
    assert num_frames == len(verts_seq), f"Frame count mismatch: {num_frames} vs {len(verts_seq)}"

    sample_img = cv2.imread(frame_paths[0])
    img_h, img_w = sample_img.shape[:2]
    cx, cy = img_w / 2.0, img_h / 2.0

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(args.out_path, fourcc, args.fps, (img_w, img_h))

    studio_bg_template = draw_studio_background(img_w, img_h)

    for idx in range(num_frames):
        verts_frame = verts_seq[idx]
        pred_joints = joints_seq[idx]
        gt_joints = gt_joints_3d[idx]

        # 1. Procrustes Alignment (SVD)
        pred_c = pred_joints.astype(np.float64) - pred_joints.mean(axis=0, keepdims=True)
        gt_c = gt_joints.astype(np.float64) - gt_joints.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(pred_c.T @ gt_c)
        R = u @ vt
        if np.linalg.det(R) < 0:
            vt[-1, :] *= -1
            s[-1] *= -1
            R = u @ vt
        scale = np.sum(s) / (np.sum(pred_c ** 2) + 1e-12)

        aligned_pred_j = scale * (pred_c @ R)
        joint_errs = np.linalg.norm(aligned_pred_j - gt_c, axis=-1) * scale_unit
        instantaneous_mpjpe = float(np.mean(joint_errs))

        # 2. Inverse-distance interpolation from 21 joints to 778 vertices
        dist_matrix = np.linalg.norm(verts_frame[:, None, :] - pred_joints[None, :, :], axis=-1)
        weights = 1.0 / (dist_matrix ** 2 + 1e-6)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        vert_errors = np.sum(weights * joint_errs[None, :], axis=-1)

        # 3. Render 3D Heatmap Hand onto dark studio canvas (no title banner)
        canvas = studio_bg_template.copy()
        rendered_frame = render_shaded_error_mesh(
            canvas, verts_frame, faces, vert_errors, scaled_focal, cx, cy, max_err=args.max_err
        )

        # 4. Calibrated colorbar on the right
        bar_w, bar_h = 24, 280
        bar_x, bar_y = img_w - 110, (img_h - bar_h) // 2
        draw_scientific_colorbar(rendered_frame, bar_x, bar_y, bar_w, bar_h, max_err=args.max_err)

        # 5. Instantaneous PA-MPJPE readout at bottom-left
        cv2.putText(
            rendered_frame, f"Error: {instantaneous_mpjpe:.2f} mm",
            (50, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA
        )

        video_writer.write(rendered_frame)
        if (idx + 1) % 10 == 0 or idx == num_frames - 1:
            print(f"  --> Rendered Frame [{idx + 1:02d}/{num_frames:02d}] | PA-MPJPE: {instantaneous_mpjpe:.2f} mm")

    video_writer.release()
    print(f"\n[SUCCESS] Ours 3D Heatmap video saved: {args.out_path}")


if __name__ == "__main__":
    main()