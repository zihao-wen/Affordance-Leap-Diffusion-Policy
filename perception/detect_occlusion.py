import os
import glob
import json
import cv2
import numpy as np
import torch
import trimesh
from ultralytics import SAM


def project_3d_to_2d(joints_3d: np.ndarray, focal_length: float, img_w: int, img_h: int) -> np.ndarray:
    """
    Project 3D hand joints in camera space (origin at optical center) onto 2D image pixels.
    """
    x = joints_3d[:, 0]
    y = joints_3d[:, 1]
    z = np.clip(joints_3d[:, 2], 1e-6, None)

    u = focal_length * (x / z) + img_w / 2.0
    v = focal_length * (y / z) + img_h / 2.0
    return np.stack([u, v], axis=-1).astype(np.float32)


def detect_self_occlusion_raycast(joints_3d: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> tuple:
    """
    Cast 3D rays from each joint toward the camera optical center (0, 0, 0).
    A joint is marked self-occluded if its ray intersects the hand surface >= 2 times
    strictly between the joint and the camera (exiting own finger + hitting another part).
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    ray_origins = joints_3d.astype(np.float64)
    to_cam = -ray_origins  # Camera optical center is at (0, 0, 0)
    cam_dists = np.linalg.norm(to_cam, axis=1)
    ray_dirs = to_cam / np.clip(cam_dists[:, None], 1e-8, None)

    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_dirs,
        multiple_hits=True
    )

    num_joints = len(joints_3d)
    self_occ = np.zeros(num_joints, dtype=bool)
    hit_counts = np.zeros(num_joints, dtype=np.int32)

    for k in range(num_joints):
        mask_k = (index_ray == k)
        if not np.any(mask_k):
            continue

        pts_k = locations[mask_k]
        dists_k = np.linalg.norm(pts_k - ray_origins[k], axis=1)
        valid_dists = np.sort(dists_k[(dists_k > 1e-4) & (dists_k < cam_dists[k])])

        # Deduplicate shared triangle-edge hits within 0.5 mm numerical tolerance
        unique_hits = []
        for d in valid_dists:
            if not unique_hits or (d - unique_hits[-1]) > 5e-4:
                unique_hits.append(d)

        hit_counts[k] = len(unique_hits)
        if hit_counts[k] >= 2:
            self_occ[k] = True

    return self_occ, hit_counts


def segment_hand_mask_sam2(sam_model: SAM, img_path: str, hand_bbox: list, img_h: int, img_w: int, device: str) -> np.ndarray:
    """
    Generate a binary foreground hand mask via SAM2 prompted by the detected hand bounding box.
    """
    if not hand_bbox or len(hand_bbox) != 4 or any(v is None for v in hand_bbox):
        return np.zeros((img_h, img_w), dtype=np.uint8)

    x1 = float(np.clip(hand_bbox[0], 0, img_w - 1))
    y1 = float(np.clip(hand_bbox[1], 0, img_h - 1))
    x2 = float(np.clip(hand_bbox[2], 0, img_w - 1))
    y2 = float(np.clip(hand_bbox[3], 0, img_h - 1))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    results = sam_model.predict(
        source=img_path,
        bboxes=[[x1, y1, x2, y2]],
        device=device,
        verbose=False
    )

    if not results or results[0].masks is None or len(results[0].masks.data) == 0:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    mask = results[0].masks.data[0].detach().cpu().numpy()
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    return (mask > 0.5).astype(np.uint8)


def detect_object_occlusion_mask(joints_2d: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
    """
    Mark a 2D projected joint as object-occluded if it falls outside the SAM2 hand mask.
    """
    img_h, img_w = hand_mask.shape
    num_joints = len(joints_2d)
    obj_occ = np.zeros(num_joints, dtype=bool)

    for k in range(num_joints):
        if not np.all(np.isfinite(joints_2d[k])):
            obj_occ[k] = True
            continue

        u = int(np.round(joints_2d[k, 0]))
        v = int(np.round(joints_2d[k, 1]))
        if u < 0 or u >= img_w or v < 0 or v >= img_h:
            obj_occ[k] = True
        elif hand_mask[v, u] == 0:
            obj_occ[k] = True

    return obj_occ


def classify_occlusion_level(num_occluded: int) -> str:
    if num_occluded <= 5:
        return "Low"
    if num_occluded <= 10:
        return "Medium"
    return "High"


def render_occlusion_overlay(
    img_bgr: np.ndarray,
    hand_mask: np.ndarray,
    joints_2d: np.ndarray,
    self_occ: np.ndarray,
    obj_occ: np.ndarray,
    frame_idx: int
) -> np.ndarray:
    """
    Render translucent hand mask with color-coded 2D joints:
      Green  = Visible
      Red    = Self-occluded
      Yellow = Object-occluded
    """
    vis = img_bgr.copy()
    img_h, img_w = vis.shape[:2]

    # Purple translucent overlay for segmented hand mask
    mask_bool = hand_mask.astype(bool)
    if np.any(mask_bool):
        purple = np.full_like(vis, (210, 110, 170), dtype=np.uint8)  # BGR purple tint
        blended = cv2.addWeighted(vis, 0.55, purple, 0.45, 0)
        vis[mask_bool] = blended[mask_bool]

    # Draw 21 keypoints
    for k in range(len(joints_2d)):
        if not np.all(np.isfinite(joints_2d[k])):
            continue

        u = int(np.round(joints_2d[k, 0]))
        v = int(np.round(joints_2d[k, 1]))
        if u < 0 or u >= img_w or v < 0 or v >= img_h:
            continue

        if obj_occ[k]:
            color = (0, 235, 255)   # Yellow (BGR): object occlusion
        elif self_occ[k]:
            color = (40, 40, 240)   # Red (BGR): self occlusion
        else:
            color = (60, 230, 60)   # Green (BGR): visible

        cv2.circle(vis, (u, v), 4, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (u, v), 3, color, -1, lineType=cv2.LINE_AA)

    num_vis = int(np.sum((~self_occ) & (~obj_occ)))
    num_occ = 21 - num_vis
    num_self = int(np.sum(self_occ))
    num_obj = int(np.sum(obj_occ))

    # Top banner and compact legend
    banner_w = min(370, img_w - 8)
    cv2.rectangle(vis, (8, 8), (banner_w, 66), (20, 20, 20), -1)
    cv2.putText(
        vis, f"F{frame_idx:02d} | Vis:{num_vis}/21 Occ:{num_occ} (S:{num_self} O:{num_obj})",
        (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.circle(vis, (22, 52), 4, (60, 230, 60), -1)
    cv2.putText(vis, "Visible", (32, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.circle(vis, (105, 52), 4, (40, 40, 240), -1)
    cv2.putText(vis, "Self-occ", (115, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.circle(vis, (198, 52), 4, (0, 235, 255), -1)
    cv2.putText(vis, "Obj-occ", (208, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return vis


def main():
    print("=" * 70)
    print("[Module 3] Dual-Criterion Occlusion Detection (3D Ray Casting + SAM2)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Hardware platform: {device}")

    # 1. Load Upstream Data Contracts
    mano_path = "results/baseline_human_pose_mano.npy"
    hand_path = "results/detected_hand_bboxes.json"
    assert os.path.exists(mano_path), f"Missing MANO baseline data: {mano_path}"
    assert os.path.exists(hand_path), f"Missing hand bounding boxes: {hand_path}"

    mano_data = np.load(mano_path, allow_pickle=True).item()
    joints_3d_all = mano_data["joints3d"]          # (30, 21, 3) in camera space
    verts_all = mano_data["vertices"]              # (30, 778, 3) in camera space
    faces = mano_data["faces"].astype(np.int32)    # (1538, 3)
    scaled_focal = float(mano_data["scaled_focal"])
    assert scaled_focal > 0, f"Invalid focal length: {scaled_focal}"

    with open(hand_path, "r") as f:
        hand_data = json.load(f)
    hand_bboxes = hand_data["bboxes"] if isinstance(hand_data, dict) else hand_data

    frame_paths = sorted(glob.glob("./data/dexycb_025_mug/frame_*.jpg"))
    num_frames = len(frame_paths)
    assert num_frames > 0, "No input frames found in ./data/dexycb_025_mug/"
    assert num_frames == len(joints_3d_all), f"Frame/joints mismatch: {num_frames} vs {len(joints_3d_all)}"
    assert num_frames == len(verts_all), f"Frame/vertices mismatch: {num_frames} vs {len(verts_all)}"
    assert num_frames == len(hand_bboxes), f"Frame/bboxes mismatch: {num_frames} vs {len(hand_bboxes)}"

    # Geometric sanity check: ensure 3D joints and mesh vertices share identical coordinate frame
    max_centroid_diff = float(np.max(np.linalg.norm(
        joints_3d_all.mean(axis=1) - verts_all.mean(axis=1), axis=1
    )))
    assert max_centroid_diff < 0.05, (
        f"Coordinate mismatch between 3D joints and mesh vertices: max centroid diff = {max_centroid_diff:.4f} m"
    )
    print(f"[INFO] Loaded {num_frames} frames | scaled_focal={scaled_focal:.2f} | max_centroid_diff={max_centroid_diff*1000:.2f} mm")

    # 2. Initialize SAM2 Model
    os.makedirs("weights", exist_ok=True)
    sam_weight_path = "weights/sam2.1_b.pt"
    print(f"[INFO] Loading SAM2 segmentation model: {sam_weight_path}")
    sam_model = SAM(sam_weight_path)

    # 3. Run Frame-by-Frame Occlusion Detection
    vis_dir = "visualizations/occlusion_vis"
    os.makedirs(vis_dir, exist_ok=True)
    vis_keyframes = {0, 10, 15, 20, 29}
    summary_panels = []

    visibility_mask_all = np.zeros((num_frames, 21), dtype=np.uint8)
    joints_2d_all = np.zeros((num_frames, 21, 2), dtype=np.float32)
    diagnostics_per_frame = []

    for t, img_path in enumerate(frame_paths):
        img_bgr = cv2.imread(img_path)
        assert img_bgr is not None, f"Failed to read image: {img_path}"
        img_h, img_w = img_bgr.shape[:2]

        # Criterion 1: 3D Ray Casting for self-occlusion
        self_occ, ray_hits = detect_self_occlusion_raycast(joints_3d_all[t], verts_all[t], faces)

        # Criterion 2: SAM2 hand mask + 2D projection for object occlusion
        joints_2d = project_3d_to_2d(joints_3d_all[t], scaled_focal, img_w, img_h)
        joints_2d_all[t] = joints_2d
        hand_mask = segment_hand_mask_sam2(sam_model, img_path, hand_bboxes[t], img_h, img_w, device=device)
        obj_occ = detect_object_occlusion_mask(joints_2d, hand_mask)

        # Binary visibility label v in {0, 1}^21 (1 = visible, 0 = occluded)
        visible = (~self_occ) & (~obj_occ)
        visibility_mask_all[t] = visible.astype(np.uint8)

        num_vis = int(np.sum(visible))
        num_occ = 21 - num_vis
        level = classify_occlusion_level(num_occ)

        diagnostics_per_frame.append({
            "frame_idx": t,
            "num_visible": num_vis,
            "num_occluded": num_occ,
            "occlusion_level": level,
            "self_occluded_joints": np.where(self_occ)[0].tolist(),
            "object_occluded_joints": np.where(obj_occ)[0].tolist(),
            "ray_hit_counts": ray_hits.tolist(),
            "sam2_mask_area_px": int(np.sum(hand_mask))
        })

        if t in vis_keyframes or t % 10 == 0:
            print(
                f"  --> Frame [{t:02d}/{num_frames - 1:02d}] | "
                f"Visible: {num_vis:2d}/21 | Occ: {num_occ:2d} (Self: {int(np.sum(self_occ)):2d}, "
                f"Obj: {int(np.sum(obj_occ)):2d}) | Level: {level}"
            )

        if t in vis_keyframes:
            overlay = render_occlusion_overlay(img_bgr, hand_mask, joints_2d, self_occ, obj_occ, t)
            out_img = os.path.join(vis_dir, f"frame_{t:02d}_occlusion.jpg")
            cv2.imwrite(out_img, overlay)
            summary_panels.append(overlay)

    # Save 5-keyframe horizontal summary strip for quick visual inspection
    if summary_panels:
        grid_img = np.concatenate(summary_panels, axis=1)
        cv2.imwrite(os.path.join(vis_dir, "occlusion_summary_grid.jpg"), grid_img)

    # 4. Save Binary Visibility Mask & Diagnostics
    os.makedirs("results", exist_ok=True)
    mask_out = "results/joint_visibility_mask.npy"
    j2d_out = "results/baseline_joints_2d.npy"
    diag_out = "results/occlusion_diagnostics.json"

    np.save(mask_out, visibility_mask_all)
    np.save(j2d_out, joints_2d_all)

    avg_occ = float(np.mean([d["num_occluded"] for d in diagnostics_per_frame]))
    avg_self = float(np.mean([len(d["self_occluded_joints"]) for d in diagnostics_per_frame]))
    avg_obj = float(np.mean([len(d["object_occluded_joints"]) for d in diagnostics_per_frame]))
    level_dist = {
        lvl: sum(1 for d in diagnostics_per_frame if d["occlusion_level"] == lvl)
        for lvl in ["Low", "Medium", "High"]
    }

    with open(diag_out, "w") as f:
        json.dump({
            "num_frames": num_frames,
            "num_joints": 21,
            "summary": {
                "avg_occluded_joints": round(avg_occ, 2),
                "avg_self_occluded_joints": round(avg_self, 2),
                "avg_object_occluded_joints": round(avg_obj, 2),
                "occlusion_level_distribution": level_dist
            },
            "per_frame": diagnostics_per_frame
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"★ Visibility Mask Saved: {mask_out} | shape={visibility_mask_all.shape}")
    print(f"★ 2D Projected Joints Saved: {j2d_out} | shape={joints_2d_all.shape}")
    print(f"★ Diagnostics Saved: {diag_out}")
    print(f"★ Visual Overlays Saved: {vis_dir}/occlusion_summary_grid.jpg")
    print(f"  Summary -> Avg Occluded: {avg_occ:.2f}/21 (Self: {avg_self:.2f}, Obj: {avg_obj:.2f}) | Levels: {level_dist}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()