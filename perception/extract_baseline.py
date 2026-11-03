import os
import sys
import glob
import json
from unittest.mock import MagicMock

# Headless environment protection
os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.modules["pyrender"] = MagicMock()

import cv2
import torch
import numpy as np

# Register WiLoR package directory
WILOR_ROOT = os.path.abspath("./WiLoR")
if WILOR_ROOT not in sys.path:
    sys.path.append(WILOR_ROOT)

from ultralytics import YOLO
from wilor.models import load_wilor
from wilor.datasets.vitdet_dataset import ViTDetDataset

def recursive_to(x, device):
    """Recursively transfer tensors within collections to compute device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_to(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [recursive_to(v, device) for v in x]
    elif isinstance(x, tuple):
        return tuple(recursive_to(v, device) for v in x)
    return x

def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0):
    """Convert weak-perspective crop camera parameters to full-frame 3D translation."""
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)

def compute_pa_mpjpe(predicted_joints: np.ndarray, ground_truth_joints: np.ndarray) -> np.ndarray:
    """
    Compute Procrustes-Aligned Mean Per Joint Position Error (PA-MPJPE).

    Args:
        predicted_joints: (N, 21, 3) array of estimated 3D hand joints (meters).
        ground_truth_joints: (N, 21, 3) array of ground truth 3D joints (meters).

    Returns:
        instantaneous_errors: (N,) array of instantaneous per-frame errors e_t (millimeters).
    """
    num_frames = predicted_joints.shape[0]
    instantaneous_errors = np.zeros(num_frames, dtype=np.float32)

    for t in range(num_frames):
        pred = predicted_joints[t]
        gt = ground_truth_joints[t]

        # Centering
        pred_centered = pred - pred.mean(axis=0, keepdims=True)
        gt_centered = gt - gt.mean(axis=0, keepdims=True)

        # SVD Procrustes Alignment
        covariance_matrix = np.dot(pred_centered.T, gt_centered)
        u, singular_values, vt = np.linalg.svd(covariance_matrix)
        rotation = np.dot(vt.T, u.T)

        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = np.dot(vt.T, u.T)

        scale = np.sum(singular_values) / np.sum(pred_centered ** 2)
        aligned_pred = scale * np.dot(pred_centered, rotation.T)

        joint_distances = np.linalg.norm(aligned_pred - gt_centered, axis=-1) * 1000.0
        instantaneous_errors[t] = np.mean(joint_distances)

    return instantaneous_errors

def main():
    print("=" * 70)
    print("[Module 1] WiLoR Baseline 3D Hand Mesh & Trajectory Pipeline")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Hardware platform: {device}")

    # 1. Load Pretrained WiLoR Model and Config
    checkpoint_path = os.path.join(WILOR_ROOT, "pretrained_models/wilor_final.ckpt")
    cfg_path = os.path.join(WILOR_ROOT, "pretrained_models/model_config.yaml")
    detector_path = os.path.join(WILOR_ROOT, "pretrained_models/detector.pt")

    assert os.path.exists(checkpoint_path), f"Checkpoint missing: {checkpoint_path}"
    assert os.path.exists(cfg_path), f"Config missing: {cfg_path}"
    assert os.path.exists(detector_path), f"Detector missing: {detector_path}"

    print(f"[INFO] Loading WiLoR weights from: {checkpoint_path}")
    print(f"[INFO] Loading WiLoR config from: {cfg_path}")
    model, model_cfg = load_wilor(cfg_path=cfg_path, checkpoint_path=checkpoint_path)
    model = model.to(device)
    model.eval()

    print(f"[INFO] Loading YOLO hand detector from: {detector_path}")
    detector = YOLO(detector_path)

    # Extract MANO triangular faces and invert winding order for left hand
    mano_faces_right = model.mano.faces.astype(np.int32)
    mano_faces_left = mano_faces_right[:, [0, 2, 1]]

    # 2. Retrieve Staged 30-Frame Input Sequence
    data_dir = "./data/dexycb_025_mug"
    frame_paths = sorted(glob.glob(os.path.join(data_dir, "frame_*.jpg")))
    assert len(frame_paths) == 30, f"Expected 30 sequence frames, found: {len(frame_paths)}"
    print(f"[INFO] Successfully loaded {len(frame_paths)} input frames from {data_dir}")

    # 3. Retrieve Ground Truth for Metric Evaluation
    eval_gt_path = os.path.join(data_dir, "eval_gt.pt")
    assert os.path.exists(eval_gt_path), f"Ground truth missing: {eval_gt_path}"
    eval_gt = torch.load(eval_gt_path, map_location="cpu")
    gt_joints_3d = eval_gt["joint_3d"].numpy()

    pred_joints_global_list = []
    pred_verts_global_list = []
    pred_poses_list = []
    pred_cam_t_list = []
    hand_bboxes_list = []
    scaled_focal_value = 5000.0

    print("[INFO] Executing WiLoR inference (778-vertex skin mesh + 3D trajectory)...")
    last_known_bbox = [100.0, 100.0, 500.0, 500.0]

    with torch.no_grad():
        for idx, img_path in enumerate(frame_paths):
            img_cv2 = cv2.imread(img_path)
            detections = detector(img_cv2, conf=0.25, iou=0.5, verbose=False)[0]

            if len(detections.boxes) > 0:
                best_idx = int(detections.boxes.conf.argmax().item())
                box = detections.boxes.xyxy[best_idx].cpu().numpy().tolist()
                is_right_val = int(detections.boxes.cls[best_idx].cpu().numpy().item())
                last_known_bbox = box
            else:
                box = last_known_bbox
                is_right_val = 0  # Default to left hand for this sequence

            hand_bboxes_list.append(box)
            boxes_np = np.array([box], dtype=np.float32)
            right_np = np.array([is_right_val], dtype=np.int64)

            dataset = ViTDetDataset(model_cfg, img_cv2, boxes_np, right_np, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

            for batch in dataloader:
                batch = recursive_to(batch, device)
                out = model(batch)

                multiplier = 2 * batch["right"] - 1
                pred_cam = out["pred_cam"].clone()
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]

                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                scaled_focal_value = float(scaled_focal.item())

                cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal)
                cam_t_np = cam_t_full[0].detach().cpu().numpy()  # (3,)

                # Extract local 3D joints & 778 vertices, flip X for left hand
                joints_local = out["pred_keypoints_3d"][0].detach().cpu().numpy()
                joints_local[:, 0] = (2 * is_right_val - 1) * joints_local[:, 0]

                verts_local = out["pred_vertices"][0].detach().cpu().numpy()
                verts_local[:, 0] = (2 * is_right_val - 1) * verts_local[:, 0]

                # Transform to global camera coordinate system (reaching trajectory)
                joints_global = joints_local + cam_t_np[None, :]
                verts_global = verts_local + cam_t_np[None, :]

                pred_joints_global_list.append(joints_global)
                pred_verts_global_list.append(verts_global)
                pred_cam_t_list.append(cam_t_np)

                pose = out["pred_mano_params"]["hand_pose"][0].detach().cpu().numpy().reshape(-1)
                pred_poses_list.append(pose)

            if (idx + 1) % 10 == 0 or idx == 29:
                print(f"  --> Processed Frame [{idx+1:02d}/30]")

    # 4. Export Reconstruction Estimates and Perception Outputs
    baseline_joints = np.array(pred_joints_global_list, dtype=np.float32)  # (30, 21, 3)
    baseline_verts = np.array(pred_verts_global_list, dtype=np.float32)    # (30, 778, 3)
    baseline_poses = np.array(pred_poses_list, dtype=np.float32)           # (30, 45)
    baseline_cam_t = np.array(pred_cam_t_list, dtype=np.float32)           # (30, 3)

    os.makedirs("results", exist_ok=True)
    np.save("results/baseline_joints_3d.npy", baseline_joints)
    np.save("results/baseline_human_pose_mano.npy", {
        "pose": baseline_poses,
        "joints3d": baseline_joints,
        "vertices": baseline_verts,
        "faces": mano_faces_left,
        "cam_t": baseline_cam_t,
        "scaled_focal": scaled_focal_value
    })

    with open("results/detected_hand_bboxes.json", "w") as f:
        json.dump(hand_bboxes_list, f, indent=2)

    print("[SUCCESS] Baseline outputs serialized to results/:")
    print("  - results/baseline_joints_3d.npy (Cartesian 3D joint trajectories)")
    print("  - results/baseline_human_pose_mano.npy (Kinematic MANO parameters + 778-vertex mesh)")
    print("  - results/detected_hand_bboxes.json (Per-frame hand bounding boxes)")

    # 5. Compute Quantitative Metrics
    instantaneous_errors = compute_pa_mpjpe(baseline_joints, gt_joints_3d)
    sequence_mean_pa = float(np.mean(instantaneous_errors))
    high_occlusion_pa = float(np.mean(instantaneous_errors[13:]))  # Subsequence frames 13-29

    metrics = {
        "metric_name": "PA-MPJPE",
        "unit": "mm",
        "mean_pa_mpjpe_mm": round(sequence_mean_pa, 2),
        "high_occlusion_pa_mpjpe_mm": round(high_occlusion_pa, 2),
        "instantaneous_errors_mm": [round(float(err), 2) for err in instantaneous_errors]
    }

    with open("results/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 6. Restored Full Summary Dashboard
    print("\n" + "=" * 70)
    print("★ WiLoR Baseline Quantitative Evaluation Summary:")
    print(f"  - Full Sequence Mean PA-MPJPE: {sequence_mean_pa:.2f} mm")
    print(f"  - Severe Occlusion Segment Mean (Frames 13-29): {high_occlusion_pa:.2f} mm")
    print(f"  - Total Instantaneous Error Points (e_t) Logged: {len(instantaneous_errors)}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()