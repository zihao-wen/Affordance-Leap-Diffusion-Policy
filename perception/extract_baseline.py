import os
import sys
import glob
import json
from unittest.mock import MagicMock

# Headless environment protection: bypass pyrender OpenGL dependency
os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.modules["pyrender"] = MagicMock()

import cv2
import torch
import numpy as np
from PIL import Image

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

def compute_pa_mpjpe(predicted_joints: np.ndarray, ground_truth_joints: np.ndarray) -> np.ndarray:
    """
    Compute Procrustes-Aligned Mean Per Joint Position Error (PA-MPJPE).

    Args:
        predicted_joints: (N, 21, 3) array of estimated 3D hand joints (meters).
        ground_truth_joints: (N, 21, 3) array of ground truth 3D joints (meters).

    Returns:
        instantaneous_errors: (N,) array of instantaneous per-frame errors e_t (millimeters).
    """
    num_frames, num_joints, _ = predicted_joints.shape
    instantaneous_errors = np.zeros(num_frames, dtype=np.float32)

    for t in range(num_frames):
        pred = predicted_joints[t]   # (21, 3)
        gt = ground_truth_joints[t]  # (21, 3)

        # Center coordinates to centroid
        pred_centered = pred - pred.mean(axis=0, keepdims=True)
        gt_centered = gt - gt.mean(axis=0, keepdims=True)

        # Orthogonal Procrustes via SVD
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
    print("[Module 1] WiLoR Baseline 3D Hand Reconstruction Pipeline")
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

    # 2. Retrieve Staged 30-Frame Input Sequence
    data_dir = "./data/dexycb_025_mug"
    frame_paths = sorted(glob.glob(os.path.join(data_dir, "frame_*.jpg")))
    assert len(frame_paths) == 30, f"Expected 30 sequence frames, found: {len(frame_paths)}"
    print(f"[INFO] Successfully loaded {len(frame_paths)} input frames from {data_dir}")

    # 3. Retrieve Isolated Ground Truth for Metric Evaluation
    eval_gt_path = os.path.join(data_dir, "eval_gt.pt")
    assert os.path.exists(eval_gt_path), f"Ground truth missing: {eval_gt_path}"
    eval_gt = torch.load(eval_gt_path, map_location="cpu")
    gt_joints_3d = eval_gt["joint_3d"].numpy()

    pred_joints_list = []
    pred_poses_list = []
    hand_bboxes_list = []

    print("[INFO] Executing WiLoR baseline inference across 30 frames...")
    last_known_bbox = [100.0, 100.0, 500.0, 500.0]

    with torch.no_grad():
        for idx, img_path in enumerate(frame_paths):
            img_cv2 = cv2.imread(img_path)

            # Detect hand bounding boxes via YOLO
            detections = detector(img_cv2, conf=0.25, iou=0.5, verbose=False)[0]

            if len(detections.boxes) > 0:
                best_det_idx = int(detections.boxes.conf.argmax().item())
                box = detections.boxes.xyxy[best_det_idx].cpu().numpy().tolist()
                is_right_val = int(detections.boxes.cls[best_det_idx].cpu().numpy().item())
                last_known_bbox = box
            else:
                box = last_known_bbox
                is_right_val = 0  # Default to left hand for this sequence

            hand_bboxes_list.append(box)

            # Construct input tensor via WiLoR ViTDetDataset
            boxes_np = np.array([box], dtype=np.float32)
            right_np = np.array([is_right_val], dtype=np.int64)

            dataset = ViTDetDataset(model_cfg, img_cv2, boxes_np, right_np, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

            for batch in dataloader:
                batch = recursive_to(batch, device)
                out = model(batch)

                # Extract 3D joints and apply horizontal mirror correction for left hands
                joints = out["pred_keypoints_3d"][0].detach().cpu().numpy()  # (21, 3)
                joints[:, 0] = (2 * is_right_val - 1) * joints[:, 0]
                pred_joints_list.append(joints)

                # Extract MANO kinematic pose parameters
                if "pred_mano_params" in out and "hand_pose" in out["pred_mano_params"]:
                    pose = out["pred_mano_params"]["hand_pose"][0].detach().cpu().numpy().reshape(-1)
                else:
                    pose = np.zeros(45, dtype=np.float32)
                pred_poses_list.append(pose)

            if (idx + 1) % 10 == 0 or idx == 29:
                print(f"  --> Processed Frame [{idx+1:02d}/30]")

    # 4. Export Reconstruction Estimates and Perception Outputs
    baseline_joints = np.array(pred_joints_list, dtype=np.float32)  # (30, 21, 3)
    baseline_poses = np.array(pred_poses_list, dtype=np.float32)    # (30, 45)

    os.makedirs("results", exist_ok=True)
    np.save("results/baseline_joints_3d.npy", baseline_joints)
    np.save("results/baseline_human_pose_mano.npy", {
        "pose": baseline_poses,
        "joints3d": baseline_joints
    })

    with open("results/detected_hand_bboxes.json", "w") as f:
        json.dump(hand_bboxes_list, f, indent=2)

    print("[SUCCESS] Baseline outputs serialized to results/:")
    print("  - results/baseline_joints_3d.npy")
    print("  - results/baseline_human_pose_mano.npy")
    print("  - results/detected_hand_bboxes.json")

    # 5. Compute Quantitative Metrics
    instantaneous_errors = compute_pa_mpjpe(baseline_joints, gt_joints_3d)
    sequence_mean_pa = float(np.mean(instantaneous_errors))
    high_occlusion_pa = float(np.mean(instantaneous_errors[13:]))

    metrics = {
        "metric_name": "PA-MPJPE",
        "unit": "mm",
        "mean_pa_mpjpe_mm": round(sequence_mean_pa, 2),
        "high_occlusion_pa_mpjpe_mm": round(high_occlusion_pa, 2),
        "instantaneous_errors_mm": [round(float(err), 2) for err in instantaneous_errors]
    }

    with open("results/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "=" * 70)
    print("★ WiLoR Baseline Quantitative Evaluation Summary:")
    print(f"  - Full Sequence Mean PA-MPJPE: {sequence_mean_pa:.2f} mm")
    print(f"  - Severe Occlusion Segment Mean (Frames 13-29): {high_occlusion_pa:.2f} mm")
    print(f"  - Total Instantaneous Error Points (e_t) Logged: {len(instantaneous_errors)}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()