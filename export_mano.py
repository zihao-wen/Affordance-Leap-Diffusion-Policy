import numpy as np

# Number of frames for synthetic grasp trajectory
T = 100

# 1. Synthesize 45-dim MANO hand pose parameters (15 joints x 3 axis-angles)
t_steps = np.linspace(0, 1, T).reshape(T, 1, 1)
open_pose = np.zeros((1, 15, 3), dtype=np.float32)
grasp_pose = np.tile(np.array([0.0, 0.0, 0.4], dtype=np.float32), (1, 15, 1))
mano_pose = ((1 - t_steps) * open_pose + t_steps * grasp_pose).reshape(T, 45)

# 2. Synthesize 21 hand 3D keypoint coordinates (T, 21, 3) in meters
joints3d = np.zeros((T, 21, 3), dtype=np.float32)
joints3d[:, 0] = np.array([0.0, 0.0, 0.4])  # Wrist root coordinate

for finger_idx in range(5):
    base_idx = 1 + finger_idx * 4
    x_offset = (finger_idx - 2) * 0.025
    for joint_idx in range(4):
        kpt_id = base_idx + joint_idx
        joints3d[:, kpt_id, 0] = x_offset * (1.0 - 0.3 * t_steps.squeeze())
        joints3d[:, kpt_id, 1] = -0.03 * (joint_idx + 1) * (1.0 - 0.2 * t_steps.squeeze())
        joints3d[:, kpt_id, 2] = 0.4 + 0.03 * (joint_idx + 1)

mano_dict = {
    "pose": mano_pose,
    "joints3d": joints3d,
    "rot": np.zeros((T, 3), dtype=np.float32),
    "trans": np.tile(np.array([0.0, 0.0, 0.4], dtype=np.float32), (T, 1))
}

# 3. Serialize hand demonstration trajectory
np.save("human_pose_mano.npy", mano_dict)
print(f"[INFO] Successfully generated MANO trajectory: human_pose_mano.npy {joints3d.shape}")
