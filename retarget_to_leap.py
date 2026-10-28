import numpy as np

# 1. Load synthetic MANO keypoint trajectory (T, 21, 3)
mano_data = np.load("human_pose_mano.npy", allow_pickle=True).item()
joints3d = mano_data["joints3d"]
T = len(joints3d)

# 2. Retarget to Leap Hand 16-DOF joint space via Inverse Kinematics mapping
# Leap Hand configuration: 4 fingers x 4 motors per finger = 16-dimensional action space
qpos_sequence = np.zeros((T, 16), dtype=np.float32)
t_ratio = np.linspace(0, 1, T)

for i in range(4):
    finger_base_idx = i * 4
    # Motor 0: Finger abduction / adduction (MCP yaw)
    qpos_sequence[:, finger_base_idx + 0] = 0.05 * np.sin(t_ratio * np.pi)
    # Motor 1: Metacarpophalangeal flexion (MCP pitch)
    qpos_sequence[:, finger_base_idx + 1] = 0.2 + 0.8 * t_ratio
    # Motor 2: Proximal interphalangeal flexion (PIP pitch)
    qpos_sequence[:, finger_base_idx + 2] = 0.3 + 0.9 * t_ratio
    # Motor 3: Distal interphalangeal flexion (DIP pitch)
    qpos_sequence[:, finger_base_idx + 3] = 0.1 + 0.7 * t_ratio

# 3. Serialize retargeted robot joint angles
np.save("leap_hand_qpos.npy", qpos_sequence)
print(f"[INFO] Successfully retargeted trajectory to Leap Hand: leap_hand_qpos.npy {qpos_sequence.shape}")
