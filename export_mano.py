import numpy as np

# 抓水杯动作序列：100 帧
T = 100

# 1. 构造 45 维 MANO 手指关节姿态参数 (15 关节 x 3 轴角)
t_steps = np.linspace(0, 1, T).reshape(T, 1, 1)
open_pose = np.zeros((1, 15, 3), dtype=np.float32)
grasp_pose = np.tile(np.array([0.0, 0.0, 0.4], dtype=np.float32), (1, 15, 1))
mano_pose = ((1 - t_steps) * open_pose + t_steps * grasp_pose).reshape(T, 45)

# 2. 构造 21 个手部关键点 3D 空间坐标 (T, 21, 3)，单位：米 (m)
joints3d = np.zeros((T, 21, 3), dtype=np.float32)
joints3d[:, 0] = np.array([0.0, 0.0, 0.4]) # 手腕基底

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

np.save("human_pose_mano.npy", mano_dict)
print(">>> [成功] 交付物 1: human_pose_mano.npy 已生成！Shape:", joints3d.shape)
