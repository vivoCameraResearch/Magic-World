import os
import sys
import argparse
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# === 路径注入（保持与你原始工程一致）===
current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# ---------- 基础工具 ----------
def extract_intrinsics_and_extrinsics(frame_data: List[float]):
    intrinsics = frame_data[:7]
    extrinsics = np.array(frame_data[7:]).reshape(3, 4)
    return intrinsics, extrinsics

def build_camera_row(intrinsics, extrinsics):
    return list(intrinsics) + list(extrinsics.flatten())

def generate_camera_trajectory_from_last(
    base_camera_array: np.ndarray,
    direction="w",
    num_frames=4,
    step_magnitude=0.1,
    total_angle_deg=15,
):
    """
    从 base_camera_array 的最后一帧出发，生成一个方向上的连续相机轨迹。
    返回 shape=[num_frames, 19] 的数组（不含 base 的最后一帧）。
    """
    trajectory = []
    last_intrinsics, last_extrinsics = extract_intrinsics_and_extrinsics(base_camera_array[-1])
    R_current = last_extrinsics[:, :3]
    t_current = last_extrinsics[:, 3:]

    if direction in ["w", "s"]:
        # 前进/后退
        forward = -R_current[:, 2]  # -z 轴为向前
        if direction == "s":
            forward = -forward
        for _ in range(num_frames):
            delta_t = forward * step_magnitude
            t_current = t_current + delta_t.reshape(3, 1)
            frame_data = build_camera_row(last_intrinsics, np.hstack([R_current, t_current]))
            trajectory.append(frame_data)

    elif direction in ["a", "d"]:
        # 左右转 + 小幅前进
        angle = np.deg2rad(-total_angle_deg if direction == "d" else total_angle_deg)
        R_delta = R.from_euler('y', angle, degrees=False)
        R_target = R_current @ R_delta.as_matrix()
        key_times = [0, 1]
        key_rots = R.from_matrix([R_current, R_target])
        slerp = Slerp(key_times, key_rots)
        times = np.linspace(0, 1, num_frames)
        Rs_interp = slerp(times).as_matrix()

        for Ri in Rs_interp:
            forward = -Ri[:, 2]
            delta_t = forward * step_magnitude * 0.3
            t_current = t_current + delta_t.reshape(3, 1)
            frame_data = build_camera_row(last_intrinsics, np.hstack([Ri, t_current]))
            trajectory.append(frame_data)
            R_current = Ri

    elif direction in ["q", "e"]:
        # 上升/下降（纯平移）
        up = R_current[:, 1]
        if direction == "e":
            up = -up
        for _ in range(num_frames):
            delta_t = up * step_magnitude
            t_current = t_current + delta_t.reshape(3, 1)
            frame_data = build_camera_row(last_intrinsics, np.hstack([R_current, t_current]))
            trajectory.append(frame_data)

    elif direction in ["r", "t"]:
        # 左/右平移（纯平移）
        right = -R_current[:, 0]  # 加负号使视觉与轨迹一致
        if direction == "r":
            right = -right
        for _ in range(num_frames):
            delta_t = right * step_magnitude
            t_current = t_current + delta_t.reshape(3, 1)
            frame_data = build_camera_row(last_intrinsics, np.hstack([R_current, t_current]))
            trajectory.append(frame_data)

    elif direction in ["z", "c"]:
        # 原地水平转向（纯旋转，平移不变）
        angle = np.deg2rad(-total_angle_deg if direction == "c" else total_angle_deg)
        R_delta = R.from_euler('y', angle, degrees=False)
        R_target = R_current @ R_delta.as_matrix()
        key_times = [0, 1]
        key_rots = R.from_matrix([R_current, R_target])
        slerp = Slerp(key_times, key_rots)
        times = np.linspace(0, 1, num_frames)
        Rs_interp = slerp(times).as_matrix()

        for Ri in Rs_interp:
            frame_data = build_camera_row(last_intrinsics, np.hstack([Ri, t_current.copy()]))
            trajectory.append(frame_data)
            R_current = Ri

    elif direction in ["u", "n"]:
        # 原地俯仰（绕 x 轴纯旋转）：u 抬头(负角)，n 低头(正角)
        angle = np.deg2rad(-total_angle_deg if direction == "u" else total_angle_deg)
        R_delta = R.from_euler('x', angle, degrees=False)
        R_target = R_current @ R_delta.as_matrix()
        key_times = [0, 1]
        key_rots = R.from_matrix([R_current, R_target])
        slerp = Slerp(key_times, key_rots)
        times = np.linspace(0, 1, num_frames)
        Rs_interp = slerp(times).as_matrix()

        for Ri in Rs_interp:
            frame_data = build_camera_row(last_intrinsics, np.hstack([Ri, t_current.copy()]))
            trajectory.append(frame_data)
            R_current = Ri

    else:
        raise ValueError(f"Invalid direction: {direction}. Use one of [w,a,s,d,q,e,r,t,z,c,u,n].")

    return np.array(trajectory, dtype=np.float32)


def load_start_frame(start_frame_txt: str):
    """
    从 txt 读取一行 19 项作为起始帧（允许多余列，只取前19个；允许空格/逗号分隔）。
    """
    if not os.path.exists(start_frame_txt):
        raise FileNotFoundError(start_frame_txt)
    with open(start_frame_txt, "r") as f:
        line = f.readline().strip()
    parts = [p for p in line.replace(",", " ").split() if p]
    vals = list(map(float, parts[:19]))
    if len(vals) != 19:
        raise ValueError(f"start_frame_txt must provide 19 values, got {len(vals)}")
    return vals


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirs",
        type=str,
        required=True,
        help="用逗号分隔的4个方向，例如: w,a,d,s；可用集合: [w,a,s,d,q,e,r,t,z,c,u,n]"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录；会写入：segment_{idx}_{dir}.txt 与 trajectory_full.txt"
    )
    parser.add_argument("--segment_frames", type=int, default=33, help="每段生成的帧数")
    parser.add_argument("--step", type=float, default=0.1, help="平移步长（每帧）")
    parser.add_argument("--angle_deg", type=float, default=10.0, help="旋转总角度（整段）")
    parser.add_argument("--start_frame_txt", type=str, default="", help="可选：自定义起始帧 txt（首行19项）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 固定起始帧（若提供 start_frame_txt 则覆盖）
    if args.start_frame_txt:
        fixed_start_frame = load_start_frame(args.start_frame_txt)
    else:
        fixed_start_frame = [
            0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

    dirs = [d.strip().lower() for d in args.dirs.split(",") if d.strip()]
    if len(dirs) != 4:
        raise ValueError(f"--dirs 需要恰好 4 个方向，当前为 {len(dirs)}: {dirs}")

    allowed = set(list("wasdqertzcun"))
    for d in dirs:
        if d not in allowed:
            raise ValueError(f"非法方向: {d}. 允许集合: {sorted(list(allowed))}")

    # 全局轨迹（含固定起始帧）
    current_trajectory = np.array([fixed_start_frame], dtype=np.float32)

    # 逐段生成并保存单段轨迹
    for idx, d in enumerate(dirs, start=1):
        # 单段起始帧 = 当前全局轨迹最后一帧
        start_of_segment = current_trajectory[-1].copy()

        seg_generated = generate_camera_trajectory_from_last(
            current_trajectory,
            direction=d,
            num_frames=args.segment_frames,
            step_magnitude=args.step,
            total_angle_deg=args.angle_deg,
        )
        # 单段文件：包含该段起始帧 + 该段生成的帧
        seg_traj = np.vstack([start_of_segment, seg_generated]) if seg_generated.size > 0 else np.array([start_of_segment])

        seg_path = os.path.join(args.output_dir, f"segment_{idx}_{d}.txt")
        np.savetxt(seg_path, seg_traj, fmt="%.10f")
        print(f"[OK] Saved segment {idx} ({d}) -> {seg_path}  (rows={len(seg_traj)})")

        # 更新全局轨迹（仅拼接生成部分；全局第一行仍保持唯一的固定起始帧）
        if seg_generated.size > 0:
            current_trajectory = np.concatenate([current_trajectory, seg_generated], axis=0)

    # 保存完整拼接轨迹
    full_path = os.path.join(args.output_dir, "trajectory_full.txt")
    np.savetxt(full_path, current_trajectory, fmt="%.10f")
    print(f"[OK] Saved FULL trajectory -> {full_path}  (rows={len(current_trajectory)})")


if __name__ == "__main__":
    main()

"""
python asset/gen_traj.py \
  --dirs w,w,w,w \
  --output_dir asset/bench/W \
  --segment_frames 33 \
  --step 0.1 \
  --angle_deg 10

python asset/gen_traj.py \
  --dirs d,d,d,d \
  --output_dir asset/bench/WD_new \
  --segment_frames 100 \
  --step 1 \
  --angle_deg 50

python asset/gen_traj.py \
  --dirs s,s,s,s \
  --output_dir asset/bench/S \
  --segment_frames 33 \
  --step 0.15 \
  --angle_deg 10

python asset/gen_traj.py \
  --dirs s,a,w,D \
  --output_dir asset/bench/S_A_W_D \
  --segment_frames 33 \
  --step 0.15 \
  --angle_deg 10
"""