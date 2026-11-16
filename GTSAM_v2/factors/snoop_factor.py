# snoop_factor.py
from typing import Any
import numpy as np
import gtsam  # type: ignore


def evaluate_snoop_geometry(
    pose_i: Any,
    pose_t: Any,
    pose_r1: Any,
    pose_r2: Any,
    measured_delta_t: float,
    sound_speed: float = 1500.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Evaluate the SNOOP residual and Pose2-local Jacobians.

    Keeping this calculation outside the ``CustomFactor`` callback makes the
    exact same nonlinear model available to the pre-update NIS diagnostic.
    """
    x_i, y_i, th_i = pose_i.x(), pose_i.y(), pose_i.theta()
    x_t, y_t, th_t = pose_t.x(), pose_t.y(), pose_t.theta()
    x_r1, y_r1, th_r1 = pose_r1.x(), pose_r1.y(), pose_r1.theta()
    x_r2, y_r2, th_r2 = pose_r2.x(), pose_r2.y(), pose_r2.theta()

    d_it = np.hypot(x_i - x_t, y_i - y_t) + 1e-6
    d_tr2 = np.hypot(x_t - x_r2, y_t - y_r2) + 1e-6
    d_ir1 = np.hypot(x_i - x_r1, y_i - y_r1) + 1e-6

    measured_distance_difference = float(measured_delta_t) * sound_speed
    residual = np.array(
        [d_it + d_tr2 - d_ir1 - measured_distance_difference],
        dtype=float,
    )

    dx_i_global = (x_i - x_t) / d_it - (x_i - x_r1) / d_ir1
    dy_i_global = (y_i - y_t) / d_it - (y_i - y_r1) / d_ir1
    h_i = np.array(
        [[
            dx_i_global * np.cos(th_i) + dy_i_global * np.sin(th_i),
            -dx_i_global * np.sin(th_i) + dy_i_global * np.cos(th_i),
            0.0,
        ]],
        dtype=float,
    )

    dx_t_global = (x_t - x_i) / d_it + (x_t - x_r2) / d_tr2
    dy_t_global = (y_t - y_i) / d_it + (y_t - y_r2) / d_tr2
    h_t = np.array(
        [[
            dx_t_global * np.cos(th_t) + dy_t_global * np.sin(th_t),
            -dx_t_global * np.sin(th_t) + dy_t_global * np.cos(th_t),
            0.0,
        ]],
        dtype=float,
    )

    dx_r1_global = (x_i - x_r1) / d_ir1
    dy_r1_global = (y_i - y_r1) / d_ir1
    h_r1 = np.array(
        [[
            dx_r1_global * np.cos(th_r1) + dy_r1_global * np.sin(th_r1),
            -dx_r1_global * np.sin(th_r1) + dy_r1_global * np.cos(th_r1),
            0.0,
        ]],
        dtype=float,
    )

    dx_r2_global = (x_r2 - x_t) / d_tr2
    dy_r2_global = (y_r2 - y_t) / d_tr2
    h_r2 = np.array(
        [[
            dx_r2_global * np.cos(th_r2) + dy_r2_global * np.sin(th_r2),
            -dx_r2_global * np.sin(th_r2) + dy_r2_global * np.cos(th_r2),
            0.0,
        ]],
        dtype=float,
    )
    return residual, [h_i, h_t, h_r1, h_r2]


def create_quaternary_snoop_factor(
    idx_init,
    idx_turn,
    idx_recv1,
    idx_recv2,
    measured_delta_t,
    noise_model,
    sound_speed: float = 1500.0,
    recv1_offset: Any = None,
    recv2_offset: Any = None,
) -> Any:
    """
    创建一个连接四个节点的四元 TDOA 被动监听因子 (Quaternary Snoop Factor)

    参数:
        idx_init: 发起通讯的节点 ID (如 B_send)
        idx_turn: 收到通讯并瞬间折返的节点 ID (如 A_turnaround)
        idx_recv1: 监听者首次收到发起波的节点 ID (如 C_recv1)
        idx_recv2: 监听者收到折返波的节点 ID (如 C_recv2)
        measured_delta_t: 监听者测得的两次信号到达的物理时间差 (标量, 秒)
        noise_model: 噪声模型 (在距离量纲下，应包含根号2倍的方差放大)
    """

    recv1_offset = recv1_offset if recv1_offset is not None else gtsam.Pose2()
    recv2_offset = recv2_offset if recv2_offset is not None else gtsam.Pose2()
    logical_keys = [idx_init, idx_turn, idx_recv1, idx_recv2]
    unique_keys = []
    logical_to_unique = []
    unique_index_by_key: dict[int, int] = {}
    for key in logical_keys:
        key_int = int(key)
        if key_int not in unique_index_by_key:
            unique_index_by_key[key_int] = len(unique_keys)
            unique_keys.append(key)
        logical_to_unique.append(unique_index_by_key[key_int])

    def error_func(this, values, H):
        pose_i = values.atPose2(idx_init)  # B_send
        pose_t = values.atPose2(idx_turn)  # A_turn
        pose_r1 = values.atPose2(idx_recv1).compose(recv1_offset)
        pose_r2 = values.atPose2(idx_recv2).compose(recv2_offset)
        error, jacobians = evaluate_snoop_geometry(
            pose_i,
            pose_t,
            pose_r1,
            pose_r2,
            measured_delta_t,
            sound_speed,
        )
        if H is not None:
            # Receiver Jacobians are initially with respect to the interpolated
            # event poses.  Map them back to their sensor-node poses.
            jacobians[2] = jacobians[2] @ np.asarray(
                recv1_offset.inverse().AdjointMap(), dtype=float
            )
            jacobians[3] = jacobians[3] @ np.asarray(
                recv2_offset.inverse().AdjointMap(), dtype=float
            )
            accumulated = [
                np.zeros((1, 3), dtype=float) for _ in unique_keys
            ]
            for logical_index, jacobian in enumerate(jacobians):
                accumulated[logical_to_unique[logical_index]] += jacobian
            for unique_index, jacobian in enumerate(accumulated):
                H[unique_index] = jacobian

        return error

    return gtsam.CustomFactor(noise_model, unique_keys, error_func)  # type: ignore
