from typing import Any
import numpy as np
import gtsam  # type: ignore


def create_ternary_twtt_factor(idx_send, idx_recv, idx_beacon, measured_distance, noise_model) -> Any:
    """
    创建一个连接 发送节点、接收节点 和 信标节点 的三元 TWTT 因子
    """

    def error_func(this, values, H):
        # 1. 提取三个节点的位姿 (Pose2)
        pose_s = values.atPose2(idx_send)
        pose_r = values.atPose2(idx_recv)
        pose_b = values.atPose2(idx_beacon)

        x_s, y_s, th_s = pose_s.x(), pose_s.y(), pose_s.theta()
        x_r, y_r, th_r = pose_r.x(), pose_r.y(), pose_r.theta()
        x_b, y_b, th_b = pose_b.x(), pose_b.y(), pose_b.theta()

        # 2. 计算预测距离
        d_s = np.sqrt((x_s - x_b) ** 2 + (y_s - y_b) ** 2) + 1e-6
        d_r = np.sqrt((x_r - x_b) ** 2 + (y_r - y_b) ** 2) + 1e-6

        # 3. 计算残差
        error = np.array([d_s + d_r - measured_distance])

        # 4. 计算雅可比矩阵
        if H is not None:
            # --- H1: 对发送节点的导数 ---
            dx_s_global = (x_s - x_b) / d_s
            dy_s_global = (y_s - y_b) / d_s
            H[0] = np.array([[dx_s_global * np.cos(th_s) + dy_s_global * np.sin(th_s),
                              -dx_s_global * np.sin(th_s) + dy_s_global * np.cos(th_s),
                              0.0]], dtype=float)

            # --- H2: 对接收节点的导数 ---
            dx_r_global = (x_r - x_b) / d_r
            dy_r_global = (y_r - y_b) / d_r
            H[1] = np.array([[dx_r_global * np.cos(th_r) + dy_r_global * np.sin(th_r),
                              -dx_r_global * np.sin(th_r) + dy_r_global * np.cos(th_r),
                              0.0]], dtype=float)

            # --- H3: 对信标节点的导数 ---
            # 根据链式法则：de/dx_b = -(x_s-x_b)/d_s - (x_r-x_b)/d_r
            dx_b_global = - (x_s - x_b) / d_s - (x_r - x_b) / d_r
            dy_b_global = - (y_s - y_b) / d_s - (y_r - y_b) / d_r

            H[2] = np.array([[dx_b_global * np.cos(th_b) + dy_b_global * np.sin(th_b),
                              -dx_b_global * np.sin(th_b) + dy_b_global * np.cos(th_b),
                              0.0]], dtype=float)

        return error

    return gtsam.CustomFactor(noise_model, [idx_send, idx_recv, idx_beacon], error_func)  # type: ignore
