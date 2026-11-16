# per_auv_estimator.py
# ============================================================================
# GTSAM_v2 — 单 AUV 因子图估计器（泛化版）
#
# 职责:
#   1. 每个 slave AUV 持有一个独立的 iSAM2 实例
#   2. 管理"自己"和"远程 AUV"的节点序列与初始估计
#   3. 处理里程计边、TWTT 因子、SNOOP 因子
#   4. 从旧版 GTSAM/TWTT+SNOOP/gtsam_estimator.py 泛化而来
#
# 关键变化:
#   - 旧版硬编码 agents = {'A':..., 'B':..., 'C':...}
#   - 新版: agents = {owner_id: {...}, remote_id_1: {...}, ...}  自动推导
# ============================================================================

import math
from typing import Any, Optional
import numpy as np
import gtsam  # type: ignore

from factors.twtt_factor import create_ternary_twtt_factor
from factors.snoop_factor import (
    create_quaternary_snoop_factor,
    evaluate_snoop_geometry,
)
from estimator.consistency_diagnostics import (
    chi_square_interval,
    pose_consistency,
    scalar_nis,
)
from estimator.odometry_preintegration import (
    preintegrate_constant_measurement,
)


class PerAUVEstimator:
    """
    单个从 AUV 的因子图估计器。

    参数:
        owner_profile   : 本 AUV 的配置 dict (来自 experiment_config.AUV_LIST)
        all_profiles    : 所有 AUV 的配置列表
        remote_ids      : 本 AUV 需要跟踪的远程 AUV ID 列表 (由 comms 自动推导)
        global_cfg      : experiment_config.GLOBAL
        noise_cfg       : experiment_config.NOISE
        prior_cfg       : experiment_config.PRIOR
    """

    def __init__(
        self,
        owner_profile: dict,
        all_profiles: list,
        remote_ids: list[str],
        global_cfg: dict,
        noise_cfg: dict,
        prior_cfg: dict,
        estimator_cfg: Optional[dict] = None,
    ):
        self.owner_id = owner_profile["id"]
        self.remote_ids = remote_ids
        self.global_cfg = global_cfg
        self.noise_cfg = noise_cfg
        self.prior_cfg = prior_cfg
        self.estimator_cfg = estimator_cfg or {}

        # ---- 所有需要管理的 AUV ID（自己 + 远程） ----
        self.all_managed_ids = [self.owner_id] + remote_ids

        # ---- iSAM2 初始化 ----
        isam2_params = gtsam.ISAM2Params()
        isam2_params.setRelinearizeThreshold(0.1)
        isam2_params.relinearizeSkip = 1
        self.isam2 = gtsam.ISAM2(isam2_params)

        # ---- 因子图缓冲 ----
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()

        # ---- 并行增量固定滞后平滑器（运行估计） ----
        self.fixed_lag_enabled = bool(
            self.estimator_cfg.get("fixed_lag_enabled", False)
        )
        self.fixed_lag_seconds = float(
            self.estimator_cfg.get("fixed_lag_seconds", 0.0)
        )
        self.fixed_lag_smoother = None
        self.fixed_lag_graph = gtsam.NonlinearFactorGraph()
        self.fixed_lag_initial_estimates = gtsam.Values()
        self.fixed_lag_timestamps = None
        self.fixed_lag_latest_time = 0.0
        if self.fixed_lag_enabled:
            if self.fixed_lag_seconds <= 0.0:
                raise ValueError("fixed_lag_seconds must be positive")
            try:
                import gtsam_unstable  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "fixed_lag_enabled requires gtsam_unstable"
                ) from exc
            self.fixed_lag_smoother = (
                gtsam_unstable.IncrementalFixedLagSmoother(
                    self.fixed_lag_seconds,
                    isam2_params,
                )
            )
            self.fixed_lag_timestamps = (
                gtsam_unstable.FixedLagSmootherKeyTimestampMap()
            )

        # ---- 每个 AUV 的状态追踪 ----
        # agents[agent_id] = {
        #     "time": 上一次打点的时间,
        #     "idx":  当前节点序号,
        #     "pose": 最新估计位姿 (gtsam.Pose2),
        # }
        self.agents: dict[str, dict] = {}

        for auv_id in self.all_managed_ids:
            profile = self._find_profile(auv_id, all_profiles)
            init_pose = profile["init_pose"]
            # 从 AUV 自身估计的角度，初始航向永远是 0（不知道真实偏置）
            pose = gtsam.Pose2(init_pose[0], init_pose[1], 0.0)

            self.agents[auv_id] = {
                "time": 0.0,
                "idx": 0,
                "pose": pose,
            }

        # ---- 符号字典 (time → gtsam.Symbol) ----
        # sym_dict[agent_id][time] = symbol
        self.sym_dict: dict[str, dict[float, gtsam.Symbol]] = {}
        for auv_id in self.all_managed_ids:
            self.sym_dict[auv_id] = {}
            sym = gtsam.symbol(auv_id, 0)
            self.sym_dict[auv_id][0.0] = sym

        # ---- 真值历史（用于远程 AUV 的 Oracle 注入） ----
        self.gt_history: dict[str, dict[float, np.ndarray]] = {}
        for auv_id in self.all_managed_ids:
            self.gt_history[auv_id] = {}

        # ---- 初始化先验 ----
        self._initialize_priors(owner_profile, all_profiles)

    def _fixed_lag_factor_keys_are_active(self, factor: Any) -> bool:
        """判断延迟因子的全部变量是否仍位于固定滞后窗口内。

        IncrementalFixedLagSmoother 会在窗口左端点处淘汰变量。声学事件
        到达较晚时，完整历史 iSAM2 仍可合法加入该因子，但固定滞后副本不
        能再引用已边缘化的键，否则 calculateEstimate() 会抛出缺键异常。
        """
        cutoff = self.fixed_lag_latest_time - self.fixed_lag_seconds
        latest_time_by_key: dict[int, float] = {}
        for symbols_by_time in self.sym_dict.values():
            for state_time, symbol in symbols_by_time.items():
                key = int(symbol)
                latest_time_by_key[key] = max(
                    latest_time_by_key.get(key, -math.inf),
                    float(state_time),
                )
        return all(
            latest_time_by_key.get(int(key), -math.inf) > cutoff + 1e-9
            for key in factor.keys()
        )

    def _add_factor(
        self,
        factor: Any,
        *,
        fixed_lag_requires_active_keys: bool = False,
    ) -> None:
        """Add one factor to the full-history and fixed-lag buffers."""
        self.graph.add(factor)
        if (
            self.fixed_lag_enabled
            and (
                not fixed_lag_requires_active_keys
                or self._fixed_lag_factor_keys_are_active(factor)
            )
        ):
            self.fixed_lag_graph.add(factor)

    def _insert_initial(
        self,
        symbol: Any,
        pose: Any,
        timestamp: float,
    ) -> None:
        """Insert one new variable into both estimator buffers."""
        self.initial_estimates.insert(symbol, pose)
        if self.fixed_lag_enabled:
            self.fixed_lag_initial_estimates.insert(symbol, pose)
            self.fixed_lag_timestamps.insert(
                (symbol, float(timestamp))
            )
            self.fixed_lag_latest_time = max(
                self.fixed_lag_latest_time,
                float(timestamp),
            )

    # ==================================================================
    # 先验初始化
    # ==================================================================

    def _find_profile(self, auv_id: str, all_profiles: list) -> dict:
        for p in all_profiles:
            if p["id"] == auv_id:
                return p
        raise KeyError(f"Profile for AUV '{auv_id}' not found")

    def _initialize_priors(self, owner_profile: dict, all_profiles: list):
        """为所有管理的 AUV 注入适当的初始先验"""
        # 独立随机数生成器，保证实验可复现
        seed_base = self.global_cfg["random_seed"]

        for auv_id in self.all_managed_ids:
            profile = self._find_profile(auv_id, all_profiles)
            role = profile["role"]
            sym = gtsam.symbol(auv_id, 0)
            init_pos = profile["init_pose"]

            if auv_id == self.owner_id:
                # === 本 AUV：在真实坐标附近随机化初始猜测 ===
                sigma_xy = self.prior_cfg["slave_sigma_xy"]
                sigma_theta = self.prior_cfg["slave_sigma_theta"]
                # 用 ord(auv_id[0]) 替代 hash()，保证跨进程确定性
                rng = np.random.RandomState(seed_base + ord(auv_id[0]) * 100)
                bias_deg = profile["init_heading_bias_deg"]
                zero_initial_error = bool(
                    self.prior_cfg.get("zero_initial_pose_error", False)
                )
                if zero_initial_error:
                    x_guess = float(init_pos[0])
                    y_guess = float(init_pos[1])
                    theta_guess = math.radians(bias_deg)
                else:
                    x_guess = init_pos[0] + rng.normal(0, sigma_xy)
                    y_guess = init_pos[1] + rng.normal(0, sigma_xy)
                    theta_guess = (
                        math.radians(bias_deg)
                        + rng.normal(0, sigma_theta)
                    )
                pose = gtsam.Pose2(x_guess, y_guess, theta_guess)

                print(f"[Init] {auv_id} | 真实位置: ({init_pos[0]:.1f}, {init_pos[1]:.1f}) "
                      f"真实航向: {bias_deg:.1f}°\n"
                      f"      先验均值: 位置 ({init_pos[0]:.1f}, {init_pos[1]:.1f}) "
                      f"航向均值: {bias_deg:.1f}°\n"
                      f"      先验方差: sigma_xy^2={sigma_xy**2:.2f} m^2  "
                      f"sigma_theta^2={math.degrees(sigma_theta)**2:.2f} deg^2\n"
                      f"      零初始位姿误差: {zero_initial_error}\n"
                      f"      初始猜测: ({x_guess:.2f}, {y_guess:.2f})  "
                      f"θ={math.degrees(theta_guess):.2f}°")
            elif role == "master":
                # === 远程 Master AUV：极高置信度锚定在真实坐标（含航向） ===
                sigma_xy = self.prior_cfg["master_sigma_xy"]
                sigma_theta = self.prior_cfg["master_sigma_theta"]
                heading_deg = profile["init_heading_bias_deg"]
                pose = gtsam.Pose2(init_pos[0], init_pos[1], math.radians(heading_deg))
            else:
                # === 远程 Slave AUV：仅锁 XY，释放航向 ===
                sigma_xy = self.prior_cfg["slave_sigma_xy"]
                sigma_theta = self.prior_cfg["slave_sigma_theta"]
                pose = gtsam.Pose2(init_pos[0], init_pos[1], 0.0)

            noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([sigma_xy, sigma_xy, sigma_theta])
            )
            self._add_factor(gtsam.PriorFactorPose2(sym, pose, noise))
            self._insert_initial(sym, pose, 0.0)
            self.agents[auv_id]["pose"] = pose

    # ==================================================================
    # 里程计节点
    # ==================================================================

    def add_odom_node(
        self,
        agent_id: str,
        current_time: float,
        v: float,
        omega: float,
        true_pose: Optional[np.ndarray] = None,
        true_cov: Optional[np.ndarray] = None,
        is_remote: bool = False,
    ) -> Any:
        """
        为指定 AUV 加入里程计节点。

        参数:
            agent_id   : 打节点的 AUV（可以是自己或远程 AUV）
            current_time: 物理时间
            v, omega   : 线/角速度
            true_pose  : 如果非 None，注入 PriorFactor（Oracle 模式 / 协方差传递）
            true_cov   : 协方差矩阵（配合 true_pose 使用）
            is_remote  : 如果为 True，里程计边噪声极大（防止错误拉扯）

        返回:
            新节点的 gtsam.Symbol
        """
        agent = self.agents[agent_id]
        dt = current_time - agent["time"]
        # 最小时间阈值：防止 dt→0 导致信息矩阵数值爆炸
        MIN_DT = self.global_cfg["min_dt"]
        if dt <= 0 or dt < MIN_DT:
            # 时间太近，复用上一个节点，防止 σ 过小导致信息矩阵病态
            sym_existing = gtsam.symbol(agent_id, agent["idx"])
            self.sym_dict[agent_id][round(current_time, 4)] = sym_existing
            # 即使复用节点，仍需注入 Oracle 先验（如 A/B 的绝对坐标）
            if true_pose is not None:
                true_pose2 = gtsam.Pose2(true_pose[0], true_pose[1], true_pose[2])
                if true_cov is not None:
                    prior_noise = gtsam.noiseModel.Gaussian.Covariance(true_cov)
                else:
                    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                        np.array([1e-6, 1e-6, 1e-6])
                    )
                self._add_factor(
                    gtsam.PriorFactorPose2(
                        sym_existing, true_pose2, prior_noise
                    )
                )
            return sym_existing

        new_idx = agent["idx"] + 1
        sym_old = gtsam.symbol(agent_id, agent["idx"])
        sym_new = gtsam.symbol(agent_id, new_idx)

        # ---- 精确单轮车预积分与完整局部协方差 ----
        dvl_sigma = self.noise_cfg["dvl_vel_sigma"]
        lateral_sigma = self.noise_cfg.get(
            "odom_lateral_sigma", dvl_sigma
        )
        gyro_sigma_deg = self.noise_cfg["imu_gyro_sigma_deg"]
        gyro_sigma = math.radians(gyro_sigma_deg)
        delta_pose, odom_covariance = preintegrate_constant_measurement(
            velocity=v,
            angular_rate=omega,
            dt=dt,
            velocity_sigma=dvl_sigma,
            angular_rate_sigma=gyro_sigma,
            lateral_velocity_sigma=lateral_sigma,
        )

        # 远程节点：里程计边极松
        if is_remote:
            remote_noise = self.prior_cfg["remote_tracking_noise"]
            odom_covariance = np.eye(3, dtype=float) * remote_noise**2

        noise_model = gtsam.noiseModel.Gaussian.Covariance(
            odom_covariance
        )

        self._add_factor(
            gtsam.BetweenFactorPose2(sym_old, sym_new, delta_pose, noise_model)
        )

        # ---- 初始猜测 (暂存内存，不提交) ----
        final_pose_estimate = agent["pose"].compose(delta_pose)

        # ---- Oracle / 协方差注入 ----
        if true_pose is not None:
            true_pose2 = gtsam.Pose2(true_pose[0], true_pose[1], true_pose[2])
            if true_cov is not None:
                prior_noise = gtsam.noiseModel.Gaussian.Covariance(true_cov)
            else:
                prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                    np.array([1e-6, 1e-6, 1e-6])
                )
            self._add_factor(
                gtsam.PriorFactorPose2(
                    sym_new, true_pose2, prior_noise
                )
            )
            final_pose_estimate = true_pose2

        # ---- 唯一一次 insert ----
        self._insert_initial(sym_new, final_pose_estimate, current_time)

        # ---- 更新状态 ----
        agent["time"] = current_time
        agent["idx"] = new_idx
        agent["pose"] = final_pose_estimate

        # ---- 更新符号字典 ----
        self.sym_dict[agent_id][round(current_time, 4)] = sym_new

        return sym_new

    # ==================================================================
    # TWTT 因子
    # ==================================================================

    def handle_twtt_event(
        self,
        sym_send: Any,
        sym_recv: Any,
        sym_beacon: Any,
        measured_distance: float,
    ):
        """
        处理 TWTT 事件：注入三元测距因子。

        连接: send(发起者) → recv(发起者) → beacon(响应者折返瞬间)
        """
        acoustic_sigma = self.noise_cfg["acoustic_time_sigma"]
        sound_speed = self.global_cfg["sound_speed"]
        sigma_dist = acoustic_sigma * sound_speed
        noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma_dist)

        factor = create_ternary_twtt_factor(
            idx_send=sym_send,
            idx_recv=sym_recv,
            idx_beacon=sym_beacon,
            measured_distance=measured_distance,
            noise_model=noise_model,
        )
        # 完整历史图始终接收该声学因子；固定滞后副本仅在三个状态仍处于
        # 活动窗口时接收，避免延迟事件重新引用已边缘化变量。
        self._add_factor(factor, fixed_lag_requires_active_keys=True)

    # ==================================================================
    # SNOOP 因子
    # ==================================================================

    def interpolate_odom_state(
        self,
        agent_id: str,
        event_time: float,
        velocity: float,
        angular_rate: float,
    ) -> dict[str, Any]:
        """Interpolate from the latest sensor node to an event time.

        No additional graph node or odometry factor is created.  The returned
        covariance is later projected into the scalar SNOOP noise, avoiding
        independent double counting when one ZOH sensor sample is split by an
        acoustic arrival.
        """
        available = [
            (float(time), symbol)
            for time, symbol in self.sym_dict[agent_id].items()
            if float(time) <= float(event_time) + 1e-9
            and self.isam2.valueExists(symbol)
        ]
        if not available:
            raise KeyError(
                f"no odometry symbol for {agent_id} at/before {event_time}"
            )
        base_time, symbol = max(available, key=lambda item: item[0])
        dt = max(0.0, float(event_time) - base_time)
        dvl_sigma = float(self.noise_cfg["dvl_vel_sigma"])
        lateral_sigma = float(
            self.noise_cfg.get("odom_lateral_sigma", dvl_sigma)
        )
        gyro_sigma = math.radians(
            float(self.noise_cfg["imu_gyro_sigma_deg"])
        )
        offset, covariance = preintegrate_constant_measurement(
            velocity=velocity,
            angular_rate=angular_rate,
            dt=dt,
            velocity_sigma=dvl_sigma,
            angular_rate_sigma=gyro_sigma,
            lateral_velocity_sigma=lateral_sigma,
        )
        return {
            "symbol": symbol,
            "base_time": base_time,
            "event_time": float(event_time),
            "dt": dt,
            "offset": offset,
            "covariance": covariance,
        }

    def evaluate_snoop_innovation(
        self,
        sym_init: Any,
        sym_turn: Any,
        sym_recv1: Any,
        sym_recv2: Any,
        measured_delta_t: float,
        confidence: float = 0.95,
        recv1_offset: Any = None,
        recv2_offset: Any = None,
        recv1_offset_covariance: Optional[np.ndarray] = None,
        recv2_offset_covariance: Optional[np.ndarray] = None,
    ) -> dict[str, float | bool]:
        """Evaluate the strict pre-update scalar SNOOP NIS.

        The joint marginal is assembled in the exact key/Jacobian order.  This
        matters because ``JointMarginal.fullMatrix()`` follows sorted key order,
        which need not match the factor callback order.
        """
        keys = [sym_init, sym_turn, sym_recv1, sym_recv2]
        values = self.isam2.calculateEstimate()
        if not all(self.isam2.valueExists(key) for key in keys):
            return {
                "residual_m": float("nan"),
                "innovation_variance_m2": float("nan"),
                "innovation_sigma_m": float("nan"),
                "nis": float("nan"),
                "normalized_innovation": float("nan"),
                "nis_lower": float("nan"),
                "nis_upper": float("nan"),
                "nis_in_interval": False,
            }

        recv1_offset = (
            recv1_offset if recv1_offset is not None else gtsam.Pose2()
        )
        recv2_offset = (
            recv2_offset if recv2_offset is not None else gtsam.Pose2()
        )
        poses = [
            values.atPose2(sym_init),
            values.atPose2(sym_turn),
            values.atPose2(sym_recv1).compose(recv1_offset),
            values.atPose2(sym_recv2).compose(recv2_offset),
        ]
        residual, jacobians = evaluate_snoop_geometry(
            *poses,
            measured_delta_t=measured_delta_t,
            sound_speed=self.global_cfg["sound_speed"],
        )
        receiver_event_jacobians = [
            np.asarray(jacobians[2], dtype=float).copy(),
            np.asarray(jacobians[3], dtype=float).copy(),
        ]
        jacobians[2] = jacobians[2] @ np.asarray(
            recv1_offset.inverse().AdjointMap(), dtype=float
        )
        jacobians[3] = jacobians[3] @ np.asarray(
            recv2_offset.inverse().AdjointMap(), dtype=float
        )

        # Duplicate symbols are legal in the physical model (e.g. both wave
        # arrivals associated with one odometry node).  Merge their Jacobians
        # before requesting a joint marginal.
        unique_keys: list[Any] = []
        jacobian_by_key: dict[int, np.ndarray] = {}
        for key, jacobian in zip(keys, jacobians):
            key_int = int(key)
            if key_int not in jacobian_by_key:
                unique_keys.append(key)
                jacobian_by_key[key_int] = np.asarray(
                    jacobian, dtype=float
                ).copy()
            else:
                jacobian_by_key[key_int] += np.asarray(
                    jacobian, dtype=float
                )

        key_vector = gtsam.KeyVector()
        for key in unique_keys:
            key_vector.append(key)
        marginals = gtsam.Marginals(
            self.isam2.getFactorsUnsafe(),
            values,
        )
        joint = marginals.jointMarginalCovariance(key_vector)
        covariance = np.block([
            [
                np.asarray(joint.at(row_key, column_key), dtype=float)
                for column_key in unique_keys
            ]
            for row_key in unique_keys
        ])
        h = np.hstack([
            jacobian_by_key[int(key)] for key in unique_keys
        ])
        acoustic_sigma = (
            float(self.noise_cfg["acoustic_time_sigma"])
            * float(self.global_cfg["sound_speed"])
            * math.sqrt(2.0)
        )
        interpolation_variance = 0.0
        for event_jacobian, offset_covariance in zip(
            receiver_event_jacobians,
            [recv1_offset_covariance, recv2_offset_covariance],
        ):
            if offset_covariance is None:
                continue
            offset_covariance = np.asarray(
                offset_covariance, dtype=float
            )
            h_offset = event_jacobian.reshape(3)
            projected_offset = [
                sum(
                    float(offset_covariance[row, column])
                    * float(h_offset[column])
                    for column in range(3)
                )
                for row in range(3)
            ]
            interpolation_variance += sum(
                float(h_offset[index])
                * float(projected_offset[index])
                for index in range(3)
            )
        measurement_variance = (
            acoustic_sigma * acoustic_sigma
            + max(interpolation_variance, 0.0)
        )
        h_row = h.reshape(-1)
        projected = [
            sum(
                float(covariance[row, column]) * float(h_row[column])
                for column in range(len(h_row))
            )
            for row in range(len(h_row))
        ]
        state_variance = sum(
            float(h_row[index]) * float(projected[index])
            for index in range(len(h_row))
        )
        innovation_variance = float(
            state_variance + measurement_variance
        )
        nis, normalized = scalar_nis(
            float(residual[0]),
            innovation_variance,
        )
        nis_lower, nis_upper = chi_square_interval(1, confidence)
        return {
            "residual_m": float(residual[0]),
            "innovation_variance_m2": innovation_variance,
            "innovation_sigma_m": float(
                math.sqrt(max(innovation_variance, 0.0))
            ),
            "acoustic_sigma_m": acoustic_sigma,
            "interpolation_sigma_m": float(
                math.sqrt(max(interpolation_variance, 0.0))
            ),
            "measurement_sigma_m": float(
                math.sqrt(max(measurement_variance, 0.0))
            ),
            "nis": nis,
            "normalized_innovation": normalized,
            "nis_lower": nis_lower,
            "nis_upper": nis_upper,
            "nis_in_interval": bool(nis_lower <= nis <= nis_upper),
        }

    def evaluate_pose_consistency(
        self,
        symbol: Any,
        true_pose: np.ndarray,
        confidence: float = 0.95,
    ) -> dict[str, float | bool]:
        """Evaluate simulation-only position/Pose2 NEES at ``symbol``."""
        covariance = self.get_pose_covariance(symbol)
        if covariance is None or not self.isam2.valueExists(symbol):
            return {
                "position_error_m": float("nan"),
                "heading_error_rad": float("nan"),
                "nees_xy": float("nan"),
                "nees_pose": float("nan"),
                "nees_xy_lower": float("nan"),
                "nees_xy_upper": float("nan"),
                "nees_pose_lower": float("nan"),
                "nees_pose_upper": float("nan"),
                "nees_xy_in_interval": False,
                "nees_pose_in_interval": False,
            }
        estimate = self.isam2.calculateEstimatePose2(symbol)
        truth = gtsam.Pose2(
            float(true_pose[0]),
            float(true_pose[1]),
            float(true_pose[2]),
        )
        return pose_consistency(
            estimate,
            truth,
            covariance,
            confidence,
        )

    def handle_snoop_event(
        self,
        sym_init: Any,
        sym_turn: Any,
        sym_recv1: Any,
        sym_recv2: Any,
        measured_delta_t: float,
        t_C1: Optional[float] = None,
        recv1_offset: Any = None,
        recv2_offset: Any = None,
        measurement_sigma_m: Optional[float] = None,
    ):
        """
        处理 SNOOP 事件：注入四元 TDOA 因子。

        连接: initiator(发起) → turn(折返) → recv1(首次听到) → recv2(二次听到)
        """
        acoustic_sigma = self.noise_cfg["acoustic_time_sigma"]
        sound_speed = self.global_cfg["sound_speed"]
        sigma_dist = (
            float(measurement_sigma_m)
            if measurement_sigma_m is not None
            else acoustic_sigma * sound_speed * 1.41421356
        )
        noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma_dist)

        # t_recv1 和 t_recv2 统一使用最近邻里程计节点
        # 不再创建幽灵节点（σ=1e2），避免精度泄露
        if sym_recv1 is None:
            return  # 查找失败，跳过本次因子注入

        factor = create_quaternary_snoop_factor(
            idx_init=sym_init,
            idx_turn=sym_turn,
            idx_recv1=sym_recv1,
            idx_recv2=sym_recv2,
            measured_delta_t=measured_delta_t,
            noise_model=noise_model,
            sound_speed=self.global_cfg["sound_speed"],
            recv1_offset=recv1_offset,
            recv2_offset=recv2_offset,
        )
        # SNOOP往返延迟可能使C1落在固定滞后窗口左边界；完整历史图仍可
        # 使用该因子，固定滞后副本必须跳过已淘汰的四元因子。
        self._add_factor(factor, fixed_lag_requires_active_keys=True)

    # ==================================================================
    # 优化
    # ==================================================================

    def update(self):
        """执行 iSAM2 增量优化"""
        if self.graph.size() > 0:
            self.isam2.update(self.graph, self.initial_estimates)
            self.graph.resize(0)
            self.initial_estimates.clear()

            # 提取最新估计
            for auv_id in self.all_managed_ids:
                idx = self.agents[auv_id]["idx"]
                sym = gtsam.symbol(auv_id, idx)
                if self.isam2.valueExists(sym):
                    self.agents[auv_id]["pose"] = self.isam2.calculateEstimatePose2(sym)

        if (
            self.fixed_lag_enabled
            and self.fixed_lag_smoother is not None
            and (
                self.fixed_lag_graph.size() > 0
                or self.fixed_lag_initial_estimates.size() > 0
            )
        ):
            self.fixed_lag_smoother.update(
                self.fixed_lag_graph,
                self.fixed_lag_initial_estimates,
                self.fixed_lag_timestamps,
            )
            self.fixed_lag_graph.resize(0)
            self.fixed_lag_initial_estimates.clear()
            self.fixed_lag_timestamps.clear()

        return self.isam2.calculateEstimate()

    def get_fixed_lag_estimate(
        self,
        output_time: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Return the owner pose nearest the delayed operational epoch."""
        if (
            not self.fixed_lag_enabled
            or self.fixed_lag_smoother is None
        ):
            return None
        output_time = (
            float(self.agents[self.owner_id]["time"])
            if output_time is None
            else float(output_time)
        )
        if output_time + 1e-9 < self.fixed_lag_seconds:
            return None
        estimate = self.fixed_lag_smoother.calculateEstimate()
        target_time = output_time - self.fixed_lag_seconds
        candidates = [
            (float(time), symbol)
            for time, symbol in self.sym_dict[self.owner_id].items()
            if estimate.exists(symbol)
        ]
        if not candidates:
            return None
        state_time, symbol = min(
            candidates,
            key=lambda item: (
                abs(item[0] - target_time),
                item[0],
            ),
        )
        pose = estimate.atPose2(symbol)
        covariance = np.asarray(
            self.fixed_lag_smoother.getISAM2().marginalCovariance(
                symbol
            ),
            dtype=float,
        )
        covariance = 0.5 * (covariance + covariance.T)
        p_xy = covariance[:2, :2]
        trace_xy = float(p_xy[0, 0] + p_xy[1, 1])
        discriminant = math.sqrt(
            max(
                0.0,
                (float(p_xy[0, 0]) - float(p_xy[1, 1])) ** 2
                + 4.0 * float(p_xy[0, 1]) ** 2,
            )
        )
        maximum_eigenvalue = 0.5 * (trace_xy + discriminant)
        owner_symbols_retained = sum(
            1
            for _, candidate_symbol in candidates
            if estimate.exists(candidate_symbol)
        )
        return {
            "output_time": output_time,
            "state_time": state_time,
            "effective_lag_s": output_time - state_time,
            "pose": pose,
            "symbol": symbol,
            "covariance": covariance,
            "position_trace": trace_xy,
            "position_max_eigenvalue": maximum_eigenvalue,
            "owner_states_retained": owner_symbols_retained,
            "total_states_retained": int(estimate.size()),
        }

    def get_current_estimate(self) -> Any:
        """获取本 AUV 的最新位姿估计"""
        idx = self.agents[self.owner_id]["idx"]
        sym = gtsam.symbol(self.owner_id, idx)
        return self.isam2.calculateEstimatePose2(sym)

    def get_pose_covariance(self, symbol: Optional[Any] = None) -> Optional[np.ndarray]:
        """只读获取 Pose2 边缘协方差。

        该接口用于严格信息增益的离线/日志评价，不向外部控制层暴露后验协方差。
        """
        try:
            if symbol is None:
                idx = self.agents[self.owner_id]["idx"]
                symbol = gtsam.symbol(self.owner_id, idx)
            if not self.isam2.valueExists(symbol):
                return None
            covariance = np.asarray(
                self.isam2.marginalCovariance(symbol), dtype=float
            )
            covariance = 0.5 * (covariance + covariance.T)
            if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
                return None
            return covariance
        except Exception:
            return None
