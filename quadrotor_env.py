import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class QuadrotorEnv(gym.Env):
    """
    A high-level wrapper around gym-pybullet-drones for Continual VND tracking.

    The internal DSLPIDControl handles low-level motor mixing, so algorithms see
    a clean 3D position-correction interface — analogous to the 2D UAVEnv.

    State (6D):  [px, py, pz, vx, vy, vz]
    Observation: [state(6), ref_t(3), ref_{t+1}(3), ..., ref_{t+H-1}(3)]  → 21D
    Action (3D): [dx, dy, dz]  position offset from reference, bounded [-0.5, 0.5] m
    Hidden:      Wind force (XY horizontal, small), learned by VND residual.

    The PID target is: ref_pos + action.
    With action=0, the PID drives toward the reference directly.
    Wind pushes the drone off; VND learns the residual correction.
    """
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, render_mode=None, enable_wind=True):
        super().__init__()
        self.render_mode = render_mode
        self.enable_wind = enable_wind

        gui = (render_mode == "human")
        self.drone_env = CtrlAviary(gui=gui, record=False, ctrl_freq=48)
        self.pid = DSLPIDControl(drone_model=self.drone_env.DRONE_MODEL)

        self.dt = self.drone_env.CTRL_TIMESTEP   # 1/48 s ≈ 0.0208 s
        self.max_steps = 500
        self.wind_change_freq = 50
        self.lookahead = 15

        # Action: position correction [dx, dy, dz] in meters (safe micro-drone flight envelope)
        self.action_space = spaces.Box(low=-0.15, high=0.15, shape=(3,), dtype=np.float32)

        # Observation: [pos(3), vel(3), ref×H(3 each)] = 6 + 3*15 = 51D
        obs_dim = 6 + 3 * self.lookahead
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def _get_reference(self, t):
        """3D Lissajous figure-8 trajectory."""
        time = t * self.dt
        A, B, C = 1.0, 1.0, 0.3
        a, b, c = 0.5, 1.0, 0.5
        z_offset = 1.0

        px = A * np.sin(a * time)
        py = B * np.sin(b * time)
        pz = z_offset + C * np.sin(c * time)
        return np.array([px, py, pz], dtype=np.float32)

    def _sample_wind(self):
        if not self.enable_wind:
            self.wind_force = np.zeros(3)
            return
        # Horizontal-only wind (~10-30% of drone weight = 0.027*9.81 ≈ 0.265 N)
        mag = self.np_random.uniform(0.02, 0.07)
        phi = self.np_random.uniform(0, 2 * np.pi)
        self.wind_force = np.array([mag * np.cos(phi), mag * np.sin(phi), 0.0])

    def _apply_wind(self):
        if not self.enable_wind:
            return
        import pybullet as p
        drone_pos = self.state[0:3].tolist()
        p.applyExternalForce(
            self.drone_env.DRONE_IDS[0], -1,
            forceObj=self.wind_force.tolist(),
            posObj=drone_pos,
            flags=p.WORLD_FRAME,
            physicsClientId=self.drone_env.CLIENT
        )

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        init_pos = self._get_reference(0)
        self.drone_env.INIT_XYZS = np.array([init_pos])
        obs_raw, _ = self.drone_env.reset(seed=seed)

        self.step_count = 0
        self._sample_wind()
        self.pid.reset()

        self._update_state(obs_raw[0])
        return self._get_obs(), {"true_state": self._get_obs()}

    def _update_state(self, kin_obs):
        """Extract pos + vel from CtrlAviary 20D KIN obs."""
        # KIN obs: [pos(3), quat(4), rpy(3), vel(3), ang_vel(3), last_action(4)] = 20D
        pos = kin_obs[0:3]
        vel = kin_obs[10:13]
        self.state = np.concatenate([pos, vel])
        # Save full obs for PID (needs quat, ang_vel)
        self._full_kin = kin_obs

    def _get_obs(self):
        obs = [self.state]
        for i in range(self.lookahead):
            obs.append(self._get_reference(self.step_count + i))
        return np.concatenate(obs).astype(np.float32)

    # ------------------------------------------------------------------
    def step(self, action):
        self.step_count += 1

        if self.step_count % self.wind_change_freq == 0:
            self._sample_wind()

        # Apply wind force before physics step
        self._apply_wind()

        # Action = position offset from reference [dx, dy, dz]
        action = np.clip(action, -0.15, 0.15)

        ref_pos = self._get_reference(self.step_count)
        cur_pos = self.state[0:3]
        cur_vel = self.state[3:6]
        cur_quat = self._full_kin[3:7]
        cur_ang_vel = self._full_kin[13:16]

        # PID target = reference position corrected by policy action
        target_pos = (ref_pos + action).astype(np.float64)

        # PID computes RPMs
        rpm, _, _ = self.pid.computeControl(
            control_timestep=self.dt,
            cur_pos=cur_pos.astype(np.float64),
            cur_quat=cur_quat.astype(np.float64),
            cur_vel=cur_vel.astype(np.float64),
            cur_ang_vel=cur_ang_vel.astype(np.float64),
            target_pos=target_pos,
        )

        pb_action = np.array([rpm])
        obs_raw, _, _, _, _ = self.drone_env.step(pb_action)
        self._update_state(obs_raw[0])

        pos = self.state[0:3]
        ref_pos_now = self._get_reference(self.step_count)
        pos_error = np.linalg.norm(pos - ref_pos_now)

        reward = -pos_error

        terminated = pos_error > 2.0 or pos[2] < 0.05
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward -= 50.0

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {"true_state": self._get_obs()}

    def render(self):
        pass
