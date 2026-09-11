import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

class QuadrotorEnv(gym.Env):
    """
    A wrapper around gym-pybullet-drones specifically tailored for
    the Continual VND tracking task with future lookahead and hidden wind.

    The base environment is HoverAviary, which we adapt for Trajectory Tracking.
    State (12D): [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    """
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, render_mode=None, enable_wind=True):
        super().__init__()
        self.render_mode = render_mode
        self.enable_wind = enable_wind

        # PyBullet drone physics
        gui = (render_mode == "human")
        self.drone_env = HoverAviary(gui=gui,
                                     record=False,
                                     obs=ObservationType.KIN,
                                     act=ActionType.RPM)

        self.dt = self.drone_env.CTRL_TIMESTEP
        self.max_steps = 300
        self.wind_change_freq = 50
        self.lookahead = 5

        # RPM Action space mapped from [-1, 1] bounds from PyTorch policy
        # We will scale this internally to motor RPMs
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: [state(12), ref_t(3), ref_{t+1}(3), ..., ref_{t+H-1}(3)]
        # For PyBullet drones, state is typically 12D for 3D physics (pos, orn, vel, ang_vel)
        # We'll stick to 3D position reference tracking
        obs_dim = 12 + 3 * self.lookahead
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _get_reference(self, t):
        # 3D Figure-8 Lissajous curve
        time = t * self.dt
        a, b, c = 0.5, 1.0, 0.5 # Frequencies
        A, B, C = 1.0, 1.0, 0.5 # Amplitudes
        z_offset = 1.0

        px = A * np.sin(a * time)
        py = B * np.sin(b * time)
        pz = z_offset + C * np.sin(c * time)

        return np.array([px, py, pz], dtype=np.float32)

    def _sample_wind(self):
        if not self.enable_wind:
            self.wind_force = np.array([0.0, 0.0, 0.0])
            return

        # Random wind force vector in 3D
        mag = self.np_random.uniform(0.1, 0.5) # Force in Newtons
        phi = self.np_random.uniform(0, 2*np.pi)
        theta = self.np_random.uniform(0, np.pi)

        wx = mag * np.sin(theta) * np.cos(phi)
        wy = mag * np.sin(theta) * np.sin(phi)
        wz = mag * np.cos(theta)

        self.wind_force = np.array([wx, wy, wz])

    def _apply_wind(self):
        if self.enable_wind and hasattr(self, 'wind_force'):
            # Directly apply wind force to the base link of the drone in PyBullet
            import pybullet as p
            p.applyExternalForce(
                self.drone_env.DRONE_IDS[0],
                -1, # Base link
                forceObj=self.wind_force.tolist(),
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.drone_env.CLIENT
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset internal pybullet environment
        init_pos = self._get_reference(0)
        # We have to bypass standard reset to force a specific starting position
        self.drone_env.INIT_XYZS = np.array([init_pos])
        obs_raw, info_raw = self.drone_env.reset(seed=seed)

        self.step_count = 0
        self._sample_wind()

        # obs_raw[0] contains the KIN obs for drone 0
        # KIN obs format: [pos(3), quat(4), rpy(3), vel(3), ang_vel(3), last_clipped_action(4)] = 20
        # We will extract standard 12D: pos, rpy, vel, ang_vel
        self._update_state(obs_raw[0])

        return self._get_obs(), {"true_state": self._get_obs()}

    def _update_state(self, kin_obs):
        pos = kin_obs[0:3]
        rpy = kin_obs[7:10]
        vel = kin_obs[10:13]
        ang_vel = kin_obs[13:16]
        self.state = np.concatenate([pos, rpy, vel, ang_vel])

    def _get_obs(self):
        obs = [self.state]
        for i in range(self.lookahead):
            obs.append(self._get_reference(self.step_count + i))
        return np.concatenate(obs).astype(np.float32)

    def step(self, action):
        self.step_count += 1

        if self.step_count % self.wind_change_freq == 0:
            self._sample_wind()

        self._apply_wind()

        # Map policy action [-1, 1] to hover RPM range
        # Hover RPM is roughly 14468 for default CF2x drone
        hover_rpm = self.drone_env.HOVER_RPM
        max_rpm = self.drone_env.MAX_RPM

        # action bounds mapping
        # 0 -> hover, -1 -> min, 1 -> max
        action_scaled = np.zeros(4)
        for i in range(4):
            if action[i] > 0:
                action_scaled[i] = hover_rpm + action[i] * (max_rpm - hover_rpm) * 0.5
            else:
                action_scaled[i] = hover_rpm + action[i] * hover_rpm * 0.5

        # Must be nested array for single drone in gym-pybullet-drones
        pb_action = np.array([action_scaled])

        obs_raw, _, terminated_raw, truncated_raw, info_raw = self.drone_env.step(pb_action)
        self._update_state(obs_raw[0])

        # Calculate custom tracking reward
        pos = self.state[0:3]
        ref_pos = self._get_reference(self.step_count)

        pos_error = np.linalg.norm(pos - ref_pos)
        reward = -pos_error

        # Terminate if the drone falls or flies way off track
        terminated = terminated_raw or pos_error > 2.0 or pos[2] < 0.05
        truncated = truncated_raw or self.step_count >= self.max_steps

        if terminated:
            reward -= 50.0

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {"true_state": self._get_obs()}

    def render(self):
        pass
