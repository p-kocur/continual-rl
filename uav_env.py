import gymnasium as gym
from gymnasium import spaces
import numpy as np

class UAVEnv(gym.Env):
    """
    Continuous 2D UAV Trajectory Tracking Environment.
    State: [px, py, vx, vy] - Position and velocity of the UAV.
    Observation: [px, py, vx, vy, ref_px, ref_py, ref_vx, ref_vy]
    Action: [ax, ay] - Commanded accelerations.
    Hidden Context: Wind [wx, wy] altering the acceleration dynamics.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None, enable_wind=True):
        super().__init__()
        self.render_mode = render_mode
        self.enable_wind = enable_wind

        self.dt = 0.1
        self.max_steps = 200
        self.wind_change_freq = 50

        # Action: [ax, ay] bounded between [-5, 5]
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)

        # Observation: [state(4), ref_state(4)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.reset()

    def _get_reference(self, t):
        # Figure-8 (Lissajous) trajectory
        # x(t) = A * sin(a * t)
        # y(t) = B * sin(b * t)
        # For a standard figure 8, a=1, b=2
        time = t * self.dt
        A, B = 2.0, 2.0
        a, b = 1.0, 2.0

        px = A * np.sin(a * time)
        py = B * np.sin(b * time)

        vx = A * a * np.cos(a * time)
        vy = B * b * np.cos(b * time)

        return np.array([px, py, vx, vy], dtype=np.float32)

    def _sample_wind(self):
        if not self.enable_wind:
            self.wind = np.array([0.0, 0.0], dtype=np.float32)
            return

        # Wind is a constant force [wx, wy]
        angle = self.np_random.uniform(0, 2 * np.pi)
        mag = self.np_random.uniform(0.5, 2.0)
        self.wind = np.array([mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Start exactly on the trajectory
        self.state = self._get_reference(self.step_count)
        self._sample_wind()

        return self._get_obs(), {"true_state": self.state.copy()}

    def _get_obs(self):
        ref_state = self._get_reference(self.step_count)
        return np.concatenate([self.state, ref_state]).astype(np.float32)

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, -5.0, 5.0)

        if self.step_count % self.wind_change_freq == 0:
            self._sample_wind()

        # True Dynamics:
        # v_{t+1} = v_t + (a_t + wind) * dt
        # p_{t+1} = p_t + v_t * dt

        px, py, vx, vy = self.state
        ax, ay = action
        wx, wy = self.wind

        new_vx = vx + (ax + wx) * self.dt
        new_vy = vy + (ay + wy) * self.dt

        new_px = px + new_vx * self.dt
        new_py = py + new_vy * self.dt

        self.state = np.array([new_px, new_py, new_vx, new_vy], dtype=np.float32)

        # Reward is negative tracking error
        ref_state = self._get_reference(self.step_count)
        pos_error = np.linalg.norm(self.state[:2] - ref_state[:2])
        vel_error = np.linalg.norm(self.state[2:] - ref_state[2:])

        reward = -(pos_error + 0.1 * vel_error)

        terminated = False # Continuous tracking task
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, {"true_state": self.state.copy()}
