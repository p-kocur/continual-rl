import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

class ShipSailingEnv(gym.Env):
    """
    A continuous grid environment (POMDP) where a ship sails through a sea with obstacles.
    State space: 2D continuous coordinates [x, y] in range [0, 50].
    Action space: Discrete 4 (0: left, 1: right, 2: up, 3: down).
    Hidden context: Wind changing every 15 turns, adding a force [w_x, w_y].
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.grid_size = 50.0

        # State: [x, y]
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype=np.float32)
        # Action: Discrete 4 (0: left, 1: right, 2: up, 3: down)
        self.action_space = spaces.Discrete(4)

        # Map discrete actions to continuous forces for the actual dynamics
        self.action_map = {
            0: np.array([-1.0, 0.0], dtype=np.float32), # Left
            1: np.array([1.0, 0.0], dtype=np.float32),  # Right
            2: np.array([0.0, -1.0], dtype=np.float32), # Up
            3: np.array([0.0, 1.0], dtype=np.float32),  # Down
        }

        self.target = np.array([45.0, 45.0], dtype=np.float32)

        # Define some obstacles as [x_min, x_max, y_min, y_max]
        self.obstacles = [
            [10, 15, 0, 30],
            [25, 30, 20, 50],
            [40, 45, 0, 40]
        ]

        self.max_steps = 200
        self.wind_change_freq = 15
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([5.0, 5.0], dtype=np.float32)
        self.step_count = 0
        self._sample_wind()

        return self.state, {}

    def _sample_wind(self):
        # Wind pushes in a random direction with magnitude up to 0.5
        angle = self.np_random.uniform(0, 2 * np.pi)
        mag = self.np_random.uniform(0.1, 0.5)
        self.wind = np.array([mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float32)
        self.wind = 0 # For now to test PPO

    def _in_obstacle(self, pos):
        x, y = pos
        for obs in self.obstacles:
            if obs[0] <= x <= obs[1] and obs[2] <= y <= obs[3]:
                return True
        return False

    def step(self, action):
        self.step_count += 1

        # Map discrete action to continuous force
        if isinstance(action, np.ndarray) and action.shape == (2,):
            # If continuous action is passed (e.g. from differentiable BPTT policy)
            continuous_action = np.clip(action, -1.0, 1.0)
        else:
            # If discrete action is passed (e.g. from normal gym stepping)
            if isinstance(action, np.ndarray) and action.size == 1:
                action = int(action.item())
            elif isinstance(action, torch.Tensor):
                action = int(action.item())
            else:
                action = int(action)
            continuous_action = self.action_map[action]

        # Update wind
        if self.step_count % self.wind_change_freq == 0:
            self._sample_wind()

        # Actual transition: s_next = s + a + wind
        next_state = self.state + continuous_action + self.wind

        # Collision checking (bounce back)
        if self._in_obstacle(next_state) or \
           next_state[0] < 0 or next_state[0] > self.grid_size or \
           next_state[1] < 0 or next_state[1] > self.grid_size:
            # Simple bounce (stay in place)
            next_state = self.state
            reward = -10.0  # Penalty for collision
        else:
            # Reward is negative distance to target
            dist = np.linalg.norm(next_state - self.target)
            reward = -dist * 0.1

        self.state = np.clip(next_state, 0, self.grid_size).astype(np.float32)

        terminated = np.linalg.norm(self.state - self.target) < 5.0
        if terminated:
            reward += 1000.0

        truncated = self.step_count >= self.max_steps

        return self.state, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "ansi":
            return

        # Downscale 50x50 to a manageable 25x25 grid for terminal
        scale = 2.0
        grid = [["." for _ in range(25)] for _ in range(25)]

        # Draw obstacles
        for obs in self.obstacles:
            x_min, x_max, y_min, y_max = [int(v / scale) for v in obs]
            for i in range(x_min, min(x_max + 1, 25)):
                for j in range(y_min, min(y_max + 1, 25)):
                    if 0 <= i < 25 and 0 <= j < 25:
                        grid[j][i] = "#"

        # Draw target
        tx, ty = int(self.target[0] / scale), int(self.target[1] / scale)
        if 0 <= tx < 25 and 0 <= ty < 25:
            grid[ty][tx] = "T"

        # Draw ship
        sx, sy = int(self.state[0] / scale), int(self.state[1] / scale)
        if 0 <= sx < 25 and 0 <= sy < 25:
            grid[sy][sx] = "S"

        # Print with Wind info
        out = "\n".join(["".join(row) for row in grid])
        out = f"Step: {self.step_count} | Wind: [{self.wind[0]:.2f}, {self.wind[1]:.2f}]\n" + out
        return out
