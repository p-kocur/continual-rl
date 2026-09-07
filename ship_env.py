import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math

class ShipSailingEnv(gym.Env):
    """
    A continuous grid environment (POMDP) where a ship sails through a sea with obstacles.
    State space: 2D continuous coordinates [x, y] in range [0, 20].
    Observation space: 10D vector:
        - [dx, dy] to target
        - 8 raycasts (distances to nearest obstacle/wall)
    Action space: Discrete 4 (0: left, 1: right, 2: up, 3: down).
    Hidden context: Wind disabled for PPO baseline test.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None, enable_wind=True):
        super().__init__()
        self.render_mode = render_mode
        self.grid_size = 20.0
        self.enable_wind = enable_wind

        # Observation: target_dx, target_dy, 8 raycasts
        self.observation_space = spaces.Box(low=-self.grid_size, high=self.grid_size, shape=(10,), dtype=np.float32)
        # Action: Discrete 4 (0: left, 1: right, 2: up, 3: down)
        self.action_space = spaces.Discrete(4)

        # Map discrete actions to continuous forces for the actual dynamics
        self.action_map = {
            0: np.array([-1.0, 0.0], dtype=np.float32), # Left
            1: np.array([1.0, 0.0], dtype=np.float32),  # Right
            2: np.array([0.0, -1.0], dtype=np.float32), # Up
            3: np.array([0.0, 1.0], dtype=np.float32),  # Down
        }

        self.target = np.array([18.0, 20.0], dtype=np.float32)

        # Define some obstacles as [x_min, x_max, y_min, y_max] scaled down for 20x20
        self.obstacles = [
            [4, 6, 6, 12],
            [10, 12, 8, 15],
            [13, 16, 9, 12],
            [14, 20, 0, 5],
            [0, 15, 17, 20],
        ]

        self.max_steps = 100
        self.wind_change_freq = 15
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([2.0, 2.0], dtype=np.float32)
        self.step_count = 0
        self._sample_wind()

        return self._get_obs(), {"true_state": self.state.copy()}

    def _get_obs(self):
        # 1. Target vector
        target_vec = self.target - self.state

        # 2. Raycasts in 8 directions
        angles = [i * (np.pi / 4) for i in range(8)]
        rays = []

        max_dist = 20.0
        step_size = 0.5

        for angle in angles:
            dir_vec = np.array([np.cos(angle), np.sin(angle)])
            dist = 0.0
            pos = self.state.copy()
            hit = False

            while dist < max_dist and not hit:
                pos += dir_vec * step_size
                dist += step_size

                # Check bounds
                if pos[0] < 0 or pos[0] > self.grid_size or pos[1] < 0 or pos[1] > self.grid_size:
                    hit = True
                    break

                # Check obstacles
                if self._in_obstacle(pos):
                    hit = True
                    break

            rays.append(dist)

        return np.concatenate([target_vec, rays]).astype(np.float32)

    def _sample_wind(self):
        if self.enable_wind:
            angle = self.np_random.uniform(0, 2 * np.pi)
            mag = self.np_random.uniform(0.1, 0.5)
            self.wind = np.array([mag * np.cos(angle), mag * np.sin(angle)], dtype=np.float32)
        else:
            self.wind = np.array([0.0, 0.0], dtype=np.float32)

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

        terminated = np.linalg.norm(self.state - self.target) < 2.5
        if terminated:
            reward += 100.0

        truncated = self.step_count >= self.max_steps

        # For compatibility with VND's f_prior which needs absolute coordinates,
        # we provide the true 2D state in the info dict.
        return self._get_obs(), reward, terminated, truncated, {"true_state": self.state.copy()}

    def render(self):
        if self.render_mode != "ansi":
            return

        # 1:1 scale for 20x20
        grid = [["." for _ in range(21)] for _ in range(21)]

        # Draw obstacles
        for obs in self.obstacles:
            x_min, x_max, y_min, y_max = [int(v) for v in obs]
            for i in range(x_min, min(x_max + 1, 21)):
                for j in range(y_min, min(y_max + 1, 21)):
                    if 0 <= i < 21 and 0 <= j < 21:
                        grid[j][i] = "#"

        # Draw target
        tx, ty = int(self.target[0]), int(self.target[1])
        if 0 <= tx < 21 and 0 <= ty < 21:
            grid[ty][tx] = "T"

        # Draw ship
        sx, sy = int(self.state[0]), int(self.state[1])
        if 0 <= sx < 21 and 0 <= sy < 21:
            grid[sy][sx] = "S"

        # Print
        out = "\n".join(["".join(row) for row in grid])
        out = f"Step: {self.step_count} | Pos: [{self.state[0]:.1f}, {self.state[1]:.1f}]\n" + out
        return out
