import torch
import numpy as np

class MPCTrainer:
    """
    Model Predictive Control using Cross-Entropy Method (CEM).
    Since it plans online, 'Trainer' here just means the decision-making agent.
    """
    def __init__(self, action_dim, f_prior, reward_fn, horizon=15, num_particles=200, num_iters=3, elite_frac=0.1, device="cpu"):
        self.action_dim = action_dim
        self.f_prior = f_prior
        self.reward_fn = reward_fn

        self.horizon = horizon
        self.num_particles = num_particles
        self.num_iters = num_iters
        self.num_elites = max(1, int(self.num_particles * elite_frac))
        self.device = device

        self.action_min = -5.0
        self.action_max = 5.0

    def select_action(self, state):
        """
        Uses CEM to find the best action sequence and returns the first action.
        state: 1D numpy array representing the current state observation
        """
        with torch.no_grad():
            # Initial action distribution: mean 0, variance covers the range
            action_mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
            action_std = torch.ones(self.horizon, self.action_dim, device=self.device) * 2.5

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            for _ in range(self.num_iters):
                # Sample action sequences: [num_particles, horizon, action_dim]
                actions = action_mean.unsqueeze(0) + action_std.unsqueeze(0) * torch.randn(
                    self.num_particles, self.horizon, self.action_dim, device=self.device
                )
                actions = torch.clamp(actions, self.action_min, self.action_max)

                # Evaluate trajectories
                returns = torch.zeros(self.num_particles, device=self.device)

                # Expand state to match number of particles
                curr_states = state_tensor.repeat(self.num_particles, 1)

                for t in range(self.horizon):
                    curr_actions = actions[:, t, :]

                    # Predict next state using the known physics model
                    next_states = self.f_prior(curr_states, curr_actions)

                    # Compute rewards
                    rewards = self.reward_fn(curr_states, curr_actions, next_states)
                    returns += rewards

                    curr_states = next_states

                # Select elites
                _, elite_indices = torch.topk(returns, self.num_elites)
                elite_actions = actions[elite_indices] # [num_elites, horizon, action_dim]

                # Update distribution
                action_mean = elite_actions.mean(dim=0)
                action_std = elite_actions.std(dim=0)
                # Add small noise to prevent premature convergence to 0 variance
                action_std = torch.clamp(action_std, min=0.1)

            # Return the first action of the optimized mean sequence
            best_action = action_mean[0].detach().cpu().numpy()

        return np.clip(best_action, self.action_min, self.action_max)
