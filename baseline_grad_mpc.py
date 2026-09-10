import torch
import torch.optim as optim
import numpy as np

class GradMPCTrainer:
    """
    Gradient-Based Model Predictive Control (Shooting Method).
    Exploits PyTorch autograd to compute exact gradients of the return
    with respect to the action sequence, vastly outperforming sampling-based CEM.
    """
    def __init__(self, action_dim, f_prior, reward_fn, horizon=15, num_iters=10, lr=0.1, device="cpu"):
        self.action_dim = action_dim
        self.f_prior = f_prior
        self.reward_fn = reward_fn

        self.horizon = horizon
        self.num_iters = num_iters
        self.lr = lr
        self.device = device

        self.action_min = -5.0
        self.action_max = 5.0

        # Warm start memory: shift the sequence for the next timestep
        self.action_seq = torch.zeros(self.horizon, self.action_dim, device=self.device)

    def select_action(self, state):
        """
        Uses Gradient Descent to optimize the action sequence and returns the first action.
        state: 1D numpy array representing the current state observation
        """
        # Shift warm-start sequence: a_t becomes a_{t-1}, last action is duplicated
        shifted_seq = torch.cat([self.action_seq[1:], self.action_seq[-1:]], dim=0)

        action_seq = shifted_seq.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([action_seq], lr=self.lr)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        for _ in range(self.num_iters):
            curr_state = state_tensor.clone()
            total_loss = 0.0

            for t in range(self.horizon):
                curr_action = torch.tanh(action_seq[t:t+1]) * self.action_max
                next_state = self.f_prior(curr_state, curr_action)

                # In our VND reward_fn_uav formulation, it expects the NEXT state's target to be compared with the NEXT state.
                # f_prior_uav correctly shifts the target variables into the [4:8] indices of next_state.
                reward = self.reward_fn(curr_state, curr_action, next_state)

                total_loss = total_loss - reward.sum()
                curr_state = next_state

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # The true optimal action is the tanh mapped one
        optimized_seq = torch.tanh(action_seq) * self.action_max

        # Save un-mapped sequence for the next timestep's warm start
        self.action_seq = action_seq.detach()

        return optimized_seq[0].detach().cpu().numpy()
