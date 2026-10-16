import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from .models import HistoryEncoder, ResidualDynamics, ContextDecoder, Policy
from .losses import mmd_loss, huber_loss
from .buffer import TrajectoryBuffer

class ContinualVNDTrainer:
    def __init__(self,
                 state_dim,
                 action_dim,
                 f_prior,
                 reward_fn,
                 context_len=10,
                 latent_dim=16,
                 hidden_dim=128,
                 lr_vnd=1e-3,
                 lr_policy=1e-3,
                 lambda_rec=1.0,
                 lambda_mmd=0.1,
                 gamma=0.99,
                 buffer_capacity=100000,
                 action_scale=None,
                 dyn_dim=None,
                 z_ema_alpha=0.15,
                 device="cpu"):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.f_prior = f_prior
        self.reward_fn = reward_fn
        self.context_len = context_len
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.device = device
        self.z_ema_alpha = z_ema_alpha

        if action_scale is None:
            self.action_scale = 5.0 if (self.state_dim == 24 and self.action_dim == 2) else 1.0
        else:
            self.action_scale = action_scale

        if dyn_dim is None:
            if self.state_dim == 24 and self.action_dim == 2:
                self.dyn_dim = 4
            elif self.action_dim == 3:
                self.dyn_dim = 6
            else:
                self.dyn_dim = self.state_dim
        else:
            self.dyn_dim = dyn_dim

        self.lambda_rec = lambda_rec
        self.lambda_mmd = lambda_mmd

        # Models
        self.encoder = HistoryEncoder(state_dim, action_dim, hidden_dim, latent_dim).to(device)
        self.residual = ResidualDynamics(state_dim, action_dim, latent_dim, hidden_dim).to(device)
        self.decoder = ContextDecoder(latent_dim, state_dim, action_dim, context_len, hidden_dim).to(device)
        self.policy = Policy(state_dim, latent_dim, action_dim, hidden_dim).to(device) # We assume obs == state for simplicity, can be changed.

        # Optimizers
        self.vnd_optimizer = optim.Adam(list(self.encoder.parameters()) +
                                        list(self.residual.parameters()) +
                                        list(self.decoder.parameters()), lr=lr_vnd)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)

        # Buffer
        self.buffer = TrajectoryBuffer(state_dim, action_dim, buffer_capacity, context_len)

    def _get_context(self, history):
        """Converts history deque to tensor"""
        states = [h[0] for h in history]
        actions = [h[1] for h in history]

        # Pad if not enough history (for beginning of episodes)
        while len(states) < self.context_len:
            states.insert(0, np.zeros(self.state_dim))
            actions.insert(0, np.zeros(self.action_dim))

        states = np.array(states)
        actions = np.array(actions)

        return (torch.FloatTensor(states).unsqueeze(0).to(self.device),
                torch.FloatTensor(actions).unsqueeze(0).to(self.device))

    def collect_experience(self, env, num_steps):
        """Collects experience from the environment using current policy and inferred latents."""
        state, _ = env.reset()
        history = deque(maxlen=self.context_len)
        z_ema = None

        episode_reward = 0
        rewards = []

        for _ in range(num_steps):
            # Infer latent z
            with torch.no_grad():
                ctx_states, ctx_actions = self._get_context(history)
                z_raw = self.encoder(ctx_states, ctx_actions)
                if self.z_ema_alpha is not None:
                    if z_ema is None:
                        z_ema = z_raw
                    else:
                        z_ema = (1.0 - self.z_ema_alpha) * z_ema + self.z_ema_alpha * z_raw
                    z = z_ema
                else:
                    z = z_raw

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_tensor = self.policy(state_tensor, z)
                action = action_tensor.squeeze(0).cpu().numpy()
                action = action * self.action_scale

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store in buffer
            self.buffer.add(state, action, next_state, done)
            history.append((state, action))

            state = next_state
            episode_reward += reward

            if done:
                state, _ = env.reset()
                history.clear()
                z_ema = None
                rewards.append(episode_reward)
                episode_reward = 0

        return np.mean(rewards) if rewards else episode_reward

    def update_vnd(self, batch_size=256):
        """Trains the Variational Neural Dynamics model (Encoder, Residual, Decoder)."""
        if self.buffer.size <= self.context_len * 2:
            return {}

        self.encoder.train()
        self.residual.train()
        self.decoder.train()

        ctx_states, ctx_actions, states, actions, next_states = self.buffer.sample(batch_size, self.device)

        # 1. Forward encoder
        z = self.encoder(ctx_states, ctx_actions)

        # 2. Forward residual
        pred_residual = self.residual(states, actions, z)

        # Compute target residual
        with torch.no_grad():
            nominal_next_states = self.f_prior(states, actions)
            target_residual = next_states - nominal_next_states

        # Dynamics loss: only penalize physical state residual, protecting lookahead references
        if self.dyn_dim < self.state_dim:
            loss_dyn = huber_loss(pred_residual[:, :self.dyn_dim], target_residual[:, :self.dyn_dim])
        else:
            loss_dyn = huber_loss(pred_residual, target_residual)

        # 3. Forward decoder (Reconstruction loss)
        reconstructed_context = self.decoder(z) # (B, C, state_dim + action_dim)
        target_context = torch.cat([ctx_states, ctx_actions], dim=-1)
        loss_rec = torch.nn.functional.mse_loss(reconstructed_context, target_context)

        # 4. MMD Loss
        prior_z = torch.randn_like(z) # Sample from N(0, I)
        loss_mmd = mmd_loss(z, prior_z)

        # Total VND loss
        loss = loss_dyn + self.lambda_rec * loss_rec + self.lambda_mmd * loss_mmd

        self.vnd_optimizer.zero_grad()
        loss.backward()
        self.vnd_optimizer.step()

        return {
            "loss_dyn": loss_dyn.item(),
            "loss_rec": loss_rec.item(),
            "loss_mmd": loss_mmd.item(),
            "loss_vnd_total": loss.item()
        }

    def update_policy(self, rollout_length=10, batch_size=256):
        """Improves the policy via differentiable simulation over the learned latent dynamics model."""
        if self.buffer.size == 0:
            return {}

        self.encoder.eval()
        self.residual.eval()
        self.policy.train()

        # Sample initial states
        states = self.buffer.sample_initial_states(batch_size, self.device)

        # Sample latents from prior
        z = torch.randn(batch_size, self.latent_dim, device=self.device)

        loss_policy = 0
        discount = 1.0

        for t in range(rollout_length):
            # Get action from policy
            actions = self.policy(states, z)

            # Action scaling for BPTT dynamics rollout
            scaled_actions = actions * self.action_scale

            # Predict next state using differentiable dynamics model
            # We want gradients to flow through f_prior for BPTT, but we don't want to update f_prior's parameters if it has any
            nominal_next_states = self.f_prior(states, scaled_actions)

            # The residual is differentiable, but we don't want to update residual parameters here.
            # We are just computing gradients wrt actions -> policy.
            pred_residual = self.residual(states, scaled_actions, z)
            if self.dyn_dim < self.state_dim:
                # Add residual only to physical states (pos and vel); keep references untouched
                next_phys = nominal_next_states[:, :self.dyn_dim] + pred_residual[:, :self.dyn_dim]
                next_states = torch.cat([next_phys, nominal_next_states[:, self.dyn_dim:]], dim=-1)
            else:
                next_states = nominal_next_states + pred_residual

            # Compute reward
            rewards = self.reward_fn(states, scaled_actions, next_states)

            # Maximize reward is minimize -reward
            loss_policy = loss_policy - discount * rewards.mean()

            # Update state for next step
            states = next_states
            discount *= self.gamma

        self.policy_optimizer.zero_grad()
        loss_policy.backward()
        # Gradient clipping to stabilize policy learning
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        return {
            "loss_policy": loss_policy.item()
        }
