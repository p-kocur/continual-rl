import torch
import torch.nn as nn

class HistoryEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.gru = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, states, actions):
        """
        states: (B, C, state_dim)
        actions: (B, C, action_dim)
        Returns: (B, latent_dim)
        """
        x = torch.cat([states, actions], dim=-1)
        # We only care about the last hidden state from the GRU
        _, h_n = self.gru(x)
        h_n = h_n.squeeze(0) # (B, hidden_dim)
        z = self.fc(h_n)
        return z

class ResidualDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action, z):
        """
        state: (B, state_dim)
        action: (B, action_dim)
        z: (B, latent_dim)
        Returns: (B, state_dim)
        """
        x = torch.cat([state, action, z], dim=-1)
        return self.net(x)

class ContextDecoder(nn.Module):
    def __init__(self, latent_dim, state_dim, action_dim, context_len, hidden_dim=128):
        super().__init__()
        self.context_len = context_len
        self.out_dim = state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_len * self.out_dim)
        )

    def forward(self, z):
        """
        z: (B, latent_dim)
        Returns: (B, context_len, state_dim + action_dim)
        """
        out = self.net(z)
        return out.view(-1, self.context_len, self.out_dim)

class Policy(nn.Module):
    def __init__(self, obs_dim, latent_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Assuming actions are bounded between -1 and 1
        )

    def forward(self, obs, z):
        """
        obs: (B, obs_dim)
        z: (B, latent_dim)
        Returns: (B, action_dim)
        """
        x = torch.cat([obs, z], dim=-1)
        return self.net(x)
