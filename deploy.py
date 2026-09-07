import torch
import numpy as np
import time
from collections import deque
import sys

from continual_vnd import HistoryEncoder, Policy
from ship_env import ShipSailingEnv
from compare import TrueStateEnvWrapper

def get_context(history, context_len, state_dim, action_dim):
    states = [h[0] for h in history]
    actions = [h[1] for h in history]

    while len(states) < context_len:
        states.insert(0, np.zeros(state_dim))
        actions.insert(0, np.zeros(action_dim))

    states = np.array(states)
    actions = np.array(actions)

    return (torch.FloatTensor(states).unsqueeze(0),
            torch.FloatTensor(actions).unsqueeze(0))

def main():
    state_dim = 2
    action_dim = 2
    context_len = 5
    latent_dim = 16
    hidden_dim = 64

    encoder = HistoryEncoder(state_dim, action_dim, hidden_dim, latent_dim)
    policy = Policy(state_dim, latent_dim, action_dim, hidden_dim)

    try:
        encoder.load_state_dict(torch.load("checkpoints/encoder.pth", map_location="cpu"))
        policy.load_state_dict(torch.load("checkpoints/policy.pth", map_location="cpu"))
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    encoder.eval()
    policy.eval()

    env = ShipSailingEnv(render_mode="ansi")
    vnd_env = TrueStateEnvWrapper(env)
    state, _ = vnd_env.reset()

    history = deque(maxlen=context_len)

    for step in range(100):
        sys.stdout.write("\033[H\033[J")
        print(env.render())
        print(f"\nStep: {step}")

        with torch.no_grad():
            ctx_states, ctx_actions = get_context(history, context_len, state_dim, action_dim)
            z = encoder(ctx_states, ctx_actions)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = policy(state_tensor, z)
            action = action_tensor.squeeze(0).numpy()

        print(f"Action (dx, dy): [{action[0]:.2f}, {action[1]:.2f}]")

        next_state, reward, terminated, truncated, _ = vnd_env.step(action)
        history.append((state, action))
        state = next_state

        time.sleep(0.1)

        if terminated:
            print("\nTarget Reached!")
            break
        if truncated:
            print("\nMax steps reached.")
            break

if __name__ == "__main__":
    main()
