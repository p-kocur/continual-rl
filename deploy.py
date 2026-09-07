import torch
import numpy as np
import time
from collections import deque
import sys

from continual_vnd import HistoryEncoder, Policy
from ship_env import ShipSailingEnv

def get_context(history, context_len, state_dim, action_dim):
    states = [h[0] for h in history]
    actions = [h[1] for h in history]

    # Pad if not enough history
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

    # Load models
    encoder = HistoryEncoder(state_dim, action_dim, hidden_dim, latent_dim)
    policy = Policy(state_dim, latent_dim, action_dim, hidden_dim)

    try:
        encoder.load_state_dict(torch.load("checkpoints/encoder.pth", map_location="cpu"))
        policy.load_state_dict(torch.load("checkpoints/policy.pth", map_location="cpu"))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}. Please run example.py first to train.")
        sys.exit(1)

    encoder.eval()
    policy.eval()

    env = ShipSailingEnv(render_mode="ansi")
    state, _ = env.reset()

    history = deque(maxlen=context_len)

    print("\nStarting deployment...\n")

    # Run a single episode with visualization
    for step in range(150):
        # Render current state
        sys.stdout.write("\033[H\033[J") # Clear screen
        print(env.render())
        print(f"\nStep: {step}")

        # Infer latent context
        with torch.no_grad():
            ctx_states, ctx_actions = get_context(history, context_len, state_dim, action_dim)
            z = encoder(ctx_states, ctx_actions)

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = policy(state_tensor, z)
            action = action_tensor.squeeze(0).numpy()

        print(f"Action (dx, dy): [{action[0]:.2f}, {action[1]:.2f}]")

        # Environment step
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Update history
        history.append((state, action))
        state = next_state

        time.sleep(0.1) # Add a small delay so human can watch the terminal

        if terminated:
            print("\nTarget Reached! Success!")
            break
        if truncated:
            print("\nMax steps reached.")
            break

if __name__ == "__main__":
    main()
