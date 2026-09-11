import torch
import numpy as np
from quadrotor_env import QuadrotorEnv
from quadrotor_prior import QuadrotorEnvVNDWrapper, f_prior_quadrotor, reward_fn_quadrotor
from continual_vnd import ContinualVNDTrainer
from visualize import plot_uav_trajectory

def evaluate_quadrotor(trainer, vnd_env, episodes=1):
    trainer.encoder.eval()
    trainer.policy.eval()
    avg_return = 0

    for ep in range(episodes):
        state, _ = vnd_env.reset()
        history = []
        ep_return = 0

        uav_xs, uav_ys = [], []
        ref_xs, ref_ys = [], []

        for _ in range(vnd_env.max_steps):
            uav_xs.append(state[0])
            uav_ys.append(state[1])
            ref_xs.append(state[12])
            ref_ys.append(state[13])

            if len(history) < trainer.context_len:
                pad = trainer.context_len - len(history)
                states = [np.zeros(27)] * pad + [h[0] for h in history]
                actions = [np.zeros(4)] * pad + [h[1] for h in history]
            else:
                states = [h[0] for h in history[-trainer.context_len:]]
                actions = [h[1] for h in history[-trainer.context_len:]]

            ctx_states = torch.FloatTensor(np.array(states)).unsqueeze(0).to(trainer.device)
            ctx_actions = torch.FloatTensor(np.array(actions)).unsqueeze(0).to(trainer.device)

            with torch.no_grad():
                z = trainer.encoder(ctx_states, ctx_actions)
                a = trainer.policy(torch.FloatTensor(state).unsqueeze(0).to(trainer.device), z)
                # Tanh output is [-1, 1], which perfectly matches the QuadrotorEnv action_space!
                action = a.squeeze(0).cpu().numpy()

            next_state, reward, terminated, truncated, _ = vnd_env.step(action)
            ep_return += reward
            history.append((state, action))
            state = next_state

            if terminated or truncated:
                break

        if ep == 0:
            plot_uav_trajectory(uav_xs, uav_ys, ref_xs, ref_ys,
                                "Continual VND - PyBullet Quadrotor (With Wind)",
                                "trajectory_pybullet_vnd.png")

        avg_return += ep_return
    return avg_return / episodes

def main():
    env = QuadrotorEnv(enable_wind=True)
    vnd_env = QuadrotorEnvVNDWrapper(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # State: 12D + 5*3D = 27D
    # Action: 4D (Motor commands)
    trainer = ContinualVNDTrainer(
        state_dim=27,
        action_dim=4,
        f_prior=f_prior_quadrotor,
        reward_fn=reward_fn_quadrotor,
        context_len=10,
        latent_dim=16,
        hidden_dim=128,
        device=device
    )

    epochs = 3
    steps_per_epoch = 1000

    print("--- Training Continual VND on PyBullet Quadrotor ---")

    for _ in range(3):
        trainer.collect_experience(vnd_env, num_steps=steps_per_epoch)

    for epoch in range(epochs):
        vnd_losses = []
        for _ in range(50):
            logs = trainer.update_vnd(batch_size=128)
            if logs: vnd_losses.append(logs['loss_dyn'])

        pol_losses = []
        for _ in range(50):
            logs = trainer.update_policy(rollout_length=10, batch_size=128)
            if logs: pol_losses.append(logs['loss_policy'])

        avg_reward = trainer.collect_experience(vnd_env, num_steps=steps_per_epoch)

        avg_dyn = np.mean(vnd_losses) if vnd_losses else 0
        avg_pol = np.mean(pol_losses) if pol_losses else 0
        print(f"Epoch {epoch+1:02d} | Dyn Loss: {avg_dyn:.4f} | Pol Loss: {avg_pol:.4f} | Avg Reward: {avg_reward:.2f}")

    print("\n--- Evaluation ---")
    ret = evaluate_quadrotor(trainer, vnd_env, episodes=3)
    print(f"Continual VND | Avg Return: {ret:.2f}")

if __name__ == "__main__":
    main()
