import torch
import numpy as np


class QuadrotorEnvVNDWrapper:
    """Thin wrapper to expose QuadrotorEnv to ContinualVNDTrainer."""

    def __init__(self, env):
        self.env = env
        self.max_steps = env.max_steps

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)


def f_prior_quadrotor(state, action):
    """
    Closed-loop 2nd-order PD kinematics prior for PyBullet quadrotor with DSLPIDControl.

    State (21D): [px, py, pz, vx, vy, vz, ref0(3), ref1(3), ..., ref4(3)]
    Action  (3D): [dx, dy, dz] — position correction offset in meters (bounded ±0.5 m)

    Dynamics:
        target_pos = ref1 + action
        a = Kp * (target_pos - p) - Kd * v
        v_next = v + a * dt
        p_next = p + v_next * dt
    """
    dt = 1.0 / 48.0  # matches CtrlAviary ctrl_freq=48 (0.020833 s)

    # Closed-loop tracking coefficients identified from the DSLPIDControl physics
    kp = torch.tensor([4.5, 4.0, 3.0], device=state.device, dtype=state.dtype)
    kd = torch.tensor([0.5, 0.5, 1.0], device=state.device, dtype=state.dtype)

    pos = state[:, 0:3]
    vel = state[:, 3:6]
    target_ref = state[:, 9:12]  # ref1 is the reference at next step t+1

    target_pos = target_ref + action
    accel = kp * (target_pos - pos) - kd * vel

    new_vel = vel + accel * dt
    new_pos = pos + new_vel * dt

    # Shift lookahead references: ref1..ref4 become new ref0..ref3, duplicate last
    new_refs = state[:, 9:].clone()
    last_ref = state[:, -3:]

    return torch.cat([new_pos, new_vel, new_refs, last_ref], dim=-1)


def reward_fn_quadrotor(state, action, next_state):
    """
    Differentiable tracking reward: negative position error w.r.t. target reference.

    next_state (21D): [pos(3), vel(3), ref0(3), ref1..ref4(3)]
    At step t+1, ref0 (indices 6:9) holds the target position for this step.
    """
    pos = next_state[:, 0:3]
    ref_pos = next_state[:, 6:9]

    pos_error = torch.norm(pos - ref_pos, dim=1)
    act_penalty = torch.sum(action**2, dim=1)

    return -(pos_error + 0.02 * act_penalty)
