import torch

class QuadrotorEnvVNDWrapper:
    def __init__(self, env):
        self.env = env
        self.max_steps = env.max_steps

    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

def f_prior_quadrotor(state, action):
    """
    Simplified Differentiable 3D Rigid Body Kinematics for PyTorch BPTT.
    State (27D): [pos(3), rpy(3), vel(3), ang_vel(3), ref0(3), ref1(3), ..., refH(3)]
    Action (4D): [rpm1, rpm2, rpm3, rpm4] in [-1, 1] scaled range.
    """
    # PyBullet CF2X parameters
    dt = 1.0 / 240.0 * 5 # Base env does 5 sub-steps per action
    m = 0.027
    g = 9.81
    # Simple hovering thrust mapping to accelerate
    # We will highly approximate the 3D dynamics for the prior:

    pos = state[:, 0:3]
    rpy = state[:, 3:6]
    vel = state[:, 6:9]
    ang_vel = state[:, 9:12]

    # Very simplified mapping: action directly influences target angular velocities and z-acceleration
    # Since modeling full quaternion dynamics here is extremely heavy, we provide a linearised kinematic model
    # where actions correlate directly to [roll_cmd, pitch_cmd, yaw_cmd, thrust] mapped from the 4 motors.
    # The VND residual will easily pick up the unmodeled complexities (which is the point of VND!)

    # Motor mixing (CF2X setup)
    u1, u2, u3, u4 = action[:, 0], action[:, 1], action[:, 2], action[:, 3]

    thrust = (u1 + u2 + u3 + u4) * 0.25 # proxy for thrust
    roll_cmd = (u1 - u2 - u3 + u4) * 0.1
    pitch_cmd = (-u1 - u2 + u3 + u4) * 0.1
    yaw_cmd = (-u1 + u2 - u3 + u4) * 0.1

    # Integration
    new_ang_vel = ang_vel + torch.stack([roll_cmd, pitch_cmd, yaw_cmd], dim=1) * dt
    new_rpy = rpy + new_ang_vel * dt

    # Acceleration in global frame (approximate small angle)
    ax = g * new_rpy[:, 1] # pitch tilts thrust forward
    ay = -g * new_rpy[:, 0] # roll tilts thrust sideways
    az = (thrust * g) - g # 0 action = hovering

    new_vel = vel + torch.stack([ax, ay, az], dim=1) * dt
    new_pos = pos + new_vel * dt

    # Shift lookahead references
    new_refs = state[:, 15:].clone()
    last_ref = state[:, -3:]

    new_state = torch.cat([new_pos, new_rpy, new_vel, new_ang_vel, new_refs, last_ref], dim=-1)
    return new_state

def reward_fn_quadrotor(state, action, next_state):
    # Calculate position error based on current reference (indices 12:15)
    pos = next_state[:, 0:3]
    ref_pos = next_state[:, 12:15]

    pos_error = torch.norm(pos - ref_pos, dim=1)

    # Penalty for flying too low or flying away
    out_of_bounds = torch.relu(0.05 - pos[:, 2]) + torch.relu(pos_error - 2.0)

    return -pos_error - out_of_bounds * 50.0
