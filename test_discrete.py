import torch
import ship_env
env = ship_env.ShipSailingEnv()
env.reset()
s, r, _, _, _ = env.step(1)
print(f"State after moving right (discrete 1): {s}")
