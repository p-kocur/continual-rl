import matplotlib.pyplot as plt
import numpy as np
import os

def plot_uav_trajectory(uav_xs, uav_ys, ref_xs, ref_ys, title, filename):
    """
    Plots the UAV's 2D trajectory against the reference trajectory and saves it to a file.
    """
    plt.figure(figsize=(8, 6))

    # Plot reference trajectory
    plt.plot(ref_xs, ref_ys, 'k--', label='Reference (Figure-8)', linewidth=2)

    # Plot UAV trajectory
    plt.plot(uav_xs, uav_ys, 'b-', label='UAV Trajectory', linewidth=2, alpha=0.8)

    # Mark start and end points
    plt.scatter([uav_xs[0]], [uav_ys[0]], color='green', marker='o', s=100, label='Start', zorder=5)
    plt.scatter([uav_xs[-1]], [uav_ys[-1]], color='red', marker='x', s=100, label='End', zorder=5)

    plt.title(title, fontsize=14)
    plt.xlabel("X Position", fontsize=12)
    plt.ylabel("Y Position", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.axis('equal')

    # Save the figure
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved trajectory visualization to {filepath}")
    plt.close()
