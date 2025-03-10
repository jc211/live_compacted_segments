import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from live_compacted_segments import (
    TimeSeriesSamplerBuilder,
)


def generate_timesteps(T, start=0, end=10):
    return np.linspace(start, end, T)


def generate_single_body_trajectory(timesteps, body_id):
    if body_id == 0:
        px = np.sin(timesteps)
        py = np.cos(timesteps)
        pz = timesteps / 10.0
    else:
        px = timesteps / 10.0
        py = np.sin(timesteps)
        pz = np.cos(timesteps)

    angles = np.linspace(0, 2 * np.pi, len(timesteps)) + body_id * np.pi / 2
    quaternions = R.from_euler("z", angles).as_quat()

    trajectory = np.column_stack((px, py, pz, quaternions))
    return trajectory


def generate_trajectories(T):
    timesteps = generate_timesteps(T)
    trajectories = np.zeros((T, 2, 7))
    for body_id in range(2):
        trajectories[:, body_id, :] = generate_single_body_trajectory(
            timesteps, body_id
        )
    return timesteps, trajectories


def plot_frame(ax, position, quaternion, scale=0.1, color="k", alpha=1.0):
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    origin = position
    x_axis = origin + rotation_matrix[:, 0] * scale
    y_axis = origin + rotation_matrix[:, 1] * scale
    z_axis = origin + rotation_matrix[:, 2] * scale

    ax.plot(
        [origin[0], x_axis[0]],
        [origin[1], x_axis[1]],
        [origin[2], x_axis[2]],
        color="r",
        alpha=alpha,
    )
    ax.plot(
        [origin[0], y_axis[0]],
        [origin[1], y_axis[1]],
        [origin[2], y_axis[2]],
        color="g",
        alpha=alpha,
    )
    ax.plot(
        [origin[0], z_axis[0]],
        [origin[1], z_axis[1]],
        [origin[2], z_axis[2]],
        color="b",
        alpha=alpha,
    )


def plot_trajectories(timesteps, trajectories, interp_trajectories=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = ["r", "b"]

    for body_id in range(trajectories.shape[1]):
        ax.plot(
            trajectories[:, body_id, 0],
            trajectories[:, body_id, 1],
            trajectories[:, body_id, 2],
            color=colors[body_id],
            linestyle="-",
            label=f"Normal Body {body_id}",
        )

        if interp_trajectories is not None:
            ax.plot(
                interp_trajectories[:, body_id, 0],
                interp_trajectories[:, body_id, 1],
                interp_trajectories[:, body_id, 2],
                color=colors[body_id],
                linestyle="--",
                label=f"Interpolated Body {body_id}",
            )

        for t in range(0, trajectories.shape[0], 2):  # Plot frames at intervals
            plot_frame(
                ax,
                trajectories[t, body_id, :3],
                trajectories[t, body_id, 3:],
                alpha=1.0,
            )
            if interp_trajectories is not None:
                plot_frame(
                    ax,
                    interp_trajectories[t, body_id, :3],
                    interp_trajectories[t, body_id, 3:],
                    color="gray",
                    alpha=0.5,
                )

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([0, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("3D Trajectories with Interpolation and Rotations")
    plt.show()


if __name__ == "__main__":
    T = 40
    timesteps, trajectories = generate_trajectories(T)
    builder = TimeSeriesSamplerBuilder(output_type="transform")
    builder.add_series(timesteps, trajectories)
    sampler = builder.finalize(4, 100, 0.01)
    traj_interp, _ = sampler.sample()

    plot_trajectories(timesteps, trajectories, traj_interp[0].cpu().numpy())
