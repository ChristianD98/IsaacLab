import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LidarViewer:
    def __init__(self, sample_rate=10):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.sample_rate = sample_rate
        self.scatter = None
        plt.ion()  # mode interactif

    def update(self, pointcloud: torch.Tensor, env_idx: int = 0):
        pc = pointcloud[env_idx][::self.sample_rate].cpu().numpy()

        self.ax.clear()
        self.ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c=pc[:, 2], cmap='viridis')
        self.ax.set_title("Nuage de points 3D (Lidar)")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.set_xlim([-self.max_range, self.max_range])
        self.ax.set_ylim([-self.max_range, self.max_range])
        self.ax.set_zlim([-0.5, 2])
        self.ax.view_init(elev=20, azim=60)
        plt.pause(0.001)

    @property
    def max_range(self):
        return 5 # ou utilise self.cfg.max_distance_lidar
