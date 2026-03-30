import matplotlib.pyplot as plt
from typing import List

class Monitor:
    """
    Tracks metrics and emergent behavior in experiments.
    Visualizes loss, worker count, and average energy during training.
    """
    def __init__(self):
        self.loss_history: List[float] = []
        self.worker_count: List[int] = []
        self.spawn_events: List[int] = []
        self.average_energy: List[float] = []

    def log_step(self, step: int, loss: float, num_workers: int, avg_energy: float, spawned: bool = False):
        """Records data for a single step."""
        self.loss_history.append(loss)
        self.worker_count.append(num_workers)
        self.average_energy.append(avg_energy)
        if spawned:
            self.spawn_events.append(step)

    def plot_metrics(self, save_path: str = "metrics.png"):
        """Plots the tracked metrics and saves them to a file."""
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot Loss
        axs[0].plot(self.loss_history, label='Loss')
        axs[0].set_title('Loss History')
        axs[0].legend()

        # Plot Worker Count and Spawn Events
        axs[1].plot(self.worker_count, label='Worker Count', color='orange')
        for spawn_step in self.spawn_events:
            axs[1].axvline(x=spawn_step, color='r', linestyle='--', alpha=0.3)
        axs[1].set_title('Worker Count over Time')
        axs[1].legend()

        # Plot Average Energy
        axs[2].plot(self.average_energy, label='Average Energy', color='green')
        axs[2].set_title('Average Energy per Cell')
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
