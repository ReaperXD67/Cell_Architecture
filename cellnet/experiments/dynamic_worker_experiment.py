import torch
import sys
import os

# Add parent directory to path so core modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.cell_network import CellNetwork
from core.spawn_manager import SpawnManager
from core.monitor import Monitor
from trainer import Trainer

def run_experiment():
    print("=== Starting DYNAMIC WORKER Architecture Experiment ===")
    
    config = {
        "mode": "dynamic",
        "initial_workers": 2,
        "max_workers": 8,
        "spawn_threshold": 0.4,
        "spawn_interval": 10
    }
    
    # Initialize components
    network = CellNetwork(config=config)
    monitor = Monitor()
    
    spawn_manager = SpawnManager(
        network=network,
        spawn_threshold=config["spawn_threshold"],
        spawn_interval=config["spawn_interval"],
        max_workers=config["max_workers"]
    )
    
    trainer = Trainer(network=network, spawn_manager=spawn_manager, monitor=monitor)
    
    # Create dummy dataset for demonstration
    dataset = [(torch.randn(16), torch.randn(1)) for _ in range(50)]
    
    # Train
    trainer.train(dataloader=dataset, epochs=5, simulation_steps=3)
    
    # Save the plot
    monitor.plot_metrics("dynamic_worker_metrics.png")
    print("Experiment finished. Metrics saved to dynamic_worker_metrics.png.")

if __name__ == "__main__":
    run_experiment()
