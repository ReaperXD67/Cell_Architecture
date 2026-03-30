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
    print("=== Starting FIXED WORKER Architecture Experiment ===")
    
    config = {
        "mode": "fixed",
        "fixed_worker_count": 4
    }
    
    # Initialize components
    network = CellNetwork(config=config)
    monitor = Monitor()
    # SpawnManager is not needed for fixed mode, but Trainer expects the type.
    # We pass it with arbitrarily high thresholds so it never fires even if accidentally called.
    spawn_manager = SpawnManager(network, spawn_threshold=999.0, spawn_interval=9999)
    
    trainer = Trainer(network=network, spawn_manager=spawn_manager, monitor=monitor)
    
    # Create dummy dataset for demonstration
    dataset = [(torch.randn(16), torch.randn(1)) for _ in range(50)]
    
    # Train
    trainer.train(dataloader=dataset, epochs=5, simulation_steps=3)
    
    # Save the plot
    monitor.plot_metrics("fixed_worker_metrics.png")
    print("Experiment finished. Metrics saved to fixed_worker_metrics.png.")

if __name__ == "__main__":
    run_experiment()
