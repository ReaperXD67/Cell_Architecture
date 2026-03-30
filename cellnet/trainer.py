import torch
from core.cell_network import CellNetwork
from core.spawn_manager import SpawnManager
from core.monitor import Monitor

class Trainer:
    """
    Handles the training loop integrating the network, spawning mechanics, and monitoring.
    It executes multiple simulation steps per forward pass to allow temporal processing.
    """
    def __init__(self, network: CellNetwork, spawn_manager: SpawnManager, monitor: Monitor, lr: float = 1e-3):
        self.network = network
        self.spawn_manager = spawn_manager
        self.monitor = monitor
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor, step: int, simulation_steps: int = 3):
        """Runs a single forward-backward pass executing multiple inner simulation steps."""
        self.optimizer.zero_grad()
        
        # Execute multiple simulation steps for each forward pass
        for t in range(simulation_steps):
            self.network.step()
            # Detach states to prevent backward tracking across inner loop
            for cell in self.network.cells.values():
                cell.state_vector = cell.state_vector.detach().requires_grad_(True)
            
        # Get output and calculate loss
        outputs = self.network._get_network_output()
        
        assert outputs.shape == targets.shape, f"Output shape {outputs.shape} and Target shape {targets.shape} mismatch"
        # Optional debugging print
        # print(f"Output shape: {outputs.shape}, Target shape: {targets.shape}")
        
        # Mock loss for demonstration
        loss = torch.nn.functional.mse_loss(outputs, targets)
        
        loss.backward()
        self.optimizer.step()
        
        current_loss = loss.item()
        
        initial_worker_count = len(self.network.worker_cells)
        # Check and spawn workers based on moving average loss only if dynamic mode
        if self.network.config.get("mode", "dynamic") == "dynamic":
            if self.spawn_manager is not None:
                self.spawn_manager.update(current_loss, step)
        
        spawned = len(self.network.worker_cells) > initial_worker_count
        
        avg_energy = 0.0
        if self.network.cells:
            avg_energy = sum(cell.energy for cell in self.network.cells.values()) / len(self.network.cells)
            
        # Log to monitor
        self.monitor.log_step(step, current_loss, len(self.network.worker_cells), avg_energy, spawned)
        
        return current_loss

    def train(self, dataloader, epochs: int, simulation_steps: int = 3):
        """Main training loop."""
        step = 0
        for epoch in range(epochs):
            for batch_inputs, batch_targets in dataloader:
                loss = self.train_step(batch_inputs, batch_targets, step, simulation_steps)
                step += 1
                if step % 10 == 0:
                    print(f"Step {step}, Loss: {loss:.4f}, Workers: {len(self.network.worker_cells)}")
