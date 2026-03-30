import random
import torch
import torch.nn as nn
from typing import Dict, Any, List
from .base_cell import BaseCell, WorkerCell

class SpawnManager:
    """Manages dynamic spawning of worker cells based on moving average loss and mutations."""
    
    def __init__(self, network, alpha: float = 0.1, spawn_threshold: float = 1.0, spawn_interval: int = 10, max_workers: int = 16):
        self.network = network
        self.alpha = alpha
        self.spawn_threshold = spawn_threshold
        self.spawn_interval = spawn_interval
        self.max_workers = max_workers
        
        self.moving_avg_loss = 0.0
        self.previous_loss = 0.0
        self.cell_counter = 0

    def update(self, current_loss: float, step: int):
        """Updates moving average loss and spawns workers if conditions are met."""
        if step == 0:
            self.moving_avg_loss = current_loss
        else:
            self.moving_avg_loss = self.alpha * current_loss + (1 - self.alpha) * self.previous_loss
            
        self.previous_loss = current_loss
        
        if self.moving_avg_loss > self.spawn_threshold and step % self.spawn_interval == 0 and len(self.network.worker_cells) < self.max_workers:
            self.spawn_worker()

    def spawn_worker(self):
        """Spawns a new worker cell with parameter mutation from a randomly selected parent."""
        if len(self.network.worker_cells) >= self.max_workers:
            return
            
        parent_worker = None
        if self.network.worker_cells:
            parent_worker = random.choice(self.network.worker_cells)

        new_cell_id = f"worker_{self.cell_counter}"
        self.cell_counter += 1
        
        # State dim matching the parent or default
        state_dim = 16 
        if parent_worker is not None:
            state_dim = parent_worker.state_vector.shape[0]
            
        new_worker = WorkerCell(cell_id=new_cell_id, state_dim=state_dim)
        
        # Knowledge inheritance via mutation
        if parent_worker is not None:
            with torch.no_grad():
                for new_param, parent_param in zip(new_worker.parameters(), parent_worker.parameters()):
                    new_param.copy_(parent_param + torch.randn_like(parent_param) * 0.02)
                    
        self.network.add_cell(new_worker, cell_type="worker")
