import torch
import torch.nn as nn
from typing import Dict, Any, List
from .message_router import MessageRouter
from .base_cell import BaseCell, WorkerCell, MemoryCell                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

class CellNetwork(nn.Module):
    """
    Cell network container managing execution order and cell registries.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.router = MessageRouter()
        
        self.cells: Dict[str, BaseCell] = {}
        self.worker_cells: List[WorkerCell] = []
        self.memory_cells: List[MemoryCell] = []
        
        self.config = config or {"mode": "dynamic", "initial_workers": 2, "max_workers": 16}
        self.output_layer = nn.Linear(16, 1) # Assumes state_dim 16 and output_dim 1
        self._initialize_from_config()

    def _initialize_from_config(self):
        """Initializes the network's worker cells based on the provided configuration."""
        mode = self.config.get("mode", "dynamic")
        
        if mode == "fixed":
            worker_count = self.config.get("fixed_worker_count", 4)
            for i in range(worker_count):
                worker = WorkerCell(cell_id=f"worker_{i}", state_dim=16)
                self.add_cell(worker, cell_type="worker")
        elif mode == "dynamic":
            initial_workers = self.config.get("initial_workers", 2)
            for i in range(initial_workers):
                worker = WorkerCell(cell_id=f"worker_{i}", state_dim=16)
                self.add_cell(worker, cell_type="worker")
        
    def add_cell(self, cell: BaseCell, cell_type: str = "worker"):
        """Registers a new cell into the network."""
        self.cells[cell.cell_id] = cell
        self.add_module(cell.cell_id, cell)
        if cell_type == "worker" and isinstance(cell, WorkerCell):
            self.worker_cells.append(cell)
        elif cell_type == "memory" and isinstance(cell, MemoryCell):
            self.memory_cells.append(cell)

    def remove_cell(self, cell_id: str):
        """Removes a cell from the network."""
        if cell_id in self.cells:
            cell = self.cells.pop(cell_id)
            if cell in self.worker_cells:
                self.worker_cells.remove(cell)
            if cell in self.memory_cells:
                self.memory_cells.remove(cell)

    def step(self):
        """
        Executes one deterministic simulation step in strict order:
        1. router.deliver_messages()
        2. each cell processes incoming messages
        3. cells update internal state
        4. cells generate outgoing messages
        5. router collects next_messages
        6. router swaps message buffers
        """
        self.router.deliver_messages(self.cells)
        
        for cell in self.cells.values():
            cell.process_messages()
            
        for cell in self.cells.values():
            cell.update_state()
            
        self.router.collect_messages(self.cells)
        self.router.swap_buffers()

    def forward(self, inputs: Any) -> Any:
        self.step()
        return self._get_network_output()
        
    def _get_network_output(self) -> Any:
        if not self.worker_cells:
            return torch.tensor([0.0], requires_grad=True)
            
        worker_states = torch.stack([w.state_vector for w in self.worker_cells])
        aggregated_state = torch.mean(worker_states, dim=0)
        outputs = self.output_layer(aggregated_state)
        return outputs
