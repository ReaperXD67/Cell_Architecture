import torch
import torch.nn as nn
from typing import List, Tuple, Any

class BaseCell(nn.Module):
    """
    Base cell architecture for all cells in the simulated network.
    """
    def __init__(self, cell_id: str, state_dim: int, energy: float = 100.0):
        super().__init__()
        self.cell_id = cell_id
        self.state_vector = torch.zeros(state_dim)
        self.incoming_messages: List[Any] = []
        self.energy = energy

    def receive_message(self, message: Any):
        """Accumulates an incoming message."""
        self.incoming_messages.append(message)

    def process_messages(self):
        """Processes all accumulated incoming messages."""
        pass

    def update_state(self):
        """Updates internal state vector."""
        pass

    def send_messages(self) -> List[Tuple[str, Any]]:
        """
        Generates outgoing messages based on current state.
        Returns a list of tuples (target_cell_id, message_data)
        """
        return []


class WorkerCell(BaseCell):
    """
    Worker cell that processes information, communicates, and can mutate parameters.
    """
    def __init__(self, cell_id: str, state_dim: int, hidden_dim: int = 32, energy: float = 100.0):
        super().__init__(cell_id, state_dim, energy)
        self.aggregated_messages = torch.zeros(state_dim)
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def process_messages(self):
        # Mean aggregation improvement
        if len(self.incoming_messages) > 0:
            # Assuming incoming_messages are tensors of shape [state_dim]
            self.aggregated_messages = torch.mean(torch.stack(self.incoming_messages), dim=0)
            self.incoming_messages.clear()
        else:
            self.aggregated_messages = torch.zeros_like(self.state_vector)

    def update_state(self):
        # Neural update with residual connection
        combined_input = self.state_vector + self.aggregated_messages
        self.state_vector = self.state_vector + self.network(combined_input)
        # Add State Normalization to cells
        self.state_vector = torch.nn.functional.layer_norm(
            self.state_vector,
            self.state_vector.shape
        )


class MemoryCell(BaseCell):
    """
    Memory cell utilizing cosine similarity for simple retrieval.
    """
    def __init__(self, cell_id: str, memory_size: int = 64, dim: int = 16, energy: float = 100.0):
        super().__init__(cell_id, dim, energy)
        self.memory_size = memory_size
        self.dim = dim
        self.keys = nn.Parameter(torch.randn(memory_size, dim))
        self.values = nn.Parameter(torch.randn(memory_size, dim))

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Soft retrieval using softmax over cosine similarities.
        """
        query_norm = query / (query.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        keys_norm = self.keys / (self.keys.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        
        # 1. Compute cosine similarity between query and memory keys
        similarity = torch.matmul(keys_norm, query_norm.transpose(0, 1) if query.dim() > 1 else query_norm)
        if similarity.dim() > 1:
            similarity = similarity.squeeze()
            
        # 2. Apply softmax over similarities
        weights = torch.softmax(similarity, dim=0)
        
        # 3. Compute weighted sum of memory values
        retrieved = torch.matmul(weights, self.values)
        
        return retrieved
