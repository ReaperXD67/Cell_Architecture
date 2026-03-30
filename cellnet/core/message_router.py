import torch
from typing import Dict, List, Any

class MessageRouter:
    """
    Routes messages between cells using double-buffering to prevent race conditions
    and ensure deterministic simulation steps.
    """
    def __init__(self):
        self.current_messages: Dict[str, List[Any]] = {}
        self.next_messages: Dict[str, List[Any]] = {}

    def deliver_messages(self, cells: Dict[str, Any]):
        """
        Delivers messages from the current_messages buffer to the corresponding cells.
        """
        for cell_id, messages in self.current_messages.items():
            if cell_id in cells:
                for msg in messages:
                    cells[cell_id].receive_message(msg)

    def collect_messages(self, cells: Dict[str, Any]):
        """
        Collects outgoing messages from all cells into the next_messages buffer.
        """
        for cell_id, cell in cells.items():
            outgoing = cell.send_messages()
            for target_id, msg in outgoing:
                if target_id not in self.next_messages:
                    self.next_messages[target_id] = []
                self.next_messages[target_id].append(msg)

    def swap_buffers(self):
        """
        Swaps the message buffers for the next simulation step.
        """
        self.current_messages = self.next_messages
        self.next_messages = {}
