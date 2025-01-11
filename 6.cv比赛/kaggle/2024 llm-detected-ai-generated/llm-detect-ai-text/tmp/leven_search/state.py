from dataclasses import dataclass
from typing import List

from .cost import Edit, EditCostConfig


@dataclass
class State:
    word: str
    distance: int
    pos: int
    updates: List[Edit]
    cost_config: EditCostConfig

    def step(self, edit=None):
        delta_distance = self.cost_config.get_cost(edit)
        new_updates = self.updates + [edit] if edit else self.updates
        return State(self.word, self.distance - delta_distance, self.pos + 1, new_updates, self.cost_config)
