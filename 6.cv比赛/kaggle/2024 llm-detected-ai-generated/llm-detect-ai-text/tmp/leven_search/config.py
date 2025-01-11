from dataclasses import dataclass


@dataclass
class Config:
    max_distance: int

    def __init__(self, max_distance: int):
        self.max_distance = max_distance
