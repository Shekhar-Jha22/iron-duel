from dataclasses import dataclass
from typing import List

@dataclass
class Obstacle:
    x: float
    y: float
    w: float
    h: float
    type: str = "rock"

@dataclass
class Spawn:
    x: float
    y: float

@dataclass
class MapConfig:
    name: str
    width: int
    height: int
    player_spawn: Spawn
    enemy_spawn: Spawn
    obstacles: List[Obstacle]
    max_speed: float = 5.0   # add this
    accel: float = 1.2 