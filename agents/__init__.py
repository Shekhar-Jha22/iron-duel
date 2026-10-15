"""
Agent module - exports all agent types for multi-agent training
"""

from .ppo_agent import PPOAgent
from .random_agent import RandomAgent
from .rule_map1 import ChaserAgent, CirclerAgent
from .rule_map2 import (
    StationaryAgent, SlowWalkerAgent, AggressiveWalkerAgent,
    EasyStraferAgent, SimpleHiderAgent,
    DefensiveHiderAgent, MobileDefenseAgent,
    DodgerAgent, SniperAgent, HunterAgent,
)

__all__ = [
    "PPOAgent",
    "RandomAgent",
    "ChaserAgent",
    "CirclerAgent",
    "StationaryAgent",
    "SlowWalkerAgent",
    "AggressiveWalkerAgent",
    "EasyStraferAgent",
    "SimpleHiderAgent",
    "DefensiveHiderAgent",
    "MobileDefenseAgent",
    "DodgerAgent",
    "SniperAgent",
    "HunterAgent",
]

AGENT_REGISTRY = {
    "ppo":         PPOAgent,
    "random":      RandomAgent,
    "chaser":      ChaserAgent,
    "circler":     CirclerAgent,
    "stationary":  StationaryAgent,
    "walker":      SlowWalkerAgent,
    "aggwalker":   AggressiveWalkerAgent,
    "strafer":     EasyStraferAgent,
    "simplehider": SimpleHiderAgent,
    "hider":       DefensiveHiderAgent,
    "mobile":      MobileDefenseAgent,
    "dodger":      DodgerAgent,
    "sniper":      SniperAgent,
    "hunter":      HunterAgent,
}

MAP1_AGENTS = {
    "Random":  RandomAgent,
    "Chaser":  ChaserAgent,
    "Circler": CirclerAgent,
}

MAP2_AGENTS = {
    "Random":      RandomAgent,
    "Stationary":  StationaryAgent,
    "SlowWalker":  SlowWalkerAgent,
    "AggWalker":   AggressiveWalkerAgent,
    "Strafer":     EasyStraferAgent,
    "SimpleHider": SimpleHiderAgent,
    "Hider":       DefensiveHiderAgent,
    "Mobile":      MobileDefenseAgent,
    "Dodger":      DodgerAgent,
    "Sniper":      SniperAgent,
    "Hunter":      HunterAgent,
}
# Curriculum order used by train.py; checkpoint v{X}_stage_{i}_final.zip was trained against STAGE_NAMES[map][i].
STAGE_NAMES = {
    "map1": ["random", "chaser", "circler", "selfplay"],
    "map2": ["random", "stationary", "walker", "aggwalker", "strafer",
             "simplehider", "hider", "mobile", "dodger", "sniper", "hunter", "mixed"],
}
