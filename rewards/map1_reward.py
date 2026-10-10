from .shaped_reward import ShapedReward


class Map1Reward(ShapedReward):
    # Open arena: duel is about aim, lead and dodging; closing distance helps.
    W_AIM    = 1.0
    W_DIST   = 1.0
    W_THREAT = 0.3
    W_TACTIC = 0.0
