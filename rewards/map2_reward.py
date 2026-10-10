from .shaped_reward import ShapedReward


class Map2Reward(ShapedReward):
    # Cover map: engage with LOS when armed, break LOS to reload,
    # don't waste ammo into walls. Distance matters less (hiders exist).
    W_AIM        = 1.0
    W_DIST       = 0.5
    W_THREAT     = 0.3
    W_TACTIC     = 0.5
    R_SHOT_BLIND = -0.1
