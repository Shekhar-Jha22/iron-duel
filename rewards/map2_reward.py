from .shaped_reward import ShapedReward


class Map2Reward(ShapedReward):
    # Cover map: engage with LOS when armed, break LOS to reload, don't waste ammo into walls.
    W_AIM        = 0.0     # PBRS aim swung ±1 per aim notch: pure variance that drowned the aim signal
    W_DIST       = 0.5
    W_THREAT     = 0.3
    W_TACTIC     = 0.5
    R_SHOT_BLIND = -0.1
    R_TIMEOUT    = -20.0   # = R_LOSE: waiting out the clock is never safer than fighting
    R_WIN_FAST   = 10.0    # quick kills pay more (up to +10)
    R_AIM_TRACK  = 0.01    # with AIM_PENALTY: random aim ≈ −0.008/frame, on target 0
    AIM_PENALTY  = True
