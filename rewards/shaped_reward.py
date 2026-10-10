import math

from .base_reward import BaseReward
from observations import (
    lead_angle, wrap, bullet_threats, shot_on_target, SHOOT_COOLDOWN,
)
from utils import has_line_of_sight


class ShapedReward(BaseReward):
    """
    Sparse outcome rewards + potential-based shaping  F = γ·Φ(s') − Φ(s).
    PBRS telescopes over an episode, so dense terms can't be farmed (no
    aim-forever / oscillate-to-approach exploits) and the optimal policy is
    unchanged — GAMMA must match PPO's gamma.
    """

    # ── sparse ────────────────────────────────────────────────────────────
    R_HIT        =  1.0    # per bullet landed (6 hits = kill)
    R_HIT_RECV   = -0.8    # slight aggression bias vs passive bots
    R_WIN        =  20.0
    R_LOSE       = -20.0
    R_TIMEOUT    = -15.0   # opponent survived → loss (no HP-lead credit: that paid for camping)

    # ── time pressure (per frame, non-potential on purpose) ───────────────
    R_STEP       = -0.01   # every frame both are alive
    R_DIST       = -0.02   # × dist/diag every frame → being far costs more
    # Victim is also charged the remaining frames at the max rate, so dying
    # early is never cheaper than surviving: kill > timeout > death.
    # Immediate credit at fire time (hit reward lands ~40 ticks later).
    # Bounded by ammo regen (~35 shots/episode) → can't outweigh a kill.
    R_SHOT_GOOD  =  0.15   # fired with a predicted hit
    R_SHOT_BAD   = -0.05   # fired with no predicted hit
    R_SHOT_BLIND = 0.0     # extra cost for firing with no line of sight

    # ── potential weights ─────────────────────────────────────────────────
    W_AIM    = 1.0   # Φ ∝ cos(lead error), only with LOS
    W_DIST   = 1.0   # Φ ∝ −distance
    W_THREAT = 0.3   # Φ ∝ −danger of most threatening incoming bullet
    W_TACTIC = 0.0   # Φ ∝ LOS·(+1 if armed else −1)

    ARMED_AMMO = 2

    def __init__(self, gamma=0.995):
        self.gamma = gamma
        self.env = None
        self.reset()

    def attach(self, env):
        self.env = env

    def reset(self):
        self.rewards = [0.0, 0.0]
        self.prev_phi = [None, None]

    # ── events ────────────────────────────────────────────────────────────
    def on_hit(self, agent_idx, target_idx):
        self.rewards[agent_idx]  += self.R_HIT
        self.rewards[target_idx] += self.R_HIT_RECV

    def on_hit_recv(self, agent_idx):
        pass

    def on_kill(self, agent_idx, target_idx):
        self.rewards[agent_idx]  += self.R_WIN
        self.rewards[target_idx] += self.R_LOSE
        if self.env is not None:
            remaining = max(self.env.max_steps - self.env.step_count, 0)
            self.rewards[target_idx] += (self.R_STEP + self.R_DIST) * remaining

    def on_death(self, agent_idx):
        pass

    def on_timeout(self):
        pass

    def on_timeout_with_state(self, players):
        for i in range(2):
            if players[i].alive and players[1 - i].alive:
                self.rewards[i] += self.R_TIMEOUT
            # terminal state has Φ = 0: undo the γ·Φ(s_T) added in per_tick
            if self.prev_phi[i] is not None:
                self.rewards[i] -= self.gamma * self.prev_phi[i]
                self.prev_phi[i] = 0.0

    # ── potential ─────────────────────────────────────────────────────────
    def potential(self, player, opponent, bullets, obstacles):
        dx, dy = opponent.x - player.x, opponent.y - player.y
        dist = math.hypot(dx, dy)
        los = 1.0 if (not obstacles or
                      has_line_of_sight(player.x, player.y, opponent.x, opponent.y, obstacles)) else 0.0

        lead = lead_angle(player.x, player.y, opponent.x, opponent.y, opponent.vx, opponent.vy)
        phi = self.W_AIM * los * max(0.0, math.cos(wrap(lead - player.aim_angle)))

        if self.env is not None:
            diag = math.hypot(self.env.map.width, self.env.map.height)
            phi -= self.W_DIST * dist / diag

        if self.W_THREAT:
            threats = bullet_threats(player, bullets, obstacles)
            if threats:
                phi -= self.W_THREAT * threats[0][0]

        if self.W_TACTIC:
            armed = player.ammo >= self.ARMED_AMMO
            phi += self.W_TACTIC * los * (1.0 if armed else -1.0)
        return phi

    def per_tick(self, agent_idx, player, opponent, bullets, obstacles):
        if player.alive and opponent.alive:
            phi = self.potential(player, opponent, bullets, obstacles)
            self.rewards[agent_idx] += self.R_STEP
            if self.env is not None:
                diag = math.hypot(self.env.map.width, self.env.map.height)
                dist = math.hypot(opponent.x - player.x, opponent.y - player.y)
                self.rewards[agent_idx] += self.R_DIST * dist / diag
            if player.shoot_cd == SHOOT_COOLDOWN:   # fired this tick
                good = shot_on_target(player, opponent, obstacles)
                self.rewards[agent_idx] += self.R_SHOT_GOOD if good else self.R_SHOT_BAD
                if self.R_SHOT_BLIND and obstacles and not has_line_of_sight(
                        player.x, player.y, opponent.x, opponent.y, obstacles):
                    self.rewards[agent_idx] += self.R_SHOT_BLIND
        else:
            phi = 0.0   # terminal

        prev = self.prev_phi[agent_idx]
        if prev is not None:
            self.rewards[agent_idx] += self.gamma * phi - prev
        self.prev_phi[agent_idx] = phi

    def get_rewards(self):
        r = self.rewards
        self.rewards = [0.0, 0.0]
        return r
