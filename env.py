import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

from maps import MAP1
from rewards import Map1Reward
from observations import build_observation, OBS_DIM
from utils.collisions import (
    resolve_player_collision,
    bullet_hits_obstacle,
)

PLAYER_RADIUS = 18
BULLET_RADIUS = 5
BULLET_SPEED = 12
MAX_SPEED = 5
ACCEL = 1.2
FRICTION = 0.82
MAX_HEALTH = 100
BULLET_DAMAGE = 18
MAX_AMMO = 8
RELOAD_TICKS = 45
MAX_STEPS = 1200
SHOOT_COOLDOWN = 12

N_MOVE_DIRS = 9
N_AIM_DIRS = 16

MOVE_DIRS = [
    (0, 0),
    (0, -1), (1, -1), (1, 0),
    (1, 1),  (0, 1),  (-1, 1),
    (-1, 0), (-1, -1),
]
AIM_ANGLES = [i * (math.pi / 8) for i in range(16)]

class Player:
    def __init__(self, x, y, team):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = 0.0, 0.0
        self.health = MAX_HEALTH
        self.ammo = MAX_AMMO
        self.reload_tick = 0
        self.shoot_cd = 0
        self.aim_angle = 0.0
        self.team = team
        self.alive = True

    def to_obs(self, arena_w, arena_h, max_speed):
        return np.array([
            self.x / arena_w,
            self.y / arena_h,
            self.vx / max_speed,
            self.vy / max_speed,
            self.health / MAX_HEALTH,
            self.ammo / MAX_AMMO,
            self.aim_angle / (2 * math.pi),
            float(self.alive),
        ], dtype=np.float32)

class Bullet:
    def __init__(self, x, y, vx, vy, owner_team):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.owner_team = owner_team
        self.alive = True

    def step(self, arena_w, arena_h, obstacles):
        self.x += self.vx
        self.y += self.vy
        if not (0 <= self.x <= arena_w and 0 <= self.y <= arena_h):
            self.alive = False
        for obs in obstacles:
            if bullet_hits_obstacle(self.x, self.y, BULLET_RADIUS, obs):
                self.alive = False
                break

class MiniMilitiaEnv(gym.Env):
    metadata = {"render_modes": ["none"]}

    def __init__(self, map_cfg=None, reward_fn=None, max_steps=MAX_STEPS):
        super().__init__()
        self.map = map_cfg or MAP1
        self.reward_fn = reward_fn or Map1Reward()
        if hasattr(self.reward_fn, "attach"):
            self.reward_fn.attach(self)
        self.max_steps = max_steps

        # Use per-map speed/accel if defined, else fall back to globals
        self.max_speed = getattr(self.map, "max_speed", MAX_SPEED)
        self.accel     = getattr(self.map, "accel",     ACCEL)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([N_MOVE_DIRS, N_AIM_DIRS, 2])

        self.players = [None, None]
        self.bullets = []
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.players = [
            Player(self.map.player_spawn.x, self.map.player_spawn.y, team=0),
            Player(self.map.enemy_spawn.x, self.map.enemy_spawn.y, team=1),
        ]
        self.players[1].aim_angle = math.pi   # face each other → symmetric start
        self.bullets = []
        self.step_count = 0
        self.reward_fn.reset()
        return self._get_obs(0), {}

    def step(self, actions):
        assert len(actions) == 2
        self.step_count += 1

        prev_pos = [
            (self.players[0].x, self.players[0].y),
            (self.players[1].x, self.players[1].y),
        ]

        for i, (player, action) in enumerate(zip(self.players, actions)):
            if not player.alive:
                continue
            move_dir, aim_dir, shoot = int(action[0]), int(action[1]), int(action[2])

            dx, dy = MOVE_DIRS[move_dir]
            norm = math.hypot(dx, dy) or 1
            player.vx = (player.vx + dx / norm * self.accel) * FRICTION
            player.vy = (player.vy + dy / norm * self.accel) * FRICTION
            player.vx = float(np.clip(player.vx, -self.max_speed, self.max_speed))
            player.vy = float(np.clip(player.vy, -self.max_speed, self.max_speed))

            new_x = float(np.clip(player.x + player.vx, PLAYER_RADIUS, self.map.width - PLAYER_RADIUS))
            new_y = float(np.clip(player.y + player.vy, PLAYER_RADIUS, self.map.height - PLAYER_RADIUS))
            new_x, new_y = resolve_player_collision(new_x, new_y, PLAYER_RADIUS, player.x, player.y, self.map.obstacles)
            player.x, player.y = new_x, new_y

            player.aim_angle = AIM_ANGLES[aim_dir]

            if player.ammo < MAX_AMMO:
                player.reload_tick += 1
                if player.reload_tick >= RELOAD_TICKS:
                    player.ammo += 1
                    player.reload_tick = 0

            if player.shoot_cd > 0:
                player.shoot_cd -= 1

            if shoot and player.ammo > 0 and player.shoot_cd == 0:
                player.ammo -= 1
                player.shoot_cd = SHOOT_COOLDOWN
                bvx = math.cos(player.aim_angle) * BULLET_SPEED
                bvy = math.sin(player.aim_angle) * BULLET_SPEED
                self.bullets.append(Bullet(player.x, player.y, bvx, bvy, owner_team=player.team))

        for bullet in self.bullets:
            bullet.step(self.map.width, self.map.height, self.map.obstacles)

        kill_fired = [False, False]
        for bullet in self.bullets:
            if not bullet.alive:
                continue
            for j, target in enumerate(self.players):
                if not target.alive or target.team == bullet.owner_team:
                    continue
                dist = math.hypot(bullet.x - target.x, bullet.y - target.y)
                if dist < PLAYER_RADIUS + BULLET_RADIUS:
                    target.health -= BULLET_DAMAGE
                    bullet.alive = False
                    self.reward_fn.on_hit(bullet.owner_team, j)
                    if target.health <= 0:
                        target.health = 0
                        target.alive = False
                        self.reward_fn.on_kill(bullet.owner_team, j)
                        kill_fired[bullet.owner_team] = True

        self.bullets = [b for b in self.bullets if b.alive]

        for i in range(2):
            self.reward_fn.per_tick(i, self.players[i], self.players[1-i], self.bullets, self.map.obstacles)

        p0_alive = self.players[0].alive
        p1_alive = self.players[1].alive
        someone_died = not p0_alive or not p1_alive
        timed_out = not someone_died and self.step_count >= self.max_steps
        # Time limit is a real game end (outcome reward + time in obs) → terminal, not truncation.
        terminated = someone_died or timed_out
        truncated = False

        if not p1_alive and p0_alive and not kill_fired[0]:
            self.reward_fn.on_kill(0, 1)
        elif not p0_alive and p1_alive and not kill_fired[1]:
            self.reward_fn.on_kill(1, 0)
        elif timed_out:
            if hasattr(self.reward_fn, "on_timeout_with_state"):
                self.reward_fn.on_timeout_with_state(self.players)
            else:
                self.reward_fn.on_timeout()

        rewards = self.reward_fn.get_rewards()
        obs = self._get_obs(0)
        info = {
            "p0_health": self.players[0].health,
            "p1_health": self.players[1].health,
            "p0_alive": self.players[0].alive,
            "p1_alive": self.players[1].alive,
            "step": self.step_count,
            "reward_p0": rewards[0],
            "reward_p1": rewards[1],
        }
        return obs, rewards[0], terminated, truncated, info

    def _get_obs(self, agent_idx):
        return build_observation(self, agent_idx)

    def get_full_state(self):
        return {
            "arena": {"w": self.map.width, "h": self.map.height},
            "map_id": self.map.name,
            "obstacles": [
                {"x": o.x, "y": o.y, "w": o.w, "h": o.h, "type": o.type}
                for o in self.map.obstacles
            ],
            "players": [
                {
                    "x": p.x, "y": p.y,
                    "vx": p.vx, "vy": p.vy,
                    "health": p.health,
                    "ammo": p.ammo,
                    "aim_angle": p.aim_angle,
                    "alive": p.alive,
                    "team": p.team,
                }
                for p in self.players
            ],
            "bullets": [
                {"x": b.x, "y": b.y, "vx": b.vx, "vy": b.vy, "team": b.owner_team}
                for b in self.bullets
            ],
            "step": self.step_count,
        }