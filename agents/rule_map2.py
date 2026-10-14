"""Scripted map-2 opponents, ordered easiest → hardest (curriculum order in agents.STAGE_NAMES).

Every bot shares one toolkit (navigation graph around obstacles, cover/peek spots, bullet
dodging, lead aim, fire discipline) and differs only in tactics and a few difficulty knobs:

    REACT    ticks between movement re-plans (reaction delay)
    MOVE_P   share of ticks it actually moves (speed throttle)
    AIM      "direct" (current bearing) or "lead" (intercept a moving target)
    AIM_ERR  chance per tick that the turret is one notch (22.5°) off
    FIRE     "los" (fire whenever the target is visible) or "ontarget" (only predicted hits)
    FIRE_P   chance to pull the trigger when it would fire
    RANGE    max engagement distance
    DODGE    chance to notice (and side-step) each dangerous incoming bullet
"""
import heapq
import math
from types import SimpleNamespace as NS

import numpy as np

from observations import PLAYER_RADIUS, bullet_threats, lead_angle, shot_on_target, wrap
from utils import has_line_of_sight
from utils.geometry import line_rect_intersect

AIM_ANGLES = [i * math.pi / 8 for i in range(16)]
MOVE_VECS = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
MOVE_ANGLES = [None] + [math.atan2(dy, dx) for dx, dy in MOVE_VECS[1:]]

NODE_PAD = PLAYER_RADIUS + 10   # nav nodes / cover spots sit this far off obstacle faces
SEG_PAD = PLAYER_RADIUS - 2     # a straight walk is clear if it stays this far from obstacles


def _idle():
    return np.zeros(3, dtype=np.int64)


def _random_action():
    return np.array([np.random.randint(0, 9), np.random.randint(0, 16),
                     np.random.randint(0, 2)], dtype=np.int64)


def _valid(s):
    return s is not None and len(s.get("players", ())) >= 2


def aim_index(angle):
    return int(round(angle / (math.pi / 8))) % 16


# ── static map analysis (cached per obstacle layout) ─────────────────────────

class _Nav:
    def __init__(self, rects, W, H):
        self.rects, self.W, self.H = rects, W, H
        self.seg_rects = [(x - SEG_PAD, y - SEG_PAD, w + 2 * SEG_PAD, h + 2 * SEG_PAD)
                          for x, y, w, h in rects]
        p = NODE_PAD
        self.nodes = [n for x, y, w, h in rects
                      for n in ((x - p, y - p), (x + w + p, y - p), (x + w + p, y + h + p), (x - p, y + h + p))
                      if self.free(*n)]
        self.adj = {i: [(j, math.dist(a, b)) for j, b in enumerate(self.nodes)
                        if i != j and self.seg_clear(a, b)]
                    for i, a in enumerate(self.nodes)}
        # cover / peek candidates: a ring of spots around every obstacle plus an open-floor grid
        spots = []
        for x, y, w, h in rects:
            for fx in (0.0, 0.25, 0.5, 0.75, 1.0):
                spots += [(x + fx * w, y - p), (x + fx * w, y + h + p)]
            for fy in (0.25, 0.5, 0.75):
                spots += [(x - p, y + fy * h), (x + w + p, y + fy * h)]
        step = 100
        spots += [(gx, gy) for gx in range(step // 2, W, step) for gy in range(step // 2, H, step)]
        self.spots = [s for s in dict.fromkeys(spots) if self.free(*s)]

    def free(self, x, y, r=PLAYER_RADIUS + 4):
        if not (r <= x <= self.W - r and r <= y <= self.H - r):
            return False
        for ox, oy, ow, oh in self.rects:
            cx, cy = min(max(x, ox), ox + ow), min(max(y, oy), oy + oh)
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                return False
        return True

    def seg_clear(self, a, b):
        return not any(line_rect_intersect(a[0], a[1], b[0], b[1], *r) for r in self.seg_rects)

    def waypoint(self, a, b):
        """First point to walk to on the shortest obstacle-avoiding path a → b."""
        if self.seg_clear(a, b):
            return b
        n = len(self.nodes)
        start = [(j, math.dist(a, q)) for j, q in enumerate(self.nodes) if self.seg_clear(a, q)]
        goal = {j: math.dist(q, b) for j, q in enumerate(self.nodes) if self.seg_clear(q, b)}
        if not start:   # wedged against a wall: head to the nearest node
            return min(self.nodes, key=lambda q: math.dist(a, q)) if self.nodes else b
        dist, first = [math.inf] * n, [None] * n
        pq = []
        for j, d in start:
            dist[j], first[j] = d, j
            heapq.heappush(pq, (d, j))
        best, best_first = math.inf, None
        while pq:
            d, i = heapq.heappop(pq)
            if d > dist[i] or d >= best:
                continue
            if i in goal and d + goal[i] < best:
                best, best_first = d + goal[i], first[i]
            for j, w in self.adj[i]:
                if d + w < dist[j]:
                    dist[j], first[j] = d + w, first[i]
                    heapq.heappush(pq, (d + w, j))
        return self.nodes[best_first] if best_first is not None else b

    def path_len(self, a, b):
        return math.dist(a, b) if self.seg_clear(a, b) else 1.6 * math.dist(a, b)


_NAV_CACHE = {}


def _nav(state):
    obstacles = state.get("obstacles", [])
    W, H = state["arena"]["w"], state["arena"]["h"]
    key = (W, H, tuple((o["x"], o["y"], o["w"], o["h"]) for o in obstacles))
    nav = _NAV_CACHE.get(key)
    if nav is None:
        nav = _NAV_CACHE[key] = _Nav(list(key[2]), W, H)
    return nav


# ── per-tick view of the game ────────────────────────────────────────────────

class _Ctx:
    def __init__(self, state, team):
        self.state = state
        self.obstacles = state.get("obstacles", [])
        self.nav = _nav(state)
        self.me = NS(**state["players"][team])
        self.en = NS(**state["players"][1 - team])
        self.me.team, self.en.team = team, 1 - team
        self.dx, self.dy = self.en.x - self.me.x, self.en.y - self.me.y
        self.dist = math.hypot(self.dx, self.dy) or 1e-6
        self.bearing = math.atan2(self.dy, self.dx)
        self.bspeed = state.get("bullet_speed", 12.0)
        self.los = has_line_of_sight(self.me.x, self.me.y, self.en.x, self.en.y, self.obstacles)
        self._threats = None

    @property
    def pos(self):
        return (self.me.x, self.me.y)

    @property
    def epos(self):
        return (self.en.x, self.en.y)

    def threats(self):
        if self._threats is None:
            bullets = [NS(x=b["x"], y=b["y"], vx=b["vx"], vy=b["vy"], owner_team=b["team"])
                       for b in self.state.get("bullets", [])]
            self._threats = bullet_threats(self.me, bullets, self.obstacles)
        return self._threats

    def danger(self):
        t = self.threats()
        return t[0][0] if t else 0.0

    def enemy_sees(self, p):
        return has_line_of_sight(p[0], p[1], self.en.x, self.en.y, self.obstacles)

    # movement primitives (all return a move index 0-8) ------------------------
    def go(self, target, stop=6.0):
        if math.dist(self.pos, target) < stop:
            return 0
        wx, wy = self.nav.waypoint(self.pos, target)
        return self.heading(math.atan2(wy - self.me.y, wx - self.me.x))

    def heading(self, angle, look=40.0):
        """Move index closest to `angle` whose next ~2 ticks are not blocked."""
        best, bd = 0, math.inf
        for k in range(1, 9):
            d = abs(wrap(MOVE_ANGLES[k] - angle))
            dx, dy = MOVE_VECS[k]
            n = math.hypot(dx, dy)
            if d < bd and self.nav.free(self.me.x + dx / n * look, self.me.y + dy / n * look, PLAYER_RADIUS + 1):
                best, bd = k, d
        return best if bd < 2.0 else 0

    def strafe(self, side, back=0.0):
        """Circle the enemy (side = ±1); back > 0 also opens range, < 0 closes it."""
        return self.heading(self.bearing + side * (math.pi / 2 + back))

    def dodge(self):
        t = self.threats()
        if not t or t[0][0] < 0.15:
            return None
        _, rx, ry, vx, vy, tca, _ = t[0]
        cx, cy = rx + vx * tca, ry + vy * tca   # bullet position relative to me at closest approach
        away = math.atan2(-cy, -cx) if math.hypot(cx, cy) > 2 else math.atan2(vx, -vy)
        return self.heading(away, look=30.0) or self.heading(away + math.pi, look=30.0)

    def cover_spot(self, min_enemy=160.0):
        """Closest spot the enemy cannot see."""
        best, bd = None, math.inf
        for s in self.nav.spots:
            if self.enemy_sees(s):
                continue
            d = self.nav.path_len(self.pos, s) + (300.0 if math.dist(s, self.epos) < min_enemy else 0.0)
            if d < bd:
                best, bd = s, d
        return best

    def peek_spot(self, lo, hi):
        """Closest spot with a view of the enemy at a distance in [lo, hi]."""
        best, bd = None, math.inf
        for s in self.nav.spots:
            if not lo <= math.dist(s, self.epos) <= hi or not self.enemy_sees(s):
                continue
            d = self.nav.path_len(self.pos, s)
            if d < bd:
                best, bd = s, d
        return best


# ── base bot ─────────────────────────────────────────────────────────────────

class _Bot:
    name = "Bot"
    color = "#94a3b8"

    REACT = 1
    MOVE_P = 1.0
    AIM = "lead"
    AIM_ERR = 0.0
    FIRE = "ontarget"
    FIRE_P = 1.0
    RANGE = 650.0
    DODGE = 0.0

    def __init__(self, team=0):
        self.team = team
        self.reset()

    def reset(self):
        self.tick = 0
        self.move = 0
        self.mode = None
        self.side = 1 if np.random.random() < 0.5 else -1
        self.goal = None
        self.noticed = {}

    def get_action(self, obs, state):
        if not _valid(state):
            return _random_action()
        if not state["players"][self.team]["alive"] or not state["players"][1 - self.team]["alive"]:
            return _idle()
        self.tick += 1
        c = _Ctx(state, self.team)
        if self.REACT == 1 or self.tick % self.REACT == 1:
            self.move = self.plan(c)
        move = self.move
        if self.DODGE and self._notices(c):
            d = c.dodge()
            if d is not None:
                move = d
        if self.MOVE_P < 1.0 and np.random.random() > self.MOVE_P:
            move = 0
        aim, fire = self.aim_fire(c)
        return np.array([move, aim, fire], dtype=np.int64)

    def plan(self, c):
        return 0

    def _notices(self, c):
        """One roll per bullet (keyed by its velocity), so DODGE is a per-shot reaction rate."""
        t = c.threats()
        if not t or t[0][0] < 0.15:
            return False
        key = (round(t[0][3], 3), round(t[0][4], 3))
        if key not in self.noticed:
            if len(self.noticed) > 64:
                self.noticed.clear()
            self.noticed[key] = np.random.random() < self.DODGE
        return self.noticed[key]

    def aim_fire(self, c):
        if self.AIM == "lead" and c.los:
            k = aim_index(lead_angle(c.me.x, c.me.y, c.en.x, c.en.y, c.en.vx, c.en.vy, c.bspeed))
        else:
            k = aim_index(c.bearing)
        if self.AIM_ERR and np.random.random() < self.AIM_ERR:
            k = (k + (1 if np.random.random() < 0.5 else -1)) % 16
        fire = 0
        if (c.los and c.me.ammo > 0 and c.me.shoot_cd <= 1 and c.dist <= self.RANGE
                and np.random.random() < self.FIRE_P):
            if self.FIRE == "los":
                fire = 1
            elif shot_on_target(NS(x=c.me.x, y=c.me.y, aim_angle=AIM_ANGLES[k]), c.en, c.obstacles, c.bspeed):
                fire = 1
            elif self.AIM == "lead":
                # the rounded lead notch misses: try its neighbours (a perfect aimer's correction)
                for kk in ((k + 1) % 16, (k - 1) % 16):
                    if shot_on_target(NS(x=c.me.x, y=c.me.y, aim_angle=AIM_ANGLES[kk]), c.en, c.obstacles, c.bspeed):
                        if np.random.random() > self.AIM_ERR * 3:
                            k, fire = kk, 1
                        break
        return k, fire

    def _strafe_or_flip(self, c, back=0.0):
        m = c.strafe(self.side, back)
        if m == 0:
            self.side = -self.side
            m = c.strafe(self.side, back)
        return m

    def _to_cover(self, c, every, min_enemy=160.0):
        if self.goal is None or c.enemy_sees(self.goal) or self.tick % every == 0:
            self.goal = c.cover_spot(min_enemy)
        return c.go(self.goal) if self.goal else self._strafe_or_flip(c, 0.7)


# ── 1  Stationary: a turret. Sits at spawn, shoots whenever it sees you. ─────
class StationaryAgent(_Bot):
    name, color = "Stationary", "#94a3b8"
    AIM, AIM_ERR, FIRE, FIRE_P = "direct", 0.35, "los", 0.5


# ── 2  SlowWalker: wanders toward you at half speed, stops to shoot. ─────────
class SlowWalkerAgent(_Bot):
    name, color = "SlowWalker", "#64748b"
    REACT, MOVE_P = 3, 0.5
    AIM, AIM_ERR, FIRE, FIRE_P = "direct", 0.3, "los", 0.6

    def plan(self, c):
        return 0 if c.los and c.dist < 450 else c.go(c.epos)


# ── 3  AggressiveWalker: full-speed rush (paths around walls), leads shots. ──
class AggressiveWalkerAgent(_Bot):
    name, color = "AggressiveWalker", "#f97316"
    REACT = 2
    AIM, AIM_ERR, FIRE = "lead", 0.25, "los"

    def plan(self, c):
        return 0 if c.los and c.dist < 110 else c.go(c.epos)


# ── 4  Strafer: closes to mid range, then strafes while trading shots. ───────
class EasyStraferAgent(_Bot):
    name, color = "EasyStrafer", "#eab308"
    REACT = 2
    AIM, AIM_ERR, FIRE = "lead", 0.2, "ontarget"

    def plan(self, c):
        if not c.los or c.dist > 450:
            return c.go(c.epos)
        if self.tick % 30 == 0:
            self.side = -self.side
        return self._strafe_or_flip(c, 0.4 if c.dist < 220 else (-0.3 if c.dist > 380 else 0.0))


# ── 5  SimpleHider: fires, ducks into cover, peeks out on a timer. ───────────
class SimpleHiderAgent(_Bot):
    name, color = "SimpleHider", "#22c55e"
    REACT = 2
    AIM, AIM_ERR, FIRE = "lead", 0.2, "ontarget"

    def plan(self, c):
        if c.los:
            return self._to_cover(c, every=20)
        return c.go(c.epos) if self.tick % 40 < 8 else 0   # peek on a timer


# ── 6  Hider: pushes while it has ammo, falls back to cover to reload. ───────
class DefensiveHiderAgent(_Bot):
    name, color = "Hider", "#f59e0b"
    REACT = 2
    AIM, AIM_ERR, FIRE = "lead", 0.15, "ontarget"
    DODGE = 0.1

    def reset(self):
        super().reset()
        self.mode = "hunt"

    def plan(self, c):
        if self.mode == "hunt" and (c.me.ammo <= 1 or c.me.health <= 36):
            self.mode = "hide"
        elif self.mode == "hide" and c.me.ammo >= 5 and c.me.health > 36:
            self.mode = "hunt"
        if self.mode == "hide":
            return self._to_cover(c, every=16)
        if c.los and c.dist < 420:
            return self._strafe_or_flip(c, -0.2 if c.dist > 300 else 0.2)
        return c.go(c.epos)


# ── 7  Mobile: kites at long range, never stands still, dodges sometimes. ────
class MobileDefenseAgent(_Bot):
    name, color = "Mobile", "#8b5cf6"
    REACT = 2
    AIM, AIM_ERR, FIRE = "lead", 0.12, "ontarget"
    DODGE = 0.2

    def plan(self, c):
        if not c.los:
            return c.go(c.peek_spot(220, 380) or c.epos)
        if self.tick % 24 == 0:
            self.side = -self.side
        return self._strafe_or_flip(c, 0.6 if c.dist < 220 else (-0.4 if c.dist > 380 else 0.1))


# ── 8  Dodger (new): close-range duelist that side-steps almost every shot. ──
class DodgerAgent(_Bot):
    name, color = "Dodger", "#06b6d4"
    AIM, AIM_ERR, FIRE, RANGE = "lead", 0.1, "ontarget", 450
    DODGE = 0.4

    def plan(self, c):
        if not c.los or c.dist > 320:
            return c.go(c.epos)
        if self.tick % 18 == 0 or np.random.random() < 0.03:
            self.side = -self.side
        return self._strafe_or_flip(c, 0.5 if c.dist < 140 else (-0.4 if c.dist > 240 else 0.0))


# ── 9  Sniper (new): holds cover, peeks only with a loaded magazine, ducks back
#       after a burst or when shot at. Punishes anyone standing in the open. ──
class SniperAgent(_Bot):
    name, color = "Sniper", "#ec4899"
    AIM, AIM_ERR, FIRE = "lead", 0.08, "ontarget"
    DODGE = 0.3

    def reset(self):
        super().reset()
        self.mode = "hide"
        self.shots_left = 0
        self.hold = 0

    def plan(self, c):
        if self.mode == "peek":
            if c.me.shoot_cd == 12:              # fired last tick
                self.shots_left -= 1
            if self.shots_left <= 0 or c.me.ammo == 0 or c.danger() > 0.3 or c.me.health <= 36:
                self.mode, self.hold = "hide", 12
        elif self.hold > 0:
            self.hold -= 1
        elif c.me.ammo >= 3 and c.me.health > 36:
            self.mode, self.shots_left = "peek", 4

        if self.mode == "hide":
            return self._to_cover(c, every=12, min_enemy=220)
        if c.los:
            return self._strafe_or_flip(c, 0.3 if c.dist < 220 else -0.3)
        return c.go(c.peek_spot(200, 400) or c.epos)


# ── 10 Hunter (new, hardest): flanks around cover to a firing spot, trades only
#       on its own terms, reloads behind walls, rushes an enemy that is out of ammo. ─
class HunterAgent(_Bot):
    name, color = "Hunter", "#ef4444"
    AIM, AIM_ERR, FIRE, RANGE = "lead", 0.04, "ontarget", 480
    DODGE = 0.5

    def reset(self):
        super().reset()
        self.mode = "hunt"

    def plan(self, c):
        hurt = c.me.health < c.en.health and c.me.health <= 54
        if self.mode == "hunt" and (c.me.ammo <= 1 or (hurt and c.me.ammo < 4)):
            self.mode = "reload"
        elif self.mode == "reload" and c.me.ammo >= 4:
            self.mode = "hunt"
        enemy_dry = c.en.ammo == 0 or (c.en.ammo == 1 and c.en.shoot_cd > 6)

        if self.mode == "reload" and not (enemy_dry and c.me.ammo >= 2):
            return self._to_cover(c, every=8, min_enemy=240)
        if c.los:
            if self.tick % 14 == 0 or np.random.random() < 0.05:
                self.side = -self.side
            want = 100 if enemy_dry else 190
            return self._strafe_or_flip(c, 0.45 if c.dist < want - 60 else (-0.45 if c.dist > want + 80 else 0.0))
        if self.goal is None or not c.enemy_sees(self.goal) or self.tick % 10 == 0:
            self.goal = c.peek_spot(150, 320)
        return c.go(self.goal or c.epos)
