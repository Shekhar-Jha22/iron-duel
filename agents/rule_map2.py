import numpy as np
import math
from utils import has_line_of_sight


# ── shared helpers ──────────────────────────────────────────────────────────

def _angle_to_aim(angle):
    """Convert radian angle to 0-15 aim index."""
    return int(round(angle / (math.pi / 8))) % 16

def _angle_to_move(angle):
    """Convert radian angle to 1-8 movement index (0 = stay)."""
    angle_deg = (math.degrees(angle) + 90) % 360
    return (int(round(angle_deg / 45.0)) % 8) + 1

def _toward(my_pos, target_pos):
    """Return (dx, dy, dist, angle) from my_pos toward target_pos."""
    dx = target_pos['x'] - my_pos['x']
    dy = target_pos['y'] - my_pos['y']
    dist = math.hypot(dx, dy) or 1.0
    angle = math.atan2(dy, dx)
    return dx, dy, dist, angle

def _valid(env_state):
    return (env_state is not None and
            'players' in env_state and
            len(env_state['players']) >= 2)

def _random_action():
    return np.array([
        np.random.randint(0, 9),
        np.random.randint(0, 16),
        np.random.randint(0, 2),
    ], dtype=np.int64)


# ── Stage 1: Stationary ─────────────────────────────────────────────────────
# Never moves. Aims at enemy and shoots if has LOS.
# Teaches the agent: aim at a target and shoot it.

class StationaryAgent:
    name = "Stationary"
    color = "#94a3b8"

    def __init__(self, team=0):
        self.team = team

    def get_action(self, obs, env_state):
        if not _valid(env_state):
            return _random_action()
        try:
            me = env_state['players'][self.team]
            en = env_state['players'][1 - self.team]
            if not me['alive'] or not en['alive']:
                return np.array([0, 0, 0], dtype=np.int64)

            _, _, _, angle = _toward(me, en)
            aim = _angle_to_aim(angle)
            obstacles = env_state.get('obstacles', [])
            los = has_line_of_sight(me['x'], me['y'], en['x'], en['y'], obstacles)
            shoot = 1 if los and me['ammo'] > 0 else 0
            return np.array([0, aim, 0], dtype=np.int64)   # 0 = don't move
        except Exception:
            return _random_action()


# ── Stage 2: SlowWalker ─────────────────────────────────────────────────────
# Walks toward enemy at ~30% speed (moves 1 tick out of every 3).
# Stops to aim and shoot when it has LOS.
# Teaches the agent: hit a slowly approaching target.

class SlowWalkerAgent:
    name = "SlowWalker"
    color = "#64748b"

    def __init__(self, team=0):
        self.team = team
        self.tick = 0

    def get_action(self, obs, env_state):
        if not _valid(env_state):
            return _random_action()
        try:
            me = env_state['players'][self.team]
            en = env_state['players'][1 - self.team]
            if not me['alive'] or not en['alive']:
                return np.array([0, 0, 0], dtype=np.int64)

            self.tick += 1
            obstacles = env_state.get('obstacles', [])
            _, _, _, angle = _toward(me, en)
            aim = _angle_to_aim(angle)
            los = has_line_of_sight(me['x'], me['y'], en['x'], en['y'], obstacles)
            shoot = 1 if los and me['ammo'] > 0 else 0

            # move only every 3rd tick, and only if no LOS (stop to shoot)
            if self.tick % 3 == 0 and not los:
                move = _angle_to_move(angle)
            else:
                move = 0

            return np.array([move, aim, 0], dtype=np.int64)
        except Exception:
            return _random_action()


# ── Stage 3: AggressiveWalker ───────────────────────────────────────────────
# Walks toward enemy at full speed. Shoots whenever it has LOS.
# No cover seeking, no strafing. Pure aggression.
# Teaches the agent: deal with a target that closes distance while shooting.

class AggressiveWalkerAgent:
    name = "AggressiveWalker"
    color = "#f97316"

    def __init__(self, team=0):
        self.team = team

    def get_action(self, obs, env_state):
        if not _valid(env_state):
            return _random_action()
        try:
            me = env_state['players'][self.team]
            en = env_state['players'][1 - self.team]
            if not me['alive'] or not en['alive']:
                return np.array([0, 0, 0], dtype=np.int64)

            obstacles = env_state.get('obstacles', [])
            _, _, dist, angle = _toward(me, en)
            aim = _angle_to_aim(angle)
            los = has_line_of_sight(me['x'], me['y'], en['x'], en['y'], obstacles)
            shoot = 1 if los and me['ammo'] > 0 else 0

            # always walk toward enemy
            move = _angle_to_move(angle) if dist > 60 else 0

            return np.array([move, aim, shoot], dtype=np.int64)
        except Exception:
            return _random_action()


# ── Stage 4: EasyStrafer ────────────────────────────────────────────────────
# Strafes side to side (perpendicular to enemy) but stays in place overall.
# Shoots when it has LOS. Doesn't reposition or seek cover.
# Teaches the agent: lead shots on a laterally moving target.

class EasyStraferAgent:
    name = "EasyStrafer"
    color = "#eab308"

    def __init__(self, team=0):
        self.team = team
        self.tick = 0

    def get_action(self, obs, env_state):
        if not _valid(env_state):
            return _random_action()
        try:
            me = env_state['players'][self.team]
            en = env_state['players'][1 - self.team]
            if not me['alive'] or not en['alive']:
                return np.array([0, 0, 0], dtype=np.int64)

            self.tick += 1
            obstacles = env_state.get('obstacles', [])
            _, _, _, angle = _toward(me, en)
            aim = _angle_to_aim(angle)
            los = has_line_of_sight(me['x'], me['y'], en['x'], en['y'], obstacles)
            shoot = 1 if los and me['ammo'] > 0 else 0

            # alternate strafe direction every 30 ticks
            perp = angle + (math.pi / 2 if (self.tick // 30) % 2 == 0 else -math.pi / 2)
            move = _angle_to_move(perp)

            return np.array([move, aim, shoot], dtype=np.int64)
        except Exception:
            return _random_action()


# ── Stage 5: SimpleHider ────────────────────────────────────────────────────
# Moves toward nearest obstacle when exposed, peeks to shoot when hidden.
# No LOS checking for shooting accuracy — just dumb peek-and-fire.
# Easier than DefensiveHiderAgent (no aim correction, slower decisions).

class SimpleHiderAgent:
    name = "SimpleHider"
    color = "#22c55e"

    def __init__(self, team=0):
        self.team = team
        self.tick = 0

    def get_action(self, obs, env_state):
        if not _valid(env_state):
            return _random_action()
        try:
            me = env_state['players'][self.team]
            en = env_state['players'][1 - self.team]
            obstacles = env_state.get('obstacles', [])
            if not me['alive'] or not en['alive']:
                return np.array([0, 0, 0], dtype=np.int64)

            self.tick += 1
            los = has_line_of_sight(me['x'], me['y'], en['x'], en['y'], obstacles)
            _, _, _, angle_to_en = _toward(me, en)
            aim = _angle_to_aim(angle_to_en)

            if los:
                # exposed — shoot then immediately try to move to cover
                shoot = 1 if me['ammo'] > 0 else 0
                # find nearest obstacle
                best = None
                best_dist = float('inf')
                for obs in obstacles:
                    cx = obs['x'] + obs['w'] / 2
                    cy = obs['y'] + obs['h'] / 2
                    d = math.hypot(cx - me['x'], cy - me['y'])
                    if d < best_dist:
                        best_dist = d
                        best = (cx, cy)
                if best and best_dist > 40:
                    _, _, _, cover_angle = _toward(me, {'x': best[0], 'y': best[1]})
                    move = _angle_to_move(cover_angle)
                else:
                    move = 0
            else:
                # hidden — peek every 40 ticks
                shoot = 0
                if self.tick % 40 < 5:
                    move = _angle_to_move(angle_to_en)   # peek out
                else:
                    move = 0                              # stay in cover

            return np.array([move, aim, shoot], dtype=np.int64)
        except Exception:
            return _random_action()


# ── Stage 6: DefensiveHiderAgent (existing, unchanged) ──────────────────────
# Full hider with accurate LOS, cover-seeking, and proper aim.

class DefensiveHiderAgent:
    name = "Hider"
    color = "#f59e0b"

    def __init__(self, team=0):
        self.team = team
        self.action_state = "hunt"
        self.hide_timer = 0

    def get_action(self, obs, env_state):
        if env_state is None or not _valid(env_state):
            return _random_action()
        try:
            me = env_state['players'][self.team]
            en = env_state['players'][1 - self.team]
            obstacles = env_state.get('obstacles', [])

            los = has_line_of_sight(me['x'], me['y'], en['x'], en['y'], obstacles)
            _, _, dist, angle = _toward(me, en)
            aim = _angle_to_aim(angle)

            if los and dist < 400:
                self.hide_timer = 0
                move_dir = _angle_to_move(angle)
                shoot = 1 if me['ammo'] > 0 else 0
                return np.array([move_dir, aim, shoot], dtype=np.int64)
            else:
                self.hide_timer += 1
                best = None
                best_dist = float('inf')
                for ob in obstacles:
                    cx = ob['x'] + ob['w'] / 2
                    cy = ob['y'] + ob['h'] / 2
                    d = math.hypot(cx - me['x'], cy - me['y'])
                    if d < best_dist:
                        best_dist = d
                        best = (cx, cy)
                if best:
                    _, _, _, cover_angle = _toward(me, {'x': best[0], 'y': best[1]})
                    move = _angle_to_move(cover_angle)
                else:
                    move = np.random.randint(1, 9)
                return np.array([move, aim, 0], dtype=np.int64)
        except Exception as e:
            print(f"❌ DefensiveHiderAgent ERROR: {e}")
            return _random_action()


# ── Stage 7: MobileDefenseAgent (existing, with ammo fix) ───────────────────

class MobileDefenseAgent:
    name = "Mobile"
    color = "#8b5cf6"

    def __init__(self, team=0):
        self.team = team
        self.tick = 0

    def get_action(self, obs, env_state):
        if env_state is None or not _valid(env_state):
            return _random_action()
        try:
            me = env_state['players'][self.team]
            en = env_state['players'][1 - self.team]
            obstacles = env_state.get('obstacles', [])

            self.tick += 1
            los = has_line_of_sight(me['x'], me['y'], en['x'], en['y'], obstacles)
            _, _, _, angle = _toward(me, en)
            aim = _angle_to_aim(angle)

            perp = angle + math.pi / 2
            if self.tick % 20 < 10:
                move_angle = perp
            else:
                move_angle = angle + math.pi
            move = _angle_to_move(move_angle)

            if los and me['ammo'] > 0:
                shoot = 1
            else:
                shoot = 0

            return np.array([move, aim, shoot], dtype=np.int64)
        except Exception as e:
            print(f"❌ MobileDefenseAgent ERROR: {e}")
            return _random_action()