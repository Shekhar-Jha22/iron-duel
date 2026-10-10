import math
import numpy as np

from utils import has_line_of_sight

PLAYER_RADIUS  = 18
BULLET_RADIUS  = 5
BULLET_SPEED   = 12
MAX_HEALTH     = 100
MAX_AMMO       = 8
SHOOT_COOLDOWN = 12
RELOAD_TICKS   = 45

HIT_R          = PLAYER_RADIUS + BULLET_RADIUS
N_RAYS         = 16
N_BULLETS      = 4
BULLET_FEATS   = 7
BULLET_LOCAL   = 300.0   # px scale for bullet-relative positions
MAX_LEAD_T     = 100.0   # ticks; beyond this, aim directly
MAX_SHOT_T     = 100.0

# Layout (74):
#   self      10  pos(2) vel(2) hp ammo cd reload aim(2)
#   opponent  12  rel(2) dist bearing(2) vel(2) hp ammo cd threat_aim los
#   aim        6  lead_dir(2) lead_err(2) on_target can_fire
#   bullets   29  4 × [present rel(2) dir(2) t_ca miss] + max_danger
#   rays      16  free distance along the 16 aim directions
#   time       1  fraction of episode remaining
OBS_DIM = 10 + 12 + 6 + N_BULLETS * BULLET_FEATS + 1 + N_RAYS + 1

_RAY_DIRS = np.array([[math.cos(i * math.pi / 8), math.sin(i * math.pi / 8)]
                      for i in range(N_RAYS)])
_RAY_DIRS[np.abs(_RAY_DIRS) < 1e-9] = 1e-12   # slab method needs nonzero components

# Player 1 observes in a 180°-rotated frame (both maps are point-symmetric),
# so one policy serves both spawns. Actions produced from obs(1) must be mapped back.
_MOVE_ROT180 = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4])


def to_world_action(action, agent_idx):
    a = np.asarray(action, dtype=np.int64).reshape(-1).copy()
    if agent_idx == 1:
        a[0] = _MOVE_ROT180[a[0]]
        a[1] = (a[1] + N_RAYS // 2) % N_RAYS
    return a


# ── Shared geometry (also used by reward shaping) ───────────────────────────

def lead_angle(px, py, ox, oy, ovx, ovy):
    """Angle to fire so a bullet intercepts a constant-velocity target."""
    dx, dy = ox - px, oy - py
    a = ovx * ovx + ovy * ovy - BULLET_SPEED * BULLET_SPEED
    b = 2.0 * (dx * ovx + dy * ovy)
    c = dx * dx + dy * dy
    t = 0.0
    if abs(a) < 1e-9:
        if b < 0:
            t = -c / b
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            sq = math.sqrt(disc)
            roots = [r for r in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if r > 0]
            if roots:
                t = min(roots)
    if t > MAX_LEAD_T:
        t = 0.0
    return math.atan2(dy + ovy * t, dx + ovx * t)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def shot_on_target(p, opp, obstacles):
    """Would a bullet fired now along the current aim hit opp (constant velocity)?"""
    ax, ay = math.cos(p.aim_angle), math.sin(p.aim_angle)
    wx, wy = ax * BULLET_SPEED - opp.vx, ay * BULLET_SPEED - opp.vy
    dx, dy = opp.x - p.x, opp.y - p.y
    ww = wx * wx + wy * wy
    if ww < 1e-9:
        return 0.0
    t = (dx * wx + dy * wy) / ww
    if t <= 0 or t > MAX_SHOT_T:
        return 0.0
    if math.hypot(wx * t - dx, wy * t - dy) > HIT_R:
        return 0.0
    if obstacles and not has_line_of_sight(p.x, p.y, p.x + ax * BULLET_SPEED * t,
                                           p.y + ay * BULLET_SPEED * t, obstacles):
        return 0.0
    return 1.0


def bullet_threats(p, bullets, obstacles):
    """Enemy bullets with closest-approach stats, most dangerous first.
    Each entry: (danger, rx, ry, vx, vy, t_ca, miss)."""
    out = []
    for b in bullets:
        if b.owner_team == p.team:
            continue
        rx, ry = b.x - p.x, b.y - p.y
        vv = b.vx * b.vx + b.vy * b.vy
        t = -(rx * b.vx + ry * b.vy) / vv if vv > 0 else 0.0
        miss = math.hypot(rx + b.vx * t, ry + b.vy * t) if t > 0 else math.hypot(rx, ry)
        danger = 0.0
        if t > 0 and not (obstacles and not has_line_of_sight(b.x, b.y, p.x, p.y, obstacles)):
            danger = math.exp(-max(miss - HIT_R, 0.0) / 30.0) * math.exp(-t / 20.0)
        out.append((danger, rx, ry, b.vx, b.vy, t, miss))
    out.sort(key=lambda z: (-z[0], z[1] * z[1] + z[2] * z[2]))
    return out


def _rects(env):
    r = getattr(env, "_rect_cache", None)
    if r is None:
        r = np.array([[o.x, o.y, o.x + o.w, o.y + o.h] for o in env.map.obstacles],
                     dtype=np.float64).reshape(-1, 4)
        env._rect_cache = r
    return r


def raycast(px, py, rects, W, H):
    """Distance from (px,py) to the first wall/obstacle along each of the 16 directions."""
    dx, dy = _RAY_DIRS[:, 0], _RAY_DIRS[:, 1]
    tx = np.where(dx > 0, (W - px) / dx, -px / dx)
    ty = np.where(dy > 0, (H - py) / dy, -py / dy)
    t = np.minimum(tx, ty)
    if len(rects):
        ix, iy = (1.0 / dx)[:, None], (1.0 / dy)[:, None]
        t0x, t1x = (rects[:, 0] - px) * ix, (rects[:, 2] - px) * ix
        t0y, t1y = (rects[:, 1] - py) * iy, (rects[:, 3] - py) * iy
        tmin = np.maximum(np.minimum(t0x, t1x), np.minimum(t0y, t1y))
        tmax = np.minimum(np.maximum(t0x, t1x), np.maximum(t0y, t1y))
        tmin = np.maximum(tmin, 0.0)
        th = np.where(tmax >= tmin, tmin, np.inf)
        t = np.minimum(t, th.min(axis=1))
    return t


# ── Observation ──────────────────────────────────────────────────────────────

def build_observation(env, agent_idx, _c=None):
    p, opp = env.players[agent_idx], env.players[1 - agent_idx]
    W, H = env.map.width, env.map.height
    L = float(max(W, H))
    diag = math.hypot(W, H)
    vmax = env.max_speed
    obstacles = env.map.obstacles
    sg = -1.0 if agent_idx == 1 else 1.0   # 180° rotation flips every vector

    dx, dy = opp.x - p.x, opp.y - p.y
    dist = math.hypot(dx, dy) or 1e-6
    los = 1.0 if (not obstacles or has_line_of_sight(p.x, p.y, opp.x, opp.y, obstacles)) else 0.0

    # opponent aim threat: cos between opp's aim and opp→self direction
    threat_aim = (math.cos(opp.aim_angle) * -dx + math.sin(opp.aim_angle) * -dy) / dist

    lead = lead_angle(p.x, p.y, opp.x, opp.y, opp.vx, opp.vy)
    err = wrap(lead - p.aim_angle)   # rotation-invariant

    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[0:10] = (
        sg * (2 * p.x / W - 1), sg * (2 * p.y / H - 1),
        sg * p.vx / vmax, sg * p.vy / vmax,
        p.health / MAX_HEALTH, p.ammo / MAX_AMMO,
        p.shoot_cd / SHOOT_COOLDOWN, p.reload_tick / RELOAD_TICKS,
        sg * math.cos(p.aim_angle), sg * math.sin(p.aim_angle),
    )
    o[10:22] = (
        sg * dx / L, sg * dy / L, dist / diag,
        sg * dx / dist, sg * dy / dist,
        sg * opp.vx / vmax, sg * opp.vy / vmax,
        opp.health / MAX_HEALTH, opp.ammo / MAX_AMMO, opp.shoot_cd / SHOOT_COOLDOWN,
        threat_aim, los,
    )
    o[22:28] = (
        sg * math.cos(lead), sg * math.sin(lead),
        math.cos(err), math.sin(err),
        shot_on_target(p, opp, obstacles),
        1.0 if (p.ammo > 0 and p.shoot_cd == 0) else 0.0,
    )

    i = 28
    threats = bullet_threats(p, env.bullets, obstacles)
    for danger, rx, ry, bvx, bvy, t, miss in threats[:N_BULLETS]:
        o[i:i + BULLET_FEATS] = (
            1.0,
            sg * rx / BULLET_LOCAL, sg * ry / BULLET_LOCAL,
            sg * bvx / BULLET_SPEED, sg * bvy / BULLET_SPEED,
            t / 30.0, miss / 150.0,
        )
        i += BULLET_FEATS
    i = 28 + N_BULLETS * BULLET_FEATS
    o[i] = threats[0][0] if threats else 0.0
    i += 1

    free = np.maximum(raycast(p.x, p.y, _rects(env), W, H) - PLAYER_RADIUS, 0.0) / (0.5 * L)
    if agent_idx == 1:
        free = np.roll(free, -N_RAYS // 2)   # canonical ray k = world ray k+8
    o[i:i + N_RAYS] = free
    i += N_RAYS

    o[i] = 1.0 - env.step_count / env.max_steps
    return np.clip(o, -1.0, 1.0)
