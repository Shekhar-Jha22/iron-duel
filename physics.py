import math
import numpy as np

from utils import (
    resolve_player_collision,
    bullet_hits_obstacle,
)

PLAYER_RADIUS = 18
BULLET_RADIUS = 5
BULLET_SPEED = 12

MAX_SPEED = 5
ACCEL = 1.2
FRICTION = 0.82

BULLET_DAMAGE = 18


def update_player(
    player,
    move_dir,
    move_dirs,
    map_cfg,
):
    """
    Apply movement physics.
    """

    dx, dy = move_dirs[move_dir]

    norm = math.hypot(dx, dy) or 1.0

    # Use per-map speed/accel if defined, else fall back to globals
    accel     = getattr(map_cfg, "accel",     ACCEL)
    max_speed = getattr(map_cfg, "max_speed", MAX_SPEED)

    player.vx = (player.vx + dx / norm * accel) * FRICTION
    player.vy = (player.vy + dy / norm * accel) * FRICTION

    player.vx = float(np.clip(player.vx, -max_speed, max_speed))
    player.vy = float(np.clip(player.vy, -max_speed, max_speed))

    new_x = float(
        np.clip(
            player.x + player.vx,
            PLAYER_RADIUS,
            map_cfg.width - PLAYER_RADIUS,
        )
    )

    new_y = float(
        np.clip(
            player.y + player.vy,
            PLAYER_RADIUS,
            map_cfg.height - PLAYER_RADIUS,
        )
    )

    new_x, new_y = resolve_player_collision(
        new_x,
        new_y,
        PLAYER_RADIUS,
        player.x,
        player.y,
        map_cfg.obstacles,
    )

    player.x = new_x
    player.y = new_y


def update_aim(
    player,
    aim_dir,
    aim_angles,
):
    player.aim_angle = aim_angles[aim_dir]


def spawn_bullet(
    player,
):
    bvx = math.cos(player.aim_angle) * BULLET_SPEED
    bvy = math.sin(player.aim_angle) * BULLET_SPEED

    return {
        "x": player.x,
        "y": player.y,
        "vx": bvx,
        "vy": bvy,
        "owner_team": player.team,
    }


def update_reload(player, max_ammo, reload_ticks):
    if player.ammo < max_ammo:
        player.reload_tick += 1

        if player.reload_tick >= reload_ticks:
            player.ammo += 1
            player.reload_tick = 0


def update_shoot_cooldown(player):
    if player.shoot_cd > 0:
        player.shoot_cd -= 1


def update_bullets(
    bullets,
    map_cfg,
):
    for bullet in bullets:

        bullet.x += bullet.vx
        bullet.y += bullet.vy

        if not (
            0 <= bullet.x <= map_cfg.width
            and
            0 <= bullet.y <= map_cfg.height
        ):
            bullet.alive = False
            continue

        for obs in map_cfg.obstacles:

            if bullet_hits_obstacle(
                bullet.x,
                bullet.y,
                BULLET_RADIUS,
                obs,
            ):
                bullet.alive = False
                break


def resolve_bullet_hits(
    bullets,
    players,
    reward_fn,
):
    for bullet in bullets:

        if not bullet.alive:
            continue

        for target in players:

            if (
                not target.alive
                or
                target.team == bullet.owner_team
            ):
                continue

            dist = math.hypot(
                bullet.x - target.x,
                bullet.y - target.y,
            )

            if dist < PLAYER_RADIUS + BULLET_RADIUS:

                target.health -= BULLET_DAMAGE

                bullet.alive = False

                reward_fn.on_hit(
                    bullet.owner_team,
                    target.team,
                )

                if target.health <= 0:

                    target.health = 0
                    target.alive = False

                    reward_fn.on_kill(
                        bullet.owner_team,
                        target.team,
                    )


def cleanup_bullets(bullets):
    return [b for b in bullets if b.alive]