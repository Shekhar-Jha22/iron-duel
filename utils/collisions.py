# from .geometry import circle_rect_distance, point_in_rect, rect_intersect

# def bullet_hits_obstacle(bullet_x, bullet_y, bullet_r, obstacle):
#     dist = circle_rect_distance(
#         bullet_x, bullet_y, bullet_r,
#         obstacle.x, obstacle.y, obstacle.w, obstacle.h
#     )
#     return dist < bullet_r

# def player_collides_obstacle(player_x, player_y, player_r, obstacle):
#     dist = circle_rect_distance(
#         player_x, player_y, player_r,
#         obstacle.x, obstacle.y, obstacle.w, obstacle.h
#     )
#     return dist < player_r

# def resolve_player_collision(new_x, new_y, player_r, old_x, old_y, obstacles):
#     for obs in obstacles:
#         if player_collides_obstacle(new_x, new_y, player_r, obs):
#             return (old_x, old_y)
#     return (new_x, new_y)
from .geometry import circle_rect_distance, point_in_rect, rect_intersect, closest_point_on_rect
import math

def bullet_hits_obstacle(bullet_x, bullet_y, bullet_r, obstacle):
    dist = circle_rect_distance(
        bullet_x, bullet_y, bullet_r,
        obstacle.x, obstacle.y, obstacle.w, obstacle.h
    )
    return dist < bullet_r

def player_collides_obstacle(player_x, player_y, player_r, obstacle):
    dist = circle_rect_distance(
        player_x, player_y, player_r,
        obstacle.x, obstacle.y, obstacle.w, obstacle.h
    )
    return dist < player_r

def resolve_player_collision(new_x, new_y, player_r, old_x, old_y, obstacles):
    """
    Resolve collisions with axis-decomposed sliding (no wall-sticking).

    For each colliding obstacle:
    1. Try the full move (new_x, new_y) — if clear, keep it.
    2. Try sliding along X only (new_x, old_y) — move parallel to horizontal wall.
    3. Try sliding along Y only (old_x, new_y) — move parallel to vertical wall.
    4. If all fail, stop fully (old_x, old_y).

    The velocity perpendicular to the wall is cancelled implicitly because
    the position update along that axis is discarded, while the parallel
    component is preserved.
    """
    result_x, result_y = new_x, new_y

    for obs in obstacles:
        if not player_collides_obstacle(result_x, result_y, player_r, obs):
            continue  # no collision with this obstacle, keep going

        # Try X-slide: keep new X, revert Y
        if not player_collides_obstacle(result_x, old_y, player_r, obs):
            result_y = old_y
        # Try Y-slide: keep new Y, revert X
        elif not player_collides_obstacle(old_x, result_y, player_r, obs):
            result_x = old_x
        else:
            # Fully blocked on both axes
            result_x, result_y = old_x, old_y

    return (result_x, result_y)