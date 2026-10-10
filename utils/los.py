from .geometry import line_rect_intersect

def has_line_of_sight(x1, y1, x2, y2, obstacles):
    for obs in obstacles:

        if isinstance(obs, dict):
            ox = obs["x"]
            oy = obs["y"]
            ow = obs["w"]
            oh = obs["h"]
        else:
            ox = obs.x
            oy = obs.y
            ow = obs.w
            oh = obs.h

        if line_rect_intersect(
            x1, y1, x2, y2,
            ox, oy, ow, oh
        ):
            return False

    return True