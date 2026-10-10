import math

def point_in_rect(px, py, rect_x, rect_y, rect_w, rect_h):
    return rect_x <= px <= rect_x + rect_w and rect_y <= py <= rect_y + rect_h

def rect_intersect(x1, y1, w1, h1, x2, y2, w2, h2):
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

def rect_center(x, y, w, h):
    return (x + w / 2, y + h / 2)

def closest_point_on_rect(px, py, rect_x, rect_y, rect_w, rect_h):
    cx = max(rect_x, min(px, rect_x + rect_w))
    cy = max(rect_y, min(py, rect_y + rect_h))
    return (cx, cy)

def circle_rect_distance(cx, cy, cr, rx, ry, rw, rh):
    close_x, close_y = closest_point_on_rect(cx, cy, rx, ry, rw, rh)
    return math.hypot(cx - close_x, cy - close_y)

def line_rect_intersect(x1, y1, x2, y2, rx, ry, rw, rh):
    dx = x2 - x1
    dy = y2 - y1
    t_min, t_max = 0.0, 1.0

    for i in range(2):
        if i == 0:
            axis_min, axis_max, val = rx, rx + rw, x1
            delta = dx
        else:
            axis_min, axis_max, val = ry, ry + rh, y1
            delta = dy

        if abs(delta) < 1e-6:
            if val < axis_min or val > axis_max:
                return False
        else:
            t1 = (axis_min - val) / delta
            t2 = (axis_max - val) / delta
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False

    return True
