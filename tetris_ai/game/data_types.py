from typing import Optional

import numpy as np

Vector2 = np.ndarray  # [x, y]
ListVector2 = np.ndarray  # [[x, y], [x, y], ...]


def rotate_point(
        point: Vector2,
        degrees: float,
        origin: Optional[Vector2] = None,
) -> Vector2:
    if origin is None:
        origin = np.array([0.0, 0.0])

    angle = np.deg2rad(degrees)

    oy, ox = origin
    py, px = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    return np.array([qy, qx])


def rotate_points(
        points: ListVector2,
        degrees: float,
        origin: Optional[Vector2] = None,
) -> ListVector2:
    if origin is None:
        origin = np.array([0.0, 0.0])

    angle = np.deg2rad(degrees)

    oy, ox = origin
    py, px = points[:, 0], points[:, 1]

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    return np.array([qy, qx]).T