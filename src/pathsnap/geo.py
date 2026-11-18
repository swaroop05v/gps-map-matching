"""Geometry helpers.

Longitude/latitude are angles, so we never use Euclidean distance in degrees.
Each trip covers a few km, so we use a local equirectangular projection centred on
the trip (x = R*dlon*cos(lat0), y = R*dlat). At this scale its error versus the
haversine distance is far below GPS noise (checked in scripts/01_eda.py).
"""
import numpy as np

EARTH_R = 6371008.8  # mean Earth radius in metres


class LocalProjection:
    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = np.deg2rad(1.0) * EARTH_R * np.cos(np.deg2rad(lat0))
        self.ky = np.deg2rad(1.0) * EARTH_R

    def to_xy(self, lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        return np.column_stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky])

    def to_lonlat(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_to_segment(P, A, B):
    """Distance from points P to segments AB (broadcasting over leading dims).

    Returns (distance, t_clipped, t_raw) where t is the projection fraction along AB.
    """
    AB = B - A
    AP = P - A
    L2 = (AB * AB).sum(-1)
    t_raw = (AP * AB).sum(-1) / np.maximum(L2, 1e-9)
    t = np.clip(t_raw, 0.0, 1.0)
    proj = A + t[..., None] * AB
    d = np.sqrt(((P - proj) ** 2).sum(-1))
    return d, t, t_raw


def bearing_deg(dx, dy):
    """Compass bearing in degrees (0 = north, clockwise) of a displacement in metres."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)
