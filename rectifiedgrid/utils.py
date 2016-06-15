import pyproj
import math
from shapely import ops
from functools import partial

EEA_GRID_RESOLUTIONS = [25, 100, 250, 500, 1000, 2500, 10000, 25000, 100000]


def calculate_gbounds(bounds, res):
    mapwidth = bounds[2] - bounds[0]
    mapheight = bounds[3] - bounds[1]
    width_reminder = (res - math.fmod(mapwidth, res))
    height_reminder = (res - math.fmod(mapheight, res))
    gbounds = [
        bounds[0] - width_reminder/2.,
        bounds[1] - height_reminder/2.,
        bounds[2] + width_reminder/2.,
        bounds[3] + height_reminder/2.,
    ]
    return gbounds


def calculate_eea_gbounds(bounds, res):
    if res not in EEA_GRID_RESOLUTIONS:
        raise ValueError("The EEA Reference Grid doesn't support the resolution {}".format(res))
    gbounds = [
        int(bounds[0] / res) * res,
        int(bounds[1] / res) * res,
        int(bounds[2] / res) * res + res,
        int(bounds[3] / res) * res + res,
    ]
    return gbounds


def parse_projection(p):
    """Initialize a pyproj.Proj object.
    """
    if isinstance(p, pyproj.Proj):
        return p
    elif isinstance(p, int):
        return pyproj.Proj(init='epsg:{}'.format(p))
    elif isinstance(p, str):
        return pyproj.Proj(init=p)
    elif isinstance(p, dict):
        return pyproj.Proj(**p)


def transform(g, from_srs, to_srs):
    project = partial(
        pyproj.transform,
        parse_projection(from_srs),
        parse_projection(to_srs))

    return ops.transform(project, g)
