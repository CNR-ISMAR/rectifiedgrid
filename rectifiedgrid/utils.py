import numpy as np
import pandas as pd
import pyproj
import math
from shapely import ops
from functools import partial
from rasterio.crs import CRS
import rasterio
from pyproj.enums import WktVersion
from matplotlib.colors import LinearSegmentedColormap

EEA_GRID_RESOLUTIONS = [25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 100000]


def calculate_gbounds(bounds, res):
    mapwidth = bounds[2] - bounds[0]
    mapheight = bounds[3] - bounds[1]
    width_reminder = (res - math.fmod(mapwidth, res))
    height_reminder = (res - math.fmod(mapheight, res))
    if width_reminder == res:
        width_reminder = 0
    if height_reminder == res:
        height_reminder = 0
    gbounds = [
        bounds[0] - width_reminder / 2.,
        bounds[1] - height_reminder / 2.,
        bounds[2] + width_reminder / 2.,
        bounds[3] + height_reminder / 2.,
    ]
    return gbounds


def calculate_rounded_gbounds(bounds, res):
    gbounds = [
        np.floor(bounds[0] / res) * res,
        np.floor(bounds[1] / res) * res,
        np.ceil(bounds[2] / res) * res,
        np.ceil(bounds[3] / res) * res,
    ]
    return gbounds


def parse_projection(p):
    """Initialize a rasterio CRS projection
    """
    # if isinstance(p, pyproj.Proj):
    #     return p
    if isinstance(p, CRS):
        return p
    elif isinstance(p, int):
        return CRS.from_epsg(p)
    elif isinstance(p, str):
        return CRS.from_string(p)
    elif isinstance(p, dict):
        return CRS.from_dict(**p)
    elif isinstance(p, pyproj.CRS):
            return rasterio.crs.CRS.from_wkt(p.to_wkt())


def transform(g, from_srs, to_srs):
    project = partial(
        pyproj.transform,
        pyproj.Proj(parse_projection(from_srs)),
        pyproj.Proj(parse_projection(to_srs))
    )

    return ops.transform(project, g)


def read_color_table(color_file, cmap_name='newcmap'):
    df = pd.read_table(color_file, sep='\s+',
                       header=None, names=['value', 'r', 'g', 'b'],
                       comment='#')
    value_norm = (df.value - df.value.min()) / (df.value.max() - df.value.min())
    df.loc[:, 'value'] = value_norm

    levels_colors = list(zip(df.value, list(zip(df.r / 255, df.g / 255, df.b / 255))))
    print(levels_colors)
    return LinearSegmentedColormap.from_list(cmap_name,
                                             levels_colors,
                                             gamma=1.0)
