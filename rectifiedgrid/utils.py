import pyproj
import math
from shapely import ops
from functools import partial
from rasterio.crs import CRS
from matplotlib.colors import LinearSegmentedColormap, from_levels_and_colors
from os.path import exists

EEA_GRID_RESOLUTIONS = [25, 100, 250, 500, 1000, 2500, 10000, 25000, 100000]


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
    if isinstance(p, CRS):
        return pyproj.Proj(p)
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


def read_color_table(color_file):
    '''
    The method for reading the color file.
    '''
    colors = []
    levels = []
    if exists(color_file) is False:
        raise Exception("Color file " + color_file + " does not exist")
    fp = open(color_file, "r")
    for line in fp:
        if line.find('#') == -1 and line.find('/') == -1:
            entry = line.split()
            levels.append(eval(entry[0]))
            colors.append((int(entry[1])/255.,int(entry[2])/255.,int(entry[3])/255.))
    fp.close()
    # cmap = LinearSegmentedColormap.from_list("my_colormap", colors, N=len(levels), gamma=1.0)
    # return levels, cmap
    return from_levels_and_colors(levels, colors, extend='min')
