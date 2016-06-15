from __future__ import absolute_import

import numpy as np
import rectifiedgrid as rg
from affine import Affine
from shapely import geometry


epsg = 3035
gtransform = Affine.from_gdal(4500000.0, 10000.0, 0.0, 1560000.0, 0.0, -10000.0)

arr1 = np.zeros((9, 9), np.float64)
arr1[:3, :3] = np.array([[-1, -2, -3],
                         [0, -1, -2],
                         [0, 0, -1]])

arr1[4:8, 4:8] = np.array([[1, 2, 3, 4],
                           [1, 2, 3, 4],
                           [1, 2, 3, 4],
                           [1, 2, 3, 4]])

arr2 = np.zeros((9, 9), np.float64)
arr2[4, 4] = 1

arr3 = np.zeros((9, 9), np.float64)
arr3[0, 0] = 1

lstring = geometry.LineString(((12.0, 36.0), (13.0, 37.0)))


def get_demo_data(name='default'):
    """Generate a demo RasterizedGrid
    """
    raster = None
    if name == 'default':
        raster = rg.RectifiedGrid(arr1, rg.parse_projection(epsg), gtransform)
    elif name == 'rg9x9':
        raster = rg.RectifiedGrid(arr2, rg.parse_projection(epsg), gtransform)
    elif name == 'line3035':
        from_srs = 4326
        to_srs = 3035
        features = [(rg.transform(lstring, from_srs, to_srs), 1)]
        raster = rg.read_features(features, 10000, 3035, eea=True)
    elif name == 'line4326':
        features = [(lstring, 1)]
        raster = rg.read_features(features, 0.1, 4326)
    return raster.copy()
