import pytest
import os
import pyproj
from rasterio.crs import CRS
import rectifiedgrid as rg
from rectifiedgrid.demo import get_demo_data
from gisdata import GOOD_DATA
import pandas as pd
from shapely import geometry
import numpy as np
from scipy import ndimage
from rasterio.warp import reproject

try:
    from rasterio.enums import Resampling
except ImportError:
    # rasterio versions 0.xx
    from rasterio.warp import RESAMPLING as Resampling

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class TestInitialization(object):
    def test_initproj(self):
        values = [4326,
                  'epsg:4326',
                  {'init': 'epsg:4326'},
                  ]

        for v in values:
            p = rg.parse_projection(v)
            assert isinstance(p, CRS)

    def test_read_vector(self):
        vector = os.path.join(GOOD_DATA, 'vector',
                              'san_andres_y_providencia_water.shp')
        grid = rg.read_vector(vector, 0.025, value=1)
        assert grid.shape == (37, 16)
        assert grid[0, 14] == 1.
        assert abs(grid.rio.resolution()[0]) == 0.025
        assert abs(grid.rio.resolution()[1]) == 0.025
        bounds = [-81.740389, 12.47477575,
                  -81.340389, 13.39977575]
        assert all([round(x - y, 6) == 0 for x, y in zip(grid.rio.bounds(), bounds)])

    # TODO: compute_area
    # def test_read_vector_area(self):
    #     vector = os.path.join(GOOD_DATA, 'vector',
    #                           'san_andres_y_providencia_water.shp')
    #     grid = rg.read_vector(vector, 0.025, value=1, compute_area=True)
    #     assert round(grid[0, 14], 3) == 0.156

    def test_read_raster(self):
        raster = os.path.join(GOOD_DATA, 'raster',
                              'test_grid.tif')
        grid = rg.read_raster(raster)
        assert grid.shape == (7, 5)

    def test_rounded_bounds(self):
        vector = os.path.join(os.path.dirname(__file__), 'air.shp')
        grid = rg.read_vector(vector, res=10000, rounded_bounds=True,
                              epsg=3035)
        assert grid.rio.bounds() == (4490000, 1520000, 5490000, 2530000)

    def test_feature(self):
        grid = get_demo_data('line3035')
        assert grid.shape == (12, 10)
        assert grid.values.sum() == 21

    def test_masked_value(self):
        grid = get_demo_data()
        grid = grid.where(grid!=0)
        assert np.round(grid.mean().values, 2) == 1.36

    def test_reproject(self):
        grid4326 = get_demo_data('line4326')
        grid3035 = get_demo_data('line3035')
        grid3035_4326 = grid3035.rio.reproject('epsg:4326', resampling=Resampling.nearest)
        assert grid4326.rio.crs == grid3035_4326.rio.crs
        assert (grid3035_4326.max().values, grid3035_4326.min().values) == (1., 0.)
        assert round(grid3035_4326.mean().values * 100) == 17

    # def test_patch(self):
    #     grid1 = get_demo_data('rg9x9')
    #     grid2 = grid1.copy()
    #     grid2[:] *= 2
    #     grid2[:2, :2] = 3
    #     grid1.patch(grid2)
    #     assert grid1.sum() == 13.
    #     grid1.patch_max(grid2)
    #     assert grid1.sum() == 14.

    def test_rescale(self):
        grid = get_demo_data('rg9x9').rg.rescale()
        assert (grid.rg.max() == 1)

    # def test_unsharedmask(self):
    #     grid1 = get_demo_data('rg9x9')
    #     assert grid1.sharedmask is False

    # TODO: check nodata for raster file
    # def test_fix_fill_value(self):
    #     file_path = os.path.join(TESTDATA_DIR, 'wrong_fill_value.tiff')
    #     grid = rg.read_raster(str(file_path))
    #     assert (grid.rio.nodata == 32767)
