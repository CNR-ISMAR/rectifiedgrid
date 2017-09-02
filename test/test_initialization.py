import pytest
import os
import pyproj
import rectifiedgrid as rg
from rectifiedgrid.demo import get_demo_data
from gisdata import GOOD_DATA
from shapely import geometry
import numpy as np
from scipy import ndimage
from rasterio.warp import reproject
try:
    from rasterio.enums import Resampling
except:
    # rasterio versions 0.xx
    from rasterio.warp import RESAMPLING as Resampling


class TestInitialization(object):
    def test_initproj(self):
        values = [4326,
                  'epsg:4326',
                  {'init': 'epsg:4326'},
                  pyproj.Proj(init='epsg:4326')]

        for v in values:
            p = rg.parse_projection(v)
            assert isinstance(p, pyproj.Proj)

    def test_subclassing(self):
        grid = rg.RectifiedGrid([1, 2, 3, 4], 4326, None)
        assert isinstance(grid, rg.RectifiedGrid)

        _grid = grid + 1
        assert isinstance(grid, rg.RectifiedGrid)
        assert _grid.proj.srs == grid.proj.srs

    def test_read_vector(self):
        vector = os.path.join(GOOD_DATA, 'vector',
                              'san_andres_y_providencia_water.shp')
        grid = rg.read_vector(vector, 0.025, value=1)
        assert grid.shape == (37, 16)
        assert grid[0, 14] == 1.
        assert grid.resolution == 0.025
        bounds = [-81.740389, 12.47477575,
                  -81.340389, 13.39977575]
        assert all([round(x - y, 6) == 0 for x, y in zip(grid.bounds, bounds)])

    def test_read_vector_area(self):
        vector = os.path.join(GOOD_DATA, 'vector',
                              'san_andres_y_providencia_water.shp')
        grid = rg.read_vector(vector, 0.025, value=1, compute_area=True)
        assert round(grid[0, 14], 3) == 0.156

    def test_read_raster(self):
        raster = os.path.join(GOOD_DATA, 'raster',
                              'test_grid.tif')
        grid = rg.read_raster(raster)
        assert grid.shape == (7, 5)

    def test_eea(self):
        vector = os.path.join(os.path.dirname(__file__), 'air.shp')
        grid = rg.read_vector(vector, res=10000, eea=True,
                              epsg=3035, compute_area=True)
        assert grid.bounds == (4490000, 1520000, 5490000, 2530000)

    def test_feature(self):
        grid = get_demo_data('line3035')
        assert grid.shape == (12, 10)
        assert grid.sum() == 21

    def test_masked_value(self):
        grid = get_demo_data()
        grid.masked_values(0.)
        assert round(grid.mean(), 2) == 1.36

    def test_reproject(self):
        grid4326 = get_demo_data('line4326')
        grid3035 = get_demo_data('line3035')
        rgrid = np.zeros_like(grid3035)
        assert rgrid.proj.srs == grid3035.proj.srs
        assert (rgrid.max(), rgrid.min()) == (0., 0.)
        rgrid.reproject(grid4326, Resampling.nearest)
        assert rgrid.max() == 1.
        # print "############", rgrid.mean()
        assert pytest.approx(float(rgrid.mean()), 0.001) == 0.15

    def test_patch(self):
        grid1 = get_demo_data('rg9x9')
        grid2 = grid1.copy()
        grid2[:] *= 2
        grid2[:2, :2] = 3
        grid1.patch(grid2)
        assert grid1.sum() == 13.
        grid1.patch_max(grid2)
        assert grid1.sum() == 14.

    def test_projection(self):
        grid1 = get_demo_data('rg9x9')
        grid1.norm()
        assert (1 == 1)

    def test_unsharedmask(self):
        grid1 = get_demo_data('rg9x9')
        assert (grid1.sharedmask == False)
