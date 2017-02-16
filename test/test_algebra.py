import os
import pyproj
import rectifiedgrid as rg
from rectifiedgrid.demo import get_demo_data
from gisdata import GOOD_DATA
from shapely import geometry
import numpy as np
from scipy import ndimage
from rasterio.warp import reproject


class TestAlgebra(object):

    def test_positive(self):
        grid = get_demo_data()
        grid.positive()
        assert (grid.min(), grid.max()) == (0., 4.)

    def test_gaussian_filter(self):
        grid = get_demo_data('rg9x9')
        grid.gaussian_filter(2.)
        assert round(grid.sum(), 2) == 0.95
