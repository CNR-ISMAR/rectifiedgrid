import tempfile
import os
import pyproj
import rectifiedgrid as rg
from rectifiedgrid.demo import get_demo_data
from gisdata import GOOD_DATA
from shapely import geometry
import numpy as np
from scipy import ndimage


class TestCrop(object):
    def test_crop(self):
        grid = get_demo_data()
        grid.positive()
        cropped = grid.crop(0)
        assert (grid.sum() == cropped.sum())

        cropped_bounds = (4540000.0, 1480000.0, 4580000.0, 1520000.0)
        assert (cropped.bounds == cropped_bounds)
