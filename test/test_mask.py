import tempfile

import numpy as np

import rectifiedgrid as rg
from rectifiedgrid.demo import get_demo_data


class TestMask(object):
    def test_read_write(self):
        grid = get_demo_data()
        grid[grid < 0] = np.ma.masked
        num, filepath = tempfile.mkstemp()
        grid.write_raster(filepath)
        _grid = rg.read_raster(filepath, masked=True)
        assert (grid == _grid).all()
        assert (grid.sum() == _grid.sum())
