from rectifiedgrid.demo import get_demo_data
import numpy as np


class TestAlgebra(object):

    def test_positive(self):
        grid = get_demo_data()
        grid.positive()
        assert (grid.min(), grid.max()) == (0., 4.)

    def test_gaussian_filter(self):
        grid = get_demo_data('rg9x9')
        grid.gaussian_filter(2.)
        assert np.round(grid.sum(), 2) == 0.95
