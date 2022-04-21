from rectifiedgrid.demo import get_demo_data


class TestCrop(object):
    def test_crop(self):
        grid = get_demo_data().rg.positive()
        cropped = grid.rg.crop(0)
        assert (grid.sum() == cropped.sum())

        cropped_bounds = (4540000.0, 1480000.0, 4580000.0, 1520000.0)
        assert (cropped.rio.bounds() == cropped_bounds)
