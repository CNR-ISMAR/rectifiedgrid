from rectifiedgrid.demo import get_demo_data

try:
    from rasterio.warp import RESAMPLING as Resampling
except ImportError:
    from rasterio.enums import Resampling


class TestProjection(object):

    def test_reproject(self):
        grid4326 = get_demo_data('line4326')
        grid3035 = get_demo_data('line3035')
        assert (grid3035.reproject(grid4326, Resampling.nearest) == grid3035).all()
