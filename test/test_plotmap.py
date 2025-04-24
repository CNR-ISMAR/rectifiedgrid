import matplotlib.pyplot as plt
import rectifiedgrid as rg
import contextily as ctx
import pytest


raster = rg.read_raster('test/data/ndvi.tiff')
raster_ci = rg.read_raster('test/data/ci.tiff')

class TestPlotmap(object):
    @pytest.mark.mpl_image_compare
    def test_simpleplot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        raster.rg.plotmap(ax=ax)
        return fig

    @pytest.mark.mpl_image_compare
    def test_vmin_vmax(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        raster.rg.plotmap(ax=ax, vmin=0, vmax=1)
        return fig

    @pytest.mark.mpl_image_compare
    def test_legend(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        raster.rg.plotmap(ax=ax, vmin=0, vmax=1, legend=True)
        return fig

    @pytest.mark.mpl_image_compare
    def test_logscale(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        rescale = 1000
        (raster * rescale).rg.plotmap(ax=ax, vmin=0, vmax=rescale, logcolor=True, legend=True)
        return fig

    @pytest.mark.mpl_image_compare
    def test_basemap_provider(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        raster_ci.where(raster_ci>0).rg.plotmap(ax=ax,
                             legend=True,
                             basemap_provider=ctx.providers.OpenStreetMap.Mapnik,
                             )
        return fig

    @pytest.mark.mpl_image_compare
    def test_extent_buffer(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        raster.rg.plotmap(ax=ax,
                          vmin=0, vmax=1,
                          extent_buffer=5000,
                          basemap_provider=ctx.providers.OpenStreetMap.Mapnik,
                          )
        return fig

    @pytest.mark.mpl_image_compare
    def test_grid(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        raster.rg.plotmap(ax=ax,
                          vmin=0, vmax=1,
                          extent_buffer=5000,
                          basemap_provider=ctx.providers.OpenStreetMap.Mapnik,
                          grid=True
                          )
        return fig


    @pytest.mark.mpl_image_compare
    def test_alpha(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        raster.rg.plotmap(ax=ax,
                          vmin=0, vmax=1,
                          extent_buffer=5000,
                          basemap_provider=ctx.providers.OpenStreetMap.Mapnik,
                          grid=True,
                          alpha=0.5
                          )
        return fig
