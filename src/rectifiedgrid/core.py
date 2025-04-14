import logging
# import copy as copyp
# import numbers
import rioxarray
from rioxarray.rioxarray import affine_to_coords
import xarray
import numpy as np
import geopandas as gpd
from .utils import calculate_gbounds, calculate_rounded_gbounds, parse_projection, transform
from .hillshade import get_hs
from affine import Affine
from rasterio.features import rasterize
import rasterio
from rasterio.warp import reproject
import cartopy
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt
import pyproj

from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform

from shapely.geometry import box, Point
from shapely import ops
from rtree.index import Index as RTreeIndex
from scipy import ndimage
from scipy import interpolate

from matplotlib.colors import Normalize, SymLogNorm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import LogFormatter
import mapclassify
from rasterio.enums import MergeAlg


logger = logging.getLogger(__name__)


# TODO: check mandatory options (eg. res or grid have to be not None
def read_vector(vector, res=None, column=None, value=1., compute_area=False,
                dtype=np.float64, rounded_bounds=False, epsg=None,
                bounds=None,
                grid=None, grid_mask=True,
                all_touched=True, merge_alg=rasterio.enums.MergeAlg.replace, fillvalue=0., nodata=np.nan,
                use_centroid=False, query=None):
    """
    Constructor to create a geospatial DataArray object from a vector file with input geometry burned in.

    Parameters
    ----------
    vector: :str
        File path or file handle to read from. See Geopandas.from_file for usage details.
    res: float
        Resolution of the output object expressed in the destination coordinate reference system.
    column: str, optional
        Attribute field of the input file from which the values should be chosen. If column=None, the 'value' parameter
        is used. Only numeric column are supported.
    compute_area: bool, optional
        Ignored, Deprecated
    dtype: rasterio or numpy data type, optional
        Used as data type for results.
    rounded_bounds: bool, optional
        If True, bounds of the result xarray object are adjusted to result in round numbers.
    epsg: int, optional
        EPSG code specifying output projection. If None, projection of the input file will be used.
    bounds: list or tuple
        minx, miny, maxx, maxy values containing the bounds of the result array. Bound will be slightly adjusted to meet
        the resolution.
    grid: xarray, optional
        The shape and the geopsatial attributes of grid define the same attributes of the returned xarray.
    grid_mask: bool, optional
        Ignore, Deprecated
    all_touched: bool, optional
        If True, all pixels touched by geometries will be burned in.
        See rasterio.features.rasterize for usage details.
    merge_alg: MergeAlg, optional
        Merge algorithm to use. One of:
            MergeAlg.replace (default):
                the new value will overwrite the existing value.
            MergeAlg.add:
                the new value will be added to the existing raster.
        See rasterio.features.rasterize for usage details.
    fillvalue: int or float, optional
        Used as fill value for all areas not covered by input geometries.
        See rasterio.features.rasterize for usage details.
    nodata: float, optional
        Nodata value for the DataArray.
        See rioxarray.write_nodata for usage details.
    use_centroid: bool, optional
        If True, centroid of polygonal geometries will be burned in.
    query: str, optional
        The query string to evaluate for filtering the data before burning.
        See pandas.DataFrame.query for usage details.

    Returns
    -------
    :obj:`xarray.DataArray`:
        DataArray with geospatial (CF) attributes
    """
    logger.debug('Reading vector as geodataframe')
    gdf = gpd.GeoDataFrame.from_file(vector)
    # remove invalid geometries
    gdf = gdf[~gdf.geometry.isna()].copy()
    if query is not None:
        gdf.query(query, inplace=True)
    if use_centroid:
        gdf.geometry = gdf.geometry.centroid
    return read_df(gdf, res, column, value, compute_area,
                   dtype, rounded_bounds, epsg, bounds,
                   grid=grid, grid_mask=grid_mask,
                   all_touched=all_touched, merge_alg=merge_alg, fillvalue=fillvalue,
                   nodata=nodata)

# TODO: merge read_df and read_df_link (as in read_vector)
def read_df(gdf, res=None, column=None, value=1., compute_area=False,
            dtype=np.float64, rounded_bounds=False, epsg=None, bounds=None,
            grid=None, grid_mask=True,
            all_touched=True, merge_alg=rasterio.enums.MergeAlg.replace, fillvalue=0.,
            nodata=np.nan):
    if epsg is not None:
        gdf.to_crs(epsg=epsg, inplace=True)
        crs = parse_projection(epsg)
    else:
        crs = parse_projection(gdf.crs)

    if grid is None:
        if bounds is None:
            bounds = gdf.total_bounds
        grid = make_array(bounds, crs, res, dtype=dtype, rounded_bounds=rounded_bounds, nodata=nodata)
    else:
        grid = grid.copy()
    return read_df_like(grid, gdf, column, value, compute_area, copy=False,
                        all_touched=all_touched, merge_alg=merge_alg, fillvalue=fillvalue,
                        grid_mask=grid_mask)


# TODO: check MergeAlg
def read_df_like(rgrid, gdf, column=None, value=1., compute_area=False,
                 copy=True,
                 all_touched=True, merge_alg=rasterio.enums.MergeAlg.replace, fillvalue=0.,
                 grid_mask=True):
    """
    """
    if column is not None:
        gdf = gdf.rename(columns={column: '__rvalue__'})
    else:
        gdf['__rvalue__'] = value

    gdf.__rvalue__ = gdf.__rvalue__.fillna(rgrid.rio.nodata)
    proj_crs = pyproj.CRS.from_user_input(rgrid.rio.crs)
    gdf.to_crs(crs=proj_crs, inplace=True)

    # TODO: using an iterators
    features = list(gdf[['geometry', '__rvalue__']].itertuples(index=False,
                                                               name=None))

    return read_features_like(rgrid, features, compute_area=compute_area,
                              copy=copy,
                              all_touched=all_touched, merge_alg=merge_alg, fillvalue=fillvalue,
                              grid_mask=grid_mask)


# TODO: merge read_features ad read_features_like as in read_vector
def read_features(features, res, crs, bounds=None, compute_area=False,
                  dtype=np.float64, rounded_bounds=False,
                  all_touched=True, merge_alg=rasterio.enums.MergeAlg.replace, fillvalue=0.,
                  nodata=np.nan,
                  grid_mask=True):
    crs = parse_projection(crs)
    # guess bounds
    if bounds is None:
        if hasattr(features, 'bounds'):
            bounds = features.bounds
        else:
            b = np.array([feature[0].bounds for feature in features])
            bounds = np.min(b[:, 0]), np.min(b[:, 1]), np.max(b[:, 2]), np.max(b[:, 3])
    rgrid = make_array(bounds, crs, res, dtype=dtype, rounded_bounds=rounded_bounds, nodata=nodata)
    return read_features_like(rgrid, features, compute_area, copy=False,
                              all_touched=all_touched, merge_alg=merge_alg, fillvalue=fillvalue,
                              grid_mask=grid_mask)


def read_features_like(da, features, compute_area=False, copy=True,
                       all_touched=True, merge_alg=rasterio.enums.MergeAlg.replace, fillvalue=0.,
                       grid_mask=0):
    if compute_area:
        # TODO: to be implemented
        # raster.rasterize_features_area(features)
        pass
    else:
        # raster.rasterize_features(features, all_touched=all_touched, merge_alg=merge_alg)
        return rasterize_features(da, features,
                                  all_touched=all_touched, merge_alg=merge_alg, fillvalue=fillvalue,
                                  grid_mask=grid_mask)


def read_raster(raster, band=1, epsg=None, grid=None, resampling=Resampling.bilinear,
                fillvalue=0, **open_kwargs):
    # TODO: deal fill_value and nodata
    # deal encoded_nodata
    # manage on-fly reprojection and reproject_match
    rgrid = rioxarray.open_rasterio(raster, **open_kwargs)
    if rgrid.rio.nodata:
        rgrid = (rgrid.where(rgrid != rgrid.rio.nodata)
                 .rio.write_nodata(np.nan)
                 )
    if band:
        rgrid = rgrid.sel(band=band).copy()
    if grid is not None:
        rgrid = (rgrid
                 .rio.reproject_match(grid,
                                      nodata=fillvalue) # rio.reproject nodata is use to initlialize the destination array
                 .where(~grid.isnull())
                 )
    elif epsg is not None:
        rgrid = rgrid.rio.reproject("epsg:{}".format(epsg),
                                resampling=resampling,
                                nodata=fillvalue) # rio.reproject nodata is use to initlialize the destination array
    return rgrid

def rasterize_features(da, features, all_touched=True, merge_alg=rasterio.enums.MergeAlg.replace, fillvalue=0.,
                       grid_mask=True):
    #TODO: better manage dtype
    #TODO: features and da CRSs have to be the same
    r = rasterio.features.rasterize(
        features,
        out_shape=da.shape,
        transform=da.rio.transform(),
        fill=fillvalue,
        all_touched=all_touched,
        merge_alg=merge_alg,
        dtype = da.dtype
        # dtype=_minimize_dtype(data_values.dtype, fill),
    )
    x_dim = da.rio.x_dim
    y_dim = da.rio.y_dim
    coords = {y_dim: da.coords[y_dim].values, x_dim: da.coords[x_dim].values}
    da_r = (
        #TODO: manage nodata
        xarray.DataArray(r, coords=coords)
            .rio.write_nodata(da.rio.nodata)
            .rio.write_crs(da.rio.crs)
            .rio.write_transform(da.rio.transform())
            .rio.write_coordinate_system()
    )
    if grid_mask:
        return da_r.where(~da.isnull())
    return da_r

def make_array(bounds=None, crs=None, res=None, dtype=np.float64, rounded_bounds=False, nodata=np.nan):
    if rounded_bounds:
        gbounds = calculate_rounded_gbounds(bounds, res)
    else:
        gbounds = calculate_gbounds(bounds, res)

    cols = int(round((gbounds[2] - gbounds[0]) / res))
    rows = int(round((gbounds[3] - gbounds[1]) / res))
    _gtransform = (gbounds[0], res, 0.0, gbounds[3], 0.0, -res)
    gtransform = Affine.from_gdal(*_gtransform)
    r = np.zeros((rows, cols))

    coords = affine_to_coords(gtransform, cols, rows)
    da = (
        xarray.DataArray(r, coords=coords)
            .astype(dtype)
            .rio.write_nodata(nodata)
            .rio.write_crs(crs)
            .rio.write_transform(gtransform)
            .rio.write_coordinate_system()
    )
    return da

@xarray.register_dataarray_accessor("rg")
class RgAccessor:
    resolution_z = None

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def rasterize_features(self, features, mode='replace', all_touched=True,
                           merge_alg=rasterio.enums.MergeAlg.replace):
        # TODO: make value substitutio more robust
        raster = rasterize_features(self._obj, features)
        if mode == 'replace':
            self._obj[:] = raster
        elif mode == 'patch':
            self.patch(raster)
        elif mode == 'patch_max':
            self.patch_max(raster)
        return self._obj

    # def add(self, array):
    #     np.add(self, array, self)

    # TODO: to update
    def patch(self, raster):
        condition = (self._obj == 0) & (raster != 0)
        self._obj[condition] = raster[condition]

    # TODO: to update
    def patch_max(self, raster):
        np.maximum(self, raster, self)


    def test(self):
        return True

    def positive(self):
        raster = self._obj.copy()
        raster.values[raster.values < 0] = 0
        return raster

    def logrescale(self):
        return (self._obj.copy()
                .rg.log()
                .rg.rescale()
                )

    def replace_value(self, oldvalue, value):
        raster = self._obj.copy()
        raster.values[raster.values == oldvalue] = value
        return raster

    def log(self):
        raster = self._obj.copy()
        raster.values[:] = np.log(raster.values + 1)[:]
        return raster

    def rescale(self):
        raster = self._obj.copy()
        maxval = raster.max().values
        if maxval != 0:
            raster.values[:] = (raster.values / maxval)[:]
        return raster

    def min(self):
        return self._obj.min().values

    def max(self):
        return self._obj.max().values

    def gaussian_conv(self, geosigma, mode="constant",
                      **kwargs):
        # geosigma order: x, y[, z]
        _res = self._obj.rg.resolution2d3d()
        sigma = (np.array(geosigma) /
                 np.abs(np.array(_res))
                 )
        return self.gaussian_filter(sigma,
                                    mode=mode,
                                    **kwargs)

    def gaussian_filter(self, sigma, mode="constant", # copy=False,
                        **kwargs):
        # sigma order: x, y[, z]
        if isinstance(sigma, int) or isinstance(sigma, float) or len(sigma)==1:
            _sigma = sigma
        elif len(sigma)==2:
            _sigma = [sigma[1], sigma[0]]
        else:
            _sigma = [sigma[2], sigma[1], sigma[0]]
        print(_sigma)
        _kwargs = dict(kwargs, sigma=_sigma, mode=mode)
        rgauss = xarray.apply_ufunc(ndimage.gaussian_filter,
                                    # align dimension to sigma
                                    self._obj.fillna(0),
                                    # input_core_dims=[[]],
                                    # output_core_dims=[[]],
                                    kwargs=_kwargs
                                    )
        # TODO: make nodata check more robust: 
        return rgauss.where(~self._obj.isnull())

    def plotmap(self,
                legend=False,
                arcgis=False,
                coast=False,
                coast_resolution='50m',
                countries=False,
                rivers=False,
                grid=False,
                gridrange=2,
                bluemarble=False,
                etopo=False,
                maptype=None,
                cmap=None,
                norm=None,
                logcolor=False,
                vmin=None,
                vmax=None,
                ax=None,
                basemap=None,
                ticks=None,
                minor_thresholds=(np.inf, np.inf),
                arcgisxpixels=1000,
                zoomlevel=2,
                hillshade=False,
                scheme=None,
                ncolors=10,
                alpha=None,
                extent_buffer=0,
                imshow=True
                ):

        if ax is None:
            cprj = cartopy.crs.Mercator()
            ax = plt.gca(projection=cprj)
        elif not hasattr(ax, "projection"):
            raise AttributeError("Passed axes doesn't have projection attribute")
        r = self._obj.rio.reproject(ax.projection.proj4_params,
                                    resampling=Resampling.bilinear,
                                    nodata=np.nan)
        bounds = r.rio.bounds()
        res = r.rio.resolution()
        img_extent = [bounds[0],
                      bounds[2],
                      bounds[1],
                      bounds[3]]
        img_extent_buffer = [bounds[0] - extent_buffer,
                             bounds[2] + extent_buffer,
                             bounds[1] - extent_buffer,
                             bounds[3] + extent_buffer]

        if maptype == 'minimal':
            coast = True,
            countries = True
        elif maptype == 'full':
            coast = True,
            countries = True
            rivers = True
            arcgis = True
            grid = True

        if vmax is None:
            vmax = self._obj.max()
        if vmin is None:
            vmin = self._obj.min()

        # if basemap is None:
        #     m = self.get_basemap(ax=ax)
        # else:
        #     m = basemap

        if cmap is not None and isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        if norm is None:
            if logcolor:
                norm = SymLogNorm(linthresh=5, linscale=1,
                                  vmin=vmin, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)

        if scheme is not None:
            # TODO: add check options compatibility (es. schema override norm, log=True and schema doesn't work together)
            scheme = getattr(mapclassify, scheme)(self.values.flatten(), ncolors)
            bins = scheme.bins
            bounds = [scheme.yb.min()] + scheme.bins.tolist()
            cm = cmap
            scheme = cm(1. * np.arange(len(bins)) / len(bins))
            cmap = colors.ListedColormap(scheme)
            norm = colors.BoundaryNorm(bounds, len(bins))
            ticks = bounds

        # if bluemarble:
        #     m.bluemarble()

        if etopo:
            ax.add_image(cimgt.Stamen('terrain-background'), zoomlevel)

        # if arcgis:
        #     m.arcgisimage(service='ESRI_Imagery_World_2D',
        #                   xpixels=arcgisxpixels, verbose=True)

        # mapimg = m.imshow(np.flipud(self), cmap=cmap, norm=norm,
        #                 vmin=vmin, vmax=vmax)

        if hillshade:
            r = get_hs(r.values, cmap, norm=norm,
                       # blend_mode='soft'
                       )
        if imshow:
            mapimg = r.plot.imshow(ax=ax,
                                   cmap=cmap,
                                   norm=norm,
                                   zorder=1,
                                   alpha=alpha,
                                   add_colorbar=False,
                                   add_labels=False)
        else:
            mapimg = r.plot(ax=ax,
                            cmap=cmap,
                            norm=norm,
                            zorder=1,
                            alpha=alpha)

        # ax.add_feature(cpf.LAND)
        # ax.add_feature(cpf.OCEAN)
        #
        # ax.add_feature(cpf.BORDERS, linestyle=':')
        # ax.add_feature(cpf.LAKES,   alpha=0.5)
        #

        if countries:
            ax.add_feature(cpf.BORDERS, linestyle=':', zorder=2)

        if coast:
            # ax.add_feature(cpf.COASTLINE)
            ax.coastlines(resolution=coast_resolution, zorder=3)

        if rivers:
            ax.add_feature(cpf.RIVERS, zorder=4)
        if grid:
            # m.drawparallels(np.arange(-90, 90, gridrange), labels=[1, 0, 0, 0], fontsize=10)
            # m.drawmeridians(np.arange(-90, 90, gridrange), labels=[0, 0, 0, 1], fontsize=10)
            gl = ax.gridlines(draw_labels=True)
            gl.xlabels_top = gl.ylabels_right = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        if legend:
            if logcolor:
                formatter = LogFormatter(10,
                                         labelOnlyBase=False,
                                         minor_thresholds=minor_thresholds
                                         )
                plt.colorbar(mapimg, orientation='vertical', ax=ax, ticks=ticks, format=formatter)
            else:
                plt.colorbar(mapimg, orientation='vertical', ax=ax, ticks=ticks)

        ax.set_extent(img_extent_buffer, crs=ax.projection)
        return ax, mapimg

    def crop(self, value=np.nan):
        # TODO: improve use of isnull and novalue
        raster = self._obj.copy().fillna(value)
        rcrop = raster.where(raster != value, drop=True)
        minx, miny, maxx, maxy = rcrop.rio.bounds()
        return raster.rio.clip_box(minx=minx, maxx=maxx,
                                   miny=miny, maxy=maxy).where(~self._obj.isnull())

    def resolution_z(self):
        minz = float(self._obj['z'][0])
        maxz = float(self._obj['z'][-1])
        sizez = self._obj['z'].size
        return (maxz - minz)/(sizez - 1)

    def resolution2d3d(self):
        if 'z' in self._obj.dims:
            return (
                self._obj.rio.resolution()
                + (self.resolution_z(),)
            )
        else:
            return self._obj.rio.resolution()

    def to_3d(self, top_surface=None, bottom_surface=None,
              top=None, bottom=None, resolution_z=1, num=None,
              mode="full", top_layer=5, bottom_layer=5):
        if (top_surface is None and top is None):
            raise ValueError('Neither "top_surface" nor "top" have a valid value.')
        if (bottom_surface is None and bottom is None):
            raise ValueError('Neither "bottom_surface" nor "bottom" have a valid value.')
        if (resolution_z is None and num is None):
            raise ValueError('Neither "resolution_z" nor "num" have a valid value.')
        raster = self._obj

        if top is not None:
           zmax = float(top)
           top_surface = raster.where(raster.isnull(), top)
        else:
            top_surface = (top_surface
                           .rio.reproject_match(raster, nodata=np.nan)
                           .where(~raster.isnull())
                           )
            zmax = top_surface.max().values

        if bottom is not None:
            zmin = float(bottom)
            bottom_surface = raster.where(raster.isnull(), bottom)
        else:
            bottom_surface = (bottom_surface
                              .rio.reproject_match(raster, nodata=np.nan)
                              .where(~raster.isnull())
                              )
            zmin = float(bottom_surface.min())

        if resolution_z is not None:
            # align coordinates around zero
            _zmin = np.floor(zmin / resolution_z) * resolution_z
            _zmax = np.ceil(zmax / resolution_z) * resolution_z
            coords = np.arange(_zmin, _zmax, resolution_z) + resolution_z/2.
        else:
            coords, resolution_z = np.linspace(zmin, zmax, num+1, retstep=True)
            coords = coords[:-1] + resolution_z / 2

        top_surface_z = top_surface.expand_dims({'z': coords})
        bottom_surface_z = bottom_surface.expand_dims({'z': coords})
        r = (raster.expand_dims({'z': coords})
             .where((bottom_surface_z <= bottom_surface_z.z) & (top_surface_z >= top_surface_z.z))
             )
        if mode == 'full':
            pass
        elif mode == 'top':
            r = r.where((r.isnull()) | (r.z >= top_surface_z-top_layer), 0)
        elif mode == 'bottom':
            r = r.where((r.isnull()) | (r.z <= bottom_surface_z+bottom_layer), 0)
        elif mode == 'middle':
            r = (r.where((r.isnull())
                         | ((r.z < top_surface_z-top_layer)
                            & (r.z > bottom_surface_z+bottom_layer)
                            ),
                         0)
                 )

        z_coord_attrs = dict(r.coords['z'].attrs)
        z_coord_attrs["axis"] = "Z"
        z_coord_attrs["long_name"] = "height"
        z_coord_attrs["standard_name"] = "height_above_mean_sea_level"
        z_coord_attrs["units"] = "metre"
        z_coord_attrs["positive"] = "up"
        r.coords['z'].attrs = z_coord_attrs

        return r


#%%
