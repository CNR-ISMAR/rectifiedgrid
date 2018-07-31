from __future__ import absolute_import

import logging
import copy
import numbers
import numpy as np
from geopandas import GeoDataFrame
from .utils import calculate_gbounds, calculate_eea_gbounds, parse_projection, transform
from affine import Affine
from rasterio.features import rasterize
import rasterio
from rasterio.warp import reproject
try:
    from rasterio.warp import RESAMPLING as Resampling
except:
    from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform

from shapely.geometry import box, Point
from shapely import ops
from rtree.index import Index as RTreeIndex
from scipy import ndimage
from scipy import interpolate
from itertools import izip
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import LogFormatter

BASEMAP = False

try:
    from mpl_toolkits import basemap
    BASEMAP = True
except:
    pass

logger = logging.getLogger(__name__)


def read_vector(vector, res, column=None, value=1., compute_area=False,
                dtype=np.float64, eea=False, epsg=None,
                bounds=None, grid=None, all_touched=True, fillvalue=0.,
                use_centroid=False):
    logger.debug('Reading vector as geodataframe')
    gdf = GeoDataFrame.from_file(vector)
    if use_centroid:
        gdf.geometry = gdf.geometry.centroid
    return read_df(gdf, res, column, value, compute_area,
                   dtype, eea, epsg, bounds, grid,
                   all_touched=all_touched, fillvalue=fillvalue)


def read_df(gdf, res, column=None, value=1., compute_area=False,
            dtype=np.float64, eea=False, epsg=None, bounds=None,
            grid=None, all_touched=True, fillvalue=0.):
    if epsg is not None:
        gdf.to_crs(epsg=epsg, inplace=True)
        proj = parse_projection(epsg)
    else:
        proj = parse_projection(gdf.crs)

    if grid is None:
        if bounds is None:
            bounds = gdf.total_bounds
        grid = _geofactory(bounds, proj, res, dtype, eea)
    else:
        grid = grid.copy()

    return read_df_like(grid, gdf, column, value, compute_area, copy=False,
                        all_touched=all_touched, fillvalue=fillvalue)


def read_df_like(rgrid, gdf, column=None, value=1., compute_area=False,
                 copy=True, all_touched=True, fillvalue=0.):
    """
    """
    if column is not None:
        gdf = gdf.rename(columns={column: '__rvalue__'})
    else:
        gdf['__rvalue__'] = value

    gdf.__rvalue__ = gdf.__rvalue__.fillna(fillvalue)
    gdf.to_crs(crs=rgrid.crs, inplace=True)

    features = list(gdf[['geometry', '__rvalue__']].itertuples(index=False,
                                                               name=None))

    return read_features_like(rgrid, features, compute_area=compute_area,
                              copy=copy, all_touched=all_touched)


def read_features(features, res, crs, bounds=None, compute_area=False,
                  dtype=np.float64, eea=False, all_touched=True):
    proj = parse_projection(crs)
    # guess bounds
    if bounds is None:
        if hasattr(features, 'bounds'):
            bounds = features.bounds
        else:
            b = np.array([feature[0].bounds for feature in features])
            bounds = np.min(b[:,0]), np.min(b[:,1]), np.max(b[:,2]), np.max(b[:,3])
    rgrid = _geofactory(bounds, proj, res, dtype, eea)
    return read_features_like(rgrid, features, compute_area, copy=False,
                              all_touched=all_touched)


def read_features_like(rgrid, features, compute_area=False, copy=True, all_touched=True):
    if copy:
        raster = rgrid.copy()
    else:
        raster = rgrid
    raster[:] = 0.
    if compute_area:
        raster.rasterize_features_area(features)
    else:
        raster.rasterize_features(features, all_touched=all_touched)
    return raster


def read_raster(raster, masked=True):
    src = rasterio.open(raster)
    if src.count > 1:
        src.close()
        raise NotImplementedError('Cannot load a multiband layer')
    if src.crs.is_valid:
        proj = parse_projection(src.crs)
    else:
        proj = None
    if masked:
        _raster = src.read(1, masked=masked)
        # return _raster
        rgrid = RectifiedGrid(_raster,
                              proj,
                              src.affine,
                              mask=_raster.mask)
    else:
        rgrid = RectifiedGrid(src.read(1),
                              proj,
                              src.affine,
                              mask=np.ma.nomask)
    src.close()
    # check and fix fill_value dtype
    if not np.can_cast(rgrid.fill_value, rgrid.dtype, casting='safe'):
        fill_value = guess_fill_value(rgrid)
        rgrid.set_fill_value(fill_value)
        logger.warning("read_raster: the fill_value has been changed to {}".format(fill_value))

    return rgrid


def guess_fill_value(obj):
    if issubclass(obj.dtype.type, numbers.Integral):
        fill_value = np.iinfo(obj.dtype.type).max
    elif issubclass(obj.dtype.type, numbers.Real):
        fill_value = np.finfo(obj.dtype.type).max
    if fill_value in obj.data:
        raise Exception('Cannot guess the fill_value. The value "{}" is present in the array'.format(fill_value))
    return fill_value


def _geofactory(bounds, proj, res, dtype=np.float64, eea=False):
    if eea:
        gbounds = calculate_eea_gbounds(bounds, res)
    else:
        gbounds = calculate_gbounds(bounds, res)

    cols = int(round((gbounds[2] - gbounds[0]) / res))
    rows = int(round((gbounds[3] - gbounds[1]) / res))
    _gtransform = (gbounds[0], res, 0.0, gbounds[3], 0.0, -res)
    gtransform = Affine.from_gdal(*_gtransform)
    # we use copy=True in order to avoid sharedmask=True
    return RectifiedGrid(np.zeros((rows, cols), dtype),
                         proj,
                         gtransform)


class SubRectifiedGrid(np.ndarray):
    """Defines a base np.ndarray subclass, that stores rectified grid metadata.
    """
    def __new__(cls, data, proj, gtransform, dtype=None, order=None):
        obj = np.asanyarray(data, dtype, order).view(cls)
        obj.proj = copy.deepcopy(parse_projection(proj))
        obj.gtransform = copy.deepcopy(gtransform)
        return obj

    def __array_finalize__(self, obj):
        if callable(getattr(super(SubRectifiedGrid, self),
                            '__array_finalize__', None)):
            super(SubRectifiedGrid, self).__array_finalize__(obj)

        self.proj = copy.deepcopy(getattr(obj, 'proj', None))
        self.gtransform = copy.deepcopy(getattr(obj, 'gtransform', None))
        # self.proj = getattr(obj, 'proj', None)
        # self.gtransform = getattr(obj, 'gtransform', None)
        return

    def copy(self, *args, **kwargs):
        obj = super(SubRectifiedGrid, self).copy(*args, **kwargs)
        obj.proj = copy.deepcopy(getattr(self, 'proj', None))
        obj.gtransform = copy.deepcopy(getattr(self, 'gtransform', None))
        return obj

    def __getitem__(self, *args, **kwargs):
        rslice = None
        cslice = None
        rstart = 0
        cstart = 0
        if isinstance(args[0], slice):
            rslice = args[0]
            cslice = slice(None, None, None)
        if isinstance(args[0], tuple) and len(args[0]) == 2:
            if isinstance(args[0], tuple) and isinstance(args[0][0], slice) and isinstance(args[0][1], slice):
                rslice, cslice = args[0]
            if isinstance(args[0], tuple) and isinstance(args[0][0], np.ndarray) and isinstance(args[0][1], np.ndarray):
                rstart = args[0][0].min()
                cstart = args[0][1].min()
        obj = super(SubRectifiedGrid, self).__getitem__(*args, **kwargs)
        if rslice is not None and rslice.start is not None:
            rstart = rslice.start
        if cslice is not None and cslice.start is not None:
            cstart = cslice.start
        if rstart > 0 or cstart > 0:
            g = obj.gtransform
            xmax, ymax = g * [cstart, rstart]
            obj.gtransform = Affine(g.a, g.b, xmax, g.d, g.e, ymax)
            # self.gtransform = Affine(g.a, g.b, xmax, g.d, g.e, ymax)
        return obj

    # def __add__(self, other):
    #     result = super(SubRectifiedGrid, self).__add__(other)
    #     result.info['added'] = result.info.get('added', 0) + 1
    #     return result

    # def __iadd__(self, other):
    #     result = super(SubRectifiedGrid, self).__iadd__(other)
    #     result.info['iadded'] = result.info.get('iadded', 0) + 1
    #     return result


class RectifiedGrid(SubRectifiedGrid, np.ma.core.MaskedArray):
    # we use copy=True in order to avoid sharedmask=True
    def __new__(cls, data, proj, gtransform, mask=np.ma.nomask, copy=True, **kwargs):
        subarr = SubRectifiedGrid(data, proj, gtransform)
        _data = np.ma.core.MaskedArray.__new__(cls, data=subarr,
                                               mask=mask, copy=copy, **kwargs)
        _data.proj = subarr.proj
        _data.gtransform = subarr.gtransform
        return _data

    @property
    def resolution(self):
        """Referenced grid resolution"""
        return self.gtransform.a

    @property
    def cellarea(self):
        """Area of a grid cell"""
        return self.resolution * self.resolution

    def rasterize_features(self, features, mode='replace', all_touched=True):
        """
        """
        _array = rasterize(features,
                            fill=0,
                            transform=self.gtransform,
                            out_shape=self.shape,
                            all_touched=all_touched)
        if mode == 'replace':
            self[:] = _array
        elif mode == 'patch':
            self.patch(_array)
        elif mode == 'patch_max':
            self.patch_max(_array)

    def add(self, array):
        np.add(self, array, self)

    def patch(self, array):
        condition = (self == 0) & (array != 0)
        self[condition] = array[condition]

    def patch_max(self, array):
        np.maximum(self, array, self)

    def rasterize_features_area(self, features):
        self.rasterize_features(features, all_touched=True)
        boundary = self - ndimage.binary_erosion(self)
        if len(features) != 1:
            # create a spatialindex
            stream = ((i, geo.bounds, value) for i, (geo, value) in
                      enumerate(features))
            sindex = RTreeIndex(stream)
            _intersection = sindex.intersection

        # https://github.com/Toblerity/rtree/issues/48
        for c in izip(*np.where(boundary > 0)):
            c_poly = self.cell_as_polygon(*c)
            _u = None

            if len(features) == 1:
                _u = c_poly.intersection(features[0][0])
            else:
                for hit in _intersection(c_poly.bounds):
                    if _u is None:
                        _u = c_poly.intersection(features[hit][0])
                    else:
                        _u = ops.unary_union([_u, c_poly.intersection(features[hit][0])])
            self[c] = _u.area / self.cellarea

    def cell_as_polygon(self, row, col):
        # TODO: check for valid row and col
        minx, maxy = self.gtransform * (col, row)
        maxx, miny = self.gtransform * (col+1, row+1)
        return box(minx, miny, maxx, maxy)

    @property
    def bounds(self):
        """Grid bounds.
        """
        ll = self.gtransform * (0, self.shape[0])
        ur = self.gtransform * (self.shape[1], 0)
        return ll + ur

    @property
    def geobounds(self):
        """Grid bounds in longitude - latitude (espg:4326).
        """
        gbounds_bbox = box(*self.bounds)
        return transform(gbounds_bbox, self.proj, "epsg:4326").bounds

    @property
    def geollur(self):
        ll = self.gtransform * (0, self.shape[0])
        ur = self.gtransform * (self.shape[1], 0)

        p_ll = Point(*ll)
        p_ur = Point(*ur)
        ll = transform(p_ll, self.proj, parse_projection("epsg:4326"))
        ur = transform(p_ur, self.proj, parse_projection("epsg:4326"))
        return ll.x, ll.y, ur.x, ur.y

    @property
    def crs(self):
        crs = {}
        if not self.proj:
            return crs
        for item in self.proj.srs.split():
            k, v = item.split('=')
            try:
                v = int(v)
            except ValueError:
                pass
            if v == 'True':
                v = True
            elif v == 'False':
                v = False
            crs[k.replace('+', '')] = v
        return crs

    def write_raster(self, filepath, dtype=None, driver='GTiff', nodata=None, compress=None):
        """Write a raster file
        """
        count = 1

        if dtype is None:
            dtype = self.dtype.name
        if dtype == 'float64':
            dtype = 'float32'

        profile = {
            'count': count,
            'crs': self.crs,
            'driver': driver,
            'dtype': dtype,
            #'nodata': 0,
            #'tiled': False,
            'transform': self.gtransform,
            'width': self.shape[1],
            'height': self.shape[0],
        }

        if compress is not None:
            profile['compress'] = compress

        if nodata is not None:
            profile['nodata'] = nodata
            with rasterio.open(filepath, 'w', **profile) as dst:
                d = self.data.copy()
                d[self.mask] = nodata

                dst.write_band(1, d.astype(dtype))
            return True

        # with rasterio.drivers():
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(filepath, 'w', **profile) as dst:
                dst.write_band(1, self.astype(dtype))
                if self.mask.any():
                    dst.write_mask(255 * (~self.mask).astype('uint8'))
                dst.close()
        return True

        # with rasterio.open(filepath, 'w', **profile) as dst:
        #     dst.write_band(1, self.astype(dtype))
        #     if self.mask.any():
        #         dst.write_mask(255 * (~self.mask).astype('uint8'))
        #     dst.close()

    def masked_mask(self, mask, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        raster[mask] = np.ma.masked
        return raster

    def masked_equal(self, value, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        raster[:] = np.ma.masked_equal(raster, value, copy=True)
        return raster

    def masked_not_equal(self, value, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        raster[:] = np.ma.masked_not_equal(raster, value, copy=True)
        # raster[raster != 3.] = np.ma.masked
        return raster

    def masked_values(self, value, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        raster[:] = np.ma.masked_values(raster, value)
        return raster

    def masked_greater_equal(self, value, copy=False, **kwargs):
        raster = self
        if copy:
            raster = self.copy()
        raster[:] = np.ma.masked_greater_equal(raster, value, **kwargs)
        return raster

    def masked_less_equal(self, value, copy=False, **kwargs):
        raster = self
        if copy:
            raster = self.copy()
        raster[:] = np.ma.masked_less_equal(raster, value, **kwargs)
        return raster

    def masked_less(self, value, copy=False, **kwargs):
        raster = self
        if copy:
            raster = self.copy()
        raster[:] = np.ma.masked_less(raster, value, **kwargs)
        return raster

    def threshold_binary(self, threshold=0, equal=False, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        if equal:
            raster[:] = raster >= threshold
        else:
            raster[:] = raster > threshold
        return raster

    def positive(self, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        raster[raster < 0] = 0
        return raster

    def lognorm(self, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        raster.log()
        raster.norm()
        return raster

    def replace_value(self, oldvalue, value, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        raster[raster == oldvalue] = value
        return raster

    def log(self, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        raster[:] = np.log(raster + 1)[:]
        return raster

    def norm(self, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        max = raster.max()
        if max != 0:
            raster[:] = (raster / max)[:]
        return raster

    def gaussian_conv(self, geosigma, mode="constant", copy=False,
                      **kwargs):
        return self.gaussian_filter(geosigma/self.resolution,
                                    mode=mode, copy=copy,
                                    **kwargs)

    def gaussian_filter(self, sigma, mode="constant", copy=False,
                        **kwargs):
        raster = self
        if copy:
            raster = self.copy()
        if sigma > 0:
            # filled data with 0 to avoid influence from masked values
            raster.data[:] = ndimage.gaussian_filter(raster.filled(0),
                                                     sigma, mode=mode,
                                                     **kwargs)
        return raster

    def fill_underlying_data(self, fill_value=None, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        raster.data[:] = self.filled(fill_value)
        return raster

    def unmask(self, fill_value=None, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        raster.data[raster.mask] = fill_value
        raster.mask = False
        return raster

    def to_srs_like(self, rgrid, src_nodata=None, dst_nodata=None,
                    resampling=Resampling.bilinear):
        if src_nodata is None:
            src_nodata = self.fill_value
        if dst_nodata is None:
            dst_nodata = src_nodata
        source = self.copy()
        source.fill_underlying_data(src_nodata)

        # dst_shape = rgrid.shape
        dst_transform = rgrid.gtransform
        dst_crs = rgrid.crs
        destination = rgrid.astype(self.dtype).copy()
        reproject(
            source,
            destination=destination,
            src_transform=self.gtransform,
            src_nodata=src_nodata,
            src_crs=self.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            dst_nodata=dst_nodata)

        destination.masked_equal(dst_nodata)
        return destination

    def to_srs(self, srs, resolution=None, src_nodata=None, dst_nodata=None,
               resampling=Resampling.nearest):
        affine, width, height = calculate_default_transform(self.crs,
                                                            srs,
                                                            self.shape[1],
                                                            self.shape[0],
                                                            *self.bounds,
                                                            resolution=resolution)
        if dst_nodata is None:
            dst_nodata = self.fill_value

        destination = RectifiedGrid(np.zeros([height, width], self.dtype),
                                    srs,
                                    affine,
                                    fill_value=dst_nodata)
        return self.to_srs_like(destination, src_nodata,
                                dst_nodata, resampling)

    # TODO deal nodata
    def reproject(self, input_raster, resampling=Resampling.bilinear, copy=False):
        """Reproject the input_raster using the current grid projection,
        resolution and extension
        """
        raster = self
        if copy:
            raster = self.copy()
        dst_shape = self.shape
        dst_transform = self.gtransform
        dst_crs = self.crs
        destination = np.zeros(dst_shape, input_raster.dtype)
        reproject(
            input_raster,
            destination,
            src_transform=input_raster.gtransform,
            src_crs=input_raster.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            dst_nodata=0.)
        raster[:] = destination[:]
        return raster

    def zoom(self, zoom, resampling=Resampling.bilinear):
        res = self.resolution / zoom
        rgrid = _geofactory(self.bounds, self.proj, res)
        return rgrid.reproject(self)

    def plot(self, cmap='Greys'):
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        mapimg = plt.imshow(self, cmap=cmap)
        plt.colorbar(mapimg, orientation='vertical')

    def get_basemap(self, ax=None):
        minx, miny, maxx, maxy = self.geollur
        epsg = self.crs['init'].split(':')[1]
        return basemap.Basemap(
            llcrnrlon=minx, llcrnrlat=miny, urcrnrlon=maxx, urcrnrlat=maxy,
            resolution='h',
            epsg=epsg, ax=ax)

    def plotmap(self, legend=False, arcgis=False, coast=False, countries=False,
                rivers=False, grid=False, gridrange=2, bluemarble=False, etopo=False,
                maptype=None, cmap=None, norm=None, logcolor=False, vmin=None,
                vmax=None, ax=None, basemap=None, ticks=None, minor_thresholds=None,
                arcgisxpixels=1000):
        if not BASEMAP:
            raise ImportError("Cannot load mpl_toolkits module")
        if maptype == 'minimal':
            coast = True,
            countries = True
        elif maptype == 'full':
            coast = True,
            countries = True
            rivers = True
            arcgis = True
            grid = True

        if basemap is None:
            m = self.get_basemap(ax=ax)
        else:
            m = basemap

        if cmap is not None and isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        if logcolor:
            norm = colors.SymLogNorm(linthresh=5, linscale=1,
                                     vmin=self.min(), vmax=self.max())

        if bluemarble:
            m.bluemarble()

        if etopo:
            m.etopo()

        if arcgis:
            m.arcgisimage(service='ESRI_Imagery_World_2D',
                          xpixels=arcgisxpixels, verbose= True)

        mapimg = m.imshow(np.flipud(self), cmap=cmap, norm=norm,
                          vmin=vmin, vmax=vmax)

        if coast:
            m.drawcoastlines()
        if countries:
            m.drawcountries()
        if rivers:
            m.drawrivers(linewidth=0.2, linestyle='solid', color='b')
        if grid:
            m.drawparallels(np.arange(-90,90,gridrange),labels=[1,0,0,0],fontsize=10)
            m.drawmeridians(np.arange(-90,90,gridrange),labels=[0,0,0,1],fontsize=10)
        if legend:
            if logcolor:
                formatter = LogFormatter(10, labelOnlyBase=False, minor_thresholds=minor_thresholds)
                cbar = plt.colorbar(mapimg, orientation='vertical', ax=ax, ticks=ticks, format=formatter)
            else:
                cbar = plt.colorbar(mapimg, orientation='vertical', ax=ax, ticks=ticks)

        return m, mapimg

    def griddata(self, x, y, z, method='nearest', copy=False):
        raster = self
        if copy:
            raster = self.copy()
        xi = np.arange(0.5, self.shape[1], 1.)
        yi = np.arange(0.5, self.shape[0], 1.)
        raster[:] = interpolate.griddata((x, y), z,
                                         (xi[None, :], yi[:, None]),
                                         method=method)
        raster[np.isnan(raster)] = np.ma.masked
        return raster

    def wrap_func(self, f, *args, **kwargs):
        # TODO: deal with type (the f could change the type)
        # TODO
        raster = self.copy()
        raster[:] = f(raster, *args, **kwargs)
        raster.mask = self.mask.copy()
        return raster

    def crop(self, value=None):
        if value is None:
            m = ~self.mask
        else:
            m = self != value
        return self[np.ix_(m.any(1),
                           m.any(0))]
