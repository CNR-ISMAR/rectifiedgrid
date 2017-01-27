from __future__ import absolute_import

import logging
import copy
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
from shapely.geometry import box, Point
from shapely import ops
from rtree.index import Index as RTreeIndex
from scipy import ndimage
from itertools import izip
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits import basemap


logger = logging.getLogger(__name__)


def read_vector(vector, res, column=None, value=1., compute_area=False,
                dtype=np.float64, eea=False, epsg=None,
                bounds=None, grid=None):
    gdf = GeoDataFrame.from_file(vector)
    return read_df(gdf, res, column, value, compute_area,
                   dtype, eea, epsg, bounds, grid)


def read_df(gdf, res, column=None, value=1., compute_area=False,
            dtype=np.float64, eea=False, epsg=None, bounds=None,
            grid=None):
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

    return read_df_like(grid, gdf, column, value, compute_area, copy=False)


def read_df_like(rgrid, gdf, column=None, value=1., compute_area=False, copy=True):
    """
    quando e' presente sia column che value il value viene utilizzato per riempire gli nan
    """
    if column is not None:
        gdf = gdf.rename(columns={column: '__rvalue__'})
    else:
        gdf['__rvalue__'] = value

    gdf.__rvalue__ = gdf.__rvalue__.fillna(value)
    gdf.to_crs(crs=rgrid.crs, inplace=True)

    features = list(gdf[['geometry', '__rvalue__']].itertuples(index=False,
                                                               name=None))

    return read_features_like(rgrid, features, compute_area=compute_area, copy=copy)


def read_features(features, res, crs, bounds=None, compute_area=False,
                  dtype=np.float64, eea=False):
    proj = parse_projection(crs)
    # guess bounds
    if bounds is None:
        if hasattr(features, 'bounds'):
            bounds = features.bounds
        else:
            b = np.array([feature[0].bounds for feature in features])
            bounds = np.min(b[:,0]), np.min(b[:,1]), np.max(b[:,2]), np.max(b[:,3])
    rgrid = _geofactory(bounds, proj, res, dtype, eea)
    return read_features_like(rgrid, features, compute_area, copy=False)


def read_features_like(rgrid, features, compute_area=False, copy=True):
    if copy:
        raster = rgrid.copy()
    else:
        raster = rgrid
    raster[:] = 0.
    if compute_area:
        raster.rasterize_features_area(features)
    else:
        raster.rasterize_features(features)
    return raster


def read_raster(raster, masked=False):
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
    return rgrid


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
        return

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

    def rasterize_features(self, features, mode='replace'):
        """
        """
        _array = rasterize(features,
                            fill=0,
                            transform=self.gtransform,
                            out_shape=self.shape,
                            all_touched=True)
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
        self.rasterize_features(features)
        boundary = self - ndimage.binary_erosion(self)

        if len(features) != 1:
            # create a spatialindex
            print "create spatial index"
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

    def write_raster(self, filepath, dtype=None, driver='GTiff', nodata=None):
        """Write a raster file
        """
        count = 1

        if dtype is None:
            dtype = self.dtype

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

        if nodata is not None:
            profile['nodata'] = nodata
            with rasterio.open(filepath, 'w', **profile) as dst:
                d = self.data.copy()
                d[self.mask] = nodata

                dst.write_band(1, d.astype(rasterio.float64))
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


    def masked_equal(self, value, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        raster[:] = np.ma.masked_equal(raster, value)
        return raster

    def masked_values(self, value, copy=False):
        raster = self
        if copy:
            raster = self.copy()
        raster[:] = np.ma.masked_values(raster, value)
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
        raster[:] = (raster / raster.max())[:]
        return raster

    def gaussian_filter(self, sigma, mode="constant", copy=False, **kwargs):
        raster = self
        if copy:
            raster = self.copy()
        if sigma > 0:
            raster[:] = ndimage.gaussian_filter(raster, sigma, mode=mode, **kwargs)
        return raster

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
        plt.imshow(self, cmap=cmap)

    def get_basemap(self, ax=None):
        minx, miny, maxx, maxy = self.geollur
        epsg = self.crs['init'].split(':')[1]
        return basemap.Basemap(
            llcrnrlon=minx, llcrnrlat=miny, urcrnrlon=maxx, urcrnrlat=maxy,
            resolution='h',
            epsg=epsg, ax=ax)

    def plotmap(self, legend=False, arcgis=False, coast=False, countries=False,
                rivers=False, grid=False, bluemarble=False, etopo=False,
                maptype=None, cmap=None, norm=None, logcolor=False, vmin=None,
                vmax=None, ax=None):

        if maptype == 'minimal':
            coast = True,
            countries = True
        elif maptype == 'full':
            coast = True,
            countries = True
            rivers = True
            arcgis = True
            grid = True

        m = self.get_basemap(ax=ax)

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
                          xpixels=2000, verbose= True)

        mapimg = m.imshow(np.flipud(self), cmap=cmap, norm=norm,
                          vmin=vmin, vmax=vmax)

        if coast:
            m.drawcoastlines()
        if countries:
            m.drawcountries()
        if rivers:
            m.drawrivers(linewidth=0.2, linestyle='solid', color='b')
        if grid:
            m.drawparallels(np.arange(-90,90,2),labels=[1,0,0,0],fontsize=10)
            m.drawmeridians(np.arange(-90,90,2),labels=[0,0,0,1],fontsize=10)
        if legend:
            plt.colorbar(mapimg, orientation='vertical', ax=ax)

        return m, mapimg
