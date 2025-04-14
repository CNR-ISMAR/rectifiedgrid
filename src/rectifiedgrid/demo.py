import numpy as np
import geopandas as gpd
import rectifiedgrid as rg
from affine import Affine
from shapely import geometry
import xarray
from rioxarray.rioxarray import affine_to_coords
from rasterio.enums import Resampling


epsg = 3035
gtransform = Affine.from_gdal(4500000.0, 10000.0, 0.0, 1560000.0, 0.0, -10000.0)

arr1 = np.zeros((9, 9), np.float64)
arr1[:3, :3] = np.array([[-1, -2, -3],
                         [0, -1, -2],
                         [0, 0, -1]])

arr1[4:8, 4:8] = np.array([[1, 2, 3, 4],
                           [1, 2, 3, 4],
                           [1, 2, 3, 4],
                           [1, 2, 3, 4]])

arr2 = np.zeros((9, 9), np.float64)
arr2[4, 4] = 1

arr3 = np.zeros((9, 9), np.float64)
arr3[0, 0] = 1

# now proj complies the axes order
lstring = geometry.LineString(((12.0, 36.0), (13.0, 37.0)))
# lstring = geometry.LineString(((36.0, 12.0), (37.0, 13.0)))
df_lstring = gpd.GeoDataFrame(geometry=[lstring], crs="epsg:4326")
df_lstring['value'] = 1

def array_to_rg(_array, epsg, gtransform):
    coords = affine_to_coords(gtransform, _array.shape[1], _array.shape[0])
    raster = (
        xarray.DataArray(_array, coords=coords)
            .rio.write_nodata(np.nan)
            .rio.write_crs(rg.parse_projection(epsg))
            .rio.write_transform(gtransform)
            .rio.write_coordinate_system()
    )
    return raster


def get_demo_data(name='default'):
    """Generate a demo RasterizedGrid
    """
    raster = None
    if name == 'default':
        raster = array_to_rg(arr1, epsg, gtransform)
    elif name == 'rg9x9':
        raster = array_to_rg(arr2, epsg, gtransform)
    elif name == 'line3035':
        features = df_lstring.to_crs("epsg:3035")[['geometry', 'value']].values.tolist()
        raster = rg.read_features(features, 10000, 3035, rounded_bounds=True)
    elif name == 'line4326':
        features = df_lstring[['geometry', 'value']].values.tolist()
        raster = rg.read_features(features, 0.1, 4326)
    return raster.copy()
