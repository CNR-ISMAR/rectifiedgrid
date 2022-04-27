RectifiedGrid
=============

**WARNING**: Starting from version 2.0.0, rectifiedgrid grid has been
refactored basing basing on xarray and rioxarray packages. Several API
have been changed.

RectifiedGrid is a Python package which, combining into a single class
several python packages (e.g. xarray, rioxarray, SciPy, shapely,
fiona, geopandas, owslib, Cartopy), simplifies geospatial grid-based
analyses. Numpy is a notable package for scientific computing with a
support for large, multi-dimensional arrays and matrices: starting
from version 2.0.0, RectifiedGrid extends xarray introducing an "rg"
accessor for adding additional functionalities (e.g. rasterization, 3D
manipulation). RectifiedGrid employs rasterio and fiona under the hood
for raster and vector I/O and owslib to access data through OGC
interoperable services.

RectifiedGrid has been initially developed to support Integrated
Coastal Management and Maritime Spatial Planning analyses.

Installation
============
When using RectifiedGrid, you need to make sure that Geopandas is installed with rtree support.
Refer to following link for more informations:
* http://geopandas.org/install.html#installing-with-pip
* http://toblerity.org/rtree/install.html

If you want to install Shapely dependency from source for
compatibility with cartopy or if you want to use a different version
of GEOS than the one included in the project wheels you should first
install the GEOS library, Cython, and Numpy on your system (using apt,
yum, brew, or other means) and then direct pip to ignore the binary
wheels.

```shell
$ pip install shapely --no-binary shapely
```

Usage
=====

### Reading and plot GeoTIFF

```python
import rectifiedgrid as rg
grid = rg.read_raster('test/data/adriatic_ionian.tiff', masked=True)
grid.plotmap()
```

![Alt text](/docs/images/adriatic_ionian_grid.png?raw=true "Adriatic Ionian Grid")

### Plotting options

RectifiedGrid wraps Matplotlib Basemap Toolkit functions.

```python
grid.plotmap(rivers=True, countries=True,
             grid=True, coast=True)
```

![Alt text](/docs/images/plot_options.png?raw=true "Plotting options")


### Map algebra: Ndvi calculation

```python
import rectifiedgrid as rg

b4 = rg.read_raster('test/data/b04.tiff', masked=True)
b8 = rg.read_raster('test/data/b08.tiff', masked=True)

ndvi = (b8 - b4)/(b8 + b4)
ndvi.plotmap(cmap=cmap_ndvi, legend=True, vmin=-1, vmax=1)
```

![Alt text](/docs/images/ndvi.png?raw=true "Ndvi example")


### Wrapping array-wise functions: distance from coast

RectifiedGrid implements a function wrapper (wrap_func) to apply
array-wise functions.

In this example we use the distance_transform_bf (from
scipy.ndimage,morphology) to calculate the distance from the coast for
the Adriatic-Inonian region.

```python
from scipy.ndimage.morphology import distance_transform_bf
distances = grid.wrap_func(distance_transform_bf)

# plotting
plt.figure(figsize=[10, 8])
distances.plotmap(rivers=True, countries=True,
             grid=True, coast=True, legend=True)

```

![Alt text](/docs/images/distances.png?raw=true "Distances example")

How to Cite
===========
Please, when you use rectifiedgrid cite as:

Menegon S, Sarretta A, Barbanti A, Gissi E, Venier C. (2016) Open
source tools to support Integrated Coastal Management and Maritime
Spatial Planning. PeerJ Preprints 4:e2245v2. doi: [10.5334/jors.106]
(https://doi.org/10.7287/peerj.preprints.2245v2)
