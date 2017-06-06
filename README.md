RectifiedGrid
=============

RectifiedGrid is a Python package which, combining into a single class
several pythonpackages (e.g. Numpy, SciPy, shapely, rasterio, fiona,
geopandas, owslib, matplotlib-basemaps), simplifies geospatial
grid-based analyses. Numpy is a notable package for scientific
computing with a support for large, multi-dimensional
arrays and matrices: RectifiedGrid extends the numpy MaskedArray class
by adding geospatial functionalities (i.e. projection awareness,
boundingboxes, raster algebra). RectifiedGrid employs rasterio and
fiona under the hood for raster and vector I/O and owslibto access
data through OGC interoperable services.

RectifiedGrid has been initialy developed to support Integrated
Coastal Management and Maritime Spatial Planning analyses.


Usage
=====

### Reading a GeoTIFF

```python
import rectifiedgrid as rg
grid = rg.read_raster('test/data/adriatic_ionian.tiff', masked=True)
grid.plotmap()
```

![Alt text](/docs/images/adriatic_ionian_grid.png?raw=true "Adriatic Ionian Grid")


### Ndvi calculation

```python
import rectifiedgrid as rg

b4 = rg.read_raster('test/data/b04.tiff', masked=True)
b8 = rg.read_raster('test/data/b08.tiff', masked=True)

ndvi = (b8.astype(float) - b4.astype(float))/(b8.astype(float) + b4.astype(float))
ndvi.plotmap(cmap=cmap_ndvi, legend=True, vmin=-1, vmax=1)
```

![Alt text](/docs/images/ndvi.png?raw=true "Ndvi example")


How to Cite
===========
Please, when you use rectifiedgrid cite as:

Menegon S, Sarretta A, Barbanti A, Gissi E, Venier C. (2016) Open
source tools to support Integrated Coastal Management and Maritime
Spatial Planning. PeerJ Preprints 4:e2245v2. doi: [10.5334/jors.106]
(https://doi.org/10.7287/peerj.preprints.2245v2)
