import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "rectifiedgrid", # or RectifiedGrid
    version = "1.0.0-beta.1",
    author = "Stefano Menegon",
    author_email = "ste.menegon@gmail.com",
    description = ("RectifiedGrid is a python module to deal with rectified grid."),
    license = "GPL3",
    keywords = "",
    url = "http://todo.todo",
    packages=['rectifiedgrid'],
    # long_description=read('README'),
    install_requires = [
        # I/O
        'rasterio',
        'fiona',

        # data structures and analyses
        'numpy',
        'geopandas',
        'scipy',

        # vector data utils
        'shapely',
        'rtree==0.8.3',

        # utils
        'affine'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ]
)
