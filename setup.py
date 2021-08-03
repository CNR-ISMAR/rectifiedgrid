import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="rectifiedgrid",  # or RectifiedGrid
    version="1.2.0",
    author="Stefano Menegon",
    author_email="ste.menegon@gmail.com",
    description="RectifiedGrid is a python module to deal with rectified grid.",
    license="GPL3",
    keywords="",
    url="http://todo.todo",
    packages=['rectifiedgrid'],
    # long_description=read('README'),
    install_requires=[
        # data structures and analyses
        'numpy',
        'geopandas',
        'scipy',

        # I/O
        'rasterio',
        'fiona',

        # vector data utils
        'shapely',
        'rtree',
        'pandas',
        # utils
        'affine', 'matplotlib', 'cartopy',
        'mapclassify'
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ]
)
