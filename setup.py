import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "rectifiedgrid", # or RectifiedGrid
    version = "0.2.0",
    author = "Stefano Menegon",
    author_email = "ste.menegon@gmail.com",
    description = ("RectifiedGrid is a python module to deal with rectified grid."),
    license = "GPL3",
    keywords = "",
    url = "http://todo.todo",
    packages=['rectifiedgrid'],
    # long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ]
)
