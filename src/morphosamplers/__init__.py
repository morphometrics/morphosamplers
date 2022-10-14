"""A library for sampling image data along morphological objects."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("morphosamplers")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Kevin Yamauchi"
__email__ = "kevin.yamauchi@gmail.com"
