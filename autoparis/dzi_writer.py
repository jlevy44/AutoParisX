import cv2
import dask
import fire
import numpy as np
import os
import pandas as pd
import subprocess
import torch
import tqdm

from PIL import Image
from dask.diagnostics import ProgressBar
from sauth import SimpleHTTPAuthHandler, serve_http
from scipy.ndimage import label as scilabel
from scipy.special import softmax
from skimage.draw import circle
from skimage.measure import regionprops_table
from deepzoom import *
from deepzoom import _get_or_create_path, _get_files_path
from pathpretrain.utils import load_image

Image.MAX_IMAGE_PIXELS = None

class Numpy2DZI(ImageCreator):
    """
    A class for creating a Deep Zoom Image (DZI) from a NumPy array.

    Attributes:
        tile_size (int): The size of each tile in pixels.
        tile_overlap (int): The number of overlapping pixels between adjacent tiles.
        tile_format (str): The file format to use for each tile (e.g. "jpg", "png").
        image_quality (float): The quality of the compressed image (0.0 to 1.0).
        resize_filter (int): The filter to use for resizing the image (e.g. cv2.INTER_CUBIC).
        copy_metadata (bool): Whether to copy the metadata from the source image.
        compression (float): The compression factor to use for the image (1.0 for no compression).

    Methods:
        create(source_arr, destination): Creates a DZI from the given NumPy array and saves it to the specified destination.

    """

    def __init__(
        self,
        tile_size=254,
        tile_overlap=1,
        tile_format="jpg",
        image_quality=0.8,
        resize_filter=None,
        copy_metadata=False,
        compression=1.
    ):
        """
        Initializes a new instance of the Numpy2DZI class.

        Args:
            tile_size (int): The size of each tile in pixels.
            tile_overlap (int): The number of overlapping pixels between adjacent tiles.
            tile_format (str): The file format to use for each tile (e.g. "jpg", "png").
            image_quality (float): The quality of the compressed image (0.0 to 1.0).
            resize_filter (int): The filter to use for resizing the image (e.g. cv2.INTER_CUBIC).
            copy_metadata (bool): Whether to copy the metadata from the source image.
            compression (float): The compression factor to use for the image (1.0 for no compression).

        """
        super().__init__(tile_size,tile_overlap,tile_format,image_quality,resize_filter,copy_metadata)
        self.compression=compression

    def create(self, source_arr, destination):
        """
        Creates a DZI from the given NumPy array and saves it to the specified destination.

        Args:
            source_arr (numpy.ndarray): The NumPy array to create the DZI from.
            destination (str): The path to save the DZI to.

        Returns:
            str: The path to the saved DZI.

        """
        self.image = PIL.Image.fromarray(source_arr if self.compression==1 else cv2.resize(source_arr,None,fx=1/self.compression,fy=1/self.compression,interpolation=cv2.INTER_CUBIC))
        width, height = self.image.size
        self.descriptor = DeepZoomImageDescriptor(
            width=width,
            height=height,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            tile_format=self.tile_format,
        )
        image_files = _get_or_create_path(_get_files_path(destination))
        for level in tqdm.trange(self.descriptor.num_levels, desc='level'):
            level_dir = _get_or_create_path(os.path.join(image_files, str(level)))
            level_image = self.get_image(level)
            for (column, row) in tqdm.tqdm(self.tiles(level), desc='tiles'):
                bounds = self.descriptor.get_tile_bounds(level, column, row)
                tile = level_image.crop(bounds)
                format = self.descriptor.tile_format
                tile_path = os.path.join(level_dir, "%s_%s.%s" % (column, row, format))
                tile_file = open(tile_path, "wb")
                if self.descriptor.tile_format == "jpg":
                    jpeg_quality = int(self.image_quality * 100)
                    tile.save(tile_file, "JPEG", quality=jpeg_quality)
                else:
                    tile.save(tile_file)
        self.descriptor.save(destination)
        return destination


def npy2dzi(npy_file='', dzi_out='', compression=1.):
    """
    Converts a NumPy array to a Deep Zoom Image (DZI) and saves it to the specified destination.

    Args:
        npy_file (str): The path to the NumPy file to convert.
        dzi_out (str): The path to save the DZI to.
        compression (float): The compression factor to use for the image (1.0 for no compression).

    Returns:
        str: The path to the saved DZI.

    """
    Numpy2DZI(compression=compression).create(load_image(npy_file), dzi_out)
    
def main():
	fire.Fire(npy2dzi)

if __name__=="__main__":
    main()
