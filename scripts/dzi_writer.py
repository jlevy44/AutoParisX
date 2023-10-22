import pandas as pd, numpy as np
from scipy.ndimage import label as scilabel
from skimage.measure import regionprops_table
import cv2, os, subprocess
from deepzoom import *
from deepzoom import _get_or_create_path,_get_files_path
from PIL import Image
import tqdm
import dask
from dask.diagnostics import ProgressBar
from scipy.special import softmax
import torch
from sauth import SimpleHTTPAuthHandler, serve_http
from skimage.draw import circle
Image.MAX_IMAGE_PIXELS = None

class Numpy2DZI(ImageCreator):
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
        super().__init__(tile_size,tile_overlap,tile_format,image_quality,resize_filter,copy_metadata)
        self.compression=compression

    def create(self, source_arr, destination):
        # potentially have an option where dynamically softlink once deeper layer is made so slide is readily available, push to background process and write metadata for dash app to read
        # speed up image saving with dask https://stackoverflow.com/questions/54615625/how-to-save-dask-array-as-png-files-slice-by-slice https://github.com/dask/dask-image/issues/110
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

def npy2dzi(npy_file='',
            dzi_out='',
            compression=1.):
    Numpy2DZI(compression=compression).create(np.load(npy_file),
                                              dzi_out)