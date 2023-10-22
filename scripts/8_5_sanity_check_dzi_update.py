import fire
import os
from dzi_writer import npy2dzi

def make_images(npy_file='', out_dir='', compression=1.):
    out_dzi=os.path.join(out_dir,os.path.basename(npy_file).replace(".npy",".dzi"))
    if not os.path.exists(out_dzi):
        npy2dzi(npy_file, out_dzi, compression)
        
if __name__=="__main__":
    fire.Fire(make_images)
