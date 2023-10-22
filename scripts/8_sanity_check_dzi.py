import fire

import numpy as np
import PIL
from PIL import Image

import pandas as pd, os
from ot import sliced
import glob
import numpy as np
from scipy.special import softmax
import matplotlib
matplotlib.use("Agg")
import seaborn as sns, matplotlib.pyplot as plt
import tqdm
import cv2
from skimage.draw import circle
matplotlib.rcParams['figure.dpi']=300

import os
import sys
import deepzoom
import time
import shutil
import subprocess
import glob

import os
import time
import subprocess
from multiprocessing import Process

import fire

import numpy as np
import PIL
from PIL import Image

import os
import sys
import deepzoom
import time
import shutil

def worker(png_path, web_dir):
    # disable safety checks for large images
    PIL.Image.MAX_IMAGE_PIXELS = None

    name=os.path.basename(png_path).replace(".png","")

    prefix_path = os.path.join(web_dir, "images", name)
    dzi_path = prefix_path + ".dzi"
    if not os.path.exists(dzi_path):
        print(dzi_path)
        base_html_path = "PairedHE2Tri/npy2dzi.html"
        new_html_path = os.path.join(web_dir, "index.html")

        openseadragon_src = "PairedHE2Tri/openseadragon/"
        openseadragon_dst = os.path.join(web_dir, "openseadragon/")

        title=name

        # create dzi files
        START_TIME = time.time()
        print("Creating .dzi file")
        creator = deepzoom.ImageCreator(
            tile_size=256,
            tile_overlap=0,
            tile_format="png",
            image_quality=1.0,
        )
        creator.create(png_path, dzi_path)
        print("Execution time (s):", time.time() - START_TIME)
        print("Done.\n")

        START_TIME = time.time()
        print("Creating HTML files")
        
def png2dzi(image_path,web_dir,basename):
    PROGRAM_START_TIME = time.time()

    jobs = []
    for png_path in glob.glob(f"{image_path}/{basename}*.png"):
        p = Process(target=worker, args=(png_path, web_dir))
        jobs += [p]
        p.start()
    for job in jobs:
        job.join()
        
def edit_stats(stat):
    stats2={}
    include=softmax(stat['uro_decision'],1).argmax(1)==0#[:,0]>0.99#
    stats2['nc_ratio']=(stat['nc'][2]/(stat['nc'][2]+stat['nc'][1]))[include]
    stats2['atypia']=stat['atypia'][include]
    return pd.concat([pd.DataFrame(stats2),stat['cell_stats'].loc[include]],axis=1)


def make_images(file=''):
    print(file)
    if not os.path.exists(f"webapp_images_dzi/images/{os.path.basename(file).replace('.pkl','_crosstabulation.png')}"):
        arr=np.load(f"example_full_scans/{os.path.basename(file).replace('.pkl','.npy')}")

        stat=edit_stats(pd.read_pickle(file))
        plt.figure()
        plt.scatter(stat['nc_ratio'],stat['atypia'],alpha=0.3,s=2)
        plt.title(file)
        plt.xlabel("NC Ratio")
        plt.ylabel("Atypia")
        sns.despine()
        plt.savefig(f"webapp_images_dzi/images/{os.path.basename(file).replace('.pkl','_crosstabulation.png')}")
        try:
            xy=np.array(stat['centroid'].tolist())
            arr2=arr.copy()
            arr3=arr.copy()
            cmap=sns.color_palette("viridis", as_cmap=True)
            colors=stat['atypia'].map(lambda x: (np.array(cmap(x)[:3])*255).astype(int)).tolist()
            colors2=stat['nc_ratio'].map(lambda x: (np.array(cmap(x)[:3])*255).astype(int)).tolist()

            for i,(x,y) in tqdm.tqdm(list(enumerate(stat['centroid'].tolist()))):
                xx,yy=circle(int(round(x)), int(round(y)), 20)
                arr2[xx,yy]=colors[i]
                arr3[xx,yy]=colors2[i]
            cv2.imwrite(f"webapp_images/{os.path.basename(file).replace('.pkl','')}.png",cv2.cvtColor(cv2.resize(arr,None,fx=1/4.,fy=1/4.,interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB))
            cv2.imwrite(f"webapp_images/{os.path.basename(file).replace('.pkl','')}.atypia.png",cv2.cvtColor(cv2.resize(arr2,None,fx=1/4.,fy=1/4.,interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB))
            cv2.imwrite(f"webapp_images/{os.path.basename(file).replace('.pkl','')}.nc.png",cv2.cvtColor(cv2.resize(arr3,None,fx=1/4.,fy=1/4.,interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB))

            basename=os.path.basename(file).replace('.pkl','')#files=glob.glob(f"webapp_images/{os.path.basename(file).replace('.pkl','')}*")
            png2dzi("webapp_images/", "webapp_images_dzi/", basename)
        except:
            pass
        
if __name__=="__main__":
    fire.Fire(make_images)
