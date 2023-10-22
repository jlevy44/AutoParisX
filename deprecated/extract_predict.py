import sys
import fire
import h5py
from matplotlib import pyplot as plt
import torch.utils.data as utils
import cv2
from matplotlib import pyplot as plt
import numpy as np, pandas as pd
from skimage import morphology as morph
from scipy.ndimage import label as scilabel
from scipy.ndimage.morphology import binary_fill_holes as fill_holes
from skimage import measure
import itertools
import time
import pickle
import scipy.misc
from scipy import misc
import numpy as np
from scipy.ndimage import binary_erosion
import os
from tqdm import trange,tqdm
import sys, os
import GPUtil
from pathpretrain.train_model import train_model, generate_transformers, generate_kornia_transforms
from torch.utils.data import Dataset
import torch, pandas as pd, numpy as np
from PIL import Image
import seaborn as sns, matplotlib
from scipy.special import softmax
from sklearn.preprocessing import RobustScaler
import dask
EPS=0

class WSI_Dataset(Dataset):
    def __init__(self, stats_dict, transform, include_all=False, aux_data=None, scaler=None):
        imgs=pd.Series([stats_dict[i]['colorimage'] for i in range(len(stats_dict))])
        self.imgs=imgs
        shapes=np.array(imgs.map(lambda x: x.shape).tolist())
        include_image=np.logical_and(shapes[:,0]>=4, shapes[:,1]>=4, np.logical_or(shapes[:,0]>=10,shapes[:,1]>=10)) if not include_all else np.ones(len(stats_dict)).astype(bool)
        if not include_all: imgs=imgs[include_image]#.map(lambda x: x.astype(float)/255.)
        self.to_pil=lambda x: Image.fromarray(x)
        self.X=imgs.tolist()
        self.retained_idx=np.where(include_image)[0]
        self.length=len(self.X)
        self.transform=transform
        self.aux_data=aux_data
        self.has_aux=(self.aux_data is not None)
        if self.has_aux and isinstance(self.aux_data,pd.DataFrame): self.aux_data=self.aux_data.values
        if self.has_aux:
            assert scaler is not None
            self.aux_data=scaler.transform(self.aux_data)
            self.n_aux_features=self.aux_data.shape[1]

    def __getitem__(self,idx):
        X=self.transform(self.to_pil(self.X[idx]))
        items=(X,torch.zeros(X.shape[-2:]).unsqueeze(0).long())
        if self.has_aux: items+=(torch.tensor(self.aux_data[idx]).float(),)
        return items

    def __len__(self):
        return self.length

def load_image(path):
    ext=os.path.splitext(path)[-1]
    if ext == '.h5':
        I = h5_load(path)
    elif ext == '.jp2':
        import glymur
        jp2 = glymur.Jp2k(path)
        I = jp2[:]
    elif ext == '.tif':
        from tifffile import imread as tiff_read
        I = tiff_read(path)
    elif ext == '.npy':
        I = np.load(path)
    return I

def stats_maker(I, resize = 0, max_objects=1e5, origin=(0,0)):

    I = cv.copyMakeBorder(I,1,1,1,1,cv.BORDER_CONSTANT,value=[255,255,255])

    BW = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

    I[BW>220]=[255,255,255]

    BW = (BW<255)
    labels = scilabel(BW)[0]
    BW = morph.remove_small_objects(labels, min_size=50, connectivity = 8, in_place=False)
    BW2 = morph.remove_small_objects(labels, min_size=500000, connectivity = 8, in_place=False)
    del labels
    BW[BW2]=0
    del BW2
    BW = fill_holes(BW)
    I[~BW]=[255,255,255]
    labels,n_labels = scilabel(BW)
    assert n_labels<=max_objects
    del BW
    stats = measure.regionprops(labels, coordinates='rc')
    del labels

    stats_dict = {num:{} for num in range(len(stats))}
    for r in trange(len(stats_dict)):
        # stats_dict[r].update(stats[r])
        for key in stats[r]:
            stats_dict[r][key] = stats[r][key]
        minr = stats[r].bbox[0]
        maxr = stats[r].bbox[2]
        minc = stats[r].bbox[1]
        maxc = stats[r].bbox[3]
        img = I[minr:maxr, minc:maxc, :].copy()
        img[stats[r].image!=1]=[255,255,255]
        stats_dict[r]['colorimage'] = img

    stats = stats_dict
    df_stat=pd.DataFrame([{k:v for k,v in stats[i].items() if isinstance(v,(np.int64,int,float,tuple)) and k not in ['slice','local_centroid']} for i in range(len(stats))])
    bbox=df_stat.pop("bbox")
    df_stat=df_stat.drop(columns=['bbox_area','centroid','euler_number','label','orientation','perimeter_crofton'])
    return stats, df_stat, bbox

def chunk_wsi(I, dr, dc):
    height,width=I.shape[:2]
    if dr==0 or dc==0: return [dict(origin=(0,0),I=I)]
    r=np.arange(0,height,dr).astype(int).tolist()
    c=np.arange(0,width,dc).astype(int).tolist()
    if r[-1]>I.shape[0]-dr/2: r=r[:-1]
    if c[-1]>I.shape[1]-dc/2: c=c[:-1]
    return [dict(origin=(r,c),I=I[r:r+dr,c:c+dc]) for r,c in itertools.product(r,c)]

def process_chunk(chunk,chunk_num=0,out_dir='out_dir',
                    gpu_id=0,
                    seg_model="segmentation_practice/uro_seg/49.checkpoint.pth",#30.
                    uro_model="uro_net_aux_v4/65.epoch.checkpoint.pth",
                    aty_model="aty_net_aux_v3/98.epoch.checkpoint.pth",
                    include_all=True,
                    max_objects=1e5,
                    save_cell_loc="",
                    out_file=""):
    st_test,df_stat,bbox=stats_maker(**chunk)
    if df_stat.shape[0]<=max_objects:
        transform=generate_transformers(256,256)['test']
        custom_dataset=WSI_Dataset(st_test,transform,include_all=include_all)
        if save_cell_loc:
            if save_cell_loc.endswith(".pkl"): save_cell_loc=save_cell_loc.replace(".pkl",f"_{chunk_num}.pkl")
            else: save_cell_loc=f"{save_cell_loc}_{chunk_num}.pkl"
            custom_dataset.imgs.to_pickle(save_cell_loc)
            exit()
        Y_seg=train_model(inputs_dir='segmentation_practice/inputs',
                    architecture='resnet50',
                    batch_size=256,
                    num_classes=3,
                    predict=True,
                    model_save_loc=seg_model,
                    predictions_save_path='tmp_test.pkl',
                    predict_set='custom',
                    verbose=False,
                    class_balance=False,
                    gpu_id=gpu_id,
                    tensor_dataset=False,
                    semantic_segmentation=True,
                    custom_dataset=custom_dataset,
                    save_predictions=False)
        transform=generate_transformers(224,256)['test']
        custom_dataset=WSI_Dataset(st_test,transform,include_all=include_all, aux_data=df_stat, scaler=pd.read_pickle("images/robust_scaler.pkl"))
        Y_class1=train_model(inputs_dir='inputs',
                    architecture='resnet50',
                    batch_size=128,
                    predict=True,
                    num_classes=6,
                    model_save_loc=uro_model,
                    predictions_save_path='tmp_test.pkl',
                    predict_set='custom',
                    verbose=False,
                    class_balance=False,
                    gpu_id=gpu_id,
                    tensor_dataset=False,
                    pickle_dataset=True,
                    semantic_segmentation=False,
                    custom_dataset=custom_dataset,
                    save_predictions=False)
        Y_class2=train_model(inputs_dir='inputs',
                    architecture='resnet50',
                    batch_size=128,
                    predict=True,
                    num_classes=2,
                    model_save_loc=aty_model,
                    predictions_save_path='tmp_test.pkl',
                    predict_set='custom',
                    verbose=False,
                    class_balance=False,
                    gpu_id=gpu_id,
                    tensor_dataset=False,
                    pickle_dataset=True,
                    semantic_segmentation=False,
                    custom_dataset=custom_dataset,
                    save_predictions=False)
        Y=dict(seg=Y_seg['pred'],
          class1=Y_class1['pred'],
          class2=Y_class2['pred'])

        included_cells=custom_dataset.retained_idx

        calc_nc_ratio=lambda x: (x==2).sum()/(((x==1).sum()+(x==2).sum()+EPS))#+(x==2).sum()

        #     include_images=(Y['class1'].argmax(1)==0) # deactivate class2 to get urothelial only
        #     nc_ratio,atypia_score=np.array([calc_nc_ratio(y.argmax(0)) for y in Y['seg'][include_images]]),softmax(Y['class2'][include_images],1)[:,1]
        nc_calls=[y.argmax(0).flatten() for y in Y['seg']]#np.array([calc_nc_ratio(y) for y in nc_calls])
        n_cells=np.array([scilabel(binary_erosion(cv2.resize(softmax(Y_seg['pred'][i],0)[2],custom_dataset.X[i].shape[:-1][::-1],interpolation=cv2.INTER_CUBIC)>0.5,iterations=5))[1] for i in trange(len(custom_dataset))])
        cell_stats=pd.DataFrame([{k:v for k,v in st_test[i].items() if isinstance(v,(np.int64,int,float,tuple)) and k not in ['slice','local_centroid']} for i in tqdm(included_cells)])
        nc_ratio,atypia_score=pd.concat([pd.Series(y).value_counts() for y in nc_calls],axis=1).T.fillna(0),softmax(Y['class2'],1)[:,1]
        return Y,n_cells,cell_stats,atypia,nc,uro_decision,chunk['origin'],out_file,bbox.iloc[included_cells]

def return_results(wsi_file='wsi.npy',
                   out_dir='out_dir',
                   gpu_id=0,
                   seg_model="segmentation_practice/uro_seg/49.checkpoint.pth",#30.
                   uro_model="uro_net_aux_v4/65.epoch.checkpoint.pth",
                   aty_model="aty_net_aux_v3/98.epoch.checkpoint.pth",
                   include_all=True,
                   max_objects=1e5,
                   save_cell_loc="",
                   num_workers=None,
                   n_chunks=0
                   ):
    os.makedirs(out_dir,exist_ok=True)
    basename=os.path.basename(wsi_file)

    if not os.path.exists(out_file):
        I = load_image(wsi_file)
        height,width=I.shape
        if n_chunks: dr,dc=height//n_chunks,width//n_chunks
        else: dr,dc=0,0
        chunks=chunk_wsi(I, dr, dc)
        chunk_results=[]
        for i,chunk in enumerate(chunks):
            chunk.update(dict(resize = 0, max_objects = max_objects))
            out_file=os.path.join(out_dir,f"{basename[:basename.rfind('.')]}_{i}.pkl")
            chunk_results.append(dask.delayed(process_chunk)(chunk,
                                                                      chunk_num=i,
                                                                      out_dir=out_dir,
                                                                    gpu_id=gpu_id,
                                                                    seg_model=seg_model,#30.
                                                                    uro_model=uro_model,
                                                                    aty_model=aty_model,
                                                                    include_all=include_all,
                                                                    max_objects=max_objects,
                                                                    save_cell_loc=save_cell_loc,
                                                                    out_file=out_file))

        for Y,n_cells,cell_stats,atypia,nc,uro_decision,origin,out_file,bbox in dask.compute(*chunk_results,scheduler="processes" if len(chunks)>1 else "single-threaded",num_workers=num_workers):
            pickle.dump(dict(uro_decision=Y['class1'],
                                  atypia=atypia_score,nc=nc_ratio,
                                 cell_stats=cell_stats,
                                 n_cells=n_cells,
                                 origin=origin,
                                 bbox=bbox),open(out_file,'wb'))
            #     pd.DataFrame(dict(uro_decision=Y['class1'],
            #                       atypia=atypia_score,nc=nc_ratio,
            #                      cell_stats=cell_stats)).to_pickle(os.path.join(out_dir,f"{basename[:basename.rfind('.')]}.pkl"))

            #     pd.DataFrame(dict(atypia=atypia_score,
            #                      nc=nc_ratio)).to_pickle(os.path.join(out_dir,f"{basename[:basename.rfind('.')]}.pkl"))
def main():
    fire.Fire(return_results)

if __name__=="__main__":
    main()
