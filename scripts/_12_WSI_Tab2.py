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
# from skimage.external.tifffile import imread as tiff_read
from scipy.ndimage.morphology import binary_fill_holes as fill_holes
from skimage import measure
# import glymur
import time
import pickle
import scipy.misc
###########Neural net modules
# import torch
from scipy import misc
import numpy as np
# from torchvision import datasets, models, transforms
import os
from tqdm import trange,tqdm
import sys, os
# sys.path.insert(0,os.path.abspath("PathPretrain"))
import GPUtil
from pathpretrain.train_model import train_model, generate_transformers, generate_kornia_transforms
from torch.utils.data import Dataset
import torch, pandas as pd, numpy as np
from PIL import Image
import seaborn as sns, matplotlib
from scipy.special import softmax
from sklearn.preprocessing import RobustScaler
# add other morphometric stats
# https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.multiscale_basic_features
# maybe even projections
EPS=0#1e-8

class WSI_Dataset(Dataset):
    def __init__(self, stats_dict, transform, include_all=False, aux_data=None, scaler=None):
        imgs=pd.Series([stats_dict[i]['colorimage'] for i in range(len(stats_dict))])
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

def im_resizer(I, scale_percent):
	import numpy as np
	import cv2
	# width = int(I.shape[1] * scale_percent / 100)
	# height = int(I.shape[0] * scale_percent / 100)

	I1 = I[0:int(I.shape[0]/2), :, :].copy()
	I2 = I[int(I.shape[0]/2):, :, :].copy()
	del I

	h1 = int(I1.shape[0] * (scale_percent / 100))
	w1 = int(I1.shape[1] * (scale_percent / 100))
	h2 = int(I2.shape[0] * (scale_percent / 100))
	w2 = int(I2.shape[1] * (scale_percent / 100))

	print(h1,w1,h2,w2)

	I1 = cv2.resize(I1, (w1, h1), interpolation = cv2.INTER_CUBIC)
	I2 = cv2.resize(I2, (w2, h2), interpolation = cv2.INTER_CUBIC)

	# init resized array
	I = 255*np.ones((h1+h2, w1, 3), dtype='uint8')
	print(I1.shape, I2.shape)
	# merge resized sub arrayss
	I[0:h1, 0:w1, :] = I1

	del I1
	I[h1:h1+h2, 0:w1, :] = I2
	del I2
	print(I.shape)
	return(I)

def stats_maker(path, resize = 0):

    if path.split('.')[-1] == 'h5':
        I = h5_load(path)
    elif path.split('.')[-1] == 'jp2':
        import glymur
        jp2 = glymur.Jp2k(path)
        I = jp2[:]
    elif path.split('.')[-1] == 'tif':
        I = tiff_read(path)
    elif path.split('.')[-1] == 'npy':
        I = np.load(path)
    
    scale=2
    scale_percent = 50*scale
    
    width = int(I.shape[1] * scale_percent / 100)
    height = int(I.shape[0] * scale_percent / 100)
    
    I[(I[:,:,0]>220) & (I[:,:,1]>220) & (I[:,:,2]>220)]=[255,255,255]
        
    tops = 255*np.ones((1, I.shape[1], 3), dtype='uint8')
    I = np.vstack((tops, I))
    I = np.vstack((I, tops))
    sides = 255*np.ones((I.shape[0], 1, 3), dtype='uint8')
    I = np.hstack((sides, I))
    I = np.hstack((I, sides))
    
    BW = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    BW2 = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    BW[BW<255]=1
    BW2[BW2<255]=1
    BW[BW==255]=0
    BW2[BW2==255]=0
    labels = scilabel(BW)[0]
    labels2 = scilabel(BW2)[0]
    BW = morph.remove_small_objects(labels, min_size=50, connectivity = 8, in_place=True) # may want to do this compressed then upsample? *(scale**2) *scale *(scale**2) *scale
    BW2 = morph.remove_small_objects(labels2, min_size=500000, connectivity = 8, in_place=True)
    del labels
    del labels2
    BW[BW2!=0]=0
    del BW2
    BW = fill_holes(BW)
    I[BW!=1]=[255,255,255]
    labels,n_labels = scilabel(BW)
    assert n_labels<=1e5
    del BW
    stats = measure.regionprops(labels, coordinates='rc')
    del labels
    
    stats_dict = {num:{} for num in range(len(stats))}
    for r in trange(len(stats_dict)):
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
    #Dict of all subimages and statistics / coordinates
    return(stats)

def return_results(wsi_file='wsi.npy',
                   out_dir='out_dir',
                   gpu_id=0,
                   seg_model="segmentation_practice/uro_seg/49.checkpoint.pth",#30.
                   uro_model="uro_net_aux_v4/65.epoch.checkpoint.pth",
                   aty_model="aty_net_aux_v3/98.epoch.checkpoint.pth",
                   include_all=True
                   ):
    os.makedirs(out_dir,exist_ok=True)
    basename=os.path.basename(wsi_file)
    out_file=os.path.join(out_dir,f"{basename[:basename.rfind('.')]}.pkl")
    if not os.path.exists(out_file):
        st_test=stats_maker(wsi_file, resize = 0)
        
        df_stat=pd.DataFrame([{k:v for k,v in st_test[i].items() if isinstance(v,(np.int64,int,float,tuple)) and k not in ['slice','local_centroid']} for i in range(len(st_test))])
        df_stat=df_stat.drop(columns=['bbox','bbox_area','centroid','euler_number','label','orientation','perimeter_crofton'])
        if df_stat.shape[0]<=1e5:
            transform=generate_transformers(256,256)['test']
            custom_dataset=WSI_Dataset(st_test,transform,include_all=include_all)
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
            cell_stats=pd.DataFrame([{k:v for k,v in st_test[i].items() if isinstance(v,(np.int64,int,float,tuple)) and k not in ['slice','local_centroid']} for i in tqdm(included_cells)])
            nc_ratio,atypia_score=pd.concat([pd.Series(y).value_counts() for y in nc_calls],axis=1).T.fillna(0),softmax(Y['class2'],1)[:,1]
            pickle.dump(dict(uro_decision=Y['class1'],
                              atypia=atypia_score,nc=nc_ratio,
                             cell_stats=cell_stats),open(out_file,'wb'))
        #     pd.DataFrame(dict(uro_decision=Y['class1'],
        #                       atypia=atypia_score,nc=nc_ratio,
        #                      cell_stats=cell_stats)).to_pickle(os.path.join(out_dir,f"{basename[:basename.rfind('.')]}.pkl"))

        #     pd.DataFrame(dict(atypia=atypia_score,
        #                      nc=nc_ratio)).to_pickle(os.path.join(out_dir,f"{basename[:basename.rfind('.')]}.pkl"))

if __name__=="__main__":
    fire.Fire(return_results)