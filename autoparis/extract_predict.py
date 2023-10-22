# Import necessary libraries
import fire
import cv2
import numpy as np, pandas as pd
from skimage import morphology as morph
from scipy.ndimage.morphology import binary_fill_holes as fill_holes
from skimage import measure
import itertools
import numpy as np
from scipy.ndimage import binary_erosion
import os
import warnings
# warnings.filterwarnings("ignore")

# Import functions from pathpretrain module
from pathpretrain.train_model import train_model, generate_transformers, generate_kornia_transforms
from torch.utils.data import Dataset
import torch, pandas as pd, numpy as np
from PIL import Image
from scipy.special import softmax
import dask
from dask.diagnostics import ProgressBar
import fill_voids

# Import GPU libraries if available
CUPY_IS_AVAILABLE=False
try:
    import cupy
    from cupyx.scipy.ndimage import label as cu_label
    from cucim.skimage.morphology import remove_small_objects as cu_remove_small_objects
    CUPY_IS_AVAILABLE=True
except:
    warnings.warn("Only GPU implementation for rapid connected components, either run pip install cucim==23.8.0 cupy-cuda11x==12.2.0 or pip install autoparis[gpu]", UserWarning)
    #raise NotImplementedError("Only GPU implementation for rapid connected components, either run pip install cucim==23.8.0 cupy-cuda11x==12.2.0 or pip install autoparis[gpu]")

# Import additional libraries
from functools import reduce
from dask.distributed import Client

# Import functions from detectron_predict and refine_detectron modules
from .detectron_predict import load_predictor
from .refine_detectron import clean_instance, clean_instance_v2

# Import garbage collector and GPU statistics libraries
import gc
import gpustat

# Define auxiliary columns
AUX_COLS=['area', 'convex_area', 'eccentricity', 'equivalent_diameter', 'extent',
       'feret_diameter_max', 'filled_area', 'major_axis_length',
       'minor_axis_length', 'perimeter', 'solidity']

# Define EPS and CURRENT_PROCESS constants
EPS=0
CURRENT_PROCESS=os.getpid()

# Define WSI_Dataset_v2 class
class WSI_Dataset_v2(Dataset):
    def __init__(self, imgs, transform, include_all=False, aux_data=None, scaler=None):
        # Initialize class variables
        self.imgs=imgs
        shapes=np.array(imgs.map(lambda x: x.shape).tolist())
        include_image=np.logical_and(shapes[:,0]>=4, shapes[:,1]>=4, np.logical_or(shapes[:,0]>=10,shapes[:,1]>=10)) if not include_all else np.ones(len(imgs)).astype(bool)
        if not include_all: imgs=imgs[include_image]#.map(lambda x: x.astype(float)/255.)
        self.to_pil=lambda x: Image.fromarray(x)
        self.X=imgs.tolist()
        self.retained_idx=np.where(include_image)[0]
        self.length=len(self.X)
        self.transform=transform
        self.aux_data=aux_data
        self.has_aux=(self.aux_data is not None)
        # Check if aux_data is a pandas DataFrame and transform it if necessary
        if self.has_aux and isinstance(self.aux_data,pd.DataFrame): self.aux_data=self.aux_data.values
        # Check if aux_data is not None and transform it using the provided scaler
        if self.has_aux:
            assert scaler is not None
            self.aux_data=scaler.transform(self.aux_data)
            self.n_aux_features=self.aux_data.shape[1]

    def __getitem__(self,idx):
        # Apply the transform to the image at the given index
        X=self.transform(self.to_pil(self.X[idx]))
        # Create a tuple of items to return
        items=(X,torch.zeros(X.shape[-2:]).unsqueeze(0).long())
        # If aux_data is not None, add it to the tuple
        if self.has_aux: items+=(torch.tensor(self.aux_data[idx]).float(),)
        return items

    def __len__(self):
        # Return the length of the dataset
        return self.length

def load_image(path):
    # Get the file extension from the path
    ext=os.path.splitext(path)[-1]
    # Load the image based on the file extension
    if ext == '.h5':
        # If the file is an HDF5 file, use the h5_load function to load it
        I = h5_load(path)
    elif ext == '.jp2':
        # If the file is a JPEG 2000 file, use the glymur library to load it
        import glymur
        jp2 = glymur.Jp2k(path)
        I = jp2[:]
    elif ext == '.tif':
        # If the file is a TIFF file, use the tifffile library to load it
        from tifffile import imread as tiff_read
        I = tiff_read(path)
    elif ext == '.npy':
        # If the file is a NumPy file, use the np.load function to load it
        I = np.load(path)
    # Return the loaded image
    return I

# Define a function to return a color image based on a bounding box and a binary mask
def return_color_im(d,I):
    # Get the bounding box coordinates
    minr,minc,maxr,maxc=d['bbox']
    # Copy the image within the bounding box
    img = I[minr:maxr, minc:maxc, :].copy()
    # Set all pixels outside the binary mask to white
    img[d['image']!=1]=255
    # Add the color image to the dictionary and return it
    d['color_image']=img
    return d

# Define a function to chunk a whole slide image into smaller pieces
def chunk_wsi(I, dr, dc):
    # Get the height and width of the image
    height,width=I.shape[:2]
    # If the chunk size is 0 in either dimension, return the whole image in a dictionary
    if dr==0 or dc==0: return [dict(origin=(0,0),I=I)]
    # Create lists of row and column indices for the chunks
    r=np.arange(0,height,dr).astype(int).tolist()
    c=np.arange(0,width,dc).astype(int).tolist()
    # If the last row or column index is too close to the edge of the image, remove it
    if r[-1]>I.shape[0]-dr/2: r=r[:-1]
    if c[-1]>I.shape[1]-dc/2: c=c[:-1]
    # Create a list of dictionaries, each containing a chunk of the image and its origin coordinates
    return [dict(origin=(r,c),I=I[r:r+dr,c:c+dc]) for r,c in itertools.product(r,c)]

# Define a delayed function to predict instances in a batch using a model
@dask.delayed
def predict(batch, model):
    # Use the model to predict instances in the batch
    with torch.no_grad():
        out = model(batch)['instances'].to("cpu")
    # Return the predicted instances
    return out

# Define a function to fix a binary mask if it has too few pixels
def fix_thres(msk,t):
    # If the mask has more pixels than the threshold, return it
    if msk.sum()>=t: return msk
    # Otherwise, set all pixels to False and return the mask
    else:
        msk[...]=False
        return msk

def gen_pred_mask(instance,t=0):
    return ((((torch.stack([fix_thres(m,t) for m in instance.pred_masks[instance.pred_classes==1]])).float().sum(0)>0).numpy().astype(float)-((instance.pred_masks[instance.pred_classes!=1]).float().sum(0)>0).numpy().astype(float))>0) if sum(instance.pred_classes==1) else np.zeros(instance._image_size).astype(bool)

def process_image(I, max_objects=1e5, origin=(0,0), *args, **kwargs):
    # Convert the input image to grayscale
    BW = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    # Threshold the grayscale image to create a binary image
    BW_bool = BW > 220
    # Set the pixels in the input image to white where the binary image is True
    for i in range(3):
        np.copyto(I[...,i], 255, where=BW_bool)
    # Convert the input image to grayscale again
    BW = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    # Invert the binary image
    BW = (BW < 255)

    # Check if CuPy is available
    if CUPY_IS_AVAILABLE:
        # If CuPy is available, use the CuPy implementation of connected components labeling and morphological operations
        BW_cu = cupy.array(BW)
        labels = cupy.asnumpy(cu_label(BW_cu)[0])
        BW = morph.remove_small_objects(labels, min_size=50, connectivity=8, in_place=False)
        BW2 = morph.remove_small_objects(labels, min_size=500000, connectivity=8, in_place=False)
        del labels
        BW[BW2 > 0] = 0
        del BW2
        BW = fill_voids.fill(BW, in_place=False)
        BW_bool = BW == 0
        for i in range(3):
            np.copyto(I[...,i], 255, where=BW_bool)
        BW_cu = cupy.array(BW)
        labels, n_labels = cu_label(BW_cu)
        labels = cupy.asnumpy(labels)
    else:
        # If CuPy is not available, use the scikit-image implementation of connected components labeling and morphological operations
        labels, n_labels = measure.label(BW, connectivity=2, return_num=True)
        BW = morph.remove_small_objects(labels, min_size=50, connectivity=8, in_place=False)
        BW2 = morph.remove_small_objects(labels, min_size=500000, connectivity=8, in_place=False)
        del labels
        BW[BW2 > 0] = 0
        del BW2
        BW = fill_voids.fill(BW)
        BW_bool = BW == 0
        for i in range(3):
            np.copyto(I[...,i], 255, where=BW_bool)
        labels, n_labels = measure.label(BW, connectivity=2, return_num=True)

    # Compute the region properties of the labeled regions
    stats = [return_color_im({k:s[k] for k in s}, I) for s in measure.regionprops(labels, coordinates='rc')]

    # Create a pandas DataFrame from the region properties
    df_stat = pd.DataFrame([{k:v for k,v in stats[i].items() if isinstance(v,(np.int64,int,float,tuple)) and k not in ['slice','local_centroid']} for i in range(len(stats))])
    # Remove the 'bbox' column from the DataFrame and store it separately
    bbox = df_stat.pop("bbox")
    # Drop some unnecessary columns from the DataFrame
    df_stat = df_stat.drop(columns=['bbox_area','centroid','euler_number','label','orientation','perimeter_crofton'])
    # Shift the bounding boxes to the specified origin
    change_origin_vec = np.array(origin[:2] * 2)
    for i in range(len(stats)):
        stats[i]['bbox'] = np.array(stats[i]['bbox']) + change_origin_vec
    bbox = bbox.map(lambda x: np.array(x) + change_origin_vec)

    # Return the region properties, DataFrame, and bounding boxes
    return stats, df_stat, bbox

def make_predictions_detectron(cluster_frame_filtered_cluster,detectron_path,n_threads,n_workers_detectron):
    predictor=load_predictor(detectron_path)
    dmodel = dask.delayed(predictor)
    predictions = [predict(batch, dmodel) for batch in cluster_frame_filtered_cluster['image'].tolist()]
    with Client(threads_per_worker=n_threads,
                    n_workers=n_workers_detectron) as client:
        predictions = dask.compute(*predictions)
        client.close()
    del predictor, dmodel
    gc.collect()
    torch.cuda.empty_cache()
    return predictions

def extract_predict(wsi_file='wsi.npy',
                   out_dir='./out_dir',
                   gpu_id=0,
                   model_root_path="./",
                   seg_model="segmentation_practice/uro_seg/49.checkpoint.pth",#30.
                   uro_model="uro_net_aux_v4/65.epoch.checkpoint.pth",
                   aty_model="aty_net_aux_v3/98.epoch.checkpoint.pth",
                   max_objects=1e5,
                   n_chunks=4,
                   scaler_path="/dartfs/rc/nosnapshots/V/VaickusL-nb/users/jlevy/autoparis/clusters/notebooks/inputs/robust_scaler.pkl",
                   detectron_path="/dartfs/rc/nosnapshots/V/VaickusL-nb/users/jlevy/autoparis/clusters/notebooks",
                   n_workers_preprocess=16,
                   n_workers_detectron=4,
                   n_threads=16,
                   batch_size_seg=128,
                   batch_size_class=256,
                   compression=1.,
                   interpolation="INTER_CUBIC",
                   export_data=False,
                   zindex=-1):
    interpolation=getattr(cv2,interpolation,cv2.INTER_CUBIC)
    if gpu_id!=-1: os.environ['CUDA_VISIBLE_DEVICES']=f"{gpu_id}"
    # device = cuda.get_current_device()
    basename=os.path.splitext(os.path.basename(wsi_file))[0]
    seg_model=os.path.join(model_root_path,seg_model)
    uro_model=os.path.join(model_root_path,uro_model)
    aty_model=os.path.join(model_root_path,aty_model)
    out_file=os.path.join(out_dir,f"{basename}{'_z'+str(zindex) if zindex!=-1 else ''}.pkl")

    # LOAD and CHUNK IMAGE
    I = load_image(wsi_file)
    if len(I.shape)==4: # account for z-stack for now, will update
        I=I[(0 if zindex==-1 else zindex)]
    else: zindex=-1
    if compression>1:
        I = cv2.resize(I,None,fx=1/compression,fy=1/compression,interpolation=interpolation)
    I = cv2.copyMakeBorder(I,1,1,1,1,cv2.BORDER_CONSTANT,value=[255,255,255])
    chunks=chunk_wsi(I, I.shape[0]//n_chunks, I.shape[1]//n_chunks)

    # EXTRACT CLUSTERS FROM CHUNKS
    res_=[]
    for c in chunks:
        c.update(max_objects=max_objects)
        res_.append(dask.delayed(process_image)(**c))
    with ProgressBar():
        res_=dask.compute(*res_,scheduler='processes' if len(res_)>1 else 'single-threaded',num_workers=min(n_chunks**2,n_workers_preprocess) if len(res_)>1 else None)
    stats=list(reduce(lambda x,y:x+y,[r[0] for r in res_]))
    df_stat=pd.concat([r[1] for r in res_]).reset_index(drop=True)
    bbox=pd.concat([r[2] for r in res_]).reset_index(drop=True)
    # device.reset()
    if export_data:
        out_file=os.path.join(out_dir,f"{basename}{'_z'+str(zindex) if zindex!=-1 else ''}_data_export.pkl")
        pd.to_pickle(dict(stats=stats,
                 df_stat=df_stat,
                 bbox=bbox),out_file)

    # SPLIT CLUSTER DF INTO CLUSTER AND CELL SPECIFIC FRAME
    cluster_frame=df_stat.copy()
    aux_cols=cluster_frame.columns
    cluster_frame['image']=[s['color_image'] for s in stats]
    cluster_frame['image_shape']=cluster_frame['image'].map(lambda x: x.shape[:2])
    cluster_frame['bbox']=bbox.tolist()
    cluster_frame['cluster_label']=np.arange(len(cluster_frame))
    # cluster_frame=cluster_frame[((cluster_frame['image_shape'].map(lambda x:x[0])>20)+(cluster_frame['image_shape'].map(lambda x:x[1])>20))>0]
    cluster_frame_filtered_single=cluster_frame[cluster_frame['area'].between(256,1800)]
    cluster_frame_filtered_cluster=cluster_frame[cluster_frame['area']>=1800]

    # RUN DETECTRON, CALCULATE CLUSTER LEVEL DENSE REGION
    # memory_target_fraction=0.95,
    # memory_limit='20GB')
    # client.restart()


    cluster_frame_filtered_single['instances'] = np.nan
    cluster_frame_filtered_cluster['instances'] = make_predictions_detectron(cluster_frame_filtered_cluster,detectron_path,n_threads,n_workers_detectron)
    cluster_frame_filtered_cluster['instances_filtered']=cluster_frame_filtered_cluster['instances'].map(clean_instance).map(clean_instance_v2)
    cluster_frame_filtered_cluster_downstream=cluster_frame_filtered_cluster[cluster_frame_filtered_cluster['instances_filtered'].map(len)>0]
    cluster_frame_filtered_cluster_downstream['dense']=cluster_frame_filtered_cluster_downstream['instances_filtered'].map(lambda x: sum(x.pred_classes.numpy()==1)>0)
    cluster_frame_filtered_cluster_downstream['dense_area']=0
    cluster_frame_filtered_cluster_downstream.loc[cluster_frame_filtered_cluster_downstream['dense'],'dense_area']=cluster_frame_filtered_cluster_downstream.loc[cluster_frame_filtered_cluster_downstream['dense'],'instances_filtered'].map(lambda x: gen_pred_mask(x).sum())
    # device.reset()

    current_device=torch.ones((0,0),device='cuda').device.index
    if current_device!=-1:
        for process in gpustat.GPUStatCollection.new_query()[current_device].processes:
            if process['pid']!=CURRENT_PROCESS:
                try: os.system(f"kill {process['pid']}")
                except: pass

    torch.cuda.reset_peak_memory_stats(current_device)
    torch.cuda.set_per_process_memory_fraction(1.)

    # ISOLATE INDIVIDUAL CELLS FROM CLUSTERS
    imgs_extract=[]
    stats_cl=[]
    filtered_images=cluster_frame_filtered_cluster_downstream['image']
    filtered_instances=cluster_frame_filtered_cluster_downstream['instances_filtered']
    cluster_labels=cluster_frame_filtered_cluster_downstream['cluster_label']
    for i in range(len(filtered_images)):
        im=filtered_images.iloc[i]
        for j in range(len(filtered_instances.iloc[i])):
            if filtered_instances.iloc[i].pred_classes[j]!=1:
                ymin,xmin,ymax,xmax=np.round(filtered_instances.iloc[i].pred_boxes.tensor[j]).numpy().astype(int)
                msk=filtered_instances.iloc[i].pred_masks[j].numpy()[xmin:xmax,ymin:ymax]
                im_=im[xmin:xmax,ymin:ymax].copy()
                im_[~msk]=255
                props=measure.regionprops(msk.astype(int))[0]
                stats_cl.append({k:props[k] for k in props})
                stats_cl[-1]["cluster_label"]=cluster_labels.iloc[i]
                stats_cl[-1]['cluster_bbox']=[xmin,ymin,xmax,ymax]
                imgs_extract.append(im_)
    stats_cl_df=pd.DataFrame(stats_cl)
    stats_cl_df['image']=imgs_extract

    # RUN SINGLE CELL INFERENCE
    scaler=pd.read_pickle(scaler_path)
    seg_datasets=dict(cluster_cells=WSI_Dataset_v2(stats_cl_df['image'],generate_transformers(256,256)['test'],include_all=True),
                     isolated_cells=WSI_Dataset_v2(cluster_frame_filtered_single['image'],generate_transformers(256,256)['test'],include_all=True))
    aty_datasets=dict(cluster_cells=WSI_Dataset_v2(stats_cl_df['image'],generate_transformers(224,256)['test'],include_all=True, aux_data=stats_cl_df[aux_cols], scaler=scaler),
                     isolated_cells=WSI_Dataset_v2(cluster_frame_filtered_single['image'],generate_transformers(224,256)['test'],include_all=True, aux_data=cluster_frame_filtered_single[aux_cols], scaler=scaler))

    Y_seg={k:train_model(inputs_dir='segmentation_practice/inputs',
                    architecture='resnet50',
                    batch_size=batch_size_seg,
                    num_classes=3,
                    predict=True,
                    model_save_loc=seg_model,
                    predictions_save_path='tmp_test.pkl',
                    predict_set='custom',
                    verbose=False,
                    class_balance=False,
                    gpu_id=-1,
                    tensor_dataset=False,
                    semantic_segmentation=True,
                    custom_dataset=seg_datasets[k],
                    save_predictions=False) for k in seg_datasets}
    # device.reset()
    gc.collect()
    torch.cuda.empty_cache()

    Y_class1={k:train_model(inputs_dir='inputs',
                    architecture='resnet50',
                    batch_size=batch_size_class,
                    predict=True,
                    num_classes=6,
                    model_save_loc=uro_model,
                    predictions_save_path='tmp_test.pkl',
                    predict_set='custom',
                    verbose=False,
                    class_balance=False,
                    gpu_id=-1,
                    tensor_dataset=False,
                    pickle_dataset=True,
                    semantic_segmentation=False,
                    custom_dataset=aty_datasets[k],
                    save_predictions=False) for k in aty_datasets}

    gc.collect()
    torch.cuda.empty_cache()

    Y_class2={k:train_model(inputs_dir='inputs',
            architecture='resnet50',
            batch_size=batch_size_class,
            predict=True,
            num_classes=2,
            model_save_loc=aty_model,
            predictions_save_path='tmp_test.pkl',
            predict_set='custom',
            verbose=False,
            class_balance=False,
            gpu_id=-1,
            tensor_dataset=False,
            pickle_dataset=True,
            semantic_segmentation=False,
            custom_dataset=aty_datasets[k],
            save_predictions=False) for k in aty_datasets}

    gc.collect()
    torch.cuda.empty_cache()

    # COMPILE ATYPIA AND NC Scores
    Y=dict()
    nc_calls=dict()
    nc_ratio=dict()
    atypia_score=dict()
    for k in seg_datasets:
        Y[k]=dict(seg=Y_seg[k]['pred'],
                  class1=Y_class1[k]['pred'],
                  class2=Y_class2[k]['pred'])
        nc_calls[k]=[y.argmax(0).flatten() for y in Y[k]['seg']]
        nc_ratio[k],atypia_score[k]=pd.concat([pd.Series(y).value_counts() for y in nc_calls[k]],axis=1).T.fillna(0),softmax(Y[k]['class2'],1)[:,1]

    stats_cl_df['aty']=atypia_score['cluster_cells']
    stats_cl_df['nc_ratio']=(nc_ratio['cluster_cells'][2]/nc_ratio['cluster_cells'][[1,2]].sum(1)).values
    stats_cl_df['nuclear_area']=nc_ratio['cluster_cells'][2].values
    stats_cl_df['cyto_area']=nc_ratio['cluster_cells'][1].values
    cell_classes=[f"cell_class_{i}" for i in range(Y['cluster_cells']['class1'].shape[1])]
    cell_assign_df=pd.DataFrame(softmax(Y['cluster_cells']['class1'],1),columns=cell_classes)
    for k in cell_assign_df.columns: stats_cl_df[k]=cell_assign_df[k].values
    change_bbox_coords_df=stats_cl_df[['cluster_label','cluster_bbox']].merge(cluster_frame_filtered_cluster_downstream[['bbox','cluster_label']],on='cluster_label',how='left')
    stats_cl_df['new_bbox']=(change_bbox_coords_df['cluster_bbox'].map(np.array)+change_bbox_coords_df['bbox'].map(lambda x: np.array(x[0:2].tolist()*2))).tolist()

    cluster_frame_filtered_single['aty']=atypia_score['isolated_cells']
    cluster_frame_filtered_single['nc_ratio']=(nc_ratio['isolated_cells'][2]/nc_ratio['isolated_cells'][[1,2]].sum(1)).values
    cluster_frame_filtered_single['nuclear_area']=nc_ratio['isolated_cells'][2].values
    cluster_frame_filtered_single['cyto_area']=nc_ratio['isolated_cells'][1].values
    cell_classes=[f"cell_class_{i}" for i in range(Y['isolated_cells']['class1'].shape[1])]
    cell_assign_df=pd.DataFrame(softmax(Y['isolated_cells']['class1'],1),columns=cell_classes)
    for k in cell_assign_df.columns: cluster_frame_filtered_single[k]=cell_assign_df[k].values

    # COMPILE FINAL RIF TABLES: CLUSTERS, CLUSTER-EXTRACTED SINGLE CELL, ISOLATED SINGLE CELL
    final_cluster_frame=cluster_frame_filtered_cluster_downstream[np.hstack([aux_cols,['bbox', 'cluster_label', 'dense','dense_area']])]
    final_cluster_single_cell_frame=stats_cl_df[np.hstack([aux_cols,['bbox', 'new_bbox', 'cluster_label', 'aty','nc_ratio','nuclear_area','cyto_area'],cell_classes])]
    final_single_cell_frame=cluster_frame_filtered_single[np.hstack([aux_cols,['bbox', 'cluster_label', 'aty','nc_ratio','nuclear_area','cyto_area'],cell_classes])]

    pd.to_pickle(dict(isolated_cells=final_single_cell_frame,
                 cluster_cells=final_cluster_single_cell_frame,
                 clusters=final_cluster_frame),out_file)

def main():
    fire.Fire(extract_predict)

if __name__=="__main__":
    main()

