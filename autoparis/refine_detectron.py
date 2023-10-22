import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import warnings

from PIL import Image
from scipy.special import softmax
from skimage import measure
from sklearn.preprocessing import RobustScaler
from torchvision.ops import nms

from pathpretrain.train_model import generate_kornia_transforms, generate_transformers, train_model
from torch.utils.data import Dataset
warnings.filterwarnings("ignore")

def clean_instance(instances,remove_non_uro=True):
    instances=instances.to("cpu")
    instance_idx=np.arange(len(instances))
    if len(instance_idx):
        keep=nms(boxes = instances.pred_boxes.tensor[instances.pred_classes!=2], scores = instances.scores[instances.pred_classes!=2], iou_threshold=0.1).numpy()
        idx_keep=np.hstack([instance_idx[(instances.pred_classes!=2).numpy()][keep],instance_idx[(instances.pred_classes==2).numpy()]])
        if remove_non_uro: idx_keep=idx_keep[np.isin((instances.pred_classes).numpy()[idx_keep],[0,1,4])]
#         if sum(instances.pred_classes==2): else: idx_keep=instance_idx[(instances.pred_classes!=2).numpy()][keep]
        idx_keep=idx_keep[(instances.scores[idx_keep]>=0.2).numpy()]
        return instances[idx_keep]
    else: return instances

def clean_instance_v2(instances,remove_non_uro=True,prioritize_uro=False,iou_threshold=0.1,aty_reassign=0.2,uro_iou=0.7,keep_class=-1):
    instances=instances.to("cpu")
    instance_idx=np.arange(len(instances))
    if len(instance_idx):
        # keep uro over atypical
        if prioritize_uro:
            uro_class=torch.tensor([0,4])
            tmp_scores=instances.scores[torch.isin(instances.pred_classes,uro_class)]
            tmp_scores[instances.pred_classes[torch.isin(instances.pred_classes,uro_class)]==0]=aty_reassign
            keep=nms(boxes = instances.pred_boxes.tensor[torch.isin(instances.pred_classes,uro_class)], scores = tmp_scores, iou_threshold=uro_iou).numpy()
    #         print(keep)
            keep=np.union1d(keep,np.where(instances.pred_classes[torch.isin(instances.pred_classes,uro_class)]==4)[0])
            idx_keep=np.hstack([instance_idx[torch.isin(instances.pred_classes,uro_class).numpy()][keep],instance_idx[~torch.isin(instances.pred_classes,uro_class)]])
            instances=instances[idx_keep]
            instance_idx=np.arange(len(instances))

        keep=nms(boxes = instances.pred_boxes.tensor[instances.pred_classes!=keep_class], scores = instances.scores[instances.pred_classes!=keep_class], iou_threshold=iou_threshold).numpy()
        idx_keep=np.hstack([instance_idx[(instances.pred_classes!=keep_class).numpy()][keep],instance_idx[(instances.pred_classes==keep_class).numpy()]])
        if remove_non_uro: idx_keep=idx_keep[np.isin((instances.pred_classes).numpy()[idx_keep],[0,1,4])]
#         if sum(instances.pred_classes==2): else: idx_keep=instance_idx[(instances.pred_classes!=2).numpy()][keep]
        idx_keep=idx_keep[(instances.scores[idx_keep]>=0.2).numpy()]
        return instances[idx_keep]
    else: return instances
