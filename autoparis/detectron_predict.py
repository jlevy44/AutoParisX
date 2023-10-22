import pandas as pd, numpy as np
import pandas as pd, os
import os
import matplotlib, matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_closing,binary_opening
from skimage.morphology import disk
from skimage.morphology import skeletonize
# from skan import Skeleton, summarize
# from skan.draw import overlay_skeleton_2d_class
# from skan.csr import JunctionModes
# print(skeletonize(Y_seg['pred'][i]>=0.5,method="lee"))
from skimage.morphology import medial_axis
import seaborn as sns
from skimage.feature import peak_local_max
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import random_walker
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.draw import disk as draw_disk
from skimage.io import imread
import matplotlib.pyplot as plt, numpy as np
from PIL import Image
import matplotlib; matplotlib.rcParams['figure.dpi']=300
from torchvision.ops import nms
import fire
try:
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.structures import BoxMode
except:
    import warnings
    warnings.warn("Detectron2 could not be imported, can only export but not process data. Try installing with python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'")

def load_predictor(dirname):
    d=pd.read_pickle(os.path.join(dirname,'config.pkl'))
    thing_classes=d['thing_classes']
    stuff_classes=d['stuff_classes']
    cv_fold=0
    resume=True#d['resume']
    learning_rate=d['learning_rate']
    epochs=500#d['epochs']
    panoptic_dataset_name=d['panoptic_dataset_name']
    annotation_root=d['annotation_root']
    panoptic_root=d['panoptic_root']
    panoptic_json_name=d['panoptic_json_name']
    sem_seg_root=d['sem_seg_root']
    instances_json_name=d['instances_json_name']
    output_dir=d['output_dir']
    use_pretrained_weights=d['use_pretrained_weights']
    panoptic_name = d['panoptic_name']
    instances_json = d['instances_json']
    segmentation_id = d['max_seg_id']
    annotation_id = d['max_annotation_id']

    panoptic_name = panoptic_dataset_name + "_train"
    panoptic_root = os.path.join(dirname,annotation_root, panoptic_root)
    panoptic_json = os.path.join(dirname,annotation_root, panoptic_json_name + "_train.json")
    sem_seg_root = os.path.join(dirname,annotation_root, sem_seg_root)
    instances_json = os.path.join(dirname,annotation_root, instances_json_name + "_train.json")

    output_dir=os.path.join(dirname,"output/")
    model_path="model_final.pth"#outputs/
    threshold=0.05
    base_model="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    n=5

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_model))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = n
    cfg.OUTPUT_DIR=output_dir
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    predictor = DefaultPredictor(cfg)
    return predictor

def clean_instance(instances,remove_non_uro=True):
    instances=instances.to("cpu")
    instance_idx=np.arange(len(instances))
    if len(instance_idx):
        keep=nms(boxes = instances.pred_boxes.tensor[instances.pred_classes!=2], scores = instances.scores[instances.pred_classes!=2], iou_threshold=0.4).numpy()
        idx_keep=np.hstack([instance_idx[(instances.pred_classes!=2).numpy()][keep],instance_idx[(instances.pred_classes==2).numpy()]])
        if remove_non_uro: idx_keep=idx_keep[np.isin((instances.pred_classes).numpy()[idx_keep],[0,1,4])]
#         if sum(instances.pred_classes==2): else: idx_keep=instance_idx[(instances.pred_classes!=2).numpy()][keep]
        idx_keep=idx_keep[(instances.scores[idx_keep]>=0.2).numpy()]
        return instances[idx_keep]
    else: return instances


def main(input_file="",dirname=".",output_dir="../AP_cells_instance_predictions/"):
    predictor=load_predictor(dirname)
    imgs=pd.read_pickle(input_file)#"../AP_cells/aty_10_5_15_R_orig.pkl"
    im_shape=imgs.map(lambda x: x.shape[:2])
    imgs_new=imgs[((im_shape.map(lambda x:x[0])>40)+(im_shape.map(lambda x:x[1])>40))>0]
    instances_map=imgs_new.map(predictor)
    actual_instances=instances_map.map(lambda x: x['instances'])
    clean_instances=actual_instances.map(clean_instance)
    clean_instances.to_pickle(os.path.join(output_dir,os.path.basename(input_file)))

#     imgs_new=imgs_new.iloc[(-imgs_new.map(lambda x: np.prod(x.shape[:2]))).argsort()]
#     imgs_new_shape=imgs_new.map(lambda x: x.shape[:2])
if __name__=="__main__":
    fire.Fire(main)
