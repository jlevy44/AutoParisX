import pandas as pd
import dask, glob,os
from dask.diagnostics import ProgressBar
from functools import reduce
import os
import numpy as np
import fire

OPTIMIZED_THRESHOLDS=[0.7822224025503274,
 0.5822194967338957,
 0.14988769237238614,
 0.7867505956281785,
 0.883978541769089,
 3.1002408550284746,
 3628,
 4129,
 7121,
 13724,
 0.1]

OPTIMIZED_THRESHOLDS=dict(zip(["aty_thres",
                  "nc_thres",
                  "uro_thres",
                  "cl_cutoff_aty",
                  "cl_cutoff_nc",
                  "cl_cutoff_n_cell",
                  "dense_small_min",
                  "dense_small_max",
                  "dense_large_min",
                  "dense_large_max",
                  "cl_cutoff_aty_nc"],
                  OPTIMIZED_THRESHOLDS))

def series_mean(ds,test_val=None,is_sum=False):
    if test_val is not None: ds=ds>=test_val
    if len(ds):
        if is_sum: return ds.sum()
        return ds.mean()
    return 0

def generate_cluster_frame(d_,uro_thres=0.9,aty_thres=0.,nc_thres=0.2):
    cluster_cells=d_['cluster_cells'].copy()
    cluster_cells=cluster_cells[d_['cluster_cells']['cell_class_0']>=uro_thres]
    
    cluster_cells['is_aty']=cluster_cells['aty']>=aty_thres
    cluster_cells['is_nc']=cluster_cells['nc_ratio']>=nc_thres
    cluster_cells['is_aty_nc']=np.logical_and(cluster_cells['is_aty'],cluster_cells['is_nc'])

    is_aty=(cluster_cells['is_aty'].groupby(cluster_cells["cluster_label"])).mean().reset_index()
    is_aty.columns=['cluster_label','aty']
    
    is_nc=(cluster_cells['is_nc'].groupby(cluster_cells["cluster_label"])).mean().reset_index()
    is_nc.columns=['cluster_label','nc']

    is_aty_nc=(cluster_cells['is_aty_nc'].groupby(cluster_cells["cluster_label"])).mean().reset_index()
    is_aty_nc.columns=['cluster_label','aty_nc']
    
    n_cells=cluster_cells['aty'].groupby(cluster_cells["cluster_label"]).size().reset_index()
    n_cells.columns=['cluster_label','n_cells']
    cluster_frame=reduce(lambda x,y: pd.merge(x,y,on='cluster_label'),[cluster_cells.drop(columns='aty'),is_aty,is_nc,is_aty_nc,n_cells]).fillna(0.)
    return cluster_frame

def generate_data(dff_cluster,
                  dff_cluster_cells_orig,
                  dff_isolated_cells_orig,
                  aty_thres,
                  nc_thres,
                  uro_thres,
                  cl_cutoff_aty,
                  cl_cutoff_nc,
                  cl_cutoff_n_cell,
                  dense_small_min,
                  dense_small_max,
                  dense_large_min,
                  dense_large_max,
                  cl_cutoff_aty_nc,
                  return_all_data=False,
                  **kwargs):
    dff_cluster_cells=dff_cluster_cells_orig.copy()
    dff_cluster_cells=dff_cluster_cells[dff_cluster_cells['cell_class_0']>=uro_thres]
    dff_cluster_cells=dff_cluster_cells[dff_cluster_cells['nc_ratio']>0]
    dff_cluster_cells['is_aty']=dff_cluster_cells['aty']>=aty_thres
    dff_cluster_cells['is_nc']=dff_cluster_cells['nc_ratio']>=nc_thres
    dff_cluster_cells['is_aty_nc']=np.logical_and(dff_cluster_cells['is_aty'],dff_cluster_cells['is_nc'])
    dff_cluster_cells['n_cells']=1

    dff_cluster_cells_frame=(dff_cluster_cells.groupby(['basename','cluster_label'])[['is_aty','is_nc',"is_aty_nc",'n_cells']].agg(is_aty=("is_aty",np.mean),is_nc=("is_nc",np.mean),is_aty_nc=("is_aty_nc",np.mean),n_cells=("n_cells",np.sum))>=np.array([cl_cutoff_aty,cl_cutoff_nc,cl_cutoff_aty_nc,cl_cutoff_n_cell])).reset_index()
    dff_cluster_cells_frame=pd.merge(dff_cluster[['basename','cluster_label','dense_area']],dff_cluster_cells_frame,on=["basename",'cluster_label'],how="outer").fillna(0.)

    dff_isolated_cells=dff_isolated_cells_orig.copy()
    dff_isolated_cells=dff_isolated_cells[dff_isolated_cells['cell_class_0']>=uro_thres]
    dff_isolated_cells=dff_isolated_cells[dff_isolated_cells['nc_ratio']>0]
    dff_isolated_cells['is_aty']=dff_isolated_cells['aty']>=aty_thres
    dff_isolated_cells['is_nc']=dff_isolated_cells['nc_ratio']>=nc_thres
    dff_isolated_cells['is_aty_nc']=np.logical_and(dff_isolated_cells['is_aty'],dff_isolated_cells['is_nc'])
    dff_isolated_cells['n_cells']=1

    dff_all_cells=pd.concat([dff_cluster_cells,dff_isolated_cells])
    
    cell_scores=dict()
    cell_scores['all_cells']=dff_all_cells.groupby(['basename'])[['is_aty','is_nc',"is_aty_nc",'n_cells']].sum().reset_index()
    cell_scores['ecc']=dff_all_cells.groupby(['basename'])[['eccentricity']].mean().reset_index()
    cell_scores['iso_cells']=dff_isolated_cells.groupby(['basename'])[['is_aty','is_nc',"is_aty_nc"]].sum().reset_index()
    cell_scores['cluster_cells']=dff_cluster_cells.groupby(['basename'])[['is_aty','is_nc',"is_aty_nc"]].sum().reset_index()
    cell_scores['dense']=dff_cluster['dense_area'].between(dense_small_min,dense_small_max).groupby(dff_cluster['basename']).sum().reset_index()
    cell_scores['dense_large']=dff_cluster['dense_area'].between(dense_large_min,dense_large_max).groupby(dff_cluster['basename']).sum().reset_index()
    cell_scores['n_cluster']=dff_cluster_cells_frame['n_cells'].groupby(dff_cluster_cells_frame['basename']).size().reset_index()
    cell_scores['n_cluster'].columns=['basename','n_clusters']
    cell_scores['cluster_aty_dense']=dff_cluster_cells_frame['dense_area'].between(dense_small_min,dense_small_max).eq(dff_cluster_cells_frame['is_aty']).groupby(dff_cluster_cells_frame['basename']).mean().reset_index()
    cell_scores['cluster_aty_dense'].columns=['basename','is_dense']
    cell_scores['cluster_aty']=dff_cluster_cells_frame['is_aty'].groupby(dff_cluster_cells_frame['basename']).sum().reset_index()
    cell_scores['cluster_aty'].columns=['basename','is_aty_cluster']
    cell_scores_df=reduce(lambda x,y:pd.merge(x,y,on="basename",how="outer").fillna(0.),cell_scores.values())
    if return_all_data: 
        return dict(dff_all_cells=dff_all_cells,
                    dff_cluster_cells=dff_cluster_cells,
                    dff_isolated_cells=dff_isolated_cells,
                    dff_cluster_cells_frame=dff_cluster_cells_frame,
                   dff_cluster=dff_cluster)
    return cell_scores_df


def generate_sif_scores(input_files="results/*",
                        demo_df_file="",
                        out_file="sif_scores.pkl"):

    with ProgressBar():
        d=dict(dask.compute(*[(os.path.basename(f).replace(".pkl",""),dask.delayed(pd.read_pickle)(f)) for f in glob.glob(input_files)],scheduler="single-threaded"))#

    for k in d: 
        d[k]["cluster_cells"]['basename']=k
        d[k]["isolated_cells"]['basename']=k
        d[k]["clusters"]['basename']=k
        
    dff_cluster=pd.concat([d[k]['clusters'] for k in d if "-orig" not in k])
    dff_cluster_cells_orig=pd.concat([d[k]['cluster_cells'] for k in d if "-orig" not in k])
    dff_isolated_cells_orig=pd.concat([d[k]['isolated_cells'] for k in d if "-orig" not in k])

    cell_scores_df = generate_data(dff_cluster,
                    dff_cluster_cells_orig,
                    dff_isolated_cells_orig,
                    **OPTIMIZED_THRESHOLDS).reset_index()
    cell_scores_df['surg_id']=cell_scores_df['basename']

    if demo_df_file and os.path.exists(demo_df_file):
        demo_df=pd.read_csv(demo_df_file)
        demo_df=demo_df[['specimen_source_voided','history_hematuria','age','gender','surg_id']+(['final_dx'] if 'final_dx' in demo_df.columns else [])]
        cell_scores_df=cell_scores_df.merge(demo_df,on="surg_id",how="inner")
    cell_scores_df=cell_scores_df.set_index("surg_id").drop(columns=["basename"])

    cell_scores_df.to_pickle(out_file)

def main():
    fire.Fire(generate_sif_scores)

if __name__=="__main__":
    main()