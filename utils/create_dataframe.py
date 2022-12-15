from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
from config import get_config

def create_df(dataset_dir):
    dataset_dir = Path(dataset_dir)
    img_base_path = dataset_dir / "video_split"
    out_path = dataset_dir / "dataframes"
    out_path.mkdir(exist_ok=True)
    
    PlastSurg_df = pd.DataFrame(columns=['image_path', 'time', 'phase', 'tool', 'video_idx'])
    
    class_labels = [
        "design",
        "anesthesia",
        "incision",
        "dissection",
        "closure",
        "hemostasis"
    ]
    
    tool_labels = [
        "skinmarker",
        "syringe",
        "scalpel",
        "scissors",
        "electric cautery",
        "Suture and Needle",
        "bipolar forceps",
    ]

    for i in tqdm(range(1,16)):
        
        img_path_for_vid = img_base_path / f"video{i:02d}"
        img_list = sorted(img_path_for_vid.glob('*.png'))
        img_list = [str(i.relative_to(img_base_path)) for i in img_list]
        img_list_df = pd.DataFrame(img_list,columns=['image_path'])
        
        df = pd.read_csv(dataset_dir / "annotations" / f"video{i:02d}.csv",index_col=0)
        df = img_list_df.merge(df,left_index=True,right_index=True,how="left").dropna()
        df = df[~df['phase'].str.contains('irrelevant_frame')]
        df['video_idx'] = i
        df = pd.get_dummies(df,columns=['tool'],prefix='',prefix_sep='')
        df.drop(['Frame','empty-handed'],axis=1,inplace=True)
        
        for j,p in enumerate(class_labels):
            df['phase'] = df['phase'].map(lambda x: j if x == p else x)
            
        
        PlastSurg_df = PlastSurg_df.append(df, ignore_index=True, sort=False)
        
    PlastSurg_df.fillna(0,inplace=True)          
    for column in tool_labels:
        PlastSurg_df[column] = PlastSurg_df[column].astype(int)
    PlastSurg_df.to_pickle(out_path / "PlastSurg_df.pkl")
        

if __name__ == '__main__':
    hparams = get_config('/media/aolab/untitiled/workspace/PlastSurg/config/config_feature_extract.yaml')
    dataset_dir = hparams.dataset_dir
    create_df(dataset_dir)
    