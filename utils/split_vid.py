from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import argparse
from config import get_config


def videos_to_imgs(output_path='',input_path='',pattern='*mp4',fps=30,start_video_index=1):
    dirs = list(input_path.glob(pattern))
    dirs.sort()
    dirs = dirs[start_video_index-1:]
    output_path.mkdir(exist_ok=True)
    
    for i, vid_path in enumerate(tqdm(dirs)):
        file_name = vid_path.stem
        out_folder = output_path / file_name
        out_folder.mkdir(exist_ok=True)
        os.system(
            f'ffmpeg -i {vid_path} -vf "scale=250:250, fps=30" {out_folder/file_name}_%6d.png '
        )
        print("Done extractin: {}".format(i+1))
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-start_idx')
    args = parser.parse_args()
    start_idx = int(args.start_idx)
    
    hparams = get_config('/media/aolab/untitiled/workspace/PlastSurg/config/config_feature_extract.yaml')
    dataset_dir = hparams.dataset_dir
    dataset_dir = Path(dataset_dir)
    input_path = dataset_dir  / "videos"
    output_path = dataset_dir  / "video_split"
    
    videos_to_imgs(output_path=output_path,input_path=input_path,start_video_index=start_idx)