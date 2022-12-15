import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from albumentations import (
    Compose,
    Resize,
    Normalize,
    ShiftScaleRotate,
)
import torch
from sklearn.utils import class_weight

class PlastSurgDataset:
    def __init__(self,hparams):
        self.hparams = hparams
        self.resize = self.hparams.input_size
        self.fps_sampling = self.hparams.fps_sampling
        self.fps_sampling_test = self.hparams.fps_sampling
        self.dataset_root_dir = Path(hparams.dataset_dir)
        self.transformations = self.__get_transformations()
        self.class_labels = [
            "design",
            "anesthesia",
            "incision",
            "dissection",
            "closure",
            "hemostasis"
        ]
        self.tool_labels = [
            "skinmarker",
            "syringe",
            "scalpel",
            "scissors",
            "electric cautery",
            "Suture and Needle",
            "bipolar forceps",
        ]
        self.label_col = "phase"
        
        # ここは関数にしたほうが見やすい
        # self.df = {'all':df_all, 'train':df_train, 'val':df_val, 'test':df_test}
        self.df = {}
        self.df["all"] = pd.read_pickle(
            self.dataset_root_dir / "dataframes/PlastSurg_df.pkl"
        )
        self.vids_for_training = [i for i in range(1,10)]
        self.vids_for_val = [i for i in range(10,13)]
        self.vids_for_test = [i for i in range(13,16)]
        
        self.df["train"] = self.df["all"][self.df["all"]["video_idx"].isin(
            self.vids_for_training
        )]
        self.df["val"] = self.df["all"][self.df["all"]["video_idx"].isin(
            self.vids_for_val
        )]
        # test_extract == True のときはTestにすべての動画を入れる
        if hparams.test_extract:
            print(
                f"test extract enabled. Test will be used to extract the videos (testset = all)"
            )
            self.vids_for_test = [i for i in range(1, 16)]
            self.df["test"] = self.df["all"]
        else:
            self.df["test"] = self.df["all"][self.df["all"]["video_idx"].isin(
                self.vids_for_test)]
        
        len_org = {
            "train": len(self.df["train"]),
            "val": len(self.df["val"]),
            "test": len(self.df["test"])
        }
        # 学習時のサンプリングフレームレートが30以外のときは再度サンプリングする
        if self.fps_sampling < 30 and self.fps_sampling > 0:
            factor = int(30 / self.fps_sampling)
            print(
                f"Subsampling(factor: {factor}) data: 30fps > {self.fps_sampling}fps"
            )
            self.df["train"] = self.df["train"].iloc[::factor]
            self.df["val"] = self.df["val"].iloc[::factor]
            self.df["all"] = self.df["all"].iloc[::factor]
            for split in ["train", "val"]:
                print(
                    f"{split:>7}: {len_org[split]:8} > {len(self.df[split])}")
        if self.fps_sampling_test < 30 and self.fps_sampling_test > 0:
            factor = int(30 / self.fps_sampling_test)
            print(
                f"Subsampling(factor: {factor}) data: 30fps > {self.fps_sampling_test}fps"
            )
            self.df["test"] = self.df["test"].iloc[::factor]
            split = "test"
            print(f"{split:>7}: {len_org[split]:8} > {len(self.df[split])}")  

        self.data = {}  
        for split in ["train", "val"]:
            self.df[split] = self.df[split].reset_index()
            self.data[split] = Dataset_from_Dataframe(
                        self.df[split],
                        self.transformations[split],
                        self.label_col, #phase_label_col
                        img_root=Path(hparams.dataset_dir) / "video_split",
                        image_path_col="image_path"
                        )
        # here we want to extract all features
        #self.df["test"] = self.df["all"].reset_index()
        self.df["test"] = self.df["test"].reset_index()
        self.data["test"] = Dataset_from_Dataframe(
            self.df["test"],
            self.transformations["test"],
            self.label_col,
            img_root=Path(hparams.dataset_dir) / "video_split",
            image_path_col="image_path",
            add_label_cols=[
                        "video_idx",
                        "image_path",
                        "index",
                        "skinmarker",
                        "syringe",
                        "scalpel",
                        "scissors",
                        "electric cautery",
                        "Suture and Needle",
                        "bipolar forceps",
                    ]
            )

        y = torch.tensor(list(self.df["train"]["phase"]))
        phase_weights = class_weight.compute_class_weight('balanced',np.unique(y),y.numpy())
        self.phase_weights = np.asarray(phase_weights)
        


    def __get_transformations(self):
        norm_mean = [0.3456, 0.2281, 0.2233]
        norm_std = [0.2528, 0.2135, 0.2104]
        normalize = Normalize(mean=norm_mean, std=norm_std)
        training_augmentation = Compose([
            ShiftScaleRotate(shift_limit=0.1,
                             scale_limit=(-0.2, 0.5),
                             rotate_limit=15,
                             border_mode=0,
                             value=0,
                             p=0.7),
        ])

        data_transformations = {}
        data_transformations["train"] = Compose([
            Resize(height=self.resize, width=self.resize),
            training_augmentation,
            normalize,
            ToTensorV2(),
        ])
        data_transformations["val"] = Compose([
            Resize(height=self.resize, width=self.resize),
            normalize,
            ToTensorV2(),
        ])
        data_transformations["test"] = data_transformations["val"]
        return data_transformations        
    
class Dataset_from_Dataframe(Dataset):
    def __init__(self,
                 df,
                 transform,
                 label_col,
                 img_root="",
                 image_path_col="path",
                 add_label_cols=[
                        "skinmarker",
                        "syringe",
                        "scalpel",
                        "scissors",
                        "electric cautery",
                        "Suture and Needle",
                        "bipolar forceps",
                    ]):
        self.df = df
        self.transform = transform
        self.label_col = label_col
        self.image_path_col = image_path_col
        self.img_root = img_root
        self.add_label_cols = add_label_cols

    def __len__(self):
        return len(self.df)

    def load_from_path(self, index):
        img_path_df = self.df.loc[index, self.image_path_col]
        p = self.img_root / img_path_df
        X = Image.open(p)
        X_array = np.array(X)
        return X_array, p

    def __getitem__(self, index):
        X_array, p = self.load_from_path(index)
        if self.transform:
            X = self.transform(image=X_array)["image"]
        label = torch.tensor(int(self.df[self.label_col][index]))
        add_label = []
        for add_l in self.add_label_cols:
            add_label.append(self.df[add_l][index])
        X = X.type(torch.FloatTensor)
        return X, label, add_label
    
