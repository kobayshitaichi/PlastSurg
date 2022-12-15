from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class PlastSurgHelper(Dataset):
    def __init__(self, hparams, data_p, dataset_split=None):
        assert dataset_split != None
        self.hparams = hparams
        self.data_p = data_p
        self.data_root = Path(hparams.root_dir) / "outputs/pickle_export"
        self.number_vids = len(self.data_p)
        self.dataset_split = dataset_split
        self.factor_sampling = int(30 / self.hparams.features_subsampling)
        

    def __len__(self):
        return self.number_vids
    
    def __getitem__(self, index):
        vid_id = index
        p = self.data_root / self.data_p[vid_id]
        unpickled_x = pd.read_pickle(p)
        stem = np.asarray(unpickled_x[0],dtype=np.float32)[::self.factor_sampling]
        y_hat = np.asarray(unpickled_x[1],dtype=np.float32)[::self.factor_sampling]
        y = np.asarray(unpickled_x[2])[::self.factor_sampling]
        return stem, y_hat, y


class PlastSurg:
    def __init__(self,hparams):
        self.name = "outputs"
        self.hparams = hparams
        self.weights = {}
        self.class_labels = [
            "design",
            "anesthesia",
            "incision",
            "dissection",
            "closure",
            "hemostasis"
        ]
        self.out_features = self.hparams.out_features ###
        self.features_per_seconds = self.hparams.features_per_seconds###
        self.factor_sampling = (int(30 / self.hparams.features_subsampling))
        print(
            f"Subsampling features: 30features_ps --> {hparams.features_subsampling}features_ps (factor: {self.factor_sampling})"
        )        
        self.data_p = {}
        self.data_p["train"] = [(
            f"{self.features_per_seconds}fps/video_{i:02d}_{self.features_per_seconds}fps.pkl"
        ) for i in range(1,4)]
        self.data_p["val"] = [(
            f"{self.features_per_seconds}fps/video_{i:02d}_{self.features_per_seconds}fps.pkl"
        ) for i in range(4,5)]
        self.data_p["test"] = [(
            f"{self.features_per_seconds}fps/video_{i:02d}_{self.features_per_seconds}fps.pkl"
        ) for i in range(5,6)]        
        self.data = {}
        for split in ["train", "val", "test"]:
            self.data[split] = PlastSurgHelper(self.hparams,self.data_p[split],dataset_split=split)
        self.weights["train"] = [
            2.59319412, 
            8.80052493, 
            1.65091088, 
            0.58824561, 
            0.41211898,
            1.30112534]
        print(
            f"train size: {self.data['train'].__len__()} - val size: {self.data['val'].__len__()}"
            f"test size: {self.data['test'].__len__()}"
        )