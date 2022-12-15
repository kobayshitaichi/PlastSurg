from datasets.PlastSurg_mstcn import PlastSurg
from utils.utils import torch_fix_seed
from utils.config import get_config
from utils.metric_helper import AccuracyStages, RecallOverClasse, PrecisionOverClasses
import yaml
import warnings
warnings.simplefilter('ignore')
import numpy as np
import wandb
import logging
from pathlib import Path
import pickle
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.utils import _input_format_classification
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
# implementation adapted from:
# https://github.com/yabufarha/ms-tcn/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MultiStageModel(nn.Module):
    def __init__(self, hparams):
        self.num_stages = hparams.mstcn_stages  # 4 #2
        self.num_layers = hparams.mstcn_layers  # 10  #5
        self.num_f_maps = hparams.mstcn_f_maps  # 64 #64
        self.dim = hparams.mstcn_f_dim  #2048 # 2048
        self.num_classes = hparams.out_features  # 7
        self.causal_conv = hparams.mstcn_causal_conv
        print(
            f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
            f" {self.num_f_maps}, dim: {self.dim}")
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes,
                                       causal_conv=self.causal_conv)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes,
                                 self.num_classes,
                                 causal_conv=self.causal_conv))
            for s in range(self.num_stages - 1)
        ])
        self.smoothing = False

    def forward(self, x):
        out_classes = self.stage1(x)
        outputs_classes = out_classes.unsqueeze(0)
        for s in self.stages:
            out_classes = s(F.softmax(out_classes, dim=1))
            outputs_classes = torch.cat(
                (outputs_classes, out_classes.unsqueeze(0)), dim=0)
        return outputs_classes


class SingleStageModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 causal_conv=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2**i,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out_classes = self.conv_out_classes(out)
        return out_classes

class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 causal_conv=False,
                 kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(dilation *
                                                   (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=dilation,
                                          dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)

class DilatedSmoothLayer(nn.Module):
    def __init__(self, causal_conv=True):
        super(DilatedSmoothLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation1 = 1
        self.dilation2 = 5
        self.kernel_size = 5
        if self.causal_conv:
            self.conv_dilated1 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation1 * 2 * 2,
                                           dilation=self.dilation1)
            self.conv_dilated2 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation2 * 2 * 2,
                                           dilation=self.dilation2)

        else:
            self.conv_dilated1 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation1 * 2,
                                           dilation=self.dilation1)
            self.conv_dilated2 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation2 * 2,
                                           dilation=self.dilation2)
        self.conv_1x1 = nn.Conv1d(7, 7, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x1 = self.conv_dilated1(x)
        x1 = self.conv_dilated2(x1[:, :, :-4])
        out = F.relu(x1)
        if self.causal_conv:
            out = out[:, :, :-((self.dilation2 * 2) * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)

class TeCNO(LightningModule):
    def __init__(self, hparams, model, dataset):
        super(TeCNO, self).__init__()
        self.hparams__ = hparams
        self.batch_size = hparams.batch_size
        self.dataset = dataset
        self.model = model
        self.weights_train = np.asarray(self.dataset.weights["train"])
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.weights_train).float())
        self.init_metrics()

    def init_metrics(self):
        self.train_acc_stages = AccuracyStages(num_stages=self.hparams__.mstcn_stages)
        self.val_acc_stages = AccuracyStages(num_stages=self.hparams__.mstcn_stages)
        self.max_acc_last_stage = {"epoch":0, "acc":0}
        self.max_acc_global = {"epoch":0, "acc":0, "stage":0, "last_stage_max_acc_is_global": False}
        self.precision_metric = PrecisionOverClasses(num_classes=self.hparams__.out_features)
        self.recall_metric = RecallOverClasse(num_classes=self.hparams__.out_features)

    def forward(self, x):
        video_fe = x.transpose(2, 1)
        y_classes = self.model.forward(video_fe)
        y_classes = torch.softmax(y_classes, dim=2)
        return y_classes

    def loss_function(self, y_classes, labels):
        stages = y_classes.shape[0]
        clc_loss = 0
        for j in range(stages):  ### make the interuption free stronge the more layers.
            p_classes = y_classes[j].squeeze().transpose(1, 0)
            ce_loss = self.ce_loss(p_classes, labels.squeeze())
            clc_loss += ce_loss
        clc_loss = clc_loss / (stages * 1.0)
        return clc_loss

    def get_class_acc(self, y_true, y_classes):
        y_true = y_true.squeeze()
        y_classes = y_classes.squeeze()
        y_classes = torch.argmax(y_classes, dim=0)
        acc_classes = torch.sum(
            y_true == y_classes).float() / (y_true.shape[0] * 1.0)
        return acc_classes

    def get_class_acc_each_layer(self, y_true, y_classes):
        y_true = y_true.squeeze()
        accs_classes = []
        for i in range(y_classes.shape[0]):
            acc_classes = self.get_class_acc(y_true, y_classes[i, 0])
            accs_classes.append(acc_classes)
        return accs_classes


    '''def log_precision_and_recall(self, precision, recall, step):
        for n,p in enumerate(precision):
            if not p.isnan():
                self.log(f"{step}_precision_{self.dataset.class_labels[n]}",p ,on_step=True, on_epoch=True)
        for n,p in enumerate(recall):
            if not p.isnan():
                self.log(f"{step}_recall_{self.dataset.class_labels[n]}",p ,on_step=True, on_epoch=True)'''

    def calc_precision_and_recall(self, y_pred, y_true, step="val"):
        y_max_pred, y_true = _input_format_classification(y_pred[-1], y_true, threshold=0.5)
        precision = self.precision_metric(y_max_pred, y_true)
        recall = self.recall_metric(y_max_pred, y_true)
        #if step == "val":
        #    self.log_precision_and_recall(precision, recall, step=step)
        return precision, recall

    def log_average_precision_recall(self, outputs, step="val"):
        precision_list = [o["precision"] for o in outputs]
        recall_list = [o["recall"] for o in outputs]
        x = torch.stack(precision_list)
        y = torch.stack(recall_list)
        phase_avg_precision = [torch.mean(x[~x[:, n].isnan(), n]) for n in range(x.shape[1])]
        phase_avg_recall = [torch.mean(y[~y[:, n].isnan(), n]) for n in range(x.shape[1])]
        phase_avg_precision = torch.stack(phase_avg_precision)
        phase_avg_recall = torch.stack(phase_avg_recall)
        phase_avg_precision_over_video = phase_avg_precision[~phase_avg_precision.isnan()].mean()
        phase_avg_recall_over_video = phase_avg_recall[~phase_avg_recall.isnan()].mean()
        self.log(f"{step}_avg_precision", phase_avg_precision_over_video, on_epoch=True, on_step=False)
        self.log(f"{step}_avg_recall", phase_avg_recall_over_video, on_epoch=True, on_step=False)

    def training_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch
        y_pred = self.forward(stem)
        loss = self.loss_function(y_pred, y_true)
        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        precision, recall = self.calc_precision_and_recall(y_pred, y_true, step="train")
        acc_stages=self.train_acc_stages(y_pred, y_true)
        acc_stages_dict = {f"train_S{s+1}_acc":acc_stages[s] for s in range(len(acc_stages))}
        acc_stages_dict["train_acc"] = acc_stages_dict.pop(f"train_S{len(acc_stages)}_acc") # Renaming metric of last Stage
        self.log_dict(acc_stages_dict, on_epoch=True, on_step=False)
        return {"loss":loss, "precision": precision, "recall": recall}


    def training_epoch_end(self, outputs):
        self.log_average_precision_recall(outputs, step="train")


    def validation_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch
        y_pred = self.forward(stem)
        val_loss = self.loss_function(y_pred, y_true)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, on_step=False)
        precision, recall = self.calc_precision_and_recall(y_pred, y_true, step="val")
        self.val_acc_stages(y_pred, y_true)
        acc_stages = self.val_acc_stages.compute()
        metric_dict = {f"val_S{s + 1}_acc": acc_stages[s] for s in range(len(acc_stages))}
        metric_dict["val_acc"] = metric_dict.pop(f"val_S{len(acc_stages)}_acc") # Renaming metric of last Stage
        self.log_dict(metric_dict, on_epoch=True, on_step=False)
        metric_dict["precision"] = precision
        metric_dict["recall"] = recall
        return metric_dict


    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        self.max_acc_last_stage = {"epoch": 0, "acc": 0}
        self.max_acc_global = {"epoch": 0, "acc": 0 , "stage": 0}
        """
        val_acc_stage_last_epoch = torch.stack([o["val_acc"] for o in outputs]).mean()

        if val_acc_stage_last_epoch > self.max_acc_last_stage["acc"]:
            self.max_acc_last_stage["acc"] = val_acc_stage_last_epoch
            self.max_acc_last_stage["epoch"] = self.current_epoch

        self.log("val: max acc last Stage", self.max_acc_last_stage["acc"])
        self.log_average_precision_recall(outputs, step="val")




    def test_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch
        y_pred = self.forward(stem)
        val_loss = self.loss_function(y_pred, y_true)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, on_step=False)
        precision, recall = self.calc_precision_and_recall(y_pred, y_true, step="test")
        self.val_acc_stages(y_pred, y_true)
        acc_stages = self.val_acc_stages.compute()
        metric_dict = {f"test_S{s + 1}_acc": acc_stages[s] for s in range(len(acc_stages))}
        metric_dict["test_acc"] = metric_dict.pop(f"test_S{len(acc_stages)}_acc") # Renaming metric of last Stage
        self.log_dict(metric_dict, on_epoch=True, on_step=False)
        metric_dict["precision"] = precision
        metric_dict["recall"] = recall
        metric_dict["y_hat"] = y_hat
        metric_dict["y_true"] = y_true
        return metric_dict


    def test_epoch_end(self, outputs):
        test_acc = torch.stack([o["test_acc"] for o in outputs]).mean()
        self.log("test_acc", test_acc)
        self.log_average_precision_recall(outputs, step="test")
        y_hat = torch.cat([val['y_hat'] for val in outputs], dim=0)
        y_hat = torch.argmax(y_hat, dim=0)
        y_true = torch.cat([val['y_true'] for val in outputs], dim=0)
        
        dir_path = Path(self.hparams__.mstcn_output_path) / Path(self.hparams__.name)
        dir_path.mkdir(exist_ok=True)
        y_path = dir_path / 'y.pkl'
        y_true_path = dir_path / 'y_true.pkl'
        # with open(y_path, mode='rb') as f:
        #     tmp = pickle.load(y_path)
        # tmp.append(y_hat)
        # with open(y_path, mode='wb') as f:
        #     pickle.dump(tmp, f)
        # with open(y_true_path, mode='rb') as f:     
        #     tmp = pickle.load(y_true_path)
        # tmp.append(y_true)
        # with open(y_true_path, mode='wb') as f:     
        #     pickle.dump(tmp,f)
        
      


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams__.mstcn_learning_rate)
        return [optimizer]  #, [scheduler]

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        should_shuffle = False
        if split == "train":
            should_shuffle = True
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)
            should_shuffle = False
        print(f"split: {split} - shuffle: {should_shuffle}")
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams__.batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.hparams__.num_workers,
            pin_memory=True,
        )
        return loader

    def train_dataloader(self):
        dataloader = self.__dataloader(split="train")
        logging.info("training data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        logging.info("test data loader called  - size: {}".format(
            len(dataloader.dataset)))
        return dataloader
if __name__ == '__main__':
    torch_fix_seed()
    config_path = '/media/aolab/untitiled/workspace/PlastSurg/config/config_tecno.yaml'
    d = {}
    with open(config_path, mode="r") as f:
        d = yaml.load(f)
    hyperparams = get_config(config_path)
    wandb.init(
        project="PlastSurg",
        name='tecno_'+hyperparams.name,
        config=d
    )
    hyperparams = get_config(config_path)
    dataset = PlastSurg(hyperparams) 
    hyperparams = get_config(config_path)
    model = MultiStageModel(hyperparams)
    module = TeCNO(hyperparams,model,dataset)
    logging.disable(logging.WARNING)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hyperparams.mstcn_output_path}/checkpoints/",
        save_top_k=hyperparams.save_top_k,
        verbose=True,
        monitor=hyperparams.mstcn_early_stopping_metric,
        mode='max',
        prefix=hyperparams.name,
        filename=f'{{epoch}}-{{{hyperparams.mstcn_early_stopping_metric}:.2f}}'
    )
    early_stop_callback = EarlyStopping(
        monitor=hyperparams.mstcn_early_stopping_metric,
        min_delta=0.00,
        patience=3,
        mode='max')
    tb_logger = TensorBoardLogger(hyperparams.mstcn_output_path, name='tb')
    wandb_logger = WandbLogger(name = 'tecno'+hyperparams.name, project="PlastSurg")
    loggers = [tb_logger, wandb_logger]
    hyperparams.mstcn_min_epochs
    trainer = Trainer(
        gpus=hyperparams.gpus,
        logger=loggers,
        min_epochs=hyperparams.mstcn_min_epochs,
        max_epochs=hyperparams.mstcn_max_epochs,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stop_callback],
        weights_summary='full',
        num_sanity_val_steps=hyperparams.num_sanity_val_steps,
    )
    trainer.fit(module)
    trainer.test(module)
    #print(module.y_true,module.y_hat)

