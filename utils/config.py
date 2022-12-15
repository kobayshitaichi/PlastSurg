import yaml
import dataclasses

@dataclasses.dataclass(frozen=True)
class Config:
    root_dir : str
    dataset_dir : str
    model_name : str
    test_extract : bool
    fps_sampling : int
    fps_sampling_test : int
    batch_size : int
    num_tasks : int
    num_workers : int
    num_sanity_val_steps : int
    input_size : int
    out_features : int
    tool_features : int
    features_subsampling : int
    features_per_seconds : int
    learning_rate : float
    early_stopping_metric : str
    pretrained : bool
    save_top_k : int
    max_epocks : int
    min_epocks : int
    gpus : list
    feature_output_path : str
    name : str
    features_per_seconds : int
    features_subsampling : int
    log_every_n_steps : int
    add_tool_feats : bool


@dataclasses.dataclass(frozen=True)
class Config_tecno:
    root_dir : str
    batch_size : int  
    gpus : list
    out_features : int
    num_workers : int
    save_top_k : int
    early_stopping_metric : str
    feature_output_path : str
    mstcn_output_path : str
    log_every_n_steps : int
    mstcn_causal_conv : bool
    mstcn_learning_rate : float
    mstcn_min_epochs : int
    mstcn_max_epochs : int
    mstcn_layers : int
    mstcn_f_maps : int
    mstcn_f_dim: int
    mstcn_stages : int
    mstcn_early_stopping_metric : str
    name : str
    
def get_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config
    