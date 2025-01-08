from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBRCNNTeacherTrainer, BaselineTrainer
from detectron2.checkpoint import DetectionCheckpointer
from ubteacher.modeling import EnsembleTSModel
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
import os
from utils.train_utils import Registration

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True) #allows custom cfg keys
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    out_dir = cfg.OUTPUT_DIR
    data_dir = cfg.DATA_DIR
    is_unlabeled = cfg.DATASETS.CROSS_DATASET
    train_fraction = cfg.TRAIN_FRACTION
    cat_map = cfg.CAT_MAP
    
    if cfg.REGISTER:    
        reg = Registration(data_dir, is_unlabeled, train_fraction, cat_map, out_dir)
        dataset_dicts = reg.accumulate_annos()
        reg.register_all(dataset_dicts)
    
    # TODO: support loading from existing dataset_dicts
    
    # train
    print("Starting training...")
    if cfg.SEMISUPNET.Trainer == "ubteacher_rcnn":
        Trainer = UBRCNNTeacherTrainer
    else:
        Trainer = BaselineTrainer #Combined from ubteacher v1
        
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
        
            
            
            
        
            
        
    