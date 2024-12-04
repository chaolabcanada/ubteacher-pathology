from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBRCNNTeacherTrainer, BaselineTrainer
from detectron2.checkpoint import DetectionCheckpointer
from ubteacher.modeling import EnsembleTSModel
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
import os
from utils.train_utils import Registration, CustomRepeatScheduler

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
        label, unlabel, val = reg.accumulate_annos()
        reg.register_dataset("train_labeled", label)
        reg.register_dataset("train_unlabeled", unlabel)
        reg.register_dataset("val", val)
        

    # TODO: support loading from existing dataset_dicts
    
    # train
    print("Starting training...")
    if cfg.SEMISUPNET.Trainer == "ubteacher_rcnn":
        Trainer = UBRCNNTeacherTrainer
    else:
        Trainer = BaselineTrainer #Combined from ubteacher v1
        
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    # Get the optimizer from the trainer
    optimizer = trainer.optimizer
    
    # Create the custom scheduler
    burn_up_step = cfg.SEMISUPNET.BURN_UP_STEP  # Read from your config file
    total_iters = cfg.SOLVER.MAX_ITER
    scheduler = CustomRepeatScheduler(optimizer, burn_up_step, total_iters)

    # sanity check the scheduler
    if cfg.DEBUG:
        scheduler.sanity_check()
    
    # Add the scheduler to the trainer
    trainer.scheduler = scheduler
    
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
        
            
            
            
        
            
        
    