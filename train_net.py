    #!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from typing import Dict, Set, List, Tuple, Iterator
import os
import logging
import json
import copy

# hacky way to register
from ubteacher.modeling import *
from ubteacher.engine import *
from ubteacher.engine.trainer import UBTeacherTrainer, UBRCNNTeacherTrainer, BaselineTrainer
from ubteacher import add_ubteacher_config
from ubteacher.utils.ROI_utils import (TrainUtil, 
                                       TrainHelper, 
                                       get_categorical_map, 
                                       get_annotypes_for_dataset)

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
    parent_dir = cfg.PARENTDIR
    dataset_dirs = []
    unlabeled_dataset_names = []
    for f in os.scandir(parent_dir):
        if f.is_dir() and not f.name.startswith('.'):
            dataset_dirs.append(f.name)
    print(f"Found the following datasets: {sorted(dataset_dirs)}")
    
    labeled_dset_selection = input(
        "Enter the names of labeled datasets you want to train with"
        + "(comma separated, or 'all' to train with everything):")
    if 'all' in labeled_dset_selection.lower():
        labeled_dataset_names = dataset_dirs
    else:
        labeled_dataset_names = [
            i.strip()
            for i in labeled_dset_selection.split(",")
        ]
        
    unlabeled_dset_selection = input(
        "Enter the names of unlabeled datasets you want to train with"
        + "(comma separated, or 'none' to train with nothing):")
    if 'none' in unlabeled_dset_selection.lower():
        unlabeled_dataset_names = []
    else:
        unlabeled_dataset_names = [
            i.strip()
            for i in unlabeled_dset_selection.split(",")
        ]
        
    data_train_labeled, data_val = TrainHelper().split_dataset(cfg, labeled_dataset_names, args, set_seed=True)
    
    if cfg.DATASETS.CROSS_DATASET:
        print('Using unlabeled datasets for training')
        data_train_unlabeled = TrainHelper().get_unlabeled(cfg, unlabeled_dataset_names[0])
    
        datasets = {
                    "train_labeled": data_train_labeled, 
                    "train_unlabeled": data_train_unlabeled,
                    "val": data_val, 
                    }
    else:
        datasets = {"train": data_train_labeled, "val": data_val}

    
    global_annotypes = set()
    dset_annotypes = get_annotypes_for_dataset(data_train_labeled)
    global_annotypes.update(dset_annotypes)
    cat_map = {'Neoplastic': 0} #TODO: Unhardcode this
    print(f"Using the following categorical map: {cat_map}")

    for d in datasets:
        #TrainHelper().basic_registration(d, datasets[d]) #For use with simpletrainer
        TrainHelper().register_dataset(d, datasets[d], cat_map) #For use with ubteacher
        
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "ubteacher_rcnn":
        Trainer = UBRCNNTeacherTrainer
    else:
        Trainer = BaselineTrainer #Combined from ubteacher v1

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

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
