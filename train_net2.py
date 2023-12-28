from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, UBRCNNTeacherTrainer, BaselineTrainer
from detectron2.checkpoint import DetectionCheckpointer
from ubteacher.modeling import EnsembleTSModel
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from typing import Dict, Tuple, List, Set, Iterator, Union
import os
import glob
from pathlib import Path

from ubteacher.utils.train2_utils import (find_dirs,
                                          find_unlabeled_dirs, 
                                          select_annotypes,
                                          select_convert_annotypes, 
                                          find_file,
                                          ParseFromQuPath,
                                          ParseUnlabeled, 
                                          get_scaling, 
                                          split_dataset, 
                                          register_dataset)

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
    
    if cfg.REGISTER:
        box_only = cfg.BOX_ONLY
        img_dirs, anno_dirs = find_dirs(cfg.ANNO_DIR, cfg.IMG_DIR)
        if cfg.DATASETS.CROSS_DATASET:
            u_img_dirs = find_unlabeled_dirs(cfg.UNLABELED_DIR)
        
        if cfg.CLASS_CONVERTER:
            class_file = cfg.CLASS_CONVERTER
            classes = select_convert_annotypes(anno_dirs, class_file)
        else:
            print('No class converter specified. Using normal classes.')
            class_file = None
            classes = select_annotypes(anno_dirs)
        
    # accumulate dataset_dicts for registration
        dicts = []
        for anno_dir, img_dir in zip(anno_dirs, img_dirs):
            for json_file in glob.glob(os.path.join(anno_dir, "*.json")):
                id = os.path.basename(json_file).split('.')[0]
                try:
                    original_img_file = find_file(Path(anno_dir).parent, id, cfg.FMT)
                    img_file = os.path.join(img_dir, id + '.npy')
                    base_dim, target_dim = get_scaling(original_img_file, img_file)
                    each_dict = ParseFromQuPath(anno_dir, 
                                                img_dir, 
                                                base_dim, 
                                                target_dim, 
                                                classes,
                                                class_file,
                                                box_only
                                                ).get_coco_format(json_file)
                    dicts.append(each_dict[0])
                except:
                    print(f"Error parsing {json_file}")
                    pass
                
    # accumulate image info for unlabeled registration
        if cfg.DATASETS.CROSS_DATASET:
            unlabeled_dicts = []
            for u_img_dir in u_img_dirs:
                for img_file in glob.glob(os.path.join(u_img_dir, "*.npy")):
                    each_dict = ParseUnlabeled(u_img_dir).get_unlabeled_coco(img_file)
                    unlabeled_dicts.append(each_dict[0])     

        # split and register
        if cfg.DATASETS.CROSS_DATASET:
            train_labeled, val = split_dataset(cfg, dicts)
            register_dataset("train_unlabeled", unlabeled_dicts, classes)
            register_dataset("train_labeled", train_labeled, classes)
            register_dataset("val", val, classes)
        else:
            train, val = split_dataset(cfg, dicts)
            register_dataset("train", train, classes)
            register_dataset("val", val, classes)

# train

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
