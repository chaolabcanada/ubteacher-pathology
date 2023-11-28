from ubteacher.utils.train2_utils import find_dirs, select_annotypes, find_file, ParseFromQuPath, get_scaling
from ubteacher import add_ubteacher_config
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
import os
import glob
from pathlib import Path

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
    img_dirs, anno_dirs = find_dirs(cfg.ANNO_DIR, cfg.IMG_DIR)
    try:
        classes = cfg.DATASET.CLASSES
    except:
        classes = select_annotypes(anno_dirs)
    
# accumulate dataset_dicts for registration

    dataset_dicts = []
    image_formats = cfg.FMT.split(',')
    for anno_dir, img_dir in zip(anno_dirs, img_dirs):
        for json_file in glob.glob(os.path.join(anno_dir, "*.json")):
            id = os.path.basename(json_file).split('.')[0]
            try:
                original_img_file = find_file(Path(anno_dir).parent, id, cfg.FMT)
                img_file = os.path.join(img_dir, id + '.npy')
                base_dim, target_dim = get_scaling(original_img_file, img_file)
                dataset_dict = ParseFromQuPath(anno_dir, 
                                            img_dir, 
                                            base_dim, 
                                            target_dim, 
                                            classes
                                            ).get_coco_format(json_file)
            except:
                print(f"Error parsing {json_file}")
            dataset_dicts.append(dataset_dict)
            
# split dataset_dicts into train and val


    
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
