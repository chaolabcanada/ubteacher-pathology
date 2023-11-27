from ubteacher.utils.train2_utils import find_anno_dir, find_img_dir, select_annotypes
from ubteacher import add_ubteacher_config
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

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
    anno_dir = find_anno_dir(cfg.ANNO_DIR)
    img_dir = find_img_dir(cfg.IMG_DIR)
    try:
        classes = cfg.DATASET.CLASSES
    except:
        classes = select_annotypes(anno_dir)
    
    #keep in mind, img_dir will be a list of all the img dirs
    
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
