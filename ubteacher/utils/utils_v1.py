
import numpy as np
import logging
import os
import copy
import matplotlib.pyplot as plt #HACK
from matplotlib import patches #HACK
import time
import os
import random
import json
import torch
import torch.utils.data as torchdata
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.events import EventStorage
import detectron2.utils.comm as comm
from detectron2.engine import DefaultTrainer, hooks, SimpleTrainer, TrainerBase
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.train_loop import AMPTrainer
from fvcore.nn.precise_bn import get_bn_modules
import itertools

import detectron2.data.detection_utils as utils
from detectron2.utils.logger import setup_logger
from detectron2.utils.file_io import PathManager
import detectron2.data.transforms as T
from detectron2.data.datasets.coco import convert_to_coco_dict
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.build import (filter_images_with_few_keypoints, 
                                   filter_images_with_only_crowd_annotations,
                                    load_proposals_into_dataset,
                                    check_metadata_consistency)

from PIL import Image
from ubteacher.data.detection_utils import build_strong_augmentation

from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluator,
    print_csv_format,
    verify_results,
)

from ubteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_semisup_train_loader_two_crops,
    build_detection_test_loader,
)

from ubteacher.solver.build import build_lr_scheduler

def vis_image_with_annos(image, annotations, output):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for anno in annotations:
        #plot bbox
        x1, y1, x2, y2 = anno['bbox']
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(
            (x1, y1),
            w, h,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.annotate(f"class_ID={anno['category_id']}", (x1, y1), color='b')
        #plot poly
        x_coords = anno['segmentation'][0][::2]
        y_coords = anno['segmentation'][0][1::2]
        ax.plot(x_coords, y_coords, color='g')
    fig.savefig(output)
    plt.close()
    
def channel_last(input: np.ndarray or tuple) -> np.ndarray or tuple:
    """Return the input in channel-last format
    Args:
    input -- np.ndarray if image or tuple of array.shape
    Return:
    image as ndarray but in channel-last (h, w, c)
    """
    if type(input) == np.ndarray:
        if input.shape[0] == 3:
            return input.transpose(1, 2, 0)
        else:
            return input
    if type(input) == tuple:
        if input[0] == 3:
            return tuple(input[i] for i in [1, 2, 0])
        else:
            return input
        
def hacked_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    
    Hacked by Chao Lab for more diverse augmentations.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            ),            
        )
    if is_train and cfg.INPUT.DIVERSE_AUG != "none":
        augmentation.append(
            T.RandomBrightness(0.7, 1.3),
            T.RandomContrast(0.7, 1.3),
            T.RandomSaturation(0.7, 1.3),
        )
    return augmentation
        
def convert_to_coco_json(dataset_name: str, output_dir: str, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
    dataset_name -- reference from the config file to the catalogs
                    must be registered in DatasetCatalog and in detectron2's 
                    standard format
    output_file -- path of json file that will be saved to
    allow_cached -- if json file is already present then skip conversion
    """
    logger = setup_logger(name=__name__)
    coco_files = []
    for file in os.scandir(output_dir):
        if "coco" in file.name:
            coco_files.append(file.path)
    if coco_files:
        logger.info("Removing previously cached coco files...")
        for i in coco_files:
            logger.info(f"Removed {i}")
            os.remove(i)
    coco_dict = convert_to_coco_dict(dataset_name)
    coco_json = json.dumps(coco_dict, indent=4)
    coco_json_file = dataset_name + "_coco_format.json"
    with open(os.path.join(output_dir, coco_json_file), "w") as json_file:
        json_file.write(coco_json)
        
class DatasetMapperTwoCropSeparateV1(DatasetMapper):
    """
    This customized mapper produces two augmented images from a single image
    instance. This mapper makes sure that the two augmented images have the same
    cropping and thus the same size.

    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """


    
    def __init__(self, cfg, is_train=True):
        self.augmentation = utils.build_augmentation(cfg, is_train)
        # include crop into self.augmentation
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            self.compute_tight_boxes = True
        else:
            self.compute_tight_boxes = False
        self.strong_augmentation = build_strong_augmentation(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = np.load(dataset_dict["file_name"]) #numpy version
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = np.load(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        # apply weak augmentation
        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image_weak_aug.shape[:2]  # h, w

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            if self.compute_tight_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            bboxes_d2_format = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = bboxes_d2_format
            
        # Plot weakly augmented inputs to prove correct transformation #HACK
        
        #out_dir = os.path.join(os.getcwd(), 'training_data_vis')
        #os.makedirs(out_dir, exist_ok=True)
        #n = 0
        #vis_file = dataset_dict['image_id'] + '.png'
        #while os.path.exists(os.path.join(out_dir, vis_file)):
        #    n += 1
        #    vis_file = f"{dataset_dict['image_id']}_{n}.png"
        #    if n > 2: #For only 2 augmentations
        #        break
        #vis_image_with_annos(image_weak_aug, annos, os.path.join(out_dir, vis_file))

        # apply strong augmentation
        # We use torchvision augmentation, which is not compatiable with
        # detectron2, which use numpy format for images. Thus, we need to
        # convert to PIL format first.
        
        image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
        image_strong_aug = np.array(self.strong_augmentation(image_pil))
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        )

        dataset_dict_key = copy.deepcopy(dataset_dict)
        dataset_dict_key["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )
        assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
        assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
        return (dataset_dict, dataset_dict_key)

class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        
        # create only one model 
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # Fr training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
            
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if cfg.TEST.EVALUATOR == "COCOeval":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        # elif cfg.TEST.EVALUATOR == "COCOTIDEeval":
        #     return COCOTIDEEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparateV1(cfg, True)
        #can be replaced with a custom mapper if needed
        return build_detection_semisup_train_loader(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)
    
    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start
        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)
        
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)
                
    class TeacherUtil:
        def __init__(self, max_dimension=1333) -> None:
            self.max_dim = max_dimension  # The max num pixels of the longest edge
        @staticmethod
        
        def train_val_split(
            image_dir: str,
            annotation_dir: str,
            split_miu: float = 0.2,
        ):
            """Split image dataset and the associated annotations into "train" and "test".
            WARNING: images and annotations must have the same name

            Args:
            image_dir -- path to images
            annotation_dir -- path to annotations (e.g., .json)
            compatible_formats -- ex: 'tif', 'jpg'
            split_miu -- fraction of overall data to split as validation dataset

            Return:
            train_set -- Dict('images': [paths], 'annotations': [paths])
            val_set -- Dict('images': [paths], 'annotations': [paths])
            """
            train = []
            val = []
            anno_train = []
            anno_val = []

            for img_file in os.scandir(image_dir):
                # Find image
                if not img_file.name.startswith(".") and img_file.name.endswith('npy'):
                    image_name = os.path.splitext(img_file.name)[0]
                    # Find matching annotation json
                    anno_file = os.path.join(annotation_dir, image_name + ".json")
                    if os.path.exists(anno_file):  # Only add images with matching annos
                        anno_train.append(anno_file)
                        train.append(img_file.path)
                    else:
                        raise(FileNotFoundError(f"Cannot find annotation for {img_file.path}"))

            # Randomly split some validation data
            val_data_len = int(len(train) * split_miu)
            train_data_len = len(train) - val_data_len
            split_guide = [True] * val_data_len + [False] * train_data_len
            random.shuffle(split_guide)
            for idx, to_split in enumerate(split_guide):
                if to_split:
                    split_guide.pop(idx)
                    val.append(train.pop(idx))
                    anno_val.append(anno_train.pop(idx))
            train_set = dict()
            val_set = dict()
            for i, data_group in zip(
                (train_set, val_set), ((train, anno_train), (val, anno_val))
            ):
                for n, k in enumerate(["images", "annotations"]):
                    i[k] = data_group[n]
            return train_set, val_set
        
        
def get_detection_dataset_dicts(
    names,
    filter_empty=True,
    min_keypoints=0,
    proposal_files=None,
    check_consistency=True,
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.
        check_consistency (bool): whether to check if datasets have consistent metadata.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]

    if isinstance(dataset_dicts[0], torchdata.Dataset):
        if len(dataset_dicts) > 1:
            # ConcatDataset does not work for iterable style dataset.
            # We could support concat for iterable as well, but it's often
            # not a good idea to concat iterables anyway.
            return torchdata.ConcatDataset(dataset_dicts)
        return dataset_dicts[0]

    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if check_consistency and has_instances:
        try:
            class_names = MetadataCatalog.get(names[0]).thing_classes
            check_metadata_consistency("thing_classes", names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts
