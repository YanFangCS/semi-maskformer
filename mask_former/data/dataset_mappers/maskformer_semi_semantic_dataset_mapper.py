import os
import logging
from detectron2 import data
from detectron2.data.transforms import transform
from detectron2.structures.instances import Instances
from detectron2.structures.masks import BitMasks

import numpy as np
import torch
import copy
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.projects.point_rend.color_augmentation import ColorAugSSDTransform
from torch.nn.modules import padding

__all__ = [
    "MaskFormerSemiDatasetMapper"
]

class MaskFormerSemiDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """
    @configurable
    def __init__(
        self,
        is_train = True,
        is_semi = True,
        *,
        augmentations, 
        image_format,
        ignore_label,
        size_divisibility,
    ) -> None:
        self.is_train = is_train
        self.is_semi = is_semi
        self.tfm_gens = augmentations
        self.image_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode} : {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train = True, is_semi = True):
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "is_semi": is_semi,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        assert self.is_train, "MaskFormerSemiDatasetMapper should only be used in training stage"

        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        file_path = os.getenv("DETECTRON2_DATASETS")
        file_path = os.path.join(file_path, "VOCdevkit", "VOCAug")
        unlabel_file = os.path.join(file_path, "train_unlabeled.txt")
        with open(unlabel_file, "r") as f:
            unlabeled_list = f.readlines()

        unlabeled_data_list = [f.strip() for f in unlabeled_list]
        f = open("unlabeled_list.txt", "w")
        print("unlabeled_data_list:", file=f)
        print(unlabeled_data_list, file=f)

        sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        
        aug_input = T.AugInput(image, sem_seg = sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        image_size = (image.shape[-2], image.shape[-1])
        #print("dataset_dict[\"file_name\"]", dataset_dict["file_name"], file=f)
        if self.size_divisibility > 0:
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0, 
                self.size_divisibility - image_size[0]
            ]
            image = F.pad(image, padding_size, value = 128).contiguous()
            sem_seg_gt = F.pad(sem_seg_gt, padding_size, value = self.ignore_label).contiguous()

        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_size=image_size)
            classes = np.unique(sem_seg_gt)
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            # instances.gt_categories = classes[classes != 0]

            # print(dataset_dict["file_name"])
            file_name = dataset_dict["file_name"]
            file_name = file_name.split("/")[-1]
            if file_name not in unlabeled_data_list:
                print("file_name", file_name, file=f)
                masks = []
                for class_id in classes:
                    masks.append(sem_seg_gt == class_id)
                
                if len(masks) == 0:
                    instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
                else:
                    masks = BitMasks(
                        torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                    )
                    instances.gt_masks = masks.tensor
                
            else:
                """
                unlabeled data:
                make sem_seg_gt be a image filled with ignore_label
                """
                print("file_name", file_name, file=f)
                sem_seg_gt = np.full_like(sem_seg_gt, self.ignore_label)
                print("sem_seg_gt", sem_seg_gt, file = f)
                print("sem_seg_gt shape", sem_seg_gt.shape, file = f)
                gt_masks = torch.full((len(classes), sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]), self.ignore_label)
                instances.gt_masks = gt_masks
                print("gt_masks", instances.gt_masks, file=f)
                print("gt_masks shape", gt_masks.shape, file = f)
                sem_seg_gt = torch.as_tensor(sem_seg_gt, dtype=torch.long).contiguous()
            
            dataset_dict["instances"] = instances
            dataset_dict["sem_seg"] = sem_seg_gt
        
        return dataset_dict