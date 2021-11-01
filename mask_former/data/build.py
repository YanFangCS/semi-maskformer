import itertools
import logging
import numpy as np
import operator
import os
import pickle
from torch._C import set_autocast_enabled
import torch.utils.data
from tabulate import tabulate
from termcolor import colored

from detectron2.config import configurable
from detectron2.structures import BoxMode
from detectron2.utils import comm
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import _log_api_usage, log_first_n

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.data.build import worker_init_reset_seed, trivial_batch_collator, get_detection_dataset_dicts, build_batch_data_loader

file_path = os.getenv("DETECTRON2_DATASETS")
file_path = os.path.join(file_path, "VOCdevkit", "VOCAug")
unlabel_file = os.path.join(file_path, "train_unlabeled.txt")
with open(unlabel_file, "r") as f:
    unlabeled_list = f.readlines()

unlabeled_data_list = [f.strip() for f in unlabeled_list]

# def build_batch_data_loader(
#     dataset, sampler, total_batch_size, unlabeled_bs, *, aspect_ratio_grouping=False, num_workers=0
# ):
#     """
#     Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
#     1. support aspect ratio grouping options
#     2. use no "batch collation", because this is common for detection training

#     Args:
#         dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
#         sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
#         total_batch_size, aspect_ratio_grouping, num_workers): see
#             :func:`build_detection_train_loader`.

#     Returns:
#         iterable[list]. Length of each list is the batch size of the current
#             GPU. Each element in the list comes from the dataset.
#     """
#     world_size = get_world_size()
#     assert (
#         total_batch_size > 0 and total_batch_size % world_size == 0
#     ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
#         total_batch_size, world_size
#     )

#     batch_size = total_batch_size // world_size
#     unlabeled_size = unlabeled_bs // world_size
#     if aspect_ratio_grouping:
#         data_loader = torch.utils.data.DataLoader(
#             dataset,
#             sampler=sampler,
#             num_workers=num_workers,
#             batch_sampler=None,
#             collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
#             worker_init_fn=worker_init_reset_seed,
#         )  # yield individual mapped dict
#         return AspectRatioGroupedDataset(data_loader, batch_size)
#     else:
#         #batch_sampler = torch.utils.data.sampler.BatchSampler(
#         #    sampler, batch_size, drop_last=True
#         #)  # drop_last so the batch always have the same size
#         labeled_idxs, unlabeled_idxs = idxs_compute(unlabeled_data_list)
#         batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, total_bs=batch_size, unlabeled_bs=unlabeled_size)
#         return torch.utils.data.DataLoader(
#             dataset,
#             num_workers=num_workers,
#             #batch_sampler=sampler,
#             batch_sampler=batch_sampler,
#             collate_fn=trivial_batch_collator,
#             worker_init_fn=worker_init_reset_seed,
#         )

# implement a distribute bathsampler
# half indices generated from labeled indices, and another half generated from unlabeled indices
class TwoStreamBatchSampler(torch.utils.data.Sampler):
    """
    primary_indices : labeled data indices
    secondary_indices : unlabeled data indices
    batch_size : total batch_size, consists of labeled and unlabeled
    secondary_batch_size : unlabeled data batch size
    """
    def __init__(self, labeled_indexes = None, unlabeled_indexes = None, total_bs = 4, unlabeled_bs = 2, seed = None) -> None:
        self._labeled_indexes = labeled_indexes
        self._ublabeled_indexes = unlabeled_indexes
        if seed == None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        self._rank = comm.get_local_rank()
        self._world_size = comm.get_world_size()
        self.total_bs = total_bs
        self.labeled_bs = total_bs - unlabeled_bs
        self.unlabeled_bs = unlabeled_bs
    
    def __iter__(self):
        labeled_iter = iterate_eternally(self._labeled_indexes)
        unlabeled_iter = iterate_eternally(self._ublabeled_indexes)
        if self.labeled_bs == self.total_bs:
            return grouper(labeled_iter, self.labeled_bs)
        return (
            labeled_batch + unlabeled_batch
            for (labeled_batch, unlabeled_batch)
            in zip(
                grouper(labeled_iter, self.labeled_bs),
                grouper(unlabeled_iter, self.unlabeled_bs)
            )
        )
        

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)

def idxs_compute(unlabeled_list):
    labeled_idxs, unlabled_idxs = [], []
    image_file = os.getenv("DETECTRON2_DATASETS")
    image_file = os.path.join(image_file, "VOCdevkit", "VOCAug", "train.txt")
    with open(image_file, "r") as f:
        images_list = f.readlines()
    images_list = [t.strip() for t in images_list]
    images_all = []
    for t in images_list:
        image, annotation = t.split()
        images_all.append(image)
    for idx in range(len(images_all)):
        filename = images_all[idx]
        if filename in unlabeled_list:
            unlabled_idxs.append(idx)
        else:
            labeled_idxs.append(idx)
    return labeled_idxs, unlabled_idxs

def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    semi_split_ratio = cfg.DATASETS.SEMI_SPLIT_RATIO
    dataset_sup_len = int(len(dataset) * semi_split_ratio)
    # print(type(cfg.DATASETS.TEST))
    if cfg.DATASETS.TEST[0] == "voc2012_test_sem_seg":
        dataset_sup_len = 1464
    
    datasets = dict()
    # sup_index = np.random.choice(np.arange(len(dataset)), dataset_sup_len, replace=False)
    # unsup_index = np.array(list(set(np.arange(len(dataset)).tolist()) - set(sup_index.tolist())))
    datasets["supervised"] = dataset[:dataset_sup_len]
    datasets["unsupervised"] = dataset[dataset_sup_len:]
    print(type(datasets["supervised"]))
    sup_file = [x["file_name"] for x in datasets["supervised"]]
    with open("sup_{}.txt".format(semi_split_ratio), "a") as f:
        sep = "\n"
        f.write(sep.join(sup_file))

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": datasets,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "unlabeled_bs": cfg.SOLVER.UNLABELED_RATIO,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }

# TODO can allow dataset as an iterable or IterableDataset to make this function more general
@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader_semi(
    dataset, *, mapper, sampler=None, total_batch_size, unlabeled_bs, aspect_ratio_grouping=True, num_workers=0
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    dataset1 = dataset["supervised"]
    dataset2 = dataset["unsupervised"]
    if isinstance(dataset1, list):
        dataset1 = DatasetFromList(dataset1, copy=False)
    if isinstance(dataset2, list):
        dataset2 = DatasetFromList(dataset2, copy=False)
    if mapper is not None:
        # the mapper used in semi- setting is a tuple of DataSetMapper
        # consists of: (SemanticDatasetMapper, SemiDatasetMapper)
        dataset1 = MapDataset(dataset1, mapper[0])
        dataset2 = MapDataset(dataset2, mapper[1])
    # labeled_idxs, unlabeled_idxs = idxs_compute(unlabeled_data_list)
    #sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,                        
    #            total_batch_size, int(total_batch_size / 2))
    sampler1 = TrainingSampler(len(dataset1))
    sampler2 = TrainingSampler(len(dataset2))
    unlabeled_bs = int(total_batch_size * unlabeled_bs)
    labeled_bs = total_batch_size - unlabeled_bs
    # sampler = TwoStreamBatchSampler(labeled_indexes = labeled_idxs,unlabeled_indexes = unlabeled_idxs)
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    if unlabeled_bs == 0:
        return build_batch_data_loader(
            dataset1,
            sampler1,
            labeled_bs,
            aspect_ratio_grouping=False,
            num_workers=num_workers,
        )
    else:
        return (build_batch_data_loader(
                    dataset1,
                    sampler1,
                    labeled_bs,
                    aspect_ratio_grouping=False,
                    num_workers=num_workers,
                ), build_batch_data_loader(
                    dataset2,
                    sampler2,
                    unlabeled_bs,
                    aspect_ratio_grouping=False,
                    num_workers=num_workers,
                )
            )
