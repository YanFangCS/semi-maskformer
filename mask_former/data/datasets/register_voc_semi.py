import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

PASCALVOC_CATEGORIES = [
    {"color": [   0,   0,   0], "trainid":  0, "id":  0, "name":"background"},
    {"color": [ 128,   0,   0], "trainid":  1, "id":  1, "name":"aeroplane"},
    {"color": [   0, 128,   0], "trainid":  2, "id":  2, "name":"bicycle"},
    {"color": [ 128, 128,   0], "trainid":  3, "id":  3, "name":"bird"},
    {"color": [   0,   0, 128], "trainid":  4, "id":  4, "name":"boat"},
    {"color": [ 128,   0, 128], "trainid":  5, "id":  5, "name":"bottle"},
    {"color": [   0, 128, 128], "trainid":  6, "id":  6, "name":"bus"},
    {"color": [ 128, 128, 128], "trainid":  7, "id":  7, "name":"car"},
    {"color": [  64,   0,   0], "trainid":  8, "id":  8, "name":"cat"},
    {"color": [ 192,   0,   0], "trainid":  9, "id":  9, "name":"chair"},
    {"color": [  64, 128,   0], "trainid": 10, "id": 10, "name":"cow"},
    {"color": [ 192, 128,   0], "trainid": 11, "id": 11, "name":"dining table"},
    {"color": [  64,   0, 128], "trainid": 12, "id": 12, "name":"dog"},
    {"color": [ 192,   0, 128], "trainid": 13, "id": 13, "name":"horse"},
    {"color": [  64, 128, 128], "trainid": 14, "id": 14, "name":"motorbike"},
    {"color": [ 192, 128, 128], "trainid": 15, "id": 15, "name":"person"},
    {"color": [   0,  64,   0], "trainid": 16, "id": 16, "name":"potted plant"},
    {"color": [ 128,  64,   0], "trainid": 17, "id": 17, "name":"sheep"},
    {"color": [   0, 192,   0], "trainid": 18, "id": 18, "name":"sofa"},
    {"color": [ 128, 192,   0], "trainid": 19, "id": 19, "name":"train"},
    {"color": [   0,  64, 128], "trainid": 20, "id": 20, "name":"tv monitor"},
]

def _get_pascalvoc_meta():

    _ids = [k["id"] for k in PASCALVOC_CATEGORIES]
    assert len(_ids) == 21, len(_ids)

    stuff_dataset_id_to_contiguous_id = {k for i,k in enumerate(_ids)}
    stuff_classes = [k["name"] for k in PASCALVOC_CATEGORIES]

    ret = {
        "pascalvoc_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }

    return ret

# partition pascal voc 2012 aug training into two sub datasets
# -- mask labeled data 
# -- image labeled data 

def register_all_pascalvoc(root):
    root = os.path.join(root, "VOCdevkit", "VOCAug")
    meta = _get_pascalvoc_meta()

    for name, image_dirname, sem_seg_dirname in [
        ("train", "images_detectron2/train", "annotations_detectron2/train"),
        ("test", "images_detectron2/val", "annotations_detectron2/val")
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"voc_semi_{name}_sem_seg"

        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir : load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root = image_dir,
            sem_seg_root = gt_dir,
            evaluator_type = "sem_seg",
            ignore_label = 255,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_pascalvoc(_root)