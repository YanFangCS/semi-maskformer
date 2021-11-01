from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import shutil
import imgviz
import argparse
import os
import tqdm

def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode = "P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/root/cloud/fy7/MSCOCO", type = str)
    parser.add_argument("--split", default = "train2017")
    return parser.parse_args()

def main(args):
    annotation_file = os.path.join(args.input_dir, "annotations", "instances_{}.json".format(args.split))
    os.makedirs(os.path.join(args.input_dir, "annotations_d2"), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, "images_d2"), exist_ok=True)
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols = 100):
        img = coco.loadImgs(imgId)
        annIds = coco.getAnnIds(imgIds=img['id'], catIds = catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            for i in range(len(anns) - 1):
                mask += coco.annToMask(anns[i + 1]) * anns[i+1]['category_id']
            img_origin_path = os.path.join(args.input_dir, args.split, img['file_name'])
            img_output_path = os.path.join(args.input_dir, 'images_d2', img['file_name'])
            seg_output_path = os.path.join(args.input_dir, 'annotations_d2', img['file_name'].replace('.jpg', '.png'))
            shutil.copy(img_origin_path, img_output_path)
            save_colored_mask(seg_output_path)

if __name__ == "__main__":
    args = get_args()
    main(args)
