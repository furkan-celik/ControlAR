import torch
import os
from PIL import Image
import json

# import orjson
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl


import torch.nn.functional as F
import glob
import random

import zipfile

# import sys

# sys.path.append('../webuidata')

# import download_partial_data_webui

DEVICE_SCALE = {
    "default": 1,
    "iPad-Mini": 2,
    "iPad-Pro": 2,
    "iPhone-13 Pro": 3,
    "iPhone-SE": 3,
}


def makeMultiHotVec(idxs, num_classes):
    vec = [1 if i in idxs else 0 for i in range(num_classes)]
    return vec


class WebUIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_file,
        boxes_dir="../../downloads/webui-boxes/all_data",
        rawdata_screenshots_dir="../../downloads/ds",
        class_map_file="class_map.json",
        min_area=100,
        device_scale=DEVICE_SCALE,
        max_boxes=100,
        max_skip_boxes=100,
        image_size=128,
        layout_length=10,
        **kwargs
    ):
        super(WebUIDataset, self).__init__()
        self.max_boxes = max_boxes
        self.max_skip_boxes = max_skip_boxes
        self.keys = []

        with open(split_file, "r") as f:
            boxes_split = json.load(f)

        rawdata_directory = rawdata_screenshots_dir
        for folder in [f for f in os.listdir(boxes_dir) if f in boxes_split]:
            for file in os.listdir(os.path.join(boxes_dir, folder)):
                if os.path.exists(
                    os.path.join(
                        rawdata_directory,
                        folder,
                        file.replace(".json", "-screenshot.webp"),
                    )
                ):
                    self.keys.append(os.path.join(boxes_dir, folder, file))

        self.min_area = min_area
        self.device_scale = device_scale
        with open(class_map_file, "r") as f:
            class_map = json.load(f)
        self.computed_boxes_directory = boxes_dir
        self.rawdata_directory = rawdata_directory
        self.idx2Label = class_map["idx2Label"]
        self.label2Idx = class_map["label2Idx"]
        self.num_classes = max([int(k) for k in self.idx2Label.keys()]) + 1
        self.img_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size), antialias=True),
            ]
        )
        # image_normalize()])

        self.image_size = (image_size, image_size)
        self.layout_length = layout_length
        self.flip = False
        self.feature_dir = []
        self.aug_feature_dir = None

    def __len__(self):
        return len(self.keys)

    def total_objects(self):
        to = 0
        for i in range(len(self.keys)):
            with open(self.keys[i], "r") as f:
                key_dict = json.load(f)
            to += len(key_dict["labels"])
        return to

    def __getitem__(self, idx):
        # try:
        idx = idx % len(self.keys)
        key = self.keys[idx]
        with open(key, "r") as f:
            key_dict = json.load(f)

        img_path = key.replace(".json", "-screenshot.webp")
        img_path = img_path.replace(
            self.computed_boxes_directory, self.rawdata_directory
        )

        key_filename = img_path.split("/")[-1]
        device_name = "-".join(key_filename.split("-")[:-1])

        img_pil = Image.open(img_path).convert("RGB")
        org_size = img_pil.size # w, h
        img = self.img_transforms(img_pil)
        target = {}
        boxes = []
        masks = []
        labels = []
        labelNames = []
        scale = self.device_scale[device_name.split("_")[0]]

        inds = list(range(len(key_dict["labels"])))
        random.shuffle(inds)


        for i in inds:
            box = key_dict["contentBoxes"][i]
            box[0] *= scale # w
            box[1] *= scale # h
            box[2] *= scale # w
            box[3] *= scale # h

            box[0] = round(min(max(0, box[0]), org_size[0]) / (org_size[0] / self.image_size[0]))
            box[1] = round(min(max(0, box[1]), org_size[1]) / (org_size[1] / self.image_size[1]))
            box[2] = round(min(max(0, box[2]), org_size[0]) / (org_size[0] / self.image_size[0]))
            box[3] = round(min(max(0, box[3]), org_size[1]) / (org_size[1] / self.image_size[1]))

            # box[0] *= self.image_size[0]
            # box[1] *= self.image_size[1]
            # box[2] *= self.image_size[0]
            # box[3] *= self.image_size[1]
            box.append(len(boxes) + 1)

            h, w = img.shape[1], img.shape[2]

            mask = torch.zeros(1, w, h)
            mask[:, box[1]:box[3], box[0]:box[2]] = 1

            # skip invalid boxes
            if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
                continue
            if box[3] <= box[1] or box[2] <= box[0]:
                continue
            if (box[3] - box[1]) * (
                box[2] - box[0]
            ) <= self.min_area:  # get rid of really small elements
                continue
            boxes.append(box)
            masks.append(mask)
            label = key_dict["labels"][i]
            labelIdx = [
                (
                    self.label2Idx[label[li]]
                    if label[li] in self.label2Idx
                    else self.label2Idx["OTHER"]
                )
                for li in range(len(label))
            ]
            # labelHot = makeMultiHotVec(set(labelIdx), self.num_classes)
            labelNames.append(", ".join(label))
            labels.append(labelIdx[0])

        if len(boxes) > self.max_skip_boxes:
            # print("skipped due to too many objects", len(boxes))
            return self.__getitem__(idx + 1)

        boxes = torch.tensor(boxes, dtype=torch.float)

        if len(masks) > 0:
            masks = torch.concat(masks, dim=0)
        masks = torch.tensor(masks, dtype=torch.float)

        labels = torch.tensor(labels, dtype=torch.long)

        target["control"] = boxes if len(boxes.shape) == 2 else torch.zeros(0, 5)
        target["obj_mask"] = masks if len(masks.shape) == 3 else torch.zeros(1, img.shape[1], img.shape[2])
        target["label"] = labels
        target["prompt"] = ",".join(labelNames)
        target["image_id"] = torch.tensor([idx])
        target["valid"] = torch.ones(len(target["control"]))
        target["num_obj"] = len(inds)

        for k in target:
            if not isinstance(target[k], int):
                target[k] = target[k][: self.max_boxes]

        target["control"] = torch.nn.functional.pad(
            target["control"],
            (0, 0, 0, self.layout_length - len(target["control"])),
            mode="constant",
            value=0,
        )
        target["obj_mask"] = torch.nn.functional.pad(
            target["obj_mask"],
            (0, 0, 0, 0, 0, self.layout_length - len(target["obj_mask"])),
            mode="constant",
            value=0,
        )
        target["label"] = torch.nn.functional.pad(
            target["label"],
            (0, self.layout_length - len(target["label"])),
            mode="constant",
            value=0,
        )
        target["valid"] = torch.nn.functional.pad(
            target["valid"],
            (0, self.layout_length - len(target["valid"])),
            mode="constant",
            value=0,
        )

        target["image"] = img

        return target  # return image and target dict

    # except Exception as e:
    #     print("failed", idx, str(e))
    #     return self.__getitem__(idx + 1)


# https://github.com/pytorch/vision/blob/5985504cc32011fbd4312600b4492d8ae0dd13b4/references/detection/utils.py#L203
def wui_collate_fn_for_layout(batch):
    return tuple(zip(*batch))


class WebUIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_split_file,
        val_split_file="../../downloads/val_split_webui.json",
        test_split_file="../../downloads/test_split_webui.json",
        batch_size=8,
        num_workers=4,
    ):
        super(WebUIDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = WebUIDataset(split_file=train_split_file)
        self.val_dataset = WebUIDataset(split_file=val_split_file)
        self.test_dataset = WebUIDataset(split_file=test_split_file)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
        )  # shuffle so that we can eval on subset

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

# def wui_collate_fn(batch):
#     """
#     Collate function to be used when wrapping CocoSceneGraphDataset in a
#     DataLoader. Returns a tuple of the following:

#     - imgs: FloatTensor of shape (N, C, H, W)
#     - objs: LongTensor of shape (O,) giving object categories
#     - boxes: FloatTensor of shape (O, 4)
#     - masks: FloatTensor of shape (O, M, M)
#     - triples: LongTensor of shape (T, 3) giving triples
#     - obj_to_img: LongTensor of shape (O,) mapping objects to images
#     - triple_to_img: LongTensor of shape (T,) mapping triples to images
#     """
#     all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img = [], [], [], [], []

#     for i, (img, objs, boxes, masks) in enumerate(batch):
#         all_imgs.append(img[None])
#         O = objs.size(0)
#         all_objs.append(objs)
#         all_boxes.append(boxes)
#         all_masks.append(masks)

#         all_obj_to_img.append(torch.LongTensor(O).fill_(i))

#     all_imgs = torch.cat(all_imgs)
#     all_objs = torch.cat(all_objs)
#     all_boxes = torch.cat(all_boxes)
#     all_masks = torch.cat(all_masks)
#     all_obj_to_img = torch.cat(all_obj_to_img)

#     out = (all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img)

#     return out

def build_wui_dsets(cfg, mode="train"):
    assert mode in ["train", "val", "test"]
    params = cfg
    dataset = WebUIDataset(cfg.split_file, cfg.boxes_dir, cfg.rawdata_screenshots_dir, cfg.class_map_file)

    num_objs = dataset.total_objects()
    num_imgs = len(dataset)
    print("%s dataset has %d images and %d objects" % (mode, num_imgs, num_objs))
    print("(%.2f objects per image)" % (float(num_objs) / num_imgs))

    return dataset
