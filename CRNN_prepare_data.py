import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image
from sklearn.model_selection import train_test_split

import os
import random
import time
import timm
import xml.etree.ElementTree as ET

def extract_data_from_xml(root_dir):
    xml_path = os.path.join(root_dir,"words.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_paths = []
    img_sizes = []
    img_labels = []
    boxes = []

    for img in root.findall("image"):
        img_paths.append( os.path.join(root_dir,img[0].text) )
        img_sizes.append( (
            int( img[1].attrib["x"] ),
            int( img[1].attrib["y"] )
        ) )

        bbs_of_img = []
        labels_of_img = []        

        tagged_rectangles = img.find("taggedRectangles")
        if tagged_rectangles is not None:
            for tagged_rectangle in tagged_rectangles.findall("taggedRectangle"):

                tag = tagged_rectangle.find("tag")
                if tag is None or tag.text is None or not tag.text.isalnum():
                    continue

                if " " in tag.text:
                    continue

                bbs_of_img.append([
                    float(tagged_rectangle.attrib["x"]),
                    float(tagged_rectangle.attrib["y"]),
                    float(tagged_rectangle.attrib["width"]),
                    float(tagged_rectangle.attrib["height"]),
                ])

                labels_of_img.append(tag.text.lower())

        boxes.append(bbs_of_img)
        img_labels.append(labels_of_img)

    return img_paths, img_sizes, img_labels, boxes

def split_bounding_boxes(img_paths, img_labels, bboxes, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    labels = []

    for img_path, img_label, bbs in zip(img_paths, img_labels, bboxes):
        img = Image.open(img_path)

        for label, bb in zip(img_label, bbs):
            cropped_img = img.crop((
                bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]
            ))

            if np.mean(cropped_img) < 35 or np.mean(cropped_img) > 220:
                continue
            if cropped_img.size[0] < 10 or cropped_img.size[1] <10:
                continue

            filename = f"{count:06d}.jpg"
            cropped_img.save( os.path.join(save_dir, filename) )

            new_img_path = os.path.join(save_dir, filename)
            label = new_img_path + "\t" + label

            labels.append( label )
            count += 1
    
    with open( os.path.join(save_dir, "labels.txt"), "w") as f:
        for label in labels:
            f.write(f"{label}\n")



if __name__ == "__main__":
    data_dir = "Data"
    img_paths, img_sizes, img_labels, boxes = extract_data_from_xml(data_dir)

    save_dir = "ocr_dataset"
    split_bounding_boxes(img_paths, img_labels, boxes, save_dir)









