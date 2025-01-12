import ultralytics
import os
import shutil
import yaml

from sklearn.model_selection import train_test_split

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

def convert_to_yolo_format(image_paths, image_sizes, bounding_boxes):
    yolo_data = []

    for image_path, image_size, bboxes in zip(image_paths,
                                              image_sizes,
                                              bounding_boxes):
        img_w, img_h = image_size[0], image_size[1]
        yolo_labels = []

        for bbox in bboxes:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

            center_x = (x + w/2) / img_w
            center_y = (y + h/2) / img_h
            w = w/img_w
            h = h/img_h

            class_id = 0

            yolo_label = f"{class_id} {center_x} {center_y} {w} {h}"
            yolo_labels.append(yolo_label)
        
        yolo_data.append((image_path,yolo_labels))
    
    return yolo_data
                 
def save_data(data, src_img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    os.makedirs( os.path.join(save_dir, "images"), exist_ok=True )
    os.makedirs( os.path.join(save_dir, "labels"), exist_ok=True )

    for img_path, labels in data:
        shutil.copy(
            img_path,
            os.path.join(save_dir, "images") 
        )

        img_name = os.path.basename( img_path )
        img_name = os.path.splitext( img_name )[0]

        with open( os.path.join(save_dir, "labels", f"{img_name}.txt"), "w") as f:
            for label in labels:
                f.write(f"{label}\n")


if __name__ == "__main__":
    data_dir = "Data"
    img_paths, img_sizes, img_labels, boxes = extract_data_from_xml(data_dir)
    yolo_data = convert_to_yolo_format(image_paths= img_paths,
                                       image_sizes= img_sizes,
                                       bounding_boxes= boxes)
    
    seed = 0
    val_size = 0.2
    test_size = 0.125
    is_shuffle = True

    train_data, test_data = train_test_split(
        yolo_data,
        test_size=val_size,
        random_state=seed,
        shuffle=is_shuffle
    )
    test_data, val_data = train_test_split(
        test_data,
        test_size=test_size,
        random_state=seed,
        shuffle=is_shuffle
    )

    save_yolo_data_dir = "yolo_data"
    os.makedirs(save_yolo_data_dir, exist_ok=True )

    save_train_dir = os.path.join(save_yolo_data_dir, "train" )
    save_val_dir = os.path.join(save_yolo_data_dir, "val" )
    save_test_dir = os.path.join(save_yolo_data_dir, "test" )

    save_data(train_data, data_dir, save_train_dir)
    save_data(val_data, data_dir, save_val_dir)
    save_data(test_data, data_dir, save_test_dir)

    data_yml = {
        "path" : "./yolo_data",
        "train" : "train/images",
        "test" : "test/images",
        "val" : "val/images",
        "nc" : 1,
        "names" : ["text"]
    }

    yolo_yaml_path = os.path.join(save_yolo_data_dir, "data.yml")
    with open( yolo_yaml_path, "w") as f:
        yaml.dump(data_yml, f, default_flow_style=False)