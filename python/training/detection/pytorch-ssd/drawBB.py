import numpy as np
import pathlib
import cv2
import argparse
import pandas as pd
import copy
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str,help="location of dataset")

args = parser.parse_args()

def get_colours(num_classes):
    """
    given num_claseses, create a dictionary for drawing the colour
    """
    colour_dict = {}
    # assume RGB 3 colour scale
    COLOURS = np.random.randint(0, 255, [num_classes, 3])
    colour_dict = {i: tuple([np.asscalar(COLOURS[i][0]), np.asscalar(
        COLOURS[i][1]), np.asscalar(COLOURS[i][2])]) for i in list(range(num_classes))}
    return colour_dict


def get_class_names():
    class_names = ["ignored_regions", "pedestrian", "person", "bicycle", "car",
                   "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"]
    return class_names


def get_class_dict(class_names):
    return {class_name: i for i, class_name in enumerate(class_names)}


def _read_data(annotation_directory, img_directory):
    img_data = []
    for annotation_file in os.listdir(annotation_directory):
        df = pd.read_csv(os.path.join(annotation_directory, annotation_file),
                         names=["bbox_left", "bbox_top", "bbox_width", "bbox_height", "score", "object_category", "truncation", "occlusion"])

        #logging.info(f'annotations loaded from:  {annotation_file}')
        # image id is based on the file name which is similar to the image
        img_id = annotation_file.split(".txt")[0]
        img_path = os.path.join(img_directory, img_id + '.jpg')

        # check that the image exists in the repository
        if os.path.isfile(img_path) is False:
            logging.error(f'missing ImageID ' + img_id +
                          '.jpg - dropping from annotations')
            continue

        boxes = df.loc[:, ["bbox_left", "bbox_top", "bbox_width",
                           "bbox_height"]].values.astype(np.float32)
        labels = np.array(df["object_category"].values.tolist(), dtype=np.int64)
        img_data.append({
            'image_id': img_id,
            'boxes': boxes,
            'labels': labels
        })

    print(f"{len(img_data)} images read")

    return img_data


def draw_bb(img_path, imgs_data):
    """
    draw boundary box in boxes (2D array) for given img and its respective labels.
    data_img is a dictionary
    """

    for img_data in imgs_data:
        img = cv2.imread(os.path.join(img_path, img_data['image_id'] + '.jpg'))
        copy= img.copy() # create a copy
        class_names = get_class_names()
        COLOURS = get_colours(len(class_names))

        for i, (bbox_left, bbox_top, bbox_width, bbox_height) in enumerate(img_data['boxes']):
            #print(bbox_left, bbox_top, bbox_width, bbox_height)
            label = img_data['labels'][i]
            class_colour = COLOURS[label]
            #print(class_colour)
            # draw bb:
            cv2.rectangle(copy, (bbox_left, bbox_top), (bbox_left+bbox_width, bbox_top+bbox_height), class_colour, 2)

            #label class:
            cv2.putText(copy, str(class_names[label]), (bbox_left, bbox_top), cv2.FONT_HERSHEY_PLAIN, 1, class_colour, 2)

        print(f"{img_data['image_id']}: {i} bboxes drawn")

        cv2.imwrite(os.path.join(img_path, img_data['image_id'] + '_bbox.jpg'), copy)



if __name__ == '__main__':
    annotation_directory = f"{args.dataset_path}/annotations/"
    img_directory = f"{args.dataset_path}/images/"
    imgs_data = _read_data(annotation_directory, img_directory)
    draw_bb(img_directory, imgs_data)

